"""Self-speculative decoding spike (Apple Silicon / MLX) — issue #52.

Draft-and-verify decoding over the model's `step` recurrence. A prompt-lookup drafter
(`src.serve.spec_decode.propose`, no second model) proposes the next few tokens; the
verifier scores them in ONE batched eval (`MLXMambaModel.verify_block`) and accepts the
greedy-matching prefix. Greedy verification makes the output BYTE-IDENTICAL to plain
greedy decoding — the spike asserts that — while the batched verify amortizes the
per-token sync that bounds batch-1 SSM decode (the #30 ~94.7 tok/s record).

    # Self-contained demo on a toy split:
    .venv/bin/python scripts/spec_decode.py \\
        --config config/toy.yaml --data data/<toy split> --train-steps 200 \\
        --max-new 256 --gamma 4

    # Real model:
    .venv/bin/python scripts/spec_decode.py --config config/toy.yaml \\
        --weights run/weights.safetensors --data data/<toy split> --max-new 256

Numbers (tokens/s plain vs speculative, acceptance rate) post to #30. MLX imports are
local so `--help` works on any host.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/toy.yaml"))
    ap.add_argument("--data", type=Path, required=True,
                    help="split dir; val.bin seeds the prompt, train.bin if --train-steps")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--gamma", type=int, default=4, help="draft length per round")
    ap.add_argument("--max-n", type=int, default=8, help="longest prompt-lookup pattern")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.weights is None and args.train_steps < 1:
        ap.error("pass --weights <safetensors>, or --train-steps N to build a toy checkpoint")
    if args.gamma < 1:
        ap.error("--gamma must be >= 1")
    return args


def _train_toy_checkpoint(cfg, data_dir, steps, batch_size, lr, seed, mx):
    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_train_step import make_train_step
    from src.train.loss_scale import scaler_for_precision
    from src.data.loader import PackedLoader
    import mlx.optimizers as optim

    mx.random.seed(seed)
    model = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=lr)
    train_step = make_train_step(model, opt, grad_clip=1.0,
                                 scaler=scaler_for_precision(cfg.precision))
    loader = PackedLoader(data_dir / "train.bin", cfg.seq_len, batch_size,
                          shuffle=True, drop_last=True)
    done = 0
    while done < steps:
        for inp, tgt in loader.epoch():
            out = train_step(model, [(inp, tgt)], lr)
            done += 1
            if done % 50 == 0 or done == steps:
                print(f"  [train] step {done}/{steps}  loss {out['loss']:.4f}")
            if done >= steps:
                break
    return model


def _argmax(mx_logits, mx) -> int:
    return int(mx.argmax(mx_logits[0]).item())


def _prefill(model, prompt, mx):
    """Feed the prompt through `step`; return (next-token logits, state)."""
    state = model.init_state(1)
    logits = None
    for tok in prompt:
        logits, state = model.step(mx.array([int(tok)]), state)
    mx.eval(logits)
    return logits, state


def plain_decode(model, prompt, max_new, mx):
    """Greedy one-step-at-a-time decoding (the baseline)."""
    logits, state = _prefill(model, prompt, mx)
    generated = []
    t0 = time.perf_counter()
    for _ in range(max_new):
        x = _argmax(logits, mx)
        generated.append(x)
        logits, state = model.step(mx.array([x]), state)
        mx.eval(logits)
    return generated, time.perf_counter() - t0


def spec_decode(model, prompt, max_new, gamma, max_n, mx):
    """Greedy self-speculative decoding. Identical output to `plain_decode`."""
    from src.serve.spec_decode import first_mismatch, propose

    logits, state = _prefill(model, prompt, mx)
    context = [int(t) for t in prompt]
    generated, drafted, accepted, rounds = [], 0, 0, 0

    t0 = time.perf_counter()
    while len(generated) < max_new:
        remaining = max_new - len(generated)
        draft = propose(context, min(gamma, remaining), max_n)
        if not draft:
            # No tail recurs — take one ordinary verifier step.
            x = _argmax(logits, mx)
            logits, state = model.step(mx.array([x]), state)
            mx.eval(logits)
            generated.append(x)
            context.append(x)
            continue

        block_logits, block_states = model.verify_block(draft, state)   # one eval
        # verifier greedy token at each draft position given the accepted prefix
        preds = [_argmax(logits, mx)] + [_argmax(bl, mx) for bl in block_logits[:-1]]
        m = first_mismatch(draft, preds)            # accepted count in [0, len(draft)]

        # Roll back to the accepted prefix, then emit the verifier's own token there
        # (a correction if m < γ, the free bonus token if m == γ).
        base_logits = logits if m == 0 else block_logits[m - 1]
        base_state = state if m == 0 else block_states[m - 1]
        x = _argmax(base_logits, mx)
        emit = draft[:m] + [x]

        logits, state = model.step(mx.array([x]), base_state)
        mx.eval(logits)
        generated.extend(emit)
        context.extend(emit)
        drafted += len(draft)
        accepted += m
        rounds += 1

    elapsed = time.perf_counter() - t0
    stats = {
        "rounds": rounds, "drafted": drafted, "accepted": accepted,
        "accept_rate": (accepted / drafted) if drafted else 0.0,
        "tokens_per_round": (len(generated) / rounds) if rounds else 0.0,
    }
    return generated[:max_new], elapsed, stats


def main() -> None:
    args = _parse_args()
    try:
        import mlx.core as mx
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/spec_decode.py ...")

    import numpy as np
    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel
    from src.data.loader import PackedLoader

    cfg = load_config(str(args.config))
    print(f"[spec] config={args.config}  d_model={cfg.d_model}  n_layers={cfg.n_layers}  "
          f"vocab={cfg.vocab_size}  gamma={args.gamma}  max_new={args.max_new}")

    if args.weights is not None:
        model = MLXMambaModel(cfg)
        model.load(str(args.weights))
    else:
        print(f"[spec] no --weights; training a toy checkpoint ({args.train_steps} steps)")
        model = _train_toy_checkpoint(cfg, args.data, args.train_steps,
                                      args.batch_size, args.lr, args.seed, mx)
    mx.eval(model.parameters())

    # Prompt from real held-out tokens so the continuation is structured (the drafter
    # needs recurring n-grams to match).
    loader = PackedLoader(args.data / "val.bin", args.prompt_len, 1,
                          shuffle=False, drop_last=False)
    inputs, _ = next(iter(loader.epoch()))
    prompt = [int(t) for t in inputs[0]]

    plain, t_plain = plain_decode(model, prompt, args.max_new, mx)
    spec, t_spec, stats = spec_decode(model, prompt, args.max_new, args.gamma, args.max_n, mx)

    identical = plain == spec
    print(f"\n[correctness] speculative == plain greedy: {identical}")
    if not identical:
        first = next(i for i, (a, b) in enumerate(zip(plain, spec)) if a != b)
        raise SystemExit(f"SPEC DECODE MISMATCH at token {first}: plain={plain[first]} "
                         f"spec={spec[first]} — verification is NOT distribution-preserving")

    print(f"[accept]   {stats['accepted']}/{stats['drafted']} draft tokens accepted "
          f"({stats['accept_rate']*100:.1f}%); {stats['tokens_per_round']:.2f} tokens/round "
          f"over {stats['rounds']} rounds")
    print(f"[plain]    {len(plain)/t_plain:>8.1f} tokens/s   ({t_plain:.3f}s)")
    print(f"[spec]     {len(spec)/t_spec:>8.1f} tokens/s   ({t_spec:.3f}s)")
    print(f"[result]   speculative is {t_plain/t_spec:.2f}x plain decode wall-clock "
          f"(identical output)")


if __name__ == "__main__":
    main()
