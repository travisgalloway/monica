"""Production train-step benchmark (Apple Silicon / MLX) — the harness for issue #31.

Measures the *production* training path — `MLXMambaModel` + `optim.AdamW` +
`scaler_for_precision` + the real `make_train_step` closure, including the per-step
`.item()` overflow sync — so the optimization spike (#30) has trustworthy before/after
numbers. Every lever in the spike is judged against this harness; run it at the
standard protocol before and after each change:

    .venv/bin/python scripts/bench_train_step.py \\
        --config config/poc.yaml --batch 32 --grad-accum 4 --warmup 3 --iters 10

Two modes:
  * `--mode train` (default): times `train_step(model, micro_batches, lr)` exactly as
    `scripts/train.py` wires it. Reports s/step (mean and min), tokens/s
    (= batch x seq x grad_accum / s_step), peak GPU memory, and the first call
    separately (it will capture compile latency once #32 lands).
  * `--mode decode`: batch-1 `init_state` + `model.step` loop — a tokens/s BASELINE
    RECORD for future M7 only. Do not optimize the decode path in this spike
    (see the rejected levers in #30).

`--chunk-size` overrides the SSD scan chunk length Q (config default 64) via
`dataclasses.replace` on the loaded config — a free experiment, no model change.

MLX imports are kept local so the module stays importable for `--help` on any host.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import platform
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--mode", choices=("train", "decode"), default="train",
                    help="train: production train_step (default). "
                         "decode: batch-1 step-loop tokens/s baseline (M7 record only).")
    ap.add_argument("--batch", type=int, default=32, help="sequences per micro-batch")
    ap.add_argument("--seq", type=int, default=None, help="sequence length (default: config seq_len)")
    ap.add_argument("--grad-accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warmup steps (first one reported separately)")
    ap.add_argument("--iters", type=int, default=10, help="timed optimizer steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate (matches train.py --base-lr default)")
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="override SSD scan chunk length Q (default: config value, else 64)")
    args = ap.parse_args()
    # Fail fast on inputs that would crash or report nonsense (same as bench_precision).
    if args.iters < 1:
        ap.error("--iters must be >= 1")
    if args.warmup < 1:
        ap.error("--warmup must be >= 1 (the first call is always run untimed)")
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    if args.grad_accum < 1:
        ap.error("--grad-accum must be >= 1")
    if args.seq is not None and args.seq < 1:
        ap.error("--seq must be >= 1")
    if args.chunk_size is not None and args.chunk_size < 1:
        ap.error("--chunk-size must be >= 1")
    return args


def _bench_train(args, cfg, mx) -> None:
    """Time the production train step: model + AdamW + scaler + make_train_step."""
    import mlx.optimizers as optim
    import numpy as np

    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_train_step import make_train_step
    from src.train.loss_scale import scaler_for_precision

    seq = args.seq if args.seq is not None else cfg.seq_len
    tokens_per_step = args.batch * seq * args.grad_accum

    # The exact wiring of scripts/train.py — this measures the production path.
    mx.random.seed(args.seed)
    model = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=args.lr)
    scaler = scaler_for_precision(cfg.precision)
    train_step = make_train_step(model, opt, grad_clip=1.0, scaler=scaler)

    rng = np.random.default_rng(args.seed)
    micro_batches = [
        (mx.array(rng.integers(0, cfg.vocab_size, size=(args.batch, seq)).astype(np.int32)),
         mx.array(rng.integers(0, cfg.vocab_size, size=(args.batch, seq)).astype(np.int32)))
        for _ in range(args.grad_accum)
    ]

    # First call timed separately — it includes weight-init eval today and will
    # capture compile latency once #32 lands.
    t0 = time.perf_counter()
    out = train_step(model, micro_batches, args.lr)
    first_call_s = time.perf_counter() - t0
    for _ in range(args.warmup - 1):
        out = train_step(model, micro_batches, args.lr)

    mx.reset_peak_memory()
    times = []
    skipped = 0
    for _ in range(args.iters):
        t0 = time.perf_counter()
        out = train_step(model, micro_batches, args.lr)
        times.append(time.perf_counter() - t0)
        skipped += int(out.get("skipped", False))
    peak_gb = mx.get_peak_memory() / 2**30

    if not math.isfinite(out["loss"]):
        raise RuntimeError("non-finite loss — run is degenerate")

    mean_s = sum(times) / len(times)
    min_s = min(times)
    print(f"first call      {first_call_s:>10.3f} s   (init/compile, excluded from stats)")
    print(f"s/step          {mean_s:>10.3f} mean   {min_s:.3f} min   over {args.iters} iters")
    print(f"tokens/s        {tokens_per_step / mean_s:>10,.0f}   ({tokens_per_step} tokens/step)")
    print(f"peak memory     {peak_gb:>10.2f} GB")
    line = f"loss {out['loss']:.4f}   grad_norm {out['grad_norm']:.4f}"
    if scaler:
        line += f"   loss_scale {out['loss_scale']:.0f}   skipped {skipped}/{args.iters}"
    print(line)


def _bench_decode(args, cfg, mx) -> None:
    """Batch-1 decode tokens/s — a baseline record for future M7, nothing more."""
    import numpy as np

    from src.model.mlx_backend import MLXMambaModel

    warmup_tokens, measure_tokens = 32, 256
    mx.random.seed(args.seed)
    model = MLXMambaModel(cfg)
    mx.eval(model.parameters())

    rng = np.random.default_rng(args.seed)
    tokens = rng.integers(0, cfg.vocab_size,
                          size=warmup_tokens + measure_tokens).astype(np.int32)

    state = model.init_state(1)
    last = None
    for i in range(warmup_tokens):
        last, state = model.step(mx.array(tokens[i:i + 1]), state)
        mx.eval(last, state)

    t0 = time.perf_counter()
    for i in range(warmup_tokens, warmup_tokens + measure_tokens):
        last, state = model.step(mx.array(tokens[i:i + 1]), state)
        mx.eval(last, state)
    elapsed = time.perf_counter() - t0

    if not bool(mx.all(mx.isfinite(last)).item()):
        raise RuntimeError("non-finite logits — run is degenerate")

    print(f"decode          {measure_tokens / elapsed:>10.1f} tokens/s   "
          f"(batch 1, {measure_tokens} tokens, {warmup_tokens} warmup)")
    print("[note] M7 baseline record only — the decode path is NOT an optimization "
          "target in this spike (#30).")


def main() -> None:
    args = _parse_args()

    # MLX-only imports kept local so the seam stays clean for portable hosts.
    try:
        import mlx.core as mx
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/bench_train_step.py ...\n"
            "(mlx installs only on Apple Silicon via the '[mlx]' extra; a bare "
            "`python` likely points at a different interpreter.)"
        ) from e

    from src.model.blocks import load_config

    cfg = load_config(str(args.config))
    if args.chunk_size is not None:
        cfg = dataclasses.replace(cfg, chunk_size=args.chunk_size)
        cfg.validate()

    seq = args.seq if args.seq is not None else cfg.seq_len
    print(f"[bench/{args.mode}] mlx {mx.__version__}  {platform.platform()}")
    print(f"[bench/{args.mode}] config={args.config}  d_model={cfg.d_model}  "
          f"d_inner={cfg.d_inner}  n_layers={cfg.n_layers}  vocab={cfg.vocab_size}  "
          f"precision={cfg.precision}  grad_checkpoint={cfg.grad_checkpoint}  "
          f"chunk_size={cfg.chunk_size or 64}")
    if args.mode == "train":
        print(f"[bench/train] batch={args.batch} x seq={seq} x grad_accum={args.grad_accum} "
              f"= {args.batch * seq * args.grad_accum} tokens/step  "
              f"iters={args.iters} (+{args.warmup} warmup)\n")
        _bench_train(args, cfg, mx)
    else:
        print()
        _bench_decode(args, cfg, mx)


if __name__ == "__main__":
    main()
