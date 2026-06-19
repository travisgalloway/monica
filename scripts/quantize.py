"""Post-training quantization measurement driver (Apple Silicon / MLX) — issue #51.

Quantizes the heavy weights of a portable checkpoint with group-wise affine W8/W4
(`src/eval/quantize.py`), then reports the held-out perplexity delta and the
model-size delta — the #65 Phase-4 "quantize" numbers. Weight-only (activations stay
float); true activation quant (the "A8" in W8A8) is not done here.

    # Real numbers need a trained checkpoint (a smoke/poc run's portable weights):
    .venv/bin/python scripts/quantize.py \\
        --config config/toy.yaml --weights run/weights.safetensors \\
        --data data/<toy split> --bits 8

    # Self-contained demo: briefly train a toy model on the split, then quantize it:
    .venv/bin/python scripts/quantize.py \\
        --config config/toy.yaml --data data/<toy split> --train-steps 60 --bits 8

The portable numeric core (`src.eval.quantize`) carries the actual quantization math
and is unit-tested without a backend; this driver only wires it to a real model + the
real eval path. MLX imports are kept local so `--help` works on any host.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/toy.yaml"))
    ap.add_argument("--data", type=Path, required=True,
                    help="split dir with val.bin (and train.bin if --train-steps > 0)")
    ap.add_argument("--weights", type=Path, default=None,
                    help="portable safetensors to quantize (omit with --train-steps)")
    ap.add_argument("--train-steps", type=int, default=0,
                    help="if no --weights, train a toy checkpoint this many steps first")
    ap.add_argument("--bits", type=int, default=8, help="weight bit-width (8, then 4 as stretch)")
    ap.add_argument("--group-size", type=int, default=64, help="affine group size along the last axis")
    ap.add_argument("--symmetric", action="store_true", help="symmetric (zero-point-free) quant")
    ap.add_argument("--max-batches", type=int, default=8, help="held-out batches per eval")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.weights is None and args.train_steps < 1:
        ap.error("pass --weights <safetensors>, or --train-steps N to build a toy checkpoint")
    if args.bits < 1 or args.bits > 16:
        ap.error("--bits must be in [1, 16]")
    return args


def _train_toy_checkpoint(cfg, data_dir, steps, batch_size, lr, seed, out_path, mx):
    """Briefly train a toy model on the split so the perplexity delta is meaningful
    (a random-init model sits at ~uniform and the delta says nothing). Mirrors the
    production wiring (model + AdamW + make_train_step), kept tiny on purpose."""
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
            if done % 20 == 0 or done == steps:
                print(f"  [train] step {done}/{steps}  loss {out['loss']:.4f}")
            if done >= steps:
                break
    model.save(str(out_path))
    return out_path


def _eval_ppl(model, data_dir, batch_size, max_batches, to_numpy):
    from src.data.loader import PackedLoader
    from src.eval.val_loss import evaluate
    loader = PackedLoader(data_dir / "val.bin", model.config.seq_len, batch_size,
                          shuffle=False, drop_last=False)
    return evaluate(model, loader, max_batches=max_batches, to_numpy=to_numpy)


def main() -> None:
    args = _parse_args()
    try:
        import mlx.core as mx
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/quantize.py ...")

    import numpy as np
    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel
    from src.train.checkpoint import load_weights_dict, save_weights
    from src.eval.quantize import quantize_state_dict

    cfg = load_config(str(args.config))
    to_numpy = lambda a: np.array(a)
    print(f"[quantize] config={args.config}  d_model={cfg.d_model}  n_layers={cfg.n_layers}  "
          f"vocab={cfg.vocab_size}  W{args.bits} g{args.group_size}"
          f"{' symmetric' if args.symmetric else ''}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1) obtain a checkpoint (given, or a quick toy train so ppl is meaningful)
        weights_path = args.weights
        if weights_path is None:
            print(f"[quantize] no --weights; training a toy checkpoint ({args.train_steps} steps)")
            weights_path = _train_toy_checkpoint(
                cfg, args.data, args.train_steps, args.batch_size, args.lr,
                args.seed, tmp / "weights.safetensors", mx)

        # 2) baseline perplexity
        base = MLXMambaModel(cfg)
        base.load(str(weights_path))
        base_eval = _eval_ppl(base, args.data, args.batch_size, args.max_batches, to_numpy)
        print(f"[baseline] val_loss={base_eval['val_loss']:.4f}  "
              f"val_perplexity={base_eval['val_perplexity']:.4f}")

        # 3) quantize the portable weights (fake quant) + size accounting
        sd = load_weights_dict(str(weights_path))
        qsd, report = quantize_state_dict(sd, args.bits, args.group_size, args.symmetric)
        print(f"[quantize] quantized {report['n_quantized']}/{report['n_total']} tensors; "
              f"targeted {report['quantized_orig_bytes']/2**20:.3f}→"
              f"{report['quantized_packed_bytes']/2**20:.3f} MiB "
              f"({report['quantized_compression']:.2f}x); "
              f"whole model {report['model_compression']:.2f}x "
              f"(fp16 baseline)")
        worst = max(report["per_tensor"], key=lambda t: t["rel_rms"], default=None)
        if worst:
            print(f"[quantize] worst rel_rms error: {worst['name']} "
                  f"rel_rms={worst['rel_rms']:.4f} rel_max={worst['rel_max']:.4f}")

        # 4) reload the quantized weights through the portable bridge + re-eval
        qpath = tmp / "weights.q.safetensors"
        save_weights(qsd, str(qpath), config=cfg)
        qmodel = MLXMambaModel(cfg)
        qmodel.load(str(qpath))                       # proves the bridge round-trips
        q_eval = _eval_ppl(qmodel, args.data, args.batch_size, args.max_batches, to_numpy)
        print(f"[quantized] val_loss={q_eval['val_loss']:.4f}  "
              f"val_perplexity={q_eval['val_perplexity']:.4f}")

        d_ppl = q_eval["val_perplexity"] - base_eval["val_perplexity"]
        rel = d_ppl / base_eval["val_perplexity"] * 100 if base_eval["val_perplexity"] else 0.0
        print(f"\n[result] W{args.bits} g{args.group_size}: "
              f"Δperplexity {d_ppl:+.4f} ({rel:+.2f}%)  |  "
              f"model size {report['model_compression']:.2f}x smaller "
              f"({report['model_orig_bytes']/2**20:.2f}→"
              f"{report['model_packed_bytes']/2**20:.2f} MiB at fp16 baseline)")


if __name__ == "__main__":
    main()
