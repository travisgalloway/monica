"""Training-free long-context extension driver (Apple Silicon / MLX) — issue #54.

Measures perplexity vs sequence length for a model trained at `seq_len`, with the
LongMamba-style knob OFF (the baseline degradation past the training length) and ON
(`MambaConfig.long_ctx_factor`, the inference-time receptive-field enlargement). The
knob is training-free: the same weights are read under both settings — only the SSM
discretization step is rescaled at eval (see `SelectiveSSM._project`).

    # Real numbers want a trained checkpoint:
    .venv/bin/python scripts/long_context.py \\
        --config config/toy.yaml --weights run/weights.safetensors \\
        --data data/<toy split> --mults 1 2 4

    # Self-contained demo: briefly train a toy model, then sweep lengths:
    .venv/bin/python scripts/long_context.py \\
        --config config/toy.yaml --data data/<toy split> --train-steps 80 --mults 1 2 4

The harness (`src.eval.long_context`) is portable and unit-tested; this driver wires
the MLX model + the knob. MLX imports are local so `--help` works on any host.
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/toy.yaml"))
    ap.add_argument("--data", type=Path, required=True,
                    help="split dir with val.bin (and train.bin if --train-steps > 0)")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--train-steps", type=int, default=0,
                    help="if no --weights, train a toy checkpoint this many steps first")
    ap.add_argument("--mults", type=int, nargs="+", default=[1, 2, 4],
                    help="evaluate at these multiples of seq_len")
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.weights is None and args.train_steps < 1:
        ap.error("pass --weights <safetensors>, or --train-steps N to build a toy checkpoint")
    return args


def _train_toy_checkpoint(cfg, data_dir, steps, batch_size, lr, seed, out_path, mx):
    """Brief toy train so degradation/recovery is measurable (mirrors the prod wiring)."""
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


def main() -> None:
    args = _parse_args()
    try:
        import mlx.core as mx
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/long_context.py ...")

    import numpy as np
    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel
    from src.eval.long_context import long_context_eval, format_curve

    # Force the knob OFF for training and the OFF baseline regardless of what the YAML
    # sets — long_ctx_factor is an inference-time knob the ON path applies explicitly, so
    # a stray factor in the config must not leak into training or the baseline curve.
    cfg = dataclasses.replace(load_config(str(args.config)), long_ctx_factor=1.0)
    to_numpy = lambda a: np.array(a)
    val_path = args.data / "val.bin"
    print(f"[long-ctx] config={args.config}  seq_len={cfg.seq_len}  "
          f"d_model={cfg.d_model}  n_layers={cfg.n_layers}  mults={args.mults}")

    with tempfile.TemporaryDirectory() as tmp:
        weights_path = args.weights
        if weights_path is None:
            print(f"[long-ctx] no --weights; training a toy checkpoint ({args.train_steps} steps)")
            weights_path = _train_toy_checkpoint(
                cfg, args.data, args.train_steps, args.batch_size, args.lr,
                args.seed, Path(tmp) / "weights.safetensors", mx)

        # Knob OFF: factor 1.0 everywhere (baseline degradation past seq_len).
        base = MLXMambaModel(cfg)
        base.load(str(weights_path))
        off = long_context_eval(base, val_path, cfg.seq_len, args.batch_size,
                                 mults=args.mults, max_batches=args.max_batches,
                                 to_numpy=to_numpy)
        print(format_curve("knob OFF", off))

        # Knob ON: set the dt-scale to the extension ratio at each length. The model is
        # rebuilt per length with long_ctx_factor=mult and the SAME weights reloaded —
        # training-free (only the inference-time discretization step changes).
        print("\n[long-ctx] knob ON (long_ctx_factor = extension ratio per length):")
        on = {}
        for mult in args.mults:
            mcfg = dataclasses.replace(cfg, long_ctx_factor=float(mult))
            model = MLXMambaModel(mcfg)
            model.load(str(weights_path))
            res = long_context_eval(model, val_path, cfg.seq_len, args.batch_size,
                                    mults=[mult], max_batches=args.max_batches,
                                    to_numpy=to_numpy)
            on[mult] = res[mult]
        print(format_curve("knob ON ", on))

        # Recovery summary at the extended lengths.
        print("\n[result] perplexity at extended lengths (lower is better):")
        for mult in args.mults:
            if mult == 1 or off.get(mult) is None or on.get(mult) is None:
                continue
            po, pn = off[mult]["val_perplexity"], on[mult]["val_perplexity"]
            print(f"  {mult}x: OFF {po:.4f}  ON {pn:.4f}  "
                  f"({(po - pn) / po * 100:+.2f}% from the knob)")


if __name__ == "__main__":
    main()
