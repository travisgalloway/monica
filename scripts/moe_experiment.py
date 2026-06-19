"""MoE-Mamba capacity experiment (Apple Silicon / MLX) — issue #53.

Trains a dense (pure-Mamba) baseline and a sparse-MoE variant on the same toy split for
the same steps, then reports held-out loss alongside each model's TOTAL params (capacity)
and ACTIVE params per token (a FLOP proxy: MoE counts only the top_k routed experts). The
question MoE-Mamba/ME-Mamba pose: does sparse routing buy better loss-vs-active-FLOPs?

    .venv/bin/python scripts/moe_experiment.py \\
        --data data/<toy split> --moe-config config/toy-moe.yaml \\
        --dense-config config/toy.yaml --steps 300

The dense and MoE configs should share dims; only the MoE block placement differs. MLX
imports are local so `--help` works on any host. Numbers post to #53.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="split dir (train.bin, val.bin)")
    ap.add_argument("--moe-config", type=Path, default=Path("config/toy-moe.yaml"))
    ap.add_argument("--dense-config", type=Path, default=None,
                    help="dense baseline config; default = the MoE config's pure-Mamba "
                         "twin (same dims, moe_every=None) for a matched-depth comparison")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _train_and_eval(cfg, args, mx, np):
    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_train_step import make_train_step
    from src.train.loss_scale import scaler_for_precision
    from src.data.loader import PackedLoader
    from src.eval.val_loss import evaluate
    import mlx.optimizers as optim

    mx.random.seed(args.seed)
    model = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=args.lr)
    train_step = make_train_step(model, opt, grad_clip=1.0,
                                 scaler=scaler_for_precision(cfg.precision))
    loader = PackedLoader(args.data / "train.bin", cfg.seq_len, args.batch_size,
                          shuffle=True, drop_last=True)
    done = 0
    while done < args.steps:
        for inp, tgt in loader.epoch():
            train_step(model, [(inp, tgt)], args.lr)
            done += 1
            if done >= args.steps:
                break
    val_loader = PackedLoader(args.data / "val.bin", cfg.seq_len, args.batch_size,
                              shuffle=False, drop_last=False)
    res = evaluate(model, val_loader, max_batches=args.max_batches,
                   to_numpy=lambda a: np.array(a))
    return res


def main() -> None:
    args = _parse_args()
    try:
        import mlx.core as mx
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/moe_experiment.py ...")

    import dataclasses
    import numpy as np
    from src.model.blocks import load_config

    moe_cfg = load_config(str(args.moe_config))
    # Default dense baseline: the SAME config with MoE off (pure-Mamba twin) — identical
    # depth/dims, so the comparison isolates the MoE block rather than confounding depth.
    dense_cfg = (load_config(str(args.dense_config)) if args.dense_config is not None
                 else dataclasses.replace(moe_cfg, moe_every=None, n_experts=0))
    dense_cfg.validate()

    rows = []
    for label, cfg in (("dense", dense_cfg), ("moe", moe_cfg)):
        print(f"[moe-exp] training {label} "
              f"(n_layers={cfg.n_layers}, n_moe_layers={cfg.n_moe_layers}, "
              f"experts={cfg.n_experts} top_k={cfg.top_k})")
        res = _train_and_eval(cfg, args, mx, np)
        rows.append((label, cfg, res))
        print(f"[moe-exp] {label}: val_loss={res['val_loss']:.4f}  "
              f"val_perplexity={res['val_perplexity']:.4f}")

    print(f"\n[result] after {args.steps} steps (lower loss / fewer active params is better):")
    print(f"  {'model':<6} {'val_loss':>9} {'perplexity':>11} "
          f"{'total params':>13} {'active params':>14}")
    for label, cfg, res in rows:
        print(f"  {label:<6} {res['val_loss']:>9.4f} {res['val_perplexity']:>11.4f} "
              f"{cfg.num_parameters():>13,} {cfg.active_num_parameters():>14,}")

    dense, moe = rows[0][2], rows[1][2]
    d_total, m_total = rows[0][1].num_parameters(), rows[1][1].num_parameters()
    d_act, m_act = rows[0][1].active_num_parameters(), rows[1][1].active_num_parameters()
    print(f"\n[result] MoE adds {m_total - d_total:+,} total params "
          f"({m_total / d_total:.2f}x capacity) for {m_act - d_act:+,} active params "
          f"({m_act / d_act:.2f}x active compute); "
          f"val_loss {dense['val_loss']:.4f} (dense) -> {moe['val_loss']:.4f} (moe).")


if __name__ == "__main__":
    main()
