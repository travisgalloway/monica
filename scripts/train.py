"""Scale-run training driver (Apple Silicon / MLX) — the M5 entry point.

Wires the portable training loop (`src.train.loop.train`) to the MLX backend for a
real run on `config/poc.yaml`: model + AdamW + (dynamic fp16) loss scaling + grad
accumulation, JSONL metrics, periodic checkpoints, and held-out val-perplexity eval.
Resume is exact-from-checkpoint (portable weights + within-backend optimizer/step/
loss-scale bundle).

Success for the POC is a smoothly decreasing `val_perplexity` curve in the metrics
file (`<out>/metrics.jsonl`) with a stable `grad_norm` — not a benchmark score.

Data prep (run once, see config/poc.yaml + docs/design/04-data-pipeline.md) produces
`<data>/train.bin` and `<data>/val.bin`. Example scale run:

    .venv/bin/python scripts/train.py --config config/poc.yaml --data data/split \\
        --out runs/poc --total-tokens 3_000_000_000 --batch-size 32 --grad-accum 4

Resume after an interruption:

    .venv/bin/python scripts/train.py --config config/poc.yaml --data data/split \\
        --out runs/poc --total-tokens 3_000_000_000 --batch-size 32 --grad-accum 4 \\
        --resume runs/poc/resume

The backend (MLX or CUDA/PyTorch) is selected via `--backend {auto,mlx,cuda}`; backend
imports stay behind `src.model.backend.get_backend`, so the module stays importable for
`--help` on any host.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--data", type=Path, required=True, help="dir with train.bin/val.bin")
    ap.add_argument("--out", type=Path, default=Path("runs/poc"))
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto",
                    help="hardware backend (auto: try mlx, fall back to cuda/torch)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--total-steps", type=int, help="number of optimizer steps")
    g.add_argument("--total-tokens", type=int,
                   help="target tokens; steps = tokens // (batch*seq*grad_accum)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--base-lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="default: total_steps // 100 (min 1)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=50,
                    help="cap val batches per eval (None-like 0 = full val set)")
    ap.add_argument("--init-loss-scale", type=float, default=2.0 ** 13)
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume bundle dir; if omitted, auto-detects <out>/resume")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # Backend selection stays behind the seam factory; only portable modules are
    # imported at module scope.
    import numpy as np

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.data.loader import PackedLoader
    from src.train.loss_scale import scaler_for_precision
    from src.train.loop import TrainConfig, train
    from src.train.logging import JsonlLogger
    from src.train.checkpoint import save_resume, load_resume
    from src.eval.val_loss import evaluate

    backend = get_backend(args.backend)

    cfg = load_config(str(args.config))                 # validates (vocab < 65536, ...)

    tokens_per_step = args.batch_size * cfg.seq_len * args.grad_accum
    total_steps = (args.total_steps if args.total_steps is not None
                   else max(1, args.total_tokens // tokens_per_step))
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, total_steps // 100)

    # --- model + optimizer + (dynamic) loss scaling ----------------------------
    backend.seed(args.seed)
    model = backend.model_cls(cfg)
    opt = backend.make_optimizer(model, args.base_lr)
    scaler = scaler_for_precision(cfg.precision, args.init_loss_scale)
    if scaler is None and cfg.precision != "fp32":
        print(f"[info] precision={cfg.precision!r}: training unscaled "
              "(loss scaling is fp16-only; expected for bf16)")
    train_step = backend.make_train_step(model, opt, grad_clip=args.grad_clip, scaler=scaler)

    # --- data ------------------------------------------------------------------
    train_loader = PackedLoader(args.data / "train.bin", cfg.seq_len,
                                args.batch_size, shuffle=True, seed=args.seed)
    val_loader = PackedLoader(args.data / "val.bin", cfg.seq_len,
                              args.batch_size, shuffle=False, drop_last=False)
    np_to = backend.to_numpy
    max_b = args.eval_batches or None
    val_eval = lambda m: evaluate(m, val_loader, max_batches=max_b, to_numpy=np_to)

    # --- resume (portable weights + within-backend bundle) ---------------------
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    weights_path = str(out / "weights.safetensors")
    bundle_dir = str(out / "resume")
    resume_dir = args.resume if args.resume is not None else (
        Path(bundle_dir) if Path(bundle_dir, "resume_meta.json").exists() else None)

    start_step = 0
    if resume_dir is not None:
        model.load(weights_path)
        meta = load_resume(str(resume_dir),
                           optimizer_deserializer=lambda p: backend.load_optimizer(opt, p))
        start_step = int(meta["step"])
        if scaler is not None:
            scaler.load_state_dict(meta.get("rng_state") or {})
        print(f"[resume] from step {start_step} (out={out})")

    logger = JsonlLogger(str(out / "metrics.jsonl"), append=resume_dir is not None)

    def on_checkpoint(step: int) -> None:
        model.save(weights_path)                        # portable weights + config
        save_resume(bundle_dir, step=step,
                    rng_state=(scaler.state_dict() if scaler else None),
                    optimizer_serializer=lambda p: backend.save_optimizer(opt, p))

    tcfg = TrainConfig(
        total_steps=total_steps, base_lr=args.base_lr, warmup_steps=warmup,
        grad_accum=args.grad_accum, grad_clip=args.grad_clip,
        log_every=args.log_every, eval_every=args.eval_every,
        ckpt_every=args.ckpt_every, out_dir=str(out), seed=args.seed,
    )

    n_params = sum(int(np.asarray(v).size) for _, v in model._portable_state_dict().items())
    print(f"[run] params~{n_params/1e6:.1f}M  total_steps={total_steps}  warmup={warmup}  "
          f"tokens/step={tokens_per_step}  precision={cfg.precision}")

    train(model, train_loader, tcfg, train_step,
          val_eval=val_eval, logger=logger, on_checkpoint=on_checkpoint,
          start_step=start_step)

    # --- final checkpoint + full eval ------------------------------------------
    on_checkpoint(total_steps)
    final = evaluate(model, val_loader, max_batches=max_b, to_numpy=np_to)
    logger.close()
    print(f"[done] step={total_steps}  val_loss={final['val_loss']:.4f}  "
          f"val_perplexity={final['val_perplexity']:.4f}  metrics={out / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
