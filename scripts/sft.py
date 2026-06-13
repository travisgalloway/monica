"""SFT driver (M9, Apple Silicon / MLX): instruction-tune the pretrained base.

Loads the pretrained POC weights as initialization (NOT a resume bundle — fresh AdamW),
then trains on response-masked instruction data (`HuggingFaceH4/no_robots`) with the
shared training loop: masked cross-entropy (`make_sft_train_step`), grad accumulation,
dynamic fp16 loss scaling, JSONL metrics, periodic checkpoints, and held-out masked
val-perplexity. Success is a falling masked `val_perplexity` and cleaner on-format chat
replies — not a benchmark score (100M ceiling).

Prep the data once (writes JSONL of response-masked records):

    .venv/bin/python -m src.data.sft_data --split train --out data/sft/train.jsonl
    .venv/bin/python -m src.data.sft_data --split test  --out data/sft/val.jsonl

Then fine-tune (init from the pretrained base):

    .venv/bin/python scripts/sft.py --config config/poc.yaml --data data/sft \\
        --init runs/poc/weights.safetensors --out runs/sft \\
        --epochs 2 --batch-size 8 --grad-accum 16 --base-lr 2e-5

Resume after an interruption (auto-detects <out>/resume):

    .venv/bin/python scripts/sft.py --config config/poc.yaml --data data/sft \\
        --out runs/sft --epochs 2 --batch-size 8 --grad-accum 16 --base-lr 2e-5
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--data", type=Path, required=True,
                    help="dir with train.jsonl / val.jsonl (from src.data.sft_data)")
    ap.add_argument("--out", type=Path, default=Path("runs/sft"))
    ap.add_argument("--init", type=Path, default=Path("runs/poc/weights.safetensors"),
                    help="pretrained base weights to initialize from (fresh run only)")
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto")
    ap.add_argument("--epochs", type=int, default=2, help="passes over the SFT set")
    ap.add_argument("--total-steps", type=int, default=None,
                    help="override the epoch-derived optimizer-step count")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--base-lr", type=float, default=2e-5, help="low: ~1e-5..5e-5")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="default: total_steps // 20 (min 1)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=30,
                    help="cap val batches per eval (0 = full val set)")
    ap.add_argument("--init-loss-scale", type=float, default=2.0 ** 13)
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume bundle dir; if omitted, auto-detects <out>/resume")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    import numpy as np

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.data.sft_loader import SFTLoader
    from src.train.loss_scale import scaler_for_precision
    from src.train.loop import TrainConfig, train
    from src.train.logging import JsonlLogger
    from src.train.checkpoint import save_resume, load_resume
    from src.eval.val_loss import evaluate_masked

    backend = get_backend(args.backend)
    cfg = load_config(str(args.config))                 # validates (vocab < 65536, ...)

    # --- data (response-masked instruction records) ----------------------------
    train_loader = SFTLoader(args.data / "train.jsonl", cfg.seq_len, args.batch_size,
                             shuffle=True, seed=args.seed, vocab_size=cfg.vocab_size)
    val_loader = SFTLoader(args.data / "val.jsonl", cfg.seq_len, args.batch_size,
                           shuffle=False, drop_last=False, vocab_size=cfg.vocab_size)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = args.total_steps or args.epochs * steps_per_epoch
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, total_steps // 20)

    # --- model + optimizer + (dynamic) loss scaling ----------------------------
    backend.seed(args.seed)
    model = backend.model_cls(cfg)
    opt = backend.make_optimizer(model, args.base_lr)
    scaler = scaler_for_precision(cfg.precision, args.init_loss_scale)
    train_step = backend.make_sft_train_step(model, opt, grad_clip=args.grad_clip,
                                             scaler=scaler)

    np_to = backend.to_numpy
    max_b = args.eval_batches or None
    val_eval = lambda m: evaluate_masked(m, val_loader, max_batches=max_b, to_numpy=np_to)

    # --- init / resume ---------------------------------------------------------
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    weights_path = str(out / "weights.safetensors")
    bundle_dir = str(out / "resume")
    resume_dir = args.resume if args.resume is not None else (
        Path(bundle_dir) if Path(bundle_dir, "resume_meta.json").exists() else None)

    start_step = 0
    if resume_dir is not None:
        model.load(weights_path)                         # resume from SFT checkpoint
        meta = load_resume(str(resume_dir),
                           optimizer_deserializer=lambda p: backend.load_optimizer(opt, p))
        start_step = int(meta["step"])
        if scaler is not None:
            scaler.load_state_dict(meta.get("rng_state") or {})
        print(f"[resume] from step {start_step} (out={out})")
    else:
        model.load(str(args.init))                       # initialize from pretrained base
        print(f"[init] from pretrained base {args.init}")

    logger = JsonlLogger(str(out / "metrics.jsonl"), append=resume_dir is not None)

    def on_checkpoint(step: int) -> None:
        model.save(weights_path)                         # portable weights + config
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
    print(f"[sft] params~{n_params/1e6:.1f}M  examples={len(train_loader.records)}  "
          f"total_steps={total_steps}  warmup={warmup}  precision={cfg.precision}")

    train(model, train_loader, tcfg, train_step,
          val_eval=val_eval, logger=logger, on_checkpoint=on_checkpoint,
          start_step=start_step)

    on_checkpoint(total_steps)
    final = evaluate_masked(model, val_loader, max_batches=max_b, to_numpy=np_to)
    logger.close()
    print(f"[done] step={total_steps}  val_loss={final['val_loss']:.4f}  "
          f"val_perplexity={final['val_perplexity']:.4f}  metrics={out / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
