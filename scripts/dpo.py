"""DPO driver (M9, Apple Silicon / MLX): preference-align the SFT model.

Loads the SFT weights as BOTH the policy initialization and the frozen reference, then
trains on preference pairs (`HuggingFaceH4/ultrafeedback_binarized`) with the DPO loss
(`make_dpo_train_step`). The reference never updates (gradients flow through the policy
only). The primary health signal is a rising chosen-minus-rejected reward margin, logged
via held-out `evaluate_dpo` at eval cadence.

Memory note: policy + reference are both resident (~100M params each) and each step runs
four forwards (policy/ref x chosen/rejected), so use a small --batch-size and lean on
--grad-accum. grad_checkpoint (poc.yaml) bounds the activation memory.

Prep the data once:

    .venv/bin/python -m src.data.dpo_data --split train_prefs --out data/dpo/train.jsonl \\
        --max-examples 8000
    .venv/bin/python -m src.data.dpo_data --split test_prefs --out data/dpo/val.jsonl \\
        --max-examples 500

Then align (init + reference from the SFT checkpoint):

    .venv/bin/python scripts/dpo.py --config config/poc.yaml --data data/dpo \\
        --init runs/sft/weights.safetensors --out runs/dpo \\
        --epochs 1 --batch-size 2 --grad-accum 8 --beta 0.1 --base-lr 5e-7
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--data", type=Path, required=True,
                    help="dir with train.jsonl / val.jsonl (from src.data.dpo_data)")
    ap.add_argument("--out", type=Path, default=Path("runs/dpo"))
    ap.add_argument("--init", type=Path, default=Path("runs/sft/weights.safetensors"),
                    help="SFT weights — used as policy init AND the frozen reference")
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2, help="small: policy+ref resident")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO KL strength")
    ap.add_argument("--base-lr", type=float, default=5e-7,
                    help="much lower than SFT (~5e-7..1e-6)")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="default: total_steps // 20 (min 1)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=30)
    ap.add_argument("--init-loss-scale", type=float, default=2.0 ** 13)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    import numpy as np

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.data.dpo_loader import DPOLoader
    from src.train.loss_scale import scaler_for_precision
    from src.train.loop import TrainConfig, train
    from src.train.logging import JsonlLogger
    from src.train.checkpoint import CheckpointStore
    from src.train.dpo_math import evaluate_dpo

    backend = get_backend(args.backend)
    cfg = load_config(str(args.config))

    train_loader = DPOLoader(args.data / "train.jsonl", cfg.seq_len, args.batch_size,
                             shuffle=True, seed=args.seed, vocab_size=cfg.vocab_size)
    val_loader = DPOLoader(args.data / "val.jsonl", cfg.seq_len, args.batch_size,
                           shuffle=False, drop_last=False, vocab_size=cfg.vocab_size)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = args.total_steps or args.epochs * steps_per_epoch
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, total_steps // 20)

    # --- policy + frozen reference (both from the SFT weights) ------------------
    backend.seed(args.seed)
    policy = backend.model_cls(cfg)
    ref = backend.model_cls(cfg)
    ref.load(str(args.init))                             # frozen reference (never updated)
    opt = backend.make_optimizer(policy, args.base_lr)
    scaler = scaler_for_precision(cfg.precision, args.init_loss_scale)
    train_step = backend.make_dpo_train_step(policy, ref, opt, beta=args.beta,
                                             grad_clip=args.grad_clip, scaler=scaler)

    np_to = backend.to_numpy
    max_b = args.eval_batches or None
    val_eval = lambda m: evaluate_dpo(m, ref, val_loader, beta=args.beta,
                                      max_batches=max_b, to_numpy=np_to)

    # --- init / resume ---------------------------------------------------------
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    weights_path = str(out / "weights.safetensors")
    store = CheckpointStore(str(args.resume) if args.resume is not None
                            else str(out / "resume"))
    resuming = store.has_checkpoint()

    start_step = 0
    if resuming:
        meta = store.load(weights_deserializer=lambda p: policy.load(p),  # resume the policy
                          optimizer_deserializer=lambda p: backend.load_optimizer(opt, p))
        start_step = int(meta["step"])
        if scaler is not None:
            scaler.load_state_dict(meta.get("loss_scale_state") or {})
        print(f"[resume] from step {start_step} slot={meta['slot']} (out={out})")
    else:
        policy.load(str(args.init))                      # init policy from SFT weights
        print(f"[init] policy + reference from {args.init}")

    logger = JsonlLogger(str(out / "metrics.jsonl"), append=resuming)

    def on_checkpoint(step: int) -> None:
        store.save(step=step,
                   loss_scale_state=(scaler.state_dict() if scaler else None),
                   weights_serializer=lambda p: policy.save(p),  # checkpoint the POLICY
                   optimizer_serializer=lambda p: backend.save_optimizer(opt, p))

    tcfg = TrainConfig(
        total_steps=total_steps, base_lr=args.base_lr, warmup_steps=warmup,
        grad_accum=args.grad_accum, grad_clip=args.grad_clip,
        log_every=args.log_every, eval_every=args.eval_every,
        ckpt_every=args.ckpt_every, out_dir=str(out), seed=args.seed,
    )

    print(f"[dpo] pairs={len(train_loader.records)}  total_steps={total_steps}  "
          f"warmup={warmup}  beta={args.beta}  base_lr={args.base_lr}  "
          f"precision={cfg.precision}")

    train(policy, train_loader, tcfg, train_step,
          val_eval=val_eval, logger=logger, on_checkpoint=on_checkpoint,
          start_step=start_step)

    # Skip the terminal checkpoint if the loop already wrote one at total_steps.
    if total_steps % tcfg.ckpt_every != 0:
        on_checkpoint(total_steps)
    policy.save(weights_path)           # canonical portable policy weights for downstream
    final = evaluate_dpo(policy, ref, val_loader, beta=args.beta, max_batches=max_b,
                         to_numpy=np_to)
    logger.close()
    print(f"[done] step={total_steps}  val_loss={final['val_loss']:.4f}  "
          f"reward_margin={final['reward_margin']:.4f}  "
          f"reward_accuracy={final['reward_accuracy']:.3f}  metrics={out / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
