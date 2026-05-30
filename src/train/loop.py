"""Minimal training loop — pure orchestration (SKELETON).

Drives the model only through `ModelInterface.forward` + the checkpoint module.
Backend-free here; the backprop/optimizer primitive is backend-specific and is
injected as `train_step` (e.g. MLX's `nn.value_and_grad` + optimizer.update on the
Mac). This keeps the seam intact: the loop never imports MLX/CUDA.

Required "robust run" features (wired on the Mac):
  * mixed precision (precision decided on MLX in M1; toy/smoke = fp32)
  * warmup + cosine LR (schedule.CosineSchedule)
  * gradient accumulation
  * gradient clipping (proven necessary at toy scale)
  * checkpointing (portable weights + within-backend resume bundle)
  * logging from step 1: loss, val loss/perplexity, LR, grad norm, tokens/sec
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..model.interface import ModelInterface
from ..data.loader import PackedLoader
from .schedule import CosineSchedule


@dataclass
class TrainConfig:
    total_steps: int
    base_lr: float = 3e-4
    warmup_steps: int = 100
    grad_accum: int = 1
    grad_clip: float = 1.0
    log_every: int = 1
    eval_every: int = 50
    ckpt_every: int = 100
    out_dir: str = "runs/toy"
    seed: int = 0


# A backend-provided step: (model, inputs, targets, lr) -> dict(loss=, grad_norm=).
TrainStepFn = Callable[[ModelInterface, object, object, float], dict]


def train(
    model: ModelInterface,
    train_loader: PackedLoader,
    cfg: TrainConfig,
    train_step: TrainStepFn,
    *,
    val_eval: Optional[Callable[[ModelInterface], float]] = None,
    logger: Optional[Callable[[dict], None]] = None,
    start_step: int = 0,
) -> None:
    """Run the training loop. `train_step` and the optimizer live in the backend.

    SKELETON: the orchestration shape is here; gradient accumulation, clipping,
    checkpoint save/resume, and logging payloads are completed when the MLX
    `train_step` is wired on Apple Silicon.
    """
    schedule = CosineSchedule(cfg.base_lr, cfg.warmup_steps, cfg.total_steps)
    step = start_step
    log = logger or (lambda payload: print(payload))

    while step < cfg.total_steps:
        for inputs, targets in train_loader.epoch(reseed=cfg.seed + step):
            lr = schedule.lr_at(step)
            metrics = train_step(model, inputs, targets, lr)  # backend: fwd+bwd+opt

            if step % cfg.log_every == 0:
                payload = {"step": step, "lr": lr, **metrics}
                if val_eval and step % cfg.eval_every == 0:
                    payload["val_loss"] = val_eval(model)
                log(payload)

            # TODO[mac]: ckpt_every -> model.save(weights) + save_resume(bundle).
            step += 1
            if step >= cfg.total_steps:
                break
