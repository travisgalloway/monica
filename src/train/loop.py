"""Minimal training loop — pure orchestration (SKELETON).

Drives the model only through `ModelInterface.forward` + the checkpoint module.
Backend-free here; the backprop/optimizer primitive is backend-specific and is
injected as `train_step` (e.g. MLX's `nn.value_and_grad` + optimizer.update on the
Mac). This keeps the seam intact: the loop never imports MLX/CUDA.

Required "robust run" features (wired on the Mac):
  * mixed precision (precision decided on MLX in M1; toy/smoke = fp32)
  * warmup + cosine or WSD LR (schedule.make_schedule)
  * gradient accumulation
  * gradient clipping (proven necessary at toy scale)
  * checkpointing (portable weights + within-backend resume bundle)
  * logging from step 1: loss, val loss/perplexity, LR, grad norm, tokens/sec
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import islice
from typing import Callable, Optional

from ..model.interface import ModelInterface
from ..data.loader import PackedLoader
from .schedule import make_schedule


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
    lr_schedule: str = "cosine"
    decay_frac: float = 0.2


# A backend-provided step: (model, micro_batches, lr) -> dict(loss=, grad_norm=, ...).
# `micro_batches` is a list of `(inputs, targets)` of length `cfg.grad_accum`.
TrainStepFn = Callable[[ModelInterface, list, float], dict]


def _micro_batch_stream(train_loader: PackedLoader, seed: int, start_micro: int = 0):
    """Infinite stream of (inputs, targets), reseeding the shuffle each epoch.

    `start_micro` fast-forwards the stream to the position reached after that many
    micro-batches — this is what makes resume continue the data sequence instead of
    replaying the corpus from the top. The stream is fully deterministic given
    (seed, epoch length), so the position is reconstructed from `start_micro` alone:
    skip whole epochs for free (just advance the per-epoch reseed) and fast-forward
    the remainder within the current epoch via `epoch(skip_batches=...)`.
    """
    per_epoch = len(train_loader)
    if per_epoch <= 0:
        raise ValueError("train_loader yields no batches per epoch")
    epoch_idx, skip = divmod(start_micro, per_epoch)
    while True:
        yield from train_loader.epoch(reseed=seed + epoch_idx, skip_batches=skip)
        epoch_idx += 1
        skip = 0


def train(
    model: ModelInterface,
    train_loader: PackedLoader,
    cfg: TrainConfig,
    train_step: TrainStepFn,
    *,
    val_eval: Optional[Callable[[ModelInterface], dict]] = None,
    logger: Optional[Callable[[dict], None]] = None,
    on_checkpoint: Optional[Callable[[int], None]] = None,
    start_step: int = 0,
) -> None:
    """Run the training loop. `train_step` and the optimizer live in the backend.

    Pure orchestration: the backprop/optimizer primitive (`train_step`) and the
    checkpoint writer (`on_checkpoint(step)`, which persists portable weights + a
    within-backend resume bundle) are injected so this stays backend-free. Resume
    is driven by `start_step`. Each step consumes `cfg.grad_accum` micro-batches.
    `val_eval(model)` returns a metrics dict (e.g. {val_loss, val_perplexity}) that
    is merged into the logged payload.
    """
    schedule = make_schedule(cfg)
    step = start_step
    log = logger or (lambda payload: print(payload))
    tokens_per_step = train_loader.batch_size * train_loader.seq_len * cfg.grad_accum

    stream = _micro_batch_stream(train_loader, cfg.seed,
                                 start_micro=start_step * cfg.grad_accum)
    t0 = time.perf_counter()
    steps_since_log = 0

    while step < cfg.total_steps:
        micro = list(islice(stream, cfg.grad_accum))
        if not micro:
            break
        lr = schedule.lr_at(step)
        metrics = train_step(model, micro, lr)            # backend: fwd+bwd+opt
        steps_since_log += 1

        if step % cfg.log_every == 0:
            now = time.perf_counter()
            dt = now - t0
            tps = steps_since_log * tokens_per_step / dt if dt > 0 else 0.0
            payload = {"step": step, "lr": lr, **metrics, "tokens_per_sec": tps}
            if val_eval and step % cfg.eval_every == 0:
                payload.update(val_eval(model))
            log(payload)
            t0 = time.perf_counter()
            steps_since_log = 0

        step += 1
        if on_checkpoint and step % cfg.ckpt_every == 0:
            on_checkpoint(step)
        if step >= cfg.total_steps:
            break
