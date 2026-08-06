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
from typing import Any, Callable, Optional

from ..model.interface import ModelInterface
from ..data.loader import PackedLoader
from .curriculum import LengthCurriculum
from .schedule import make_schedule
from .stream import MicroBatchStream


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

    The pre-#216 fixed-shape stream, now a thin shim over a single-stage
    `MicroBatchStream` + `seek_micro`. Kept because it PINS the legacy semantics: the
    position is reconstructed from `start_micro` alone, which is only valid while
    `seq_len` — and hence `len(train_loader)` — is fixed for the whole run. A length
    curriculum breaks that assumption, which is why the real loop now carries explicit
    dataloader state instead (see `src/train/stream.py`).
    """
    curriculum = LengthCurriculum.single(train_loader.seq_len, train_loader.batch_size,
                                         steps=1, grad_accum=1)
    stream = MicroBatchStream(curriculum, lambda sl, bs: train_loader, seed)
    stream.seek_micro(start_micro)
    return stream


def train(
    model: ModelInterface,
    train_loader: PackedLoader,
    cfg: TrainConfig,
    train_step: TrainStepFn,
    *,
    val_eval: Optional[Callable[[ModelInterface], dict]] = None,
    logger: Optional[Callable[[dict], None]] = None,
    on_checkpoint: Optional[Callable[[int, Any], None]] = None,
    start_step: int = 0,
    curriculum: Optional[LengthCurriculum] = None,
    loader_factory: Optional[Callable[[int, int], Any]] = None,
    start_data_state: Optional[dict] = None,
    ignore_data_state: bool = False,
) -> dict:
    """Run the training loop. `train_step` and the optimizer live in the backend.

    Pure orchestration: the backprop/optimizer primitive (`train_step`) and the
    checkpoint writer (`on_checkpoint(step, data_state)`, which persists portable weights
    + a within-backend resume bundle) are injected so this stays backend-free. Resume is
    driven by `start_step` plus, since #216, an explicit `start_data_state`.
    `val_eval(model)` returns a metrics dict (e.g. {val_loss, val_perplexity}) merged into
    the logged payload.

    Length curriculum (#216). Pass a `LengthCurriculum` plus a
    `loader_factory(seq_len, batch_size) -> loader` to ramp `seq_len` (and shrink
    `batch_size`) across the run. With `curriculum=None` the loop synthesizes the
    degenerate one-stage curriculum over `train_loader`, so there is exactly ONE code
    path and the no-curriculum behavior — including for `SFTLoader`/`DPOLoader` — is
    identical to the pre-#216 loop. `tokens_per_step` is therefore read per step rather
    than computed once, and `tokens_per_sec` accumulates real tokens instead of
    multiplying a constant.

    Returns `{"step", "data_state", "tokens_seen"}` so the caller's final out-of-loop
    checkpoint can persist a correct dataloader position instead of `None`.
    """
    if curriculum is None:
        curriculum = LengthCurriculum.single(train_loader.seq_len, train_loader.batch_size,
                                             steps=cfg.total_steps, grad_accum=cfg.grad_accum)
        loader_factory = lambda sl, bs: train_loader        # noqa: E731 — trivial shim
    elif loader_factory is None:
        raise ValueError("a curriculum needs a loader_factory(seq_len, batch_size)")
    elif curriculum.total_steps != cfg.total_steps:
        # The caller derives one from the other; a drift here means the LR schedule and
        # the stage allocation disagree about where the run ends.
        raise ValueError(
            f"curriculum.total_steps ({curriculum.total_steps}) != cfg.total_steps "
            f"({cfg.total_steps}) — derive TrainConfig.total_steps from the curriculum.")

    schedule = make_schedule(cfg)
    step = start_step
    log = logger or (lambda payload: print(payload))

    stream = MicroBatchStream(curriculum, loader_factory, cfg.seed)
    if start_data_state is not None and not ignore_data_state:
        stream.load_state_dict(start_data_state)
        want = curriculum.stage_index_at(start_step)
        if want != stream.stage_idx:
            raise ValueError(
                f"resume mismatch: step {start_step} belongs to curriculum stage {want}, "
                f"but the saved dataloader state is in stage {stream.stage_idx}. The "
                "stage allocation moved under the checkpoint (usually a changed "
                "--total-tokens/--total-steps). Restore the original budget, or pass "
                "--ignore-data-state to discard the saved position and reconstruct it "
                "from the step instead.")
    elif len(curriculum.stages) == 1:
        stream.seek_micro(start_step * cfg.grad_accum)     # exactly the legacy path
    else:
        stream.seek_step(start_step)

    tokens_seen = curriculum.tokens_before(start_step)
    t0 = time.perf_counter()
    tokens_since_log = 0
    last_stage = None

    while step < cfg.total_steps:
        stage = curriculum.stage_at(step)
        if stage.index != last_stage and len(curriculum.stages) > 1:
            # One visible, timestamped line per stage change. On CUDA every torch.compile
            # recompile is preceded by this, so a stall at a boundary is attributable
            # rather than mysterious (see scripts/train.py's [curriculum] banner). Only
            # emitted for a real curriculum — a single-stage run has no boundary, and the
            # extra payload would be noise in every existing pretrain/SFT/DPO log.
            log({"event": "stage", "step": step, "stage": stage.index,
                 "seq_len": stage.seq_len, "batch_size": stage.batch_size,
                 "tokens_per_step": stage.tokens_per_step(cfg.grad_accum)})
        last_stage = stage.index

        micro = list(islice(stream, cfg.grad_accum))
        if not micro:
            break
        lr = schedule.lr_at(step)
        metrics = train_step(model, micro, lr)            # backend: fwd+bwd+opt
        tokens_per_step = stage.tokens_per_step(cfg.grad_accum)
        tokens_since_log += tokens_per_step
        tokens_seen += tokens_per_step

        if step % cfg.log_every == 0:
            now = time.perf_counter()
            dt = now - t0
            tps = tokens_since_log / dt if dt > 0 else 0.0
            payload = {"step": step, "lr": lr, **metrics, "tokens_per_sec": tps,
                       "seq_len": stage.seq_len, "batch_size": stage.batch_size,
                       "stage": stage.index, "tokens_per_step": tokens_per_step,
                       "tokens": tokens_seen}
            if val_eval and step % cfg.eval_every == 0:
                payload.update(val_eval(model))
            log(payload)
            t0 = time.perf_counter()
            tokens_since_log = 0

        step += 1
        if on_checkpoint and step % cfg.ckpt_every == 0:
            # Fired after the increment, when the stream has consumed exactly
            # step*grad_accum micro-batches — i.e. it is positioned at the START of
            # step `step`, matching the `step` written into the same bundle.
            on_checkpoint(step, stream.state_dict())
        if step >= cfg.total_steps:
            break

    return {"step": step, "data_state": stream.state_dict(), "tokens_seen": tokens_seen}
