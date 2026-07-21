"""Learning-rate schedules: warmup + cosine decay, and warmup-stable-decay (WSD).

Pure math, backend-free, fully testable in any environment. Both schedules decay
to `min_lr_ratio * base_lr` (do NOT let LR hit zero). `make_schedule(cfg)` selects
between them from a duck-typed config (`lr_schedule`, `decay_frac`, ...).

Note (#216 length curriculum): both schedules are expressed in absolute-step units
over `total_steps`, and WSD's decay window is a *fraction* (`decay_frac`) of it — so
if a future length curriculum makes `tokens_per_step` (and thus `total_steps`) vary
per stage, passing each stage's own `total_steps` yields a correctly-scaled per-stage
decay window for free; no code here depends on `total_steps` being fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Schedule(Protocol):
    def lr_at(self, step: int) -> float: ...


@dataclass
class CosineSchedule:
    base_lr: float
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float = 0.1  # floor as a fraction of base_lr; never 0

    def __post_init__(self) -> None:
        # Fail fast on misconfiguration that would yield negative/increasing LRs.
        if self.base_lr <= 0:
            raise ValueError("base_lr must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.total_steps < self.warmup_steps:
            raise ValueError("total_steps must be >= warmup_steps")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in (0, 1]")

    def lr_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be >= 0")
        floor = self.base_lr * self.min_lr_ratio

        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup from 0 up to base_lr.
            return self.base_lr * (step + 1) / self.warmup_steps

        # Cosine from base_lr down to floor over the post-warmup span.
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, (step - self.warmup_steps) / denom)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (self.base_lr - floor) * cosine


@dataclass
class WSDSchedule:
    """Linear warmup -> constant base_lr plateau -> 1-sqrt decay to a floor.

    `decay_steps` is the number of trailing steps (ending at `total_steps`) spent
    decaying; the plateau fills the gap between warmup and the decay window. Fits
    the dense-train / cheap-re-decay-per-branch pattern (a stable trunk checkpoint
    can be re-decayed independently downstream, e.g. per sparse-upcycle branch).
    """

    base_lr: float
    warmup_steps: int
    total_steps: int
    decay_steps: int
    min_lr_ratio: float = 0.1  # floor as a fraction of base_lr; never 0

    def __post_init__(self) -> None:
        # Fail fast on misconfiguration that would yield negative/increasing LRs.
        if self.base_lr <= 0:
            raise ValueError("base_lr must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.decay_steps < 0:
            raise ValueError("decay_steps must be >= 0")
        if self.warmup_steps + self.decay_steps > self.total_steps:
            raise ValueError("warmup_steps + decay_steps must be <= total_steps")
        if not 0 < self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in (0, 1]")

    def lr_at(self, step: int) -> float:
        if step < 0:
            raise ValueError("step must be >= 0")
        floor = self.base_lr * self.min_lr_ratio

        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup from 0 up to base_lr (identical to CosineSchedule).
            return self.base_lr * (step + 1) / self.warmup_steps

        decay_start = self.total_steps - self.decay_steps
        if step < decay_start:
            # Stable plateau at base_lr, between warmup and the decay window.
            return self.base_lr

        # 1-sqrt decay from base_lr down to floor (WSD standard; cheap re-decay).
        progress = min(1.0, (step - decay_start) / max(1, self.decay_steps))
        factor = 1.0 - math.sqrt(progress)
        return floor + (self.base_lr - floor) * factor


def make_schedule(cfg) -> Schedule:
    """Build a schedule from a duck-typed config (e.g. `TrainConfig`).

    Reads only `lr_schedule`, `base_lr`, `warmup_steps`, `total_steps`, and (for
    WSD) `decay_frac`; does not import `TrainConfig` to avoid a circular import
    (`loop.py` imports from this module).
    """
    if cfg.lr_schedule == "cosine":
        return CosineSchedule(cfg.base_lr, cfg.warmup_steps, cfg.total_steps)
    if cfg.lr_schedule == "wsd":
        decay_steps = max(1, int(round(cfg.decay_frac * cfg.total_steps)))
        return WSDSchedule(cfg.base_lr, cfg.warmup_steps, cfg.total_steps, decay_steps)
    raise ValueError(f"unknown lr_schedule {cfg.lr_schedule!r} (expected 'cosine' or 'wsd')")
