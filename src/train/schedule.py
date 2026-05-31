"""Learning-rate schedule: linear warmup + cosine decay to a floor.

Pure math, backend-free, fully testable in any environment. Decay stops at
`min_lr_ratio * base_lr` (do NOT let LR hit zero).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
