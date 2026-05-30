"""Unit tests for the warmup + cosine LR schedule (runs anywhere)."""

import math

from src.train.schedule import CosineSchedule


def test_warmup_ramps_to_base():
    s = CosineSchedule(base_lr=1.0, warmup_steps=10, total_steps=100)
    assert s.lr_at(0) == 0.1            # first step = base * 1/10
    assert math.isclose(s.lr_at(9), 1.0)  # end of warmup hits base_lr


def test_cosine_decays_to_floor_not_zero():
    s = CosineSchedule(base_lr=1.0, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    end = s.lr_at(100)
    assert math.isclose(end, 0.1, abs_tol=1e-6)   # floor, never 0
    assert end > 0.0


def test_monotonic_decay_after_warmup():
    s = CosineSchedule(base_lr=1.0, warmup_steps=10, total_steps=100)
    post = [s.lr_at(t) for t in range(10, 101)]
    assert all(a >= b - 1e-12 for a, b in zip(post, post[1:]))
