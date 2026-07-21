"""Unit tests for the LR schedules (cosine + WSD) and the factory (runs anywhere)."""

import math
from types import SimpleNamespace

import pytest

from src.train.schedule import CosineSchedule, WSDSchedule, make_schedule


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


def test_wsd_warmup_ends_at_base():
    s = WSDSchedule(base_lr=1.0, warmup_steps=10, total_steps=100, decay_steps=20)
    assert s.lr_at(0) == 0.1            # first step = base * 1/10
    assert math.isclose(s.lr_at(9), 1.0)  # end of warmup hits base_lr


def test_wsd_plateau_is_flat():
    s = WSDSchedule(base_lr=1.0, warmup_steps=10, total_steps=100, decay_steps=20)
    for t in range(10, 80):  # decay_start = 100 - 20 = 80
        assert s.lr_at(t) == 1.0


def test_wsd_decays_to_floor_at_total():
    s = WSDSchedule(base_lr=1.0, warmup_steps=10, total_steps=100, decay_steps=20,
                     min_lr_ratio=0.1)
    end = s.lr_at(100)
    assert math.isclose(end, 0.1, abs_tol=1e-6)   # floor, never 0
    assert end > 0.0


def test_wsd_monotonic_nonincreasing_in_decay():
    s = WSDSchedule(base_lr=1.0, warmup_steps=10, total_steps=100, decay_steps=20)
    decay = [s.lr_at(t) for t in range(80, 101)]  # decay_start = 80
    assert all(a >= b - 1e-12 for a, b in zip(decay, decay[1:]))


def test_wsd_validation():
    with pytest.raises(ValueError):
        WSDSchedule(base_lr=1.0, warmup_steps=10, total_steps=100, decay_steps=95)  # 10+95>100
    with pytest.raises(ValueError):
        WSDSchedule(base_lr=0.0, warmup_steps=10, total_steps=100, decay_steps=20)
    with pytest.raises(ValueError):
        WSDSchedule(base_lr=1.0, warmup_steps=10, total_steps=100, decay_steps=20,
                    min_lr_ratio=0.0)


def test_make_schedule_selects():
    cfg_cosine = SimpleNamespace(lr_schedule="cosine", base_lr=1.0, warmup_steps=10,
                                 total_steps=100, decay_frac=0.2)
    cfg_wsd = SimpleNamespace(lr_schedule="wsd", base_lr=1.0, warmup_steps=10,
                              total_steps=100, decay_frac=0.2)
    cfg_bad = SimpleNamespace(lr_schedule="bogus", base_lr=1.0, warmup_steps=10,
                              total_steps=100, decay_frac=0.2)

    assert isinstance(make_schedule(cfg_cosine), CosineSchedule)
    wsd = make_schedule(cfg_wsd)
    assert isinstance(wsd, WSDSchedule)
    assert wsd.decay_steps == round(cfg_wsd.decay_frac * cfg_wsd.total_steps)
    with pytest.raises(ValueError):
        make_schedule(cfg_bad)
