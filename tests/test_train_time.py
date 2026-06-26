"""Tests for the portable training-time estimator (src/model/train_time.py)."""

from __future__ import annotations

import pytest

from src.model.train_time import (
    ANCHOR_PARAMS,
    ANCHOR_SECONDS_PER_STEP,
    ANCHOR_TOKENS_PER_STEP,
    chinchilla_tokens,
    default_registry,
    format_count,
    format_time,
    parse_count,
    train_seconds,
    training_flops,
)

SECONDS_PER_DAY = 86400.0


def test_m1pro_reproduces_measured_anchor():
    """The calibrated M1 Pro must reproduce its own bench point exactly."""
    hw = default_registry()["m1pro"]
    secs = train_seconds(ANCHOR_PARAMS, ANCHOR_TOKENS_PER_STEP, hw)
    assert secs == pytest.approx(ANCHOR_SECONDS_PER_STEP, rel=1e-9)


def test_m1pro_3b_run_is_about_26_days():
    """Sanity: poc-scale @ 3B tokens on M1 Pro ≈ 26 days (CLAUDE.md claim)."""
    hw = default_registry()["m1pro"]
    days = train_seconds(ANCHOR_PARAMS, 3e9, hw) / SECONDS_PER_DAY
    assert days == pytest.approx(26.0, abs=1.0)


def test_cluster_scales_over_single_h100():
    """8×H100 effective throughput = 8 × single × scaling efficiency."""
    reg = default_registry(mfu=0.40, scaling=0.85)
    ratio = reg["8xh100"].effective_flops / reg["h100"].effective_flops
    assert ratio == pytest.approx(8 * 0.85, rel=1e-9)


def test_overrides_change_h100_only():
    """--mfu/--scaling retune the GPU entries but never the calibrated M1 Pro."""
    base = default_registry()
    tuned = default_registry(mfu=0.50, scaling=0.90)
    assert tuned["m1pro"].effective_flops == base["m1pro"].effective_flops
    assert tuned["h100"].effective_flops > base["h100"].effective_flops


def test_chinchilla_and_flops():
    assert chinchilla_tokens(1_000) == 20_000
    assert training_flops(100, 10) == 6 * 100 * 10


def test_format_time_units():
    assert format_time(30) == "30 s"
    assert format_time(90).endswith(" m")
    assert format_time(2 * 3600).endswith(" h")
    assert format_time(3 * 86400).endswith(" d")
    assert format_time(2 * 365 * 86400).endswith(" y")


@pytest.mark.parametrize("text,expected", [
    ("270M", 270_000_000),
    ("3B", 3_000_000_000),
    ("7b", 7_000_000_000),
    ("1.5B", 1_500_000_000),
    ("500K", 500_000),
    ("1000", 1000),
])
def test_parse_count(text, expected):
    assert parse_count(text) == expected


def test_format_count_roundtrip_label():
    assert format_count(126_731_712) == "127M"
    assert format_count(1_033_650_944) == "1.03B"
