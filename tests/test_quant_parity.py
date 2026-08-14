"""Portable tests (#168) for `src.conformance.quant_parity.check_quant_parity`: identical
logits agree perfectly, shuffled logits fail, and the bits-dependent thresholds gate
correctly. No backend required."""

import numpy as np
import pytest

from src.conformance.quant_parity import QUANT_THRESHOLDS, check_quant_parity


def _logits(seed, shape=(2, 8, 32)):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def test_identical_logits_are_perfect_agreement():
    fp = _logits(0)
    result = check_quant_parity(fp, fp.copy(), bits=8)
    assert result["top1_agreement"] == 1.0
    assert result["mean_kl"] == pytest.approx(0.0, abs=1e-9)
    assert result["max_abs_drift"] == 0.0
    assert result["ok"] is True


def test_small_perturbation_within_int8_threshold():
    fp = _logits(1)
    q = fp + np.random.default_rng(2).normal(scale=1e-3, size=fp.shape).astype(np.float32)
    result = check_quant_parity(fp, q, bits=8)
    assert result["top1_agreement"] > 0.99
    assert result["ok"] is True


def test_shuffled_logits_fail():
    fp = _logits(3)
    q = fp[:, :, ::-1].copy()  # reverse the vocab axis -> top-1 essentially never agrees
    result = check_quant_parity(fp, q, bits=8)
    assert result["top1_agreement"] < 0.5
    assert result["ok"] is False


def test_shape_mismatch_reports_not_ok_without_raising():
    fp = _logits(4, shape=(2, 8, 32))
    q = _logits(5, shape=(2, 8, 16))
    result = check_quant_parity(fp, q, bits=8)
    assert result["ok"] is False
    assert "error" in result


def test_missing_thresholds_for_unknown_bits_reports_not_ok():
    fp = _logits(6)
    result = check_quant_parity(fp, fp.copy(), bits=3)
    assert result["ok"] is False
    assert "error" in result


def test_explicit_thresholds_override_the_default_table():
    fp = _logits(7)
    q = fp + np.random.default_rng(8).normal(scale=0.5, size=fp.shape).astype(np.float32)
    # A deliberately impossible threshold forces ok=False even though top1/kl still compute.
    result = check_quant_parity(fp, q, bits=8, thresholds={"top1": 1.1, "kl": -1.0})
    assert result["ok"] is False


def test_default_threshold_table_shape():
    assert set(QUANT_THRESHOLDS) == {8, 4}
    for bits, thr in QUANT_THRESHOLDS.items():
        assert 0.0 < thr["top1"] <= 1.0
        assert thr["kl"] > 0.0
    # int4's bar is looser than int8's on both axes.
    assert QUANT_THRESHOLDS[4]["top1"] < QUANT_THRESHOLDS[8]["top1"]
    assert QUANT_THRESHOLDS[4]["kl"] > QUANT_THRESHOLDS[8]["kl"]
