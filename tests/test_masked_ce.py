"""Masked cross-entropy reference (portable). The SFT loss must reduce to the existing
`cross_entropy` when every token is unmasked, and to CE over the unmasked subset
otherwise."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.val_loss import cross_entropy, masked_cross_entropy

APPROX = dict(rel=1e-9, abs=1e-12)


def _logits(seed=0, n=6, v=5):
    return np.random.default_rng(seed).normal(size=(n, v))


def test_all_ones_mask_equals_cross_entropy():
    logits, targets = _logits(), np.array([0, 1, 2, 3, 4, 0])
    assert masked_cross_entropy(logits, targets, np.ones(6)) == \
        pytest.approx(cross_entropy(logits, targets), **APPROX)


def test_masked_region_equals_subset_cross_entropy():
    logits, targets = _logits(1), np.array([1, 2, 3, 4, 0, 1])
    mask = np.array([0, 0, 1, 1, 1, 0])  # keep indices 2,3,4
    keep = mask.astype(bool)
    expected = cross_entropy(logits[keep], targets[keep])
    assert masked_cross_entropy(logits, targets, mask) == pytest.approx(expected, **APPROX)


def test_all_zero_mask_is_zero():
    logits, targets = _logits(2), np.array([0, 1, 2, 3, 4, 0])
    assert masked_cross_entropy(logits, targets, np.zeros(6)) == 0.0


def test_2d_logits_are_flattened():
    # (B, L, V) logits with a (B, L) mask must agree with the flattened computation.
    logits = _logits(3, n=8).reshape(2, 4, 5)
    targets = np.array([[0, 1, 2, 3], [4, 0, 1, 2]])
    mask = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
    flat = masked_cross_entropy(logits.reshape(-1, 5), targets.reshape(-1),
                                mask.reshape(-1))
    assert masked_cross_entropy(logits, targets, mask) == pytest.approx(flat, **APPROX)
