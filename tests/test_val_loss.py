"""Unit tests for the val-loss numeric core (runs anywhere, numpy only)."""

import numpy as np
import pytest

from src.eval.val_loss import cross_entropy, perplexity, evaluate


class _EmptyLoader:
    def epoch(self):
        return iter(())


def test_evaluate_empty_loader_raises_not_false_perfect():
    # An empty val loader must fail loudly, not report perplexity=1.0 (the best possible
    # score), which would silently mask a misconfigured eval path.
    with pytest.raises(ValueError, match="empty"):
        evaluate(model=None, loader=_EmptyLoader())


def test_cross_entropy_uniform_logits():
    # Uniform logits over V classes -> CE = log(V), perplexity = V.
    V = 8
    logits = np.zeros((4, V))
    targets = np.array([0, 1, 2, 3])
    ce = cross_entropy(logits, targets)
    assert np.isclose(ce, np.log(V), atol=1e-6)
    assert np.isclose(perplexity(ce), V, atol=1e-4)


def test_cross_entropy_confident_correct_is_low():
    logits = np.array([[10.0, 0.0, 0.0]])
    ce = cross_entropy(logits, np.array([0]))
    assert ce < 1e-3


def test_cross_entropy_handles_sequence_shape():
    # (B, T, V) logits with (B, T) targets should flatten correctly.
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 5, 16))
    targets = rng.integers(0, 16, size=(2, 5))
    ce = cross_entropy(logits, targets)
    assert ce > 0
