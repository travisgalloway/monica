"""Unit tests for the val-loss numeric core (runs anywhere, numpy only)."""

import numpy as np
import pytest

from src.eval.val_loss import (bits_per_byte, cross_entropy, evaluate,
                               evaluate_masked, perplexity)


class _EmptyLoader:
    def epoch(self):
        return iter(())


def test_evaluate_empty_loader_raises_not_false_perfect():
    # An empty val loader must fail loudly, not report perplexity=1.0 (the best possible
    # score), which would silently mask a misconfigured eval path.
    with pytest.raises(ValueError, match="empty"):
        evaluate(model=None, loader=_EmptyLoader())


class _FakeModel:
    """Returns fixed uniform logits over V classes -> CE = log(V) per token, analytic."""

    def __init__(self, vocab: int):
        self.vocab = vocab

    def forward(self, inputs):
        b, t = np.asarray(inputs).shape
        return np.zeros((b, t, self.vocab))


class _FakeLoader:
    """One fixed batch, with optional corpus byte/token totals (#192)."""

    def __init__(self, batch_size=2, seq_len=4, n_bytes=None, n_tokens=None):
        self.batch_size = batch_size
        self.seq_len = seq_len
        if n_bytes is not None:
            self.n_bytes = n_bytes
        if n_tokens is not None:
            self.n_tokens = n_tokens
        self._inputs = np.zeros((batch_size, seq_len), dtype=np.int64)
        self._targets = np.zeros((batch_size, seq_len), dtype=np.int64)

    def epoch(self, reseed=None, skip_batches=0):
        yield self._inputs, self._targets


class _FakeMaskedLoader:
    """One fixed masked batch (all-ones mask), with optional byte/token totals."""

    def __init__(self, batch_size=2, seq_len=4, n_bytes=None, n_tokens=None):
        if n_bytes is not None:
            self.n_bytes = n_bytes
        if n_tokens is not None:
            self.n_tokens = n_tokens
        self._inputs = np.zeros((batch_size, seq_len), dtype=np.int64)
        self._targets = np.zeros((batch_size, seq_len), dtype=np.int64)
        self._mask = np.ones((batch_size, seq_len))

    def epoch(self, reseed=None, skip_batches=0):
        yield self._inputs, self._targets, self._mask


def test_bits_per_byte_numeric():
    total_ce = np.log(2) * 10
    assert np.isclose(bits_per_byte(total_ce, 5), 2.0)


def test_evaluate_returns_val_bpb():
    V = 8
    model = _FakeModel(V)
    batch_size, seq_len = 2, 4
    n_bytes, n_tokens = 400, 100
    loader = _FakeLoader(batch_size=batch_size, seq_len=seq_len,
                         n_bytes=n_bytes, n_tokens=n_tokens)
    result = evaluate(model, loader)
    assert "val_bpb" in result
    total_tokens = batch_size * seq_len
    total_ce = np.log(V) * total_tokens
    effective_bytes = total_tokens * (n_bytes / n_tokens)
    expected = bits_per_byte(total_ce, effective_bytes)
    assert np.isclose(result["val_bpb"], expected)


def test_evaluate_omits_val_bpb_when_no_bytes():
    model = _FakeModel(8)
    loader = _FakeLoader()  # no n_bytes/n_tokens attributes
    result = evaluate(model, loader)
    assert "val_loss" in result and "val_perplexity" in result
    assert "val_bpb" not in result


def test_evaluate_masked_val_bpb_present_and_absent():
    V = 8
    model = _FakeModel(V)
    batch_size, seq_len = 2, 4
    n_bytes, n_tokens = 200, 50
    loader_with_bytes = _FakeMaskedLoader(batch_size=batch_size, seq_len=seq_len,
                                          n_bytes=n_bytes, n_tokens=n_tokens)
    result = evaluate_masked(model, loader_with_bytes)
    assert "val_bpb" in result
    total_tokens = batch_size * seq_len
    total_ce = np.log(V) * total_tokens
    effective_bytes = total_tokens * (n_bytes / n_tokens)
    expected = bits_per_byte(total_ce, effective_bytes)
    assert np.isclose(result["val_bpb"], expected)

    loader_no_bytes = _FakeMaskedLoader(batch_size=batch_size, seq_len=seq_len)
    result_no_bytes = evaluate_masked(model, loader_no_bytes)
    assert "val_loss" in result_no_bytes and "val_perplexity" in result_no_bytes
    assert "val_bpb" not in result_no_bytes


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
