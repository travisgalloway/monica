"""Tests for sampling and logit masking composition (#226, #360).
Covers:
- sample() with allowed_ids (symbol table constraints #226)
- sample() with grammar_allowed_ids and grammar_mask (#360)
- Composition of semantic allowed_ids and grammar_allowed_ids via bitwise intersection
- Verification that logit masking causes zero logit corruption (no NaNs, stable softmax)
- Fallback behavior when all surviving logits are masked or non-finite
"""

from __future__ import annotations

import numpy as np
import pytest

from src.serve.sampling import sample


# --------------------------------------------------------------------------- #
# Basic sampling & allowed_ids
# --------------------------------------------------------------------------- #

def test_allowed_ids_none_is_unconstrained():
    logits = np.array([1.0, 5.0, 2.0, 8.0, 0.5], dtype=np.float32)
    a = sample(logits, temperature=0.7, rng=np.random.default_rng(42))
    b = sample(logits, temperature=0.7, rng=np.random.default_rng(42), allowed_ids=None)
    assert a == b


def test_allowed_ids_empty_raises():
    logits = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        sample(logits, allowed_ids=[])


def test_grammar_allowed_ids_empty_raises():
    logits = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        sample(logits, grammar_allowed_ids=[])


def test_allowed_ids_forces_token_greedy():
    logits = np.array([10.0, 9.0, 8.0, -100.0], dtype=np.float32)
    tok = sample(logits, temperature=0.0, allowed_ids=[3])
    assert tok == 3


def test_grammar_allowed_ids_forces_token_greedy():
    logits = np.array([10.0, 9.0, 8.0, -100.0], dtype=np.float32)
    tok = sample(logits, temperature=0.0, grammar_allowed_ids=[3])
    assert tok == 3


def test_grammar_mask_boolean_array():
    logits = np.array([10.0, 9.0, 8.0, 1.0], dtype=np.float32)
    mask = np.array([False, False, False, True], dtype=bool)
    tok = sample(logits, temperature=0.0, grammar_mask=mask)
    assert tok == 3


# --------------------------------------------------------------------------- #
# Composition with #226 semantic masking (bitwise intersection)
# --------------------------------------------------------------------------- #

def test_composition_bitwise_intersection():
    # Token 1 has logit 50.0, Token 2 has logit 40.0, Token 3 has logit 30.0, Token 4 has logit 20.0
    # allowed_ids (symbol table) allows [1, 2, 3]
    # grammar_allowed_ids allows [2, 3, 4]
    # Intersection is [2, 3]
    # Between 2 and 3, token 2 has the highest logit (40.0 > 30.0), so greedy picks 2!
    logits = np.array([0.0, 50.0, 40.0, 30.0, 20.0], dtype=np.float32)
    tok = sample(logits, temperature=0.0, allowed_ids=[1, 2, 3], grammar_allowed_ids=[2, 3, 4])
    assert tok == 2


def test_composition_disjoint_intersection_raises_value_error():
    # Disjoint allowed sets have empty intersection
    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Intersection of allowed_ids and grammar constraints is empty"):
        sample(logits, allowed_ids=[0, 1], grammar_allowed_ids=[2, 3])


def test_composition_no_logit_corruption():
    # Verify that composing masks never introduces NaNs or inf-inf anomalies
    logits = np.array([1.5, -2.3, 0.0, 10.2, -50.0], dtype=np.float32)
    rng = np.random.default_rng(123)
    for _ in range(50):
        tok = sample(
            logits,
            temperature=1.0,
            rng=rng,
            allowed_ids=[0, 2, 3],
            grammar_allowed_ids=[2, 3, 4],
        )
        assert tok in {2, 3}


def test_composition_fallback_stays_within_intersection():
    # Both sets intersect at {1, 2}. Repetition bans token 1, leaving token 2.
    # When repetition bans both {1, 2}, fallback draws uniformly from {1, 2} only!
    logits = np.array([10.0, 2.0, 1.0, 100.0], dtype=np.float32)
    rng = np.random.default_rng(42)
    draws = set()
    for _ in range(50):
        tok = sample(
            logits,
            temperature=1.0,
            rng=rng,
            allowed_ids=[1, 2],
            grammar_allowed_ids=[1, 2, 3],
            previous_tokens=[1, 2],
            no_repeat_ngram_size=1,
        )
        draws.add(tok)
    assert draws <= {1, 2}
    assert 0 not in draws
    assert 3 not in draws
