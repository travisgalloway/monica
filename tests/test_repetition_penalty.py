"""Repetition control in the sampler (portable, no backend).

Greedy decoding (`temperature == 0`) is an exact argmax, so the effect of the
repetition penalty / no-repeat-ngram ban on the *logits* is observable through which
token wins — no need to peek at internal arrays.
"""

from __future__ import annotations

import numpy as np

from src.serve.sampling import _banned_ngram_tokens, sample


# --- repetition penalty -------------------------------------------------------------

def test_penalty_one_is_a_noop():
    logits = np.array([0.1, 5.0, 0.2, 3.0])
    # With penalty 1.0, passing context must not change the bare-sampler result.
    assert sample(logits, temperature=0.0, previous_tokens=[1],
                  repetition_penalty=1.0) == 1
    assert sample(logits, temperature=0.0) == 1


def test_no_previous_tokens_skips_penalty():
    logits = np.array([0.1, 5.0, 0.2])
    assert sample(logits, temperature=0.0, previous_tokens=None,
                  repetition_penalty=2.0) == 1
    assert sample(logits, temperature=0.0, previous_tokens=[],
                  repetition_penalty=2.0) == 1


def test_penalty_pushes_down_seen_positive_logit():
    # token 1 is the argmax (2.0); penalizing it (÷2 -> 1.0) lets token 0 (1.0) win
    # the tie-break (argmax returns the first max).
    logits = np.array([1.0, 2.0, 0.5])
    assert sample(logits, temperature=0.0) == 1
    assert sample(logits, temperature=0.0, previous_tokens=[1],
                  repetition_penalty=2.0) == 0


def test_penalty_pushes_down_seen_negative_logit():
    # A negative logit is multiplied (more negative) for a seen token: token 1 (-0.6,
    # the argmax) becomes -1.2, dropping below token 0 (-1.0).
    logits = np.array([-1.0, -0.6])
    assert sample(logits, temperature=0.0) == 1
    assert sample(logits, temperature=0.0, previous_tokens=[1],
                  repetition_penalty=2.0) == 0


def test_penalty_applies_under_sampling_too():
    # Even with temperature/top-k, the penalized token should stop dominating.
    logits = np.array([0.0, 10.0, 0.0])
    rng = np.random.default_rng(0)
    draws = {sample(logits, temperature=1.0, previous_tokens=[1],
                    repetition_penalty=5.0, rng=rng) for _ in range(200)}
    assert draws != {1}  # token 1 no longer the only outcome


def test_invalid_penalty_raises():
    import pytest
    with pytest.raises(ValueError):
        sample(np.zeros(4), previous_tokens=[0], repetition_penalty=0.0)


# --- no-repeat-ngram ----------------------------------------------------------------

def test_no_repeat_ngram_bans_completing_token():
    # Context [5,1,5]: the trailing 1-gram (5,) occurred earlier followed by 1, so with
    # n=2 token 1 is banned (-inf) and the next-best token 5 wins.
    logits = np.array([0.0, 9.0, 0.0, 0.0, 0.0, 1.0])
    assert sample(logits, temperature=0.0) == 1
    assert sample(logits, temperature=0.0, previous_tokens=[5, 1, 5],
                  no_repeat_ngram_size=2) == 5


def test_no_repeat_ngram_size_one_bans_all_seen():
    banned = _banned_ngram_tokens(np.array([1, 3, 1]), n=1, vocab_size=8)
    assert set(banned.tolist()) == {1, 3}


def test_all_tokens_banned_falls_back_without_crash():
    # n=1 bans every previously-seen token; with prev covering the whole vocab there is
    # no legal next token. Must fall back to a valid draw (not crash on NaN probs under
    # sampling, nor return a banned argmax under greedy).
    logits = np.zeros(3)
    rng = np.random.default_rng(0)
    for temp in (0.0, 1.0):
        tok = sample(logits, temperature=temp, previous_tokens=[0, 1, 2],
                     no_repeat_ngram_size=1, rng=rng)
        assert 0 <= tok < 3


def test_no_repeat_ngram_no_match_is_noop():
    # Trailing 1-gram (2,) never occurred earlier in [0,1,2]; nothing is banned.
    banned = _banned_ngram_tokens(np.array([0, 1, 2]), n=2, vocab_size=8)
    assert banned.size == 0


def test_banned_ngram_drops_out_of_vocab_ids():
    # Trailing 1-gram (1,) was earlier followed by 99, but 99 >= vocab_size is dropped.
    banned = _banned_ngram_tokens(np.array([1, 99, 1]), n=2, vocab_size=8)
    assert banned.size == 0
