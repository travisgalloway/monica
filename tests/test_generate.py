"""Tests for the sampler + shared generation core (portable, no backend).

A deterministic FakeModel (logits one-hot at ``(token+1) % vocab``) makes greedy
decoding an exact counter, so stop conditions and state advancement are checkable with
plain arithmetic — mirroring the FakeModel approach in ``tests/test_serve.py``.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from src.serve.generate import generate
from src.serve.sampling import sample
from src.serve.sessions import SessionStore


class CounterModel:
    """ModelInterface stand-in. Greedy next token = (current token + 1) % vocab."""

    def __init__(self, vocab_size: int = 16):
        self.config = SimpleNamespace(
            n_layers=2, d_conv=4, d_inner=128, n_heads=8, head_dim=16, d_state=16,
            precision="fp32", vocab_size=vocab_size,
        )

    def init_state(self, batch_size: int):
        return np.zeros((batch_size,), dtype=np.int64)

    def step(self, token, state):
        token = np.asarray(token)
        nxt = (token + 1) % self.config.vocab_size
        logits = np.eye(self.config.vocab_size)[nxt]  # (1, vocab), one-hot
        return logits, state + token  # state = running sum (fresh array)

    def clone_state(self, state):
        return np.array(state, copy=True)


def _store():
    store = SessionStore(CounterModel())
    store.create("s")
    return store


# --- sampler ------------------------------------------------------------------------

def test_greedy_is_argmax_and_deterministic():
    logits = np.array([0.1, 5.0, 0.2, 3.0])
    assert sample(logits, temperature=0.0) == 1
    assert sample(logits, temperature=0.0) == 1


def test_top_k_one_collapses_to_argmax():
    logits = np.array([0.1, 5.0, 0.2, 3.0])
    rng = np.random.default_rng(0)
    # With only the top logit surviving, sampling must return the argmax every time.
    for _ in range(10):
        assert sample(logits, temperature=1.0, top_k=1, rng=rng) == 1


def test_top_p_restricts_to_nucleus():
    # One token dominates; nucleus of 0.5 keeps only it.
    logits = np.log(np.array([0.9, 0.05, 0.03, 0.02]))
    rng = np.random.default_rng(0)
    for _ in range(10):
        assert sample(logits, temperature=1.0, top_p=0.5, rng=rng) == 0


def test_top_p_includes_threshold_crossing_token():
    # Two near-tied leaders (0.3 each); nucleus 0.5 must include the SECOND token (the
    # one whose mass crosses 0.5), not just the first. Tokens 2 and 3 stay excluded.
    # A plain `cumulative <= top_p` would wrongly keep only token 0.
    logits = np.log(np.array([0.3, 0.3, 0.25, 0.15]))
    rng = np.random.default_rng(0)
    draws = {sample(logits, temperature=1.0, top_p=0.5, rng=rng) for _ in range(200)}
    assert draws == {0, 1}


def test_negative_temperature_raises():
    with pytest.raises(ValueError):
        sample(np.zeros(4), temperature=-1.0)


# --- generation core ----------------------------------------------------------------

def test_greedy_generation_counts_up():
    store = _store()
    greedy = partial(sample, temperature=0.0)
    out = generate(store, "s", [0], sampler=greedy, max_new_tokens=4)
    assert out == [1, 2, 3, 4]


def test_max_new_tokens_bounds_length():
    store = _store()
    out = generate(store, "s", [0], sampler=partial(sample, temperature=0.0),
                   max_new_tokens=2)
    assert out == [1, 2]


def test_eos_halts_before_appending():
    store = _store()
    # Counting up from 2 hits eos_id=5 at the 4th token; it must NOT be appended.
    out = generate(store, "s", [2], sampler=partial(sample, temperature=0.0),
                   max_new_tokens=10, eos_id=5)
    assert out == [3, 4]


def test_stop_fn_halts_generation():
    store = _store()
    out = generate(store, "s", [0], sampler=partial(sample, temperature=0.0),
                   max_new_tokens=10, stop_fn=lambda gen: len(gen) >= 3)
    assert out == [1, 2, 3]


def test_stop_fn_session_state_reflects_all_emitted_tokens():
    # Regression: when stop_fn halts, the last emitted token is fed back BEFORE the stop
    # check, so the session state reflects every id in `generated` (resume-safe).
    store = _store()
    out = generate(store, "s", [0], sampler=partial(sample, temperature=0.0),
                   max_new_tokens=10, stop_fn=lambda gen: len(gen) >= 3)
    assert out == [1, 2, 3]
    # state = running sum of all fed tokens: prefill 0 + generated 1+2+3 = 6.
    assert int(store.get_state("s")[0]) == 6


def test_on_token_streams_each_generated_id():
    store = _store()
    seen = []
    out = generate(store, "s", [0], sampler=partial(sample, temperature=0.0),
                   max_new_tokens=3, on_token=seen.append)
    assert seen == out == [1, 2, 3]


def test_generation_advances_session_state():
    store = _store()
    generate(store, "s", [1, 2], sampler=partial(sample, temperature=0.0),
             max_new_tokens=3)  # prefill 1,2 then generate 3,4,5
    # State is the running sum of every token fed. Prefill feeds 1,2; then each
    # generated token (3,4,5) is fed back to advance the recurrence. Sum = 15.
    assert int(store.get_state("s")[0]) == 15


def test_empty_prompt_raises():
    store = _store()
    with pytest.raises(ValueError):
        generate(store, "s", [], sampler=partial(sample, temperature=0.0))


def test_pass_context_forwards_prompt_plus_generated():
    # A repetition-aware sampler needs the running context; pass_context=True must hand
    # it prompt + everything emitted so far, each step.
    store = _store()
    seen = []

    def recording_sampler(logits, previous_tokens=None):
        seen.append(list(previous_tokens))
        return int(np.asarray(logits).argmax())

    out = generate(store, "s", [0], sampler=recording_sampler, max_new_tokens=3,
                   pass_context=True)
    assert out == [1, 2, 3]
    assert seen == [[0], [0, 1], [0, 1, 2]]


def test_default_uses_bare_sampler_contract():
    # Without pass_context the lm-eval adapter's `sampler(logits)` contract is preserved
    # (a sampler taking no previous_tokens kwarg still works).
    store = _store()

    def bare_sampler(logits):
        return int(np.asarray(logits).argmax())

    out = generate(store, "s", [0], sampler=bare_sampler, max_new_tokens=2)
    assert out == [1, 2]
