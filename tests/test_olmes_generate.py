"""Generative-eval compatibility: `generate_until_texts` over the shared core.

Exercises the same code path lm-eval's `generate_until` delegates to, but with a
FakeModel + a char tokenizer and NO lm-eval dependency. Proves the two behaviors the
harness relies on: stopping/truncating at an `until` string, and bounding output by
`max_gen_toks`. The lazily-imported lm-eval shell itself is covered in the live
eval-harness run (plan Verification step 7).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.eval.olmes_adapter import generate_until_texts


class CounterModel:
    """Greedy next token = (current token + 1) % vocab (matches tests/test_generate)."""

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
        return np.eye(self.config.vocab_size)[nxt], state + token

    def clone_state(self, state):
        return np.array(state, copy=True)


class CharTokenizer:
    """Maps token id i <-> letter ('a'+i). eos disabled so it never short-circuits."""

    vocab_size = 16
    eos_token_id = None

    def encode(self, s, add_special_tokens=False):
        return [(ord(c) - ord("a")) % self.vocab_size for c in s]

    def decode(self, ids):
        return "".join(chr(ord("a") + (int(i) % self.vocab_size)) for i in ids)


def _run(pairs, max_length=32):
    return generate_until_texts(CounterModel(), CharTokenizer(), pairs,
                                max_length=max_length)


def test_until_string_truncates_output():
    # Prompt "a"(=0) decodes greedily to b, c, d, ...; until=["d"] stops at d and the
    # returned text is truncated *before* d -> "bc".
    (out,) = _run([("a", {"until": ["d"], "max_gen_toks": 10})])
    assert out == "bc"


def test_max_gen_toks_bounds_output():
    (out,) = _run([("a", {"max_gen_toks": 3})])
    assert out == "bcd"  # exactly 3 generated tokens, no stop string


def test_multiple_requests_preserve_order():
    out = _run([("a", {"max_gen_toks": 2}), ("b", {"max_gen_toks": 2})])
    assert out == ["bc", "cd"]


def test_string_until_is_accepted():
    # lm-eval may pass `until` as a bare string rather than a list.
    (out,) = _run([("a", {"until": "c", "max_gen_toks": 10})])
    assert out == "b"


def test_context_is_left_truncated_to_make_room():
    # max_length 4, max_gen 3 -> keep only the last context token; still returns a
    # bounded string (regression guard on the truncation arithmetic).
    (out,) = _run([("abcdef", {"max_gen_toks": 3})], max_length=4)
    assert isinstance(out, str) and len(out) == 3


def test_max_gen_toks_capped_to_max_length():
    # max_gen_toks (100) far exceeds max_length (4); generation must be capped so
    # prompt (>=1 token) + new tokens stays within max_length -> at most 3 generated.
    (out,) = _run([("a", {"max_gen_toks": 100})], max_length=4)
    assert len(out) <= 3
