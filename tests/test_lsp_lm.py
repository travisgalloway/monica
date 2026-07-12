"""Tests for `src/lsp/lm.py`'s backend-agnostic helpers (`offset_map`,
`token_index_at`), against a tiny word-level `FakeLM` — no model, no node.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from src.lsp.lm import LMAdapter, offset_map, token_index_at


class FakeLM:
    """A trivial whitespace tokenizer standing in for a real `LMAdapter`.

    `encode` splits on spaces (keeping the trailing space attached to each word,
    so decode is a clean concatenation and prefix-consistent by construction) —
    just enough to exercise `offset_map`'s incremental-decode-length logic without
    a real BPE tokenizer's merge subtleties.
    """

    def __init__(self):
        self.n_forward_tokens = 0
        self.n_forward_tokens_nocache = 0
        self._vocab: List[str] = []
        self._index = {}

    def _tok_id(self, piece: str) -> int:
        if piece not in self._index:
            self._index[piece] = len(self._vocab)
            self._vocab.append(piece)
        return self._index[piece]

    def encode(self, text: str) -> List[int]:
        pieces = []
        rest = text
        while rest:
            sp = rest.find(" ")
            if sp == -1:
                pieces.append(rest)
                rest = ""
            else:
                pieces.append(rest[: sp + 1])
                rest = rest[sp + 1:]
        return [self._tok_id(p) for p in pieces]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self._vocab[i] for i in token_ids)

    def reset(self, context: str) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

    def step(self, token_id: int) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

    def rollback(self, n_tokens: int) -> None:
        pass


def test_fakelm_satisfies_lmadapter_protocol():
    assert isinstance(FakeLM(), LMAdapter)


def test_offset_map_basic():
    lm = FakeLM()
    context = "const x = 1"
    ids = lm.encode(context)
    offsets = offset_map(lm, context)
    assert len(offsets) == len(ids)
    assert offsets[0] == 0
    # Each offset must equal len(decode(ids[:k])), reconstructing where in
    # `context` that token starts.
    for k in range(len(ids)):
        assert offsets[k] == len(lm.decode(ids[:k]))
    # And offsets are strictly increasing (each token is non-empty here).
    assert offsets == sorted(offsets)


def test_offset_map_matches_manual_word_boundaries():
    lm = FakeLM()
    context = "foo bar baz"
    offsets = offset_map(lm, context)
    # "foo ", "bar ", "baz" -> starts at 0, 4, 8
    assert offsets == [0, 4, 8]


def test_token_index_at_exact_boundaries():
    offsets = [0, 4, 8]
    assert token_index_at(offsets, 0) == 0
    assert token_index_at(offsets, 4) == 1
    assert token_index_at(offsets, 8) == 2


def test_token_index_at_mid_token():
    offsets = [0, 4, 8]
    assert token_index_at(offsets, 2) == 0
    assert token_index_at(offsets, 6) == 1
    assert token_index_at(offsets, 100) == 2  # past the end -> last token


def test_token_index_at_before_first_token_clamps_to_zero():
    offsets = [3, 7, 11]
    assert token_index_at(offsets, 0) == 0
