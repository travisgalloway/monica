"""The `LMAdapter` seam: what the harness generates against, backend-injected.

Explicit stepwise generation state — `reset`/`step`/`rollback` over a persistent
backend cache — mirroring `src/serve/sessions.py::SessionStore.step`'s idiom. Not a
reuse of `SessionStore` itself: that class is `MambaConfig`-shaped (a fixed-size
recurrent state snapshot per session) and cannot back a transformer's growing KV
cache, which needs trim-in-place rollback instead of clone/restore.

Implementations live below the seam (`src/model/mlx_lm_adapter.py` on `mlx_lm`, or
an `hf_lm_adapter.py` fallback on `transformers`). This module only defines the
`Protocol` and the backend-agnostic helpers built on top of `encode`/`decode`
(`offset_map`, `token_index_at`) — pure stdlib, works against any adapter (including
`FakeLM` test doubles), no `mlx`/`torch` import anywhere.

ABOVE THE SEAM — stdlib only (+ numpy for the logits type). No `mlx`/`torch` import
anywhere in this module (guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import bisect
from typing import List, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class LMAdapter(Protocol):
    """Call order: `reset(context)` once, then any sequence of `step`/`rollback`.
    `reset` may be called again later to start a fresh record on the same adapter
    (a new backend cache), so an adapter instance can be reused across records.

    `n_forward_tokens` / `n_forward_tokens_nocache` are cumulative counters over the
    adapter's lifetime (the harness snapshots them before/after a generation to get
    a per-record cost): `n_forward_tokens` is the number of token positions actually
    run through the model (cache hits are free); `n_forward_tokens_nocache` is what
    the same sequence of calls would have cost a hypothetical implementation with no
    cache trimming at all (every rollback pays a full re-prefill). The gap between
    them is the caching win, and comparing `n_forward_tokens` alone across backends
    would otherwise reward whichever backend happens to implement exact cache
    trimming, rather than measuring the algorithm.
    """

    n_forward_tokens: int
    n_forward_tokens_nocache: int

    def encode(self, text: str) -> List[int]:
        """Tokenize `text`. No BOS/EOS is added beyond what the tokenizer does by
        default; the harness treats prompts as raw text spans, not chat turns."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Detokenize `token_ids` back to text. Must be prefix-consistent with
        `encode` (`decode(encode(context)[:k])` is a genuine prefix of `context`
        for every `k`) — `offset_map` below depends on this."""
        ...

    def reset(self, context: str) -> np.ndarray:
        """(Re)initialize state from `context` (a full prefill); return next-token
        logits, shape `(vocab_size,)`."""
        ...

    def step(self, token_id: int) -> np.ndarray:
        """Advance state by one already-decided token; return next-token logits."""
        ...

    def rollback(self, n_tokens: int) -> None:
        """Discard the `n_tokens` most recently `step`ped tokens from state (never
        reaches back past the `reset` context — the harness never asks for that,
        since diagnostic offsets are clamped to `generation_start` before a rollback
        target is computed; see `diagnostics.filter_diagnostics`)."""
        ...


# --------------------------------------------------------------------------- #
# Backend-agnostic helpers (work against any LMAdapter, incl. FakeLM)
# --------------------------------------------------------------------------- #

def offset_map(adapter: LMAdapter, context: str) -> List[int]:
    """The character offset, in `context`, where each of `adapter.encode(context)`'s
    tokens starts. Built from **incremental `decode()` lengths**
    (`len(decode(ids[:k]))` is token `k`'s start offset) rather than concatenating
    individually-decoded per-token strings, which is unsound for byte-level BPE
    (a single token's decode can be an invalid/different string in isolation than
    it is as part of a longer decoded prefix).
    """
    ids = adapter.encode(context)
    return [len(adapter.decode(ids[:k])) for k in range(len(ids))]


def token_index_at(offsets: Sequence[int], char_offset: int) -> int:
    """Index of the token whose span contains `char_offset`: the last token whose
    start offset is `<= char_offset`. Clamped to token 0 if `char_offset` precedes
    every token's start (shouldn't happen for a non-negative offset within the
    encoded text, but stay total rather than raising on an edge case).
    """
    idx = bisect.bisect_right(offsets, char_offset) - 1
    return max(idx, 0)
