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
# Optional capabilities
#
# Deliberately NOT part of `LMAdapter` above: `FakeLM` and the original MLX adapter
# predate them and must stay valid implementations. Probe with `supports_chat(...)` /
# `supports_snapshot(...)` rather than `isinstance`, so an adapter opts in simply by
# having the methods.
# --------------------------------------------------------------------------- #

@runtime_checkable
class ChatCapable(Protocol):
    """An adapter whose tokenizer carries a chat template (i.e. an instruct model).

    The template belongs to the tokenizer, so rendering happens below the seam; the
    portable side (`src/lsp/chat.py`) only decides what the messages *say*.
    """

    def render_chat(self, messages: Sequence[dict]) -> str:
        """Render chat `messages` into the model's prompt string, with the generation
        prompt appended (so the model's next token starts the assistant turn)."""
        ...


@runtime_checkable
class SnapshotCapable(Protocol):
    """An adapter that can checkpoint and restore generation state directly (#202).

    This exists because of a hard asymmetry between architectures, and it is the whole
    reason the SSM arm of this experiment is interesting:

    - A **transformer** KV cache stores per-token history, so rollback is a *trim*:
      move the write offset back. O(1) time, zero extra memory — but the cache itself
      grows linearly with context.
    - An **SSM** (Mamba) keeps a fixed-size running summary with no per-token history,
      so there is nothing to trim (`ArraysCache.is_trimmable()` is False). Rollback
      therefore degrades to a full re-prefill... *unless* you snapshot the state and
      restore it, which is possible precisely because the state is fixed-size.

    So the SSM cannot trim but *can* checkpoint, at a cost that is **constant in
    context length** (it is O(layers x d_inner x d_state)) where the transformer's is
    linear. That inverts the usual intuition: SSM rollback is more expensive than a
    trim at short context and *cheaper* at long context — which is exactly the regime
    this project's headline claim (long-context local inference) lives in.
    """

    def checkpoint(self) -> object:
        """Capture current generation state; returns an opaque handle."""
        ...

    def restore(self, handle: object) -> np.ndarray:
        """Restore state captured by `checkpoint`; return next-token logits."""
        ...

    def snapshot_bytes(self) -> int:
        """Size of one checkpoint handle, for the cost table."""
        ...


def supports_chat(adapter: object) -> bool:
    return callable(getattr(adapter, "render_chat", None))


def supports_snapshot(adapter: object) -> bool:
    return callable(getattr(adapter, "checkpoint", None)) and \
        callable(getattr(adapter, "restore", None))


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
