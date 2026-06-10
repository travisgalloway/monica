"""Multi-session state map: session_id -> that session's fixed-size recurrent state.

Multi-session serving is the easy regime for Mamba: constant memory per session, so
`max_concurrent = memory_budget / per_session_state`. We serialize within a session
(the recurrence is a sequential dependency) and parallelize across sessions — each
session's state is an independent value, so there is no cross-session coupling. No
batching at POC scale; this is single-threaded bookkeeping over the seam's functional
`step(token, state) -> (logits, new_state)` primitive.

Portable (no `mlx`/`torch`): state is an opaque blob from the seam; we only ever pass
it back into `model.step`, snapshot it via `model.clone_state`, and key it by id. The
per-session byte size is pure config arithmetic, so the memory budget is computable
here without a backend.

Usage::

    store = SessionStore(model, memory_budget_bytes=2 * 1024**3)
    store.create("s1")
    for tok in prompt_ids:
        logits = store.step("s1", tok)   # caller samples the next token from logits
    snapshot = store.get_state("s1")     # hand to serve.rewind.RewindTree.commit(...)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np

from ..model.blocks import MambaConfig
from ..model.interface import Array, ModelInterface, State

# Bytes per state element by declared precision. Note the conservative default below:
# the MLX backend keeps the SSM state in fp32 even when precision is fp16, so a naive
# `floats * _BYTES_PER[precision]` under-counts. Budgeting must not under-count.
_BYTES_PER = {"fp32": 4, "fp16": 2, "bf16": 2}


def per_session_state_floats(config: MambaConfig) -> int:
    """Element count of one session's recurrent state (batch=1), from config alone.

    Per layer: a conv window ``(d_conv-1, d_inner)`` plus an SSM state
    ``(n_heads, head_dim, d_state)``. Uniform across sessions — that uniformity is what
    makes the memory budget a simple division.
    """
    per_layer = (config.d_conv - 1) * config.d_inner \
        + config.n_heads * config.head_dim * config.d_state
    return config.n_layers * per_layer


def per_session_state_bytes(config: MambaConfig, *, conservative_fp32: bool = True) -> int:
    """Bytes for one session's state. Defaults to a conservative fp32 upper bound.

    `conservative_fp32=True` charges 4 bytes/element regardless of precision. This
    over-budgets slightly for fp16/bf16 conv state but never under-counts — the right
    failure direction for an admission gate (over-budget = refuse a session; under-budget
    = OOM). Pass `conservative_fp32=False` for the precision-accurate (optimistic) number
    when reporting rather than gating.
    """
    bytes_per = 4 if conservative_fp32 else _BYTES_PER[config.precision]
    return per_session_state_floats(config) * bytes_per


class SessionStore:
    """A bounded, LRU-evicting map from session id to opaque recurrent state.

    Admission is gated either by an explicit `max_concurrent` or by a memory budget
    divided by the per-session state size. When the live set would exceed the cap, the
    least-recently-*stepped* session is evicted (its conversation is dropped).
    """

    def __init__(self, model: ModelInterface, memory_budget_bytes: Optional[int] = None,
                 max_concurrent: Optional[int] = None):
        self.model = model
        self._states: "OrderedDict[str, State]" = OrderedDict()
        self._per_session_bytes = per_session_state_bytes(model.config)

        if max_concurrent is not None:
            self.max_concurrent: Optional[int] = max_concurrent
        elif memory_budget_bytes is not None:
            self.max_concurrent = memory_budget_bytes // self._per_session_bytes
            if self.max_concurrent < 1:
                raise ValueError(
                    f"memory_budget_bytes={memory_budget_bytes} cannot hold even one "
                    f"session (needs {self._per_session_bytes} bytes)."
                )
        else:
            self.max_concurrent = None  # unbounded

    # --- lifecycle ---
    def create(self, session_id: str) -> list[str]:
        """Start a fresh session. Returns the ids evicted to admit it (LRU first)."""
        if session_id in self._states:
            raise ValueError(f"session {session_id!r} already exists")
        self._states[session_id] = self.model.init_state(batch_size=1)
        return self._maybe_evict()

    def remove(self, session_id: str) -> None:
        del self._states[session_id]  # KeyError if absent — explicit

    # --- advance ---
    def step(self, session_id: str, token: int) -> Array:
        """Feed one token to a session; return its logits. Marks it most-recently-used."""
        tok = np.asarray([token], dtype=np.int64)  # (1,) — batch=1; backend casts at seam
        logits, new_state = self.model.step(tok, self._states[session_id])
        self._states[session_id] = new_state
        self._states.move_to_end(session_id)
        return logits

    # --- snapshot / restore (the bridge to serve.rewind) ---
    def get_state(self, session_id: str) -> State:
        """An independent snapshot of a session's state, safe to retain (cloned)."""
        return self.model.clone_state(self._states[session_id])

    def set_state(self, session_id: str, state: State) -> None:
        """Restore a session to a previously captured snapshot (e.g. a rewind target)."""
        self._states[session_id] = state
        self._states.move_to_end(session_id)

    # --- introspection ---
    def __contains__(self, session_id: str) -> bool:
        return session_id in self._states

    def __len__(self) -> int:
        return len(self._states)

    def session_ids(self) -> list[str]:
        """Live session ids, least- to most-recently-stepped."""
        return list(self._states)

    # --- internal ---
    def _maybe_evict(self) -> list[str]:
        evicted: list[str] = []
        if self.max_concurrent is None:
            return evicted
        while len(self._states) > self.max_concurrent:
            sid, _ = self._states.popitem(last=False)  # LRU front
            evicted.append(sid)
        return evicted
