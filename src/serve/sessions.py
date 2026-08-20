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
    logits = store.prefill("s1", prompt_ids)   # whole prompt in ONE parallel scan
    logits = store.step("s1", next_id)         # then one token at a time

    history = SessionHistory(store, "s1")      # composes the store with serve.rewind
    history.commit_turn()                      # snapshot this turn boundary into the tree
    history.rewind_turns(1)                    # ... and restore it later, in place

`SessionHistory` (below) is the composition point between this module and
`serve.rewind.RewindTree`; `scripts/generate.py --interactive` is the entry point that
drives it (#305).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np

from ..model.blocks import MambaConfig
from ..model.interface import Array, ModelInterface, State
from .rewind import RewindTree

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
        # Tokens consumed per session, tracked ONLY to enforce prefill's fresh-session-only
        # contract (#165). `None` means "unknown" — a restored snapshot's position is not
        # recoverable above the seam, and unknown must not be mistaken for fresh.
        self._positions: "dict[str, Optional[int]]" = {}
        self._per_session_bytes = per_session_state_bytes(model.config)

        if max_concurrent is not None:
            if max_concurrent < 1:
                raise ValueError(f"max_concurrent={max_concurrent} must be >= 1")
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
        self._positions[session_id] = 0
        return self._maybe_evict()

    def remove(self, session_id: str) -> None:
        del self._states[session_id]  # KeyError if absent — explicit
        self._positions.pop(session_id, None)

    # --- advance ---
    def prefill(self, session_id: str, prompt_ids: Sequence[int]) -> Array:
        """Consume a whole prompt in ONE parallel scan; return the last position's logits.

        Equivalent to `step`-ing every id in `prompt_ids` — same state, same final logits
        (gated by `src/conformance/prefill_decode_parity.py`) — but one parallel pass
        instead of `len(prompt_ids)` sequential graph evaluations.

        **Fresh sessions only.** The seam's `prefill` seeds attention RoPE from absolute
        position 0, so it cannot extend a session that has already consumed tokens; a
        session restored via `set_state` has an unknowable position and is refused too.
        Raises `ValueError` in both cases rather than silently mis-positioning a prompt.
        """
        if len(prompt_ids) == 0:
            raise ValueError("prompt_ids must be non-empty")
        pos = self._positions.get(session_id)
        if session_id not in self._states:
            raise KeyError(session_id)
        if pos != 0:
            raise ValueError(
                f"session {session_id!r} is at position {pos!r}; prefill is fresh-session "
                "only (the seam seeds attention RoPE positions from 0 — see "
                "ModelInterface.prefill). Use `step` to extend a live session, or create a "
                "fresh session and prefill the full context.")
        ids = np.asarray(prompt_ids, dtype=np.int64)[None]                 # (1, L)
        logits, state = self.model.prefill(ids, last_only=True)            # logits (1, V)
        # Clone on the way in, mirroring `set_state`'s discipline: the store must own an
        # independent copy that a later in-place `step` cannot alias.
        self._states[session_id] = self.model.clone_state(state)
        self._states.move_to_end(session_id)
        self._positions[session_id] = len(prompt_ids)
        return logits

    def step(self, session_id: str, token: int) -> Array:
        """Feed one token to a session; return its logits. Marks it most-recently-used."""
        tok = np.asarray([token], dtype=np.int64)  # (1,) — batch=1; backend casts at seam
        logits, new_state = self.model.step(tok, self._states[session_id])
        self._states[session_id] = new_state
        self._states.move_to_end(session_id)
        pos = self._positions.get(session_id)
        if pos is not None:
            self._positions[session_id] = pos + 1
        return logits

    # --- snapshot / restore (the bridge to serve.rewind) ---
    def get_state(self, session_id: str) -> State:
        """An independent snapshot of a session's state, safe to retain (cloned).

        `SessionHistory.commit_turn` (below) is the caller: it hands the clone straight to
        `RewindTree.commit`, which retains it as a turn-boundary node.
        """
        return self.model.clone_state(self._states[session_id])

    def set_state(self, session_id: str, state: State) -> None:
        """Restore a session to a previously captured snapshot (e.g. a rewind target).

        Clones on the way in — symmetric with `get_state` — so the store owns an
        independent copy. Without this, installing a `RewindTree.rewind(...)` result
        (which aliases the tree node) lets a subsequent in-place `step` mutation on a
        numpy/CUDA state silently corrupt the retained snapshot.

        `SessionHistory.rewind_to` (below) is the caller: it pairs `RewindTree.rewind`
        with this method, which is the whole of the rewind operation.
        """
        self._states[session_id] = self.model.clone_state(state)
        self._states.move_to_end(session_id)
        # A restored snapshot's token position is unknowable above the seam, so mark it
        # unknown — `prefill` must refuse it rather than assume a fresh RoPE origin.
        self._positions[session_id] = None

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
            self._positions.pop(sid, None)
            evicted.append(sid)
        return evicted


class SessionHistory:
    """Turn-boundary history for ONE session: the composition of `SessionStore` and
    `serve.rewind.RewindTree`.

    `SessionStore` owns the live state and `RewindTree` owns the retained snapshots;
    neither knows about the other. This class is the join: `commit_turn` snapshots the
    live session (`store.get_state`, which clones at the seam) into the tree, and
    `rewind_to` pulls a snapshot back out (`tree.rewind`) and reinstalls it
    (`store.set_state`). It never touches the model and never inspects a `State` — above
    the seam `State` is opaque, so the blob is only ever passed through.

    **Rewinds count RETAINED boundaries, not wall-clock turns.** The tree is LRU-capped at
    `max_depth`, and `RewindTree._detach` reparents an evicted node's children onto its
    grandparent so deeper branches outlive the cap. One "parent hop" therefore skips any
    turn that has already been evicted. `rewind_turns(n)` walks `n` retained parents; if
    the run is long enough for eviction, that is not the same as "n turns ago", and the
    caller is told the retained depth when it asks for more than exists.

    Guards **reject**; none clamps and none silently no-ops. Asking to rewind further than
    the retained history raises rather than landing on the root, because a rewind that
    quietly lands somewhere else is indistinguishable from one that worked.
    """

    def __init__(self, store: SessionStore, session_id: str, max_depth: int = 32) -> None:
        if session_id not in store:
            raise KeyError(
                f"session {session_id!r} does not exist in the store; create it before "
                "attaching a SessionHistory")
        self.store = store
        self.session_id = session_id
        self.tree = RewindTree(max_depth=max_depth)   # raises if max_depth < 1
        # Ids in commit order. The tree evicts, so this is a superset of what is retained;
        # `retained_ids` filters it through the tree's own membership test rather than
        # reaching into `RewindTree._nodes`.
        self._committed: list[int] = []

    # --- the composition ---
    def commit_turn(self) -> int:
        """Snapshot the live session as a turn boundary. Returns the new node id."""
        node_id = self.tree.commit(self.store.get_state(self.session_id))
        # Drop ids the tree has already evicted before appending the new one, so `_committed`
        # stays bounded by `max_depth` too. The tree is LRU-capped; without this the id list
        # would grow without bound over a long-lived session even though the tree itself
        # never does — `retained_ids()` masked the leak by filtering on every read, but never
        # freed the stale entries themselves.
        self._committed = self.retained_ids()
        self._committed.append(node_id)
        return node_id

    def rewind_turns(self, n: int) -> int:
        """Walk `n` retained parents from the current node and restore that snapshot.

        Returns the node id landed on. Raises `ValueError` — without touching the session
        — if `n < 1`, if nothing has been committed, or if fewer than `n` retained
        ancestors exist (the message names both `n` and the available depth).
        """
        if n < 1:
            raise ValueError(f"n must be >= 1 (got {n})")
        current = self.tree.current()
        if current is None:
            raise ValueError(
                f"session {self.session_id!r} has no committed turn boundaries; "
                "call commit_turn() before rewinding")
        target = current
        for _ in range(n):
            parent = self.tree.parent(target)
            if parent is None:
                available = self.depth() - 1
                raise ValueError(
                    f"cannot rewind {n} turn(s): session {self.session_id!r} retains "
                    f"{self.depth()} boundary/boundaries including the current one, so at "
                    f"most {available} rewind hop(s) are possible "
                    f"(retained ids: {self.retained_ids()})")
            target = parent
        return self.rewind_to(target)

    def rewind_to(self, node_id: int) -> int:
        """Restore an absolute node id. Raises `LookupError` if it is unknown or evicted."""
        try:
            state = self.tree.rewind(node_id)
        except KeyError:
            raise LookupError(
                f"node {node_id} is not retained by session {self.session_id!r}'s rewind "
                f"tree (retained ids: {self.retained_ids()})") from None
        self.store.set_state(self.session_id, state)
        return node_id

    # --- introspection ---
    def current(self) -> Optional[int]:
        return self.tree.current()

    def __len__(self) -> int:
        """Retained nodes, as the tree itself counts them (the LRU cap's own view)."""
        return len(self.tree)

    def retained_ids(self) -> list[int]:
        """Committed ids that survived eviction, in commit order."""
        return [i for i in self._committed if i in self.tree]

    def depth(self) -> int:
        """Retained ancestors of the current node, inclusive. 0 if nothing is committed."""
        node = self.tree.current()
        n = 0
        while node is not None:
            n += 1
            node = self.tree.parent(node)
        return n

    def nodes(self) -> list[tuple[int, Optional[int], list[int]]]:
        """`(id, parent_id, children)` for every retained node, in commit order."""
        return [(i, self.tree.parent(i), self.tree.children(i)) for i in self.retained_ids()]

    @property
    def max_depth(self) -> int:
        return self.tree.max_depth

    def per_node_bytes(self) -> int:
        """Conservative bytes for ONE retained snapshot — pure config arithmetic."""
        return per_session_state_bytes(self.store.model.config)

    def budget_bytes(self) -> int:
        """Ceiling on what the retained tree can cost: `max_depth x per_node_bytes()`.

        Conservative (fp32-charged) and exact in shape: snapshots are uniform because the
        recurrent state is fixed-size, which is what makes a flat node count the right
        budget in the first place.
        """
        return self.max_depth * self.per_node_bytes()
