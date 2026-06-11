"""Per-turn snapshot tree for undo/branch.

Snapshot the full, consistent cross-section of recurrent state at each turn boundary
(the resume point) as a node in a tree; `rewind` to any retained node makes it the new
branch point, so a later `commit` forks history there. Cap retained nodes (LRU) — states
are uniform size, so a flat count is the right budget. When an evicted node has children,
we reparent them onto its parent so deeper branches outlive the cap.

NOTE: this rewinds the *running summary* carried in the fixed-size state; it does NOT
restore exact per-item recall (a fixed-state architectural limit, not a cache bug).

Portable (no `mlx`/`torch`): nodes hold opaque `State` blobs only. Snapshots are taken
upstream via `SessionStore.get_state` (which clones at the seam), so this module never
touches the model — `commit` stores the blob it is handed, `rewind` returns one. The
caller reinstalls a rewound state with `SessionStore.set_state`.

Usage::

    tree = RewindTree(max_depth=32)
    n0 = tree.commit(store.get_state("s1"))   # turn boundary
    # ... more turns, each ending in commit(store.get_state("s1")) ...
    store.set_state("s1", tree.rewind(n0))    # undo: session branches from n0
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from ..model.interface import State


@dataclass
class _Node:
    node_id: int
    parent_id: Optional[int]
    state: State
    children: list[int] = field(default_factory=list)


class RewindTree:
    """LRU-capped tree of per-turn state snapshots for one conversation."""

    def __init__(self, max_depth: int = 32):
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.max_depth = max_depth
        self._nodes: "OrderedDict[int, _Node]" = OrderedDict()
        self._next_id = 0
        self._current_id: Optional[int] = None

    def commit(self, state: State) -> int:
        """Record a turn-boundary snapshot as a child of the current node. Returns its id.

        `state` should be an independent snapshot (e.g. from `SessionStore.get_state`,
        which clones at the seam).
        """
        node_id = self._next_id
        self._next_id += 1
        parent_id = self._current_id
        self._nodes[node_id] = _Node(node_id, parent_id, state)
        if parent_id is not None:
            self._nodes[parent_id].children.append(node_id)
        self._current_id = node_id
        self._nodes.move_to_end(node_id)  # most-recently-used
        self._evict()
        return node_id

    def rewind(self, node_id: int) -> State:
        """Make `node_id` current and return its snapshot. KeyError if unknown/evicted."""
        node = self._nodes[node_id]
        self._current_id = node_id
        self._nodes.move_to_end(node_id)  # touch — protects it from eviction
        return node.state

    # --- introspection ---
    def current(self) -> Optional[int]:
        return self._current_id

    def parent(self, node_id: int) -> Optional[int]:
        return self._nodes[node_id].parent_id

    def children(self, node_id: int) -> list[int]:
        return list(self._nodes[node_id].children)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: int) -> bool:
        return node_id in self._nodes

    # --- internal ---
    def _evict(self) -> None:
        while len(self._nodes) > self.max_depth:
            victim = self._pick_victim()
            if victim is None:  # only the current node remains
                break
            self._detach(victim)

    def _pick_victim(self) -> Optional[int]:
        for node_id in self._nodes:  # front = least-recently-used first
            if node_id != self._current_id:
                return node_id
        return None

    def _detach(self, node_id: int) -> None:
        node = self._nodes.pop(node_id)
        # Reparent children onto the victim's parent so deeper history survives the cap.
        parent_id = node.parent_id
        if parent_id is not None:
            siblings = self._nodes[parent_id].children
            siblings.remove(node_id)
            siblings.extend(node.children)
        for child_id in node.children:
            self._nodes[child_id].parent_id = parent_id
