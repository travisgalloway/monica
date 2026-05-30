"""Per-turn snapshot tree for undo/branch (DEFERRED stub, optional even at scale).

Snapshot the full, consistent cross-section of state at each turn boundary (the
resume point). Cap history depth — LRU is correct for pure Mamba since states are
uniform size. NOTE: this rewinds the running summary; it does NOT restore exact
per-item recall (a fixed-state architectural limit, not a cache bug).
"""

from __future__ import annotations

from ..model.interface import ModelInterface


class RewindTree:  # pragma: no cover - deferred
    def __init__(self, model: ModelInterface, max_depth: int = 32):
        raise NotImplementedError("Rewind layer is deferred for the POC.")
