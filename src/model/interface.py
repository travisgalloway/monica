"""The seam: the abstract model protocol.

THIS MODULE MUST NOT IMPORT ANY BACKEND (no `mlx`, no `torch`/CUDA). Everything
above the seam (train/serve/eval/conformance) depends only on this interface and
on `blocks.MambaConfig`. Each backend (`mlx_backend`, `cuda_backend`) provides a
concrete subclass implementing exactly these methods.

`State` is intentionally typed as `Any`: its concrete representation is
backend-specific (an MLX array tuple, a torch tensor, ...). Code above the seam
treats it as an opaque, fixed-size blob that it can snapshot and restore.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

from .blocks import MambaConfig

# Opaque, backend-defined recurrent state.
State = Any
# Opaque, backend-defined logits / token-batch arrays.
Array = Any


class ModelInterface(ABC):
    """Contract every backend implements. Lock this before building on top of it."""

    #: Single source of truth for architecture parameters.
    config: MambaConfig

    # --- training path ---
    @abstractmethod
    def forward(self, token_batch: Array) -> Array:
        """Full-sequence parallel forward. `token_batch` is (batch, seq_len) ids.

        Returns logits (batch, seq_len, vocab_size). Uses the parallel scan.
        """

    # --- inference path ---
    @abstractmethod
    def step(self, token: Array, state: State) -> Tuple[Array, State]:
        """Single-token recurrence. Returns (logits, new_state).

        Must agree with `forward` within tolerance for the same inputs
        (guarded by conformance/forward_step_parity).
        """

    @abstractmethod
    def init_state(self, batch_size: int) -> State:
        """Fresh, zeroed recurrent state for `batch_size` sequences."""

    # --- snapshot / restore (serve + rewind) ---
    @abstractmethod
    def get_state(self) -> State:
        """Return a copy of the current recurrent state."""

    @abstractmethod
    def set_state(self, state: State) -> None:
        """Restore recurrent state previously produced by get_state/step."""

    @abstractmethod
    def clone_state(self, state: State) -> State:
        """Return an independent snapshot of `state`, safe to retain while stepping.

        The serving layer (serve/sessions, serve/rewind) holds many states at once and
        snapshots them at turn boundaries. On an immutable-array backend (MLX) a
        structural copy suffices; a backend whose `step` mutates buffers in place must
        deep-copy here so the snapshot cannot be aliased by later steps.
        """

    # --- checkpointing (weights via checkpoint module) ---
    @abstractmethod
    def save(self, path: str) -> None:
        """Persist weights in a portable format (safetensors). See train/checkpoint."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load weights from a portable checkpoint produced by `save`."""
