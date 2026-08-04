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
    def forward(self, token_batch: Array, seg_ids: Array = None) -> Array:
        """Full-sequence parallel forward. `token_batch` is (batch, seq_len) ids.

        Returns logits (batch, seq_len, vocab_size). Uses the parallel scan.

        `seg_ids` (optional, (batch, seq_len) int) is a per-position **document id** for
        packing-aware training (#68): positions in different documents never interact, so
        recurrent SSM state and attention can't bleed across packed document boundaries.
        `None` (the default) is the original single-segment behavior. Document boundaries
        must be **chunk-aligned** (each document starts at a multiple of `chunk_size`) —
        `src/data/shard.py::pack_sequences` enforces this when packing with `chunk_align`
        set; `src/conformance/doc_boundary_parity.py` verifies a packed multi-doc forward
        equals the per-document forwards.
        """

    # --- inference path ---
    @abstractmethod
    def step(self, token: Array, state: State) -> Tuple[Array, State]:
        """Single-token recurrence. Returns (logits, new_state).

        Must agree with `forward` within tolerance for the same inputs
        (guarded by conformance/forward_step_parity).
        """

    @abstractmethod
    def prefill(self, token_batch: Array, seg_ids: Array = None, *,
                last_only: bool = False) -> Tuple[Array, State]:
        """Consume a whole prompt in ONE parallel scan, returning (logits, state) (#165).

        The recurrence is O(prompt_len) sequential graph evaluations; the SSD chunked
        scan already consumes a full sequence in one pass AND computes the carry-out
        state it used to throw away. This surfaces that state, so serving pays one
        parallel pass for a prompt instead of one `step` per prompt token.

        Shapes. `token_batch` is (batch, seq_len) ids.
          * `last_only=False` -> logits (batch, seq_len, vocab_size), identical to
            `forward` on the same input.
          * `last_only=True`  -> logits (batch, vocab_size), the last position only —
            the same shape `step` returns, which is what the serving loop wants. It
            also lets the backend skip the vocab head over the first `L-1` positions
            (V is 50k+ at poc scale, so this is a real saving).

        The returned `State` is **the state `step` would have produced** after
        consuming all `L` tokens starting from `init_state(batch)`. That equivalence
        is the contract, and it is gated element-wise by
        `src/conformance/prefill_decode_parity.py` in fp32 at ~1e-4.

        **Fresh-session only (v1).** Attention RoPE positions are seeded from absolute
        position 0 here (as in `forward`), while `step` seeds from the KV cache length.
        So `prefill` may only be used from a zeroed state — never to extend a session
        that has already consumed tokens. `SessionStore.prefill` enforces this.
        Follow-up: thread a position offset through the seam so a prompt can be
        appended to a live session.

        `seg_ids` is accepted for signature symmetry with `forward` and currently
        raises `NotImplementedError`. Three things block it: (1) the SSD inter-chunk
        decay matrix's last row is deliberately masked to zero under `seg_ids`
        (the `-2` sentinel in `_chunk_seg_mask`), so the carry-out would read as
        zeros; (2) the conv window's trailing `d_conv-1` rows can straddle a document
        boundary; (3) `AttentionBlock.step` has no per-document masking, so decode
        would attend across boundaries in a multi-document KV cache.
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
