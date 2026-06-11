"""CUDA backend (scale-up milestone) — STUB.

Built at the CUDA scale-up against the now-finalized `ModelInterface`, using the
`mamba-ssm` fused selective-scan kernel for `forward` and the recurrence form for
`step`. Certified against the MLX backend by `conformance/backend_parity.py`
(comparison run in fp32, ~1e-4 relative tolerance).

No `torch` import here yet so the module is inspectable without a CUDA stack; the
import is added when the backend is implemented.
"""

from __future__ import annotations

from typing import Tuple

from .blocks import MambaConfig
from .interface import ModelInterface, State, Array


class CUDAMambaModel(ModelInterface):
    """Placeholder. Implement with PyTorch + mamba-ssm at the scale-up milestone."""

    def __init__(self, config: MambaConfig):
        config.validate()
        self.config = config
        raise NotImplementedError(
            "CUDA backend is built at the scale-up milestone (mamba-ssm fused kernel). "
            "It must implement ModelInterface and pass backend_parity in fp32."
        )

    def forward(self, token_batch: Array) -> Array:  # pragma: no cover
        raise NotImplementedError

    def step(self, token: Array, state: State) -> Tuple[Array, State]:  # pragma: no cover
        raise NotImplementedError

    def init_state(self, batch_size: int) -> State:  # pragma: no cover
        raise NotImplementedError

    def get_state(self) -> State:  # pragma: no cover
        raise NotImplementedError

    def set_state(self, state: State) -> None:  # pragma: no cover
        raise NotImplementedError

    def clone_state(self, state: State) -> State:  # pragma: no cover
        raise NotImplementedError

    def save(self, path: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def load(self, path: str) -> None:  # pragma: no cover
        raise NotImplementedError
