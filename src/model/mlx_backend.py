"""MLX backend for the Mamba POC (Apple Silicon).

SKELETON. The structure, method signatures, and the load-bearing initialization
notes are in place; the heavy SSM math (parallel scan, chunked scan) is marked
TODO and is completed + run on Apple Silicon. This file imports `mlx`, so it does
NOT import on Linux/CUDA hosts — that is intentional and allowed: it lives below
the seam and nothing portable imports it.

Implements `ModelInterface` exactly. Reference for the selective-scan math:
the standard Mamba block (Gu & Dao). Keep all comparisons against the sequential
reference in fp32 (conformance tolerance ~1e-4 relative).
"""

from __future__ import annotations

import math
from typing import Tuple

import mlx.core as mx  # noqa: F401  (Apple Silicon only)
import mlx.nn as nn

from .blocks import MambaConfig
from .interface import ModelInterface, State, Array


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((d_model,))

    def __call__(self, x: Array) -> Array:
        # TODO[mac]: x * rsqrt(mean(x^2) + eps) * weight
        raise NotImplementedError("Complete RMSNorm on Apple Silicon.")


class SelectiveSSM(nn.Module):
    """Diagonal-A selective state space with input-dependent B, C, delta.

    Two code paths that MUST agree (forward_step_parity):
      * `parallel(x)`  : cumsum-based closed-form scan over the full sequence.
      * `recurrence(x, state)` : one-step update used by `step`.

    Chunking (`config.chunk_size`) bounds the scan's working set and prevents
    `exp` overflow for long sequences. At seq_len <= ~2k it is unused.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner, d_state = config.d_inner, config.d_state
        dt_rank = config.dt_rank_resolved

        # x_proj produces (delta, B, C); dt_proj maps dt_rank -> d_inner.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Diagonal A stored as log for stability: A = -exp(A_log).
        self.A_log = mx.zeros((d_inner, d_state))  # TODO[mac]: init to log of 1..d_state
        self.D = mx.ones((d_inner,))

        self._init_dt_bias()

    def _init_dt_bias(self) -> None:
        """LOAD-BEARING dt-projection bias init (inverse-softplus into a small
        positive range). Without this the model fails to learn recall.

            dt = uniform(log(dt_min), log(dt_max)).exp().clamp(min=dt_init_floor)
            bias = dt + log(-expm1(-dt))   # inverse softplus
        """
        # TODO[mac]: implement with mx; set self.dt_proj.bias to inv_softplus(dt).
        raise NotImplementedError("Complete dt-bias init on Apple Silicon.")

    def parallel(self, x: Array) -> Array:
        # TODO[mac]: cumsum closed-form selective scan (chunked if chunk_size set).
        raise NotImplementedError("Complete parallel scan on Apple Silicon.")

    def recurrence(self, x: Array, state: State) -> Tuple[Array, State]:
        # TODO[mac]: single-step state update; must match `parallel` in fp32.
        raise NotImplementedError("Complete recurrence on Apple Silicon.")


class MambaBlock(nn.Module):
    """input proj -> split main+gate -> causal depthwise conv -> SiLU -> SSM
    -> * SiLU(gate) -> output proj. Wrapped pre-norm with a residual outside."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner = config.d_inner
        self.in_proj = nn.Linear(config.d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, config.d_conv,
                              groups=d_inner, padding=config.d_conv - 1)
        self.ssm = SelectiveSSM(config)
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)

    def forward_seq(self, x: Array) -> Array:
        # TODO[mac]: parallel-scan path for training.
        raise NotImplementedError("Complete MambaBlock.forward_seq on Apple Silicon.")

    def step(self, x: Array, state: State) -> Tuple[Array, State]:
        # TODO[mac]: recurrence path for inference (conv state + ssm state).
        raise NotImplementedError("Complete MambaBlock.step on Apple Silicon.")


# --------------------------------------------------------------------------- #
# Top-level model implementing the seam
# --------------------------------------------------------------------------- #
class MLXMambaModel(ModelInterface, nn.Module):
    def __init__(self, config: MambaConfig):
        nn.Module.__init__(self)
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [MambaBlock(config) for _ in range(config.n_layers)]
        self.norm_f = RMSNorm(config.d_model)
        # Tied LM head: head weight IS the embedding weight (mandatory at scale).
        self._tie_embeddings = config.tie_embeddings
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    # --- ModelInterface ---
    def forward(self, token_batch: Array) -> Array:
        raise NotImplementedError("Complete forward on Apple Silicon.")

    def step(self, token: Array, state: State) -> Tuple[Array, State]:
        raise NotImplementedError("Complete step on Apple Silicon.")

    def init_state(self, batch_size: int) -> State:
        raise NotImplementedError("Complete init_state on Apple Silicon.")

    def get_state(self) -> State:
        raise NotImplementedError("Complete get_state on Apple Silicon.")

    def set_state(self, state: State) -> None:
        raise NotImplementedError("Complete set_state on Apple Silicon.")

    def save(self, path: str) -> None:
        # Delegates to train.checkpoint.save_weights (portable safetensors).
        from ..train.checkpoint import save_weights
        save_weights(self._portable_state_dict(), path, config=self.config)

    def load(self, path: str) -> None:
        from ..train.checkpoint import load_weights
        load_weights(self, path)

    def _portable_state_dict(self) -> dict:
        # TODO[mac]: flatten params to {name: np-convertible array} for safetensors.
        raise NotImplementedError("Complete portable state-dict export on Apple Silicon.")
