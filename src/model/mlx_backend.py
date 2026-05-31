"""MLX backend for the Mamba POC (Apple Silicon).

Implements `ModelInterface` with the standard Mamba block (Gu & Dao): a diagonal
selective SSM with input-dependent B, C, delta. Two code paths must agree
(forward_step_parity, fp32 ~1e-4 rel):

  * `parallel(x)`     : a chunked closed-form selective scan over the full
                        sequence (training path). Chunking keeps the per-chunk
                        cumulative decay bounded so `exp` does not overflow — a
                        global single-pass cumsum overflows fp32 even at modest
                        seq_len, so we always chunk (default chunk 32).
  * `recurrence(x, h)`: one-step state update (inference path).

This file imports `mlx`, so it does NOT import on Linux/CUDA hosts — intentional
and allowed: it lives below the seam and nothing portable imports it.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np

from .blocks import MambaConfig
from .interface import ModelInterface, State, Array


def _silu(x: Array) -> Array:
    return x * mx.sigmoid(x)


def _softplus(x: Array) -> Array:
    # log(1 + exp(x)), numerically stable via logaddexp(x, 0).
    return mx.logaddexp(x, mx.zeros_like(x))


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((d_model,))

    def __call__(self, x: Array) -> Array:
        norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return self.weight * (x * norm)


class SelectiveSSM(nn.Module):
    """Diagonal-A selective state space with input-dependent B, C, delta."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner, d_state = config.d_inner, config.d_state
        dt_rank = config.dt_rank_resolved

        # x_proj produces (delta, B, C); dt_proj maps dt_rank -> d_inner.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Diagonal A stored as log for stability: A = -exp(A_log). Init A = -(1..d_state)
        # broadcast across channels (the standard "S4D-real" init).
        a = mx.arange(1, d_state + 1, dtype=mx.float32)          # (d_state,)
        self.A_log = mx.log(mx.ones((d_inner, d_state)) * a)     # (d_inner, d_state)
        self.D = mx.ones((d_inner,))

        self._init_dt_bias()

    def _init_dt_bias(self) -> None:
        """LOAD-BEARING dt-projection bias init (inverse-softplus into a small
        positive range). Without this the model fails to learn recall.

            dt   = uniform(log(dt_min), log(dt_max)).exp().clamp(min=dt_init_floor)
            bias = dt + log(-expm1(-dt))          # inverse softplus
        """
        c = self.config
        dt = mx.exp(mx.random.uniform(
            low=math.log(c.dt_min), high=math.log(c.dt_max),
            shape=(c.d_inner,)))
        dt = mx.maximum(dt, c.dt_init_floor)
        inv_softplus = dt + mx.log(-mx.expm1(-dt))
        self.dt_proj.bias = inv_softplus

    # --- shared projections --------------------------------------------------
    def _project(self, x: Array):
        dt_rank, d_state = self.config.dt_rank_resolved, self.config.d_state
        proj = self.x_proj(x)
        dt = proj[..., :dt_rank]
        B = proj[..., dt_rank:dt_rank + d_state]
        C = proj[..., dt_rank + d_state:]
        delta = _softplus(self.dt_proj(dt))     # (..., d_inner)
        A = -mx.exp(self.A_log)                  # (d_inner, d_state)
        return delta, A, B, C

    def parallel(self, x: Array) -> Array:
        """Chunked closed-form selective scan. x: (B, L, d_inner) -> (B, L, d_inner)."""
        B_, L, d_inner = x.shape
        delta, A, Bmat, Cmat = self._project(x)

        # Discretized terms: a = delta*A (log-decay), deltaBu = delta*B*x.
        a = delta[..., None] * A[None, None]                 # (B, L, di, ds)
        deltaBu = delta[..., None] * Bmat[:, :, None, :] * x[..., None]  # (B,L,di,ds)

        chunk = self.config.chunk_size or min(L, 32)
        d_state = self.config.d_state
        h_carry = mx.zeros((B_, d_inner, d_state))
        ys = []
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            a_c = a[:, s:e]                                  # (B, lc, di, ds)
            bu_c = deltaBu[:, s:e]
            C_c = Cmat[:, s:e]                               # (B, lc, ds)
            A_cum = mx.cumsum(a_c, axis=1)                   # inclusive log-decay
            # h_j = exp(A_cum_j) * (h_carry + sum_{i<=j} exp(-A_cum_i) * bu_i)
            inner = mx.cumsum(mx.exp(-A_cum) * bu_c, axis=1)
            h = mx.exp(A_cum) * (h_carry[:, None] + inner)   # (B, lc, di, ds)
            ys.append(mx.sum(h * C_c[:, :, None, :], axis=-1))  # (B, lc, di)
            h_carry = h[:, -1]
        y = mx.concatenate(ys, axis=1)                       # (B, L, di)
        return y + x * self.D

    def recurrence(self, x: Array, state: State) -> Tuple[Array, State]:
        """Single timestep. x: (B, d_inner), state h: (B, d_inner, d_state)."""
        delta, A, Bmat, Cmat = self._project(x)
        dA = mx.exp(delta[..., None] * A[None])              # (B, di, ds)
        dBu = delta[..., None] * Bmat[:, None, :] * x[..., None]
        h = dA * state + dBu                                 # (B, di, ds)
        y = mx.sum(h * Cmat[:, None, :], axis=-1) + x * self.D
        return y, h


class MambaBlock(nn.Module):
    """pre-norm -> input proj -> split main+gate -> causal depthwise conv -> SiLU
    -> selective SSM -> * SiLU(gate) -> output proj, with a residual."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner = config.d_inner
        self.norm = RMSNorm(config.d_model)
        self.in_proj = nn.Linear(config.d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, config.d_conv,
                              groups=d_inner, padding=config.d_conv - 1)
        self.ssm = SelectiveSSM(config)
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)

    def forward_seq(self, x: Array) -> Array:
        L = x.shape[1]
        xn = self.norm(x)
        x_main, z = mx.split(self.in_proj(xn), 2, axis=-1)   # (B,L,di) each
        # Causal depthwise conv: pad both sides (d_conv-1), keep the first L outputs.
        xc = self.conv(x_main)[:, :L]
        xc = _silu(xc)
        y = self.ssm.parallel(xc)
        y = y * _silu(z)
        return x + self.out_proj(y)

    def step(self, x: Array, state: State) -> Tuple[Array, State]:
        conv_state, ssm_state = state                        # (B,k-1,di), (B,di,ds)
        xn = self.norm(x)
        x_main, z = mx.split(self.in_proj(xn), 2, axis=-1)    # (B,di) each
        window = mx.concatenate([conv_state, x_main[:, None, :]], axis=1)  # (B,k,di)
        # depthwise conv at this timestep: sum over kernel positions.
        wk = self.conv.weight[:, :, 0].T                      # (k, di)
        conv_out = mx.sum(window * wk[None], axis=1) + self.conv.bias       # (B, di)
        xc = _silu(conv_out)
        y, new_ssm = self.ssm.recurrence(xc, ssm_state)
        y = y * _silu(z)
        out = x + self.out_proj(y)
        return out, (window[:, 1:], new_ssm)


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
        self._tie_embeddings = config.tie_embeddings
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._state = None

    def _head(self, h: Array) -> Array:
        if self._tie_embeddings:
            return h @ self.embedding.weight.T
        return self.lm_head(h)

    # --- ModelInterface ---
    def forward(self, token_batch: Array) -> Array:
        h = self.embedding(mx.array(token_batch))
        for layer in self.layers:
            h = layer.forward_seq(h)
        return self._head(self.norm_f(h))

    def step(self, token: Array, state: State) -> Tuple[Array, State]:
        h = self.embedding(mx.array(token))
        new_state = []
        for layer, st in zip(self.layers, state):
            h, st2 = layer.step(h, st)
            new_state.append(st2)
        return self._head(self.norm_f(h)), new_state

    def init_state(self, batch_size: int) -> State:
        di, ds, k = self.config.d_inner, self.config.d_state, self.config.d_conv
        return [(mx.zeros((batch_size, k - 1, di)), mx.zeros((batch_size, di, ds)))
                for _ in range(self.config.n_layers)]

    def get_state(self) -> State:
        return self._state

    def set_state(self, state: State) -> None:
        self._state = state

    def save(self, path: str) -> None:
        from ..train.checkpoint import save_weights
        save_weights(self._portable_state_dict(), path, config=self.config)

    def load(self, path: str) -> None:
        from ..train.checkpoint import load_weights
        load_weights(self, path)

    def _portable_state_dict(self) -> dict:
        # Flatten MLX params to {name: numpy array} for safetensors. With tied
        # embeddings there is no separate head param, so nothing to drop.
        return {k: np.array(v) for k, v in tree_flatten(self.parameters())}

    def _load_portable(self, weights: dict) -> None:
        params = [(k, mx.array(v)) for k, v in weights.items()]
        self.update(tree_unflatten(params))
        mx.eval(self.parameters())
