"""MLX backend for the Mamba POC (Apple Silicon).

Implements `ModelInterface` with a Mamba-2 / SSD block: a SCALAR-A selective SSM
(Dao & Gu, State Space Duality), multi-head with one shared B/C group. Scalar A is
what lets the scan become a sequence of matmuls. Two code paths must agree
(forward_step_parity, fp32 ~1e-4 rel):

  * `parallel(x)`     : the SSD chunked-matmul scan over the full sequence (training
                        path). Intra-chunk via matmul, a short recurrence across
                        chunk-states. All decays are exp of non-positive sums, so it
                        is overflow-safe; chunk length Q comes from `chunk_size`.
  * `recurrence(x, h)`: one-step state update (inference path).

Memory at depth is controlled with gradient checkpointing (recompute each layer's
forward in backward); see `MLXMambaModel.__init__`.

This file imports `mlx`, so it does NOT import on Linux/CUDA hosts — intentional
and allowed: it lives below the seam and nothing portable imports it.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint as _checkpoint
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


def _segsum(x: Array) -> Array:
    """Lower-triangular segment-sum. out[..., i, j] = sum_{j < k <= i} x_k for i >= j,
    and -inf above the diagonal. `exp(_segsum(g))` is the within-window decay matrix
    of a scalar SSM (a 1-semiseparable matrix); the -inf upper triangle exp's to 0,
    enforcing causality."""
    T = x.shape[-1]
    xc = mx.cumsum(x, axis=-1)
    seg = xc[..., :, None] - xc[..., None, :]
    mask = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    return mx.where(mask, seg, mx.array(float("-inf"), dtype=seg.dtype))


class SelectiveSSM(nn.Module):
    """Scalar-A Mamba-2 / SSD selective state space.

    d_inner is split into `n_heads` heads of width `head_dim` (P); each head has a
    SCALAR decay A (the SSD restriction that turns the scan into matmuls). B and C
    are a single group of width `d_state` (N), shared across heads. The training
    path (`parallel`) is the SSD chunked-matmul scan; the inference path
    (`recurrence`) is the matching one-step recurrence. Both compute the same
    function (forward_step_parity, fp32 ~1e-4)."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        d_inner, d_state = config.d_inner, config.d_state
        dt_rank, H = config.dt_rank_resolved, config.n_heads

        # x_proj produces (dt_pre, B, C); B and C are one group, shared across heads.
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        # dt is PER-HEAD in Mamba-2: dt_proj maps dt_rank -> n_heads.
        self.dt_proj = nn.Linear(dt_rank, H, bias=True)

        # Scalar decay A per head, stored as log: A = -exp(A_log). S4D-real init.
        self.A_log = mx.log(mx.arange(1, H + 1, dtype=mx.float32))   # (H,)
        self.D = mx.ones((H,))                                       # (H,) skip per head

        self._init_dt_bias()

    def _init_dt_bias(self) -> None:
        """LOAD-BEARING dt-projection bias init (inverse-softplus into a small
        positive range). Without this the model fails to learn recall. Now PER-HEAD
        (shape n_heads), since Mamba-2 has one dt per head.

            dt   = uniform(log(dt_min), log(dt_max)).exp().clamp(min=dt_init_floor)
            bias = dt + log(-expm1(-dt))          # inverse softplus
        """
        c = self.config
        dt = mx.exp(mx.random.uniform(
            low=math.log(c.dt_min), high=math.log(c.dt_max),
            shape=(c.n_heads,)))
        dt = mx.maximum(dt, c.dt_init_floor)
        self.dt_proj.bias = dt + mx.log(-mx.expm1(-dt))             # (H,)

    # --- shared projections --------------------------------------------------
    def _project(self, x: Array):
        """x: (..., d_inner) -> delta (..., H), a (H,), B (..., N), C (..., N)."""
        dt_rank, d_state = self.config.dt_rank_resolved, self.config.d_state
        proj = self.x_proj(x)
        dt_pre = proj[..., :dt_rank]
        B = proj[..., dt_rank:dt_rank + d_state]
        C = proj[..., dt_rank + d_state:]
        delta = _softplus(self.dt_proj(dt_pre))   # (..., H) per head
        a = -mx.exp(self.A_log)                    # (H,) scalar decay
        return delta, a, B, C

    def parallel(self, x: Array) -> Array:
        """SSD chunked-matmul scan. x: (B, L, d_inner) -> (B, L, d_inner).

        Pads L up to a multiple of the chunk length Q (padded steps carry zero
        input and are trimmed). All decays are exp of non-positive sums, so the
        scan is overflow-safe by construction."""
        B_, L, d_inner = x.shape
        H, P, N = self.config.n_heads, self.config.head_dim, self.config.d_state
        Q = self.config.chunk_size or 64
        delta, a, Bm, Cm = self._project(x)        # delta (B,L,H); Bm,Cm (B,L,N)
        X = x.reshape(B_, L, H, P)

        pad = (-L) % Q
        if pad:
            zc = lambda t, shp: mx.concatenate([t, mx.zeros(shp, dtype=t.dtype)], axis=1)
            X, delta = zc(X, (B_, pad, H, P)), zc(delta, (B_, pad, H))
            Bm, Cm = zc(Bm, (B_, pad, N)), zc(Cm, (B_, pad, N))
        Lp, nc = L + pad, (L + pad) // Q
        g = delta * a                              # (B,Lp,H) log-decay (<= 0)
        Xin = delta[..., None] * X                 # (B,Lp,H,P) input = dt * X

        gc = g.reshape(B_, nc, Q, H).transpose(0, 3, 1, 2)   # (B,H,nc,Q)
        Xc = Xin.reshape(B_, nc, Q, H, P)
        Bc = Bm.reshape(B_, nc, Q, N)
        Cc = Cm.reshape(B_, nc, Q, N)
        Acum = mx.cumsum(gc, axis=-1)                        # (B,H,nc,Q)

        # 1) intra-chunk diagonal block (attention-like, within each chunk)
        Lmask = mx.exp(_segsum(gc))                          # (B,H,nc,Q,Q)
        CB = mx.einsum("bcin,bcjn->bcij", Cc, Bc)            # (B,nc,Q,Q)
        Ydiag = mx.einsum("bhcij,bcij,bcjhp->bcihp", Lmask, CB, Xc)
        # 2) each chunk's final state, from that chunk's inputs only
        decay_end = mx.exp(Acum[..., -1:] - Acum)            # (B,H,nc,Q)
        states = mx.einsum("bhcj,bcjhp,bcjn->bchpn", decay_end, Xc, Bc)
        # 3) inter-chunk recurrence (the only scan, over nc chunk-states)
        states = mx.concatenate(
            [mx.zeros((B_, 1, H, P, N), dtype=states.dtype), states], axis=1)
        chunk_tot = mx.pad(Acum[..., -1], [(0, 0), (0, 0), (1, 0)])   # (B,H,nc+1)
        decay_chunk = mx.exp(_segsum(chunk_tot))                      # (B,H,nc+1,nc+1)
        new_states = mx.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        S_enter = new_states[:, :-1]                                  # (B,nc,H,P,N)
        # 4) off-diagonal output: entering state decayed to each position
        out_decay = mx.exp(Acum)                                      # (B,H,nc,Q)
        Yoff = mx.einsum("bcin,bchpn,bhci->bcihp", Cc, S_enter, out_decay)

        Y = (Ydiag + Yoff).reshape(B_, Lp, H, P)[:, :L]             # (B,L,H,P)
        Y = Y + X[:, :L] * self.D[None, None, :, None]             # skip
        return Y.reshape(B_, L, d_inner)

    def recurrence(self, x: Array, state: State) -> Tuple[Array, State]:
        """One timestep. x: (B, d_inner), state h: (B, H, P, N) -> y: (B, d_inner)."""
        B_ = x.shape[0]
        H, P = self.config.n_heads, self.config.head_dim
        delta, a, Bm, Cm = self._project(x)        # delta (B,H); Bm,Cm (B,N)
        Xh = x.reshape(B_, H, P)
        dA = mx.exp(delta * a)                      # (B,H)
        dBx = (delta[..., None] * Xh)[..., None] * Bm[:, None, None, :]  # (B,H,P,N)
        h = dA[:, :, None, None] * state + dBx      # (B,H,P,N)
        y = mx.sum(h * Cm[:, None, None, :], axis=-1) + Xh * self.D[None, :, None]
        return y.reshape(B_, -1), h


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
        # Gradient checkpointing: recompute each layer's forward in the backward pass
        # instead of retaining its activations. Essential at poc scale — without it the
        # 24-layer backward exceeds 32GB and swaps. `step` (inference) is unaffected.
        if config.grad_checkpoint:
            self._layer_fns = [_checkpoint(l, l.forward_seq) for l in self.layers]
        else:
            self._layer_fns = [l.forward_seq for l in self.layers]

    def _head(self, h: Array) -> Array:
        if self._tie_embeddings:
            return h @ self.embedding.weight.T
        return self.lm_head(h)

    # --- ModelInterface ---
    def forward(self, token_batch: Array) -> Array:
        h = self.embedding(mx.array(token_batch))
        for layer_fn in self._layer_fns:
            h = layer_fn(h)
        return self._head(self.norm_f(h))

    def step(self, token: Array, state: State) -> Tuple[Array, State]:
        h = self.embedding(mx.array(token))
        new_state = []
        for layer, st in zip(self.layers, state):
            h, st2 = layer.step(h, st)
            new_state.append(st2)
        return self._head(self.norm_f(h)), new_state

    def init_state(self, batch_size: int) -> State:
        c = self.config
        di, k = c.d_inner, c.d_conv
        H, P, N = c.n_heads, c.head_dim, c.d_state
        # Per layer: (conv window (B,k-1,di), SSM state (B,H,P,N)).
        return [(mx.zeros((batch_size, k - 1, di)), mx.zeros((batch_size, H, P, N)))
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
