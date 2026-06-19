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
from typing import List, Sequence, Tuple

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
# Mixed precision (issue #27): fp32 master weights + fp16/bf16 compute.
#
# Params stay fp32; we cast to `compute_dtype` (cd) AT THE MATMUL SITE. MLX
# autodiff returns fp32 grads through the `astype` cast, so AdamW keeps updating
# the fp32 masters with no optimizer change, and the fp16 loss scaler (#3) finally
# does meaningful work. For fp32 every cast is guarded to the original op verbatim,
# so the fp32 path stays bit-identical (smoke gate + conformance untouched).
# --------------------------------------------------------------------------- #
_DTYPES = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}


def _f32(t: Array) -> Array:
    """Upcast to fp32, but return `t` unchanged when already fp32 (no identity node)."""
    return t if t.dtype == mx.float32 else t.astype(mx.float32)


def _cast(t: Array, cd) -> Array:
    """Cast to compute dtype `cd`, returning `t` unchanged when already `cd` (no
    identity node) — so the fp32 path stays verbatim, like `_f32`."""
    return t if t.dtype == cd else t.astype(cd)


def _linear(layer: nn.Linear, x: Array, cd) -> Array:
    """nn.Linear with operands cast to `cd`. fp32 routes to the original call verbatim
    (fp16 @ fp32 would promote back to fp32, so BOTH operands must be cast)."""
    if cd == mx.float32:
        return layer(x)
    y = x.astype(cd) @ layer.weight.astype(cd).T
    if "bias" in layer:                              # in/x/out_proj are bias-free; dt_proj has bias
        y = y + layer.bias.astype(cd)
    return y


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((d_model,))

    def __call__(self, x: Array) -> Array:
        if x.dtype == mx.float32:                    # fp32 path: original op, bit-identical
            norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
            return self.weight * (x * norm)
        # Mixed precision: do the reduction in fp32 (weight is fp32), return in x's dtype.
        xf = x.astype(mx.float32)
        norm = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (self.weight * (xf * norm)).astype(x.dtype)


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


def _chunk_seg_mask(seg_ids: Array, B: int, Q: int, nc: int, pad: int) -> Array:
    """Boundary mask for the SSD inter-chunk decay matrix (#68).

    Doc boundaries are chunk-aligned, so each length-Q scan chunk is single-document. The
    inter-chunk recurrence carries each chunk's final state into later chunks; this mask
    zeros that carry across documents, so SSM state can't bleed past a packed boundary. The
    intra-chunk block needs no mask (single-document by construction) and chunk 0 already
    gets a zero entering state from the decay matrix's lower-triangular structure.

    `seg_ids` (B, L) document ids -> (B, nc+1, nc+1): entry [i, c] keeps the decay from
    state c (chunk c-1's final; c=0 is the zero initial state) into entering chunk i iff
    they share a document.
    """
    seg = mx.array(seg_ids)
    if pad:
        seg = mx.concatenate([seg, mx.broadcast_to(seg[:, -1:], (B, pad))], axis=1)
    seg_chunks = seg.reshape(B, nc, Q)[:, :, 0]                  # (B, nc) doc id per chunk
    sentinel_a = mx.full((B, 1), -1, dtype=seg_chunks.dtype)     # zero-state column
    sentinel_b = mx.full((B, 1), -2, dtype=seg_chunks.dtype)     # dropped output row
    src = mx.concatenate([sentinel_a, seg_chunks], axis=1)       # state c axis
    out = mx.concatenate([seg_chunks, sentinel_b], axis=1)       # entering-chunk i axis
    return (out[:, :, None] == src[:, None, :])                  # (B, nc+1, nc+1) bool


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
        """x: (..., d_inner) -> delta (..., H), a (H,), B (..., N), C (..., N).

        `x_proj` is the heavy SSM GEMM and runs in `compute_dtype`; its outputs
        (dt_pre, B, C) upcast to fp32 so the rest of the scan — dt_proj, softplus,
        the decays, cumsum/_segsum, and all state einsums — runs in fp32."""
        cd = _DTYPES[self.config.precision]
        dt_rank, d_state = self.config.dt_rank_resolved, self.config.d_state
        proj = _linear(self.x_proj, x, cd)
        dt_pre = _f32(proj[..., :dt_rank])
        B = _f32(proj[..., dt_rank:dt_rank + d_state])
        C = _f32(proj[..., dt_rank + d_state:])
        delta = _softplus(self.dt_proj(dt_pre))   # (..., H) per head, fp32
        # Long-context extension (#54): divide the discretization step so per-step decay
        # exp(delta*a) moves toward 1, enlarging the receptive field at inference. Guarded
        # so factor 1.0 (the default) leaves delta byte-identical — training/smoke untouched.
        if self.config.long_ctx_factor != 1.0:
            delta = delta / self.config.long_ctx_factor
        a = -mx.exp(self.A_log)                    # (H,) scalar decay, fp32
        return delta, a, B, C

    def parallel(self, x: Array, seg_ids: Array = None) -> Array:
        """SSD chunked-matmul scan. x: (B, L, d_inner) -> (B, L, d_inner).

        Pads L up to a multiple of the chunk length Q (padded steps carry zero
        input and are trimmed). All decays are exp of non-positive sums, so the
        scan is overflow-safe by construction.

        `seg_ids` (B, L) document ids (chunk-aligned boundaries) makes the scan
        packing-aware (#68): the inter-chunk state carry is masked so recurrent state can't
        bleed across documents. `None` is the original single-segment scan."""
        B_, L, d_inner = x.shape
        H, P, N = self.config.n_heads, self.config.head_dim, self.config.d_state
        Q = self.config.chunk_size or 64
        cd = _DTYPES[self.config.precision]
        delta, a, Bm, Cm = self._project(x)        # delta (B,L,H); Bm,Cm (B,L,N) — fp32
        # Upcast the head inputs to fp32 so the whole scan (and the pad zeros, which
        # inherit t.dtype) runs in fp32; the output casts back to cd for out_proj.
        X = _f32(x).reshape(B_, L, H, P)

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
        # 3) inter-chunk recurrence over the nc chunk-states. The canonical SSD form
        # (Dao & Gu) does this as a matmul against an (nc+1, nc+1) decay matrix rather
        # than a sequential scan: parallel/tensor-core-friendly, at the cost of O(nc^2).
        # nc = ceil(L/Q) is small (16 at poc seq 1024 / Q 64), so the matrix is tiny vs
        # the per-position activations; for very long contexts raise `chunk_size` (Q) to
        # keep nc bounded. (A true O(nc) scan would be slower at these scales.)
        states = mx.concatenate(
            [mx.zeros((B_, 1, H, P, N), dtype=states.dtype), states], axis=1)
        chunk_tot = mx.pad(Acum[..., -1], [(0, 0), (0, 0), (1, 0)])   # (B,H,nc+1)
        decay_chunk = mx.exp(_segsum(chunk_tot))                      # (B,H,nc+1,nc+1)
        if seg_ids is not None:                                       # #68: cross-doc reset
            seg_mask = _chunk_seg_mask(seg_ids, B_, Q, nc, pad)       # (B,nc+1,nc+1) bool
            decay_chunk = decay_chunk * seg_mask[:, None].astype(decay_chunk.dtype)
        new_states = mx.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        S_enter = new_states[:, :-1]                                  # (B,nc,H,P,N)
        # 4) off-diagonal output: entering state decayed to each position
        out_decay = mx.exp(Acum)                                      # (B,H,nc,Q)
        Yoff = mx.einsum("bcin,bchpn,bhci->bcihp", Cc, S_enter, out_decay)

        Y = (Ydiag + Yoff).reshape(B_, Lp, H, P)[:, :L]             # (B,L,H,P)
        Y = Y + X[:, :L] * self.D[None, None, :, None]             # skip
        return _cast(Y.reshape(B_, L, d_inner), cd)                # back to compute dtype

    def mixing_matrix(self, x: Array) -> Array:
        """Materialize the dense (B, H, L, L) 1-semiseparable mixing matrix M such that
        `einsum('bhij,bjhp->bihp', M, X) == parallel(x)` (X = the head-split input) **in the
        single-segment case** (`seg_ids=None`). This is the matrix the SSD scan applies; the
        distillation `mixing-match` stage (#100) matches it against the teacher's attention
        matrix. Folds in the per-step `delta` scaling and the per-head `D` skip, so it maps the
        RAW (unscaled) input. It does NOT model the packing-aware `seg_ids` path (#68), whose
        masked inter-chunk carry changes the scan; the equality holds only without `seg_ids`.
        Dense O(L^2) — a training-time auxiliary at modest L, not the chunked training path."""
        B_, L, _ = x.shape
        H, N = self.config.n_heads, self.config.d_state
        delta, a, Bm, Cm = self._project(x)        # delta (B,L,H); Bm,Cm (B,L,N) — fp32
        gh = (delta * a).transpose(0, 2, 1)        # (B,H,L) log-decay (<= 0)
        decay = mx.exp(_segsum(gh))                # (B,H,L,L): exp(cumA_i-cumA_j), causal
        CB = mx.einsum("bin,bjn->bij", Cm, Bm)     # (B,L,L) shared B/C group
        delta_col = delta.transpose(0, 2, 1)[:, :, None, :]          # (B,H,1,L) delta_j
        M = decay * CB[:, None] * delta_col                          # (B,H,L,L)
        eye = mx.eye(L, dtype=M.dtype)
        return M + self.D[None, :, None, None] * eye[None, None]     # + D skip on the diagonal

    def recurrence(self, x: Array, state: State) -> Tuple[Array, State]:
        """One timestep. x: (B, d_inner), state h: (B, H, P, N) -> y: (B, d_inner)."""
        B_ = x.shape[0]
        H, P = self.config.n_heads, self.config.head_dim
        cd = _DTYPES[self.config.precision]
        delta, a, Bm, Cm = self._project(x)        # delta (B,H); Bm,Cm (B,N) — fp32
        Xh = _f32(x).reshape(B_, H, P)             # scan + state stay fp32
        dA = mx.exp(delta * a)                      # (B,H)
        dBx = (delta[..., None] * Xh)[..., None] * Bm[:, None, None, :]  # (B,H,P,N)
        h = dA[:, :, None, None] * state + dBx      # (B,H,P,N) — fp32 state
        y = mx.sum(h * Cm[:, None, None, :], axis=-1) + Xh * self.D[None, :, None]
        return _cast(y.reshape(B_, -1), cd), h


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

    def _conv_seq(self, x_main: Array, cd) -> Array:
        """Causal depthwise conv in `cd`. fp32 routes to nn.Conv1d verbatim; the
        functional path mirrors the layer's own call (stride/padding/dilation/groups
        pulled from the layer so the two paths can't drift) and adds the bias."""
        if cd == mx.float32:
            return self.conv(x_main)
        c = self.conv
        y = mx.conv1d(x_main.astype(cd), c.weight.astype(cd),
                      c.stride, c.padding, c.dilation, c.groups)
        return y + c.bias.astype(cd)

    def _conv_seq_seg(self, x_main: Array, cd, seg: Array) -> Array:
        """Boundary-aware causal depthwise conv (#68): taps reaching into a previous
        document are zeroed, so the conv window can't bleed across a packed boundary (the
        conv is part of the per-doc recurrent state). `seg` is (B, L). Returns length L,
        matching the full conv exactly within a single document."""
        c = self.conv
        K = self.config.d_conv
        B_, L, _ = x_main.shape
        x = x_main.astype(cd)
        w = c.weight.astype(cd)                     # (d_inner, K, 1) MLX Conv1d layout
        acc = None
        for k in range(K):
            shift = K - 1 - k                       # how far this tap reaches into the past
            wk = w[:, k, 0]                          # (d_inner,)
            if shift == 0:
                xs = x
            elif shift >= L:
                continue                            # tap reaches entirely before the start
            else:
                xs = mx.pad(x[:, :L - shift], [(0, 0), (shift, 0), (0, 0)])   # x[t-shift]
                same = seg[:, shift:] == seg[:, :-shift]                       # (B, L-shift)
                valid = mx.pad(same, [(0, 0), (shift, 0)]).astype(cd)         # 0 across boundary
                xs = xs * valid[..., None]
            term = xs * wk
            acc = term if acc is None else acc + term
        if acc is None:                             # K > L: all taps reach before the start
            acc = mx.zeros((B_, L, x.shape[-1]), dtype=cd)
        return acc + c.bias.astype(cd)

    def forward_seq(self, x: Array, seg_ids: Array = None) -> Array:
        L = x.shape[1]
        cd = _DTYPES[self.config.precision]
        xn = self.norm(x)
        x_main, z = mx.split(_linear(self.in_proj, xn, cd), 2, axis=-1)   # (B,L,di) each
        # Causal depthwise conv: pad both sides (d_conv-1), keep the first L outputs. With
        # seg_ids the conv is boundary-aware so its window can't cross a packed doc boundary.
        if seg_ids is None:
            xc = self._conv_seq(x_main, cd)[:, :L]
        else:
            xc = self._conv_seq_seg(x_main, cd, mx.array(seg_ids))
        xc = _silu(xc)
        y = self.ssm.parallel(xc, seg_ids)
        y = y * _silu(z)
        return x + _linear(self.out_proj, y, cd)

    def mixing_matrix(self, x: Array) -> Array:
        """This block's head-split SSM mixing matrix (B, H, L, L), for distillation
        `mixing-match` (#100). Runs the block front-end (norm -> in_proj main -> causal conv
        -> SiLU) then materializes the SSM matrix on that input."""
        L = x.shape[1]
        cd = _DTYPES[self.config.precision]
        xn = self.norm(x)
        x_main, _ = mx.split(_linear(self.in_proj, xn, cd), 2, axis=-1)
        xc = _silu(self._conv_seq(x_main, cd)[:, :L])
        return self.ssm.mixing_matrix(xc)

    def step(self, x: Array, state: State) -> Tuple[Array, State]:
        conv_state, ssm_state = state                        # (B,k-1,di), (B,di,ds)
        cd = _DTYPES[self.config.precision]
        xn = self.norm(x)
        x_main, z = mx.split(_linear(self.in_proj, xn, cd), 2, axis=-1)    # (B,di) each
        window = mx.concatenate([conv_state, x_main[:, None, :]], axis=1)  # (B,k,di)
        # depthwise conv at this timestep: sum over kernel positions (in cd to match
        # the conv in forward_seq; cast no-ops for fp32).
        wk = self.conv.weight[:, :, 0].T                      # (k, di)
        conv_out = (mx.sum(window.astype(cd) * wk.astype(cd)[None], axis=1)
                    + self.conv.bias.astype(cd))              # (B, di)
        xc = _silu(conv_out)
        y, new_ssm = self.ssm.recurrence(xc, ssm_state)
        y = y * _silu(z)
        out = x + _linear(self.out_proj, y, cd)
        return out, (window[:, 1:], new_ssm)


# --------------------------------------------------------------------------- #
# Hybrid attention (#67): causal MHA with RoPE, parity-exact forward/step.
# --------------------------------------------------------------------------- #
def _rope_cos_sin(positions: Array, head_dim: int) -> Tuple[Array, Array]:
    """RoPE cos/sin for absolute `positions` (fp32). Computed on the fly (no stored
    buffer) so nothing extra lands in the parameter tree. Returns (T, head_dim) each."""
    half = head_dim // 2
    inv_freq = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
    ang = positions.astype(mx.float32)[:, None] * inv_freq[None, :]   # (T, half)
    cos = mx.concatenate([mx.cos(ang), mx.cos(ang)], axis=-1)         # (T, head_dim)
    sin = mx.concatenate([mx.sin(ang), mx.sin(ang)], axis=-1)
    return cos, sin


def _rotate_half(x: Array) -> Array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    # x: (B, H, T, Dh); cos/sin: (T, Dh) -> broadcast over (B, H).
    cos = cos[None, None]
    sin = sin[None, None]
    return x * cos + _rotate_half(x) * sin


def _softmax_lastdim(scores: Array) -> Array:
    scores = scores - mx.max(scores, axis=-1, keepdims=True)
    w = mx.exp(scores)
    return w / mx.sum(w, axis=-1, keepdims=True)


class AttentionBlock(nn.Module):
    """Pre-norm causal multi-head attention with RoPE (the hybrid's attention layer).

    Drop-in for MambaBlock: same `forward_seq(x)->x` (training) and
    `step(x, state)->(x, state)` (inference) contract. State is a (k_cache, v_cache)
    pair, each (B, H, T, Dh), grown one token per `step` — a 2-tuple of arrays, so the
    model's state plumbing (init/clone) treats it exactly like the Mamba (conv, ssm)
    pair. Scores/softmax run in fp32 (qkv/o_proj GEMMs in compute dtype) so forward and
    step agree at the fp32 ~1e-4 parity tolerance."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.H = config.n_attn_heads_resolved
        self.Dh = config.attn_head_dim
        d_attn = self.H * self.Dh
        self.norm = RMSNorm(config.d_model)
        self.qkv_proj = nn.Linear(config.d_model, 3 * d_attn, bias=False)
        self.o_proj = nn.Linear(d_attn, config.d_model, bias=False)

    def _qkv(self, xn: Array, cd):
        B = xn.shape[0]
        T = xn.shape[1] if xn.ndim == 3 else 1
        qkv = _linear(self.qkv_proj, xn, cd)                 # (B,[T,]3*d_attn)
        q, k, v = mx.split(qkv, 3, axis=-1)
        def heads(t):                                        # -> (B,H,T,Dh) fp32
            return _f32(t).reshape(B, T, self.H, self.Dh).transpose(0, 2, 1, 3)
        return heads(q), heads(k), heads(v)

    def forward_seq(self, x: Array, seg_ids: Array = None) -> Array:
        cd = _DTYPES[self.config.precision]
        L = x.shape[1]
        xn = self.norm(x)
        q, k, v = self._qkv(xn, cd)                          # (B,H,L,Dh) fp32
        cos, sin = _rope_cos_sin(mx.arange(L), self.Dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.Dh)   # (B,H,L,L)
        causal = mx.tril(mx.ones((L, L), dtype=mx.bool_))
        if seg_ids is None:
            scores = mx.where(causal, scores, mx.array(float("-inf"), dtype=scores.dtype))
        else:
            # Block-diagonal: a token only attends within its own document (#68). Exact for
            # arbitrary boundaries (attention needs no chunk-alignment, unlike the SSM scan).
            seg = mx.array(seg_ids)
            same = seg[:, :, None] == seg[:, None, :]        # (B,L,L)
            allow = causal[None] & same                      # (B,L,L)
            scores = mx.where(allow[:, None], scores,
                              mx.array(float("-inf"), dtype=scores.dtype))
        out = _softmax_lastdim(scores) @ v                   # (B,H,L,Dh)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], L, self.H * self.Dh)
        return x + _linear(self.o_proj, _cast(out, cd), cd)

    def step(self, x: Array, state: State) -> Tuple[Array, State]:
        cd = _DTYPES[self.config.precision]
        k_cache, v_cache = state                             # (B,H,T,Dh) each, fp32
        t = k_cache.shape[2]                                 # absolute position
        xn = self.norm(x)                                    # x: (B, d_model)
        q, k, v = self._qkv(xn, cd)                          # (B,H,1,Dh) fp32
        cos, sin = _rope_cos_sin(mx.arange(t, t + 1), self.Dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        k_cache = mx.concatenate([k_cache, k], axis=2)       # (B,H,t+1,Dh)
        v_cache = mx.concatenate([v_cache, v], axis=2)
        scores = (q @ k_cache.transpose(0, 1, 3, 2)) / math.sqrt(self.Dh)  # (B,H,1,t+1)
        out = _softmax_lastdim(scores) @ v_cache             # (B,H,1,Dh)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], self.H * self.Dh)
        return x + _linear(self.o_proj, _cast(out, cd), cd), (k_cache, v_cache)


# --------------------------------------------------------------------------- #
# Top-level model implementing the seam
# --------------------------------------------------------------------------- #
class MLXMambaModel(ModelInterface, nn.Module):
    def __init__(self, config: MambaConfig):
        nn.Module.__init__(self)
        config.validate()
        self.config = config
        self._cd = _DTYPES[config.precision]         # compute dtype for the heavy GEMMs
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Hybrid (#67): attention blocks replace Mamba blocks at the gated positions.
        self.layers = [AttentionBlock(config) if config.is_attention_layer(i)
                       else MambaBlock(config) for i in range(config.n_layers)]
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
        # Logits + cross-entropy run in fp32 (wide-vocab softmax stability); h is
        # upcast so the head matmul is fp32 regardless of compute dtype.
        h = _f32(h)
        if self._tie_embeddings:
            return h @ self.embedding.weight.T
        return self.lm_head(h)

    # --- ModelInterface ---
    def forward(self, token_batch: Array, seg_ids: Array = None) -> Array:
        h = _cast(self.embedding(mx.array(token_batch)), self._cd)   # activation stream in cd
        if seg_ids is None:
            for layer_fn in self._layer_fns:
                h = layer_fn(h)
        else:
            seg = mx.array(seg_ids)                                   # (B, L) document ids
            for layer_fn in self._layer_fns:
                h = layer_fn(h, seg)                                  # boundary-aware (#68)
        return self._head(self.norm_f(h))

    def step(self, token: Array, state: State) -> Tuple[Array, State]:
        h = _cast(self.embedding(mx.array(token)), self._cd)
        new_state = []
        for layer, st in zip(self.layers, state):
            h, st2 = layer.step(h, st)
            new_state.append(st2)
        return self._head(self.norm_f(h)), new_state

    def verify_block(self, tokens: Sequence[int], state: State):
        """Speculative-decoding verify pass (#52): consume `tokens` (a HOST-side sequence
        of int ids) through the `step` recurrence from `state`, returning the per-token
        next-token logits and the per-token states in a SINGLE graph eval.

        Identical in value to calling `step` token-by-token — but evaluating the whole
        block at once amortizes the per-token kernel-launch/sync that dominates batch-1
        decode, which is where speculative decoding's wall-clock win comes from. Returns
        `(logits_list, state_list)` of length len(tokens); `state_list[i]` is the state
        after consuming `tokens[:i+1]`, so the caller can roll back to the accepted prefix
        without recomputing. `tokens` is normalized to host ints ONCE up front so a stray
        MLX-array argument can't force a per-token host sync inside the loop."""
        toks = [int(t) for t in tokens]
        logits_list, state_list = [], []
        h_state = state
        for tok in toks:
            logit, h_state = self.step(mx.array([tok]), h_state)
            logits_list.append(logit)
            state_list.append(h_state)
        # One eval realizes the whole block (logits + every intermediate state) together.
        leaves = list(logits_list)
        for st in state_list:
            leaves.extend(v for _, v in tree_flatten(st))
        mx.eval(leaves)
        return logits_list, state_list

    # --- distillation matching accessors (#100) ------------------------------
    def hidden_states(self, token_batch: Array) -> Tuple[Array, ...]:
        """Per-layer hidden states for the `hidden-align` stage: the embedding output
        followed by each layer's output (length n_layers + 1) — the HF/teacher convention
        (`mlx_teacher.MLXConversionTeacher.forward(return_hidden=True)`)."""
        h = _cast(self.embedding(mx.array(token_batch)), self._cd)
        hs = [h]
        for layer in self.layers:
            h = layer.forward_seq(h)
            hs.append(h)
        return tuple(hs)

    def mixing_matrices(self, token_batch: Array) -> List[Tuple[int, Array]]:
        """For the `mixing-match` stage: each Mamba layer's head-averaged mixing matrix
        `(B, L, L)` paired with its layer index. Attention layers are skipped (their mixer is
        the teacher's attention they were matched against)."""
        h = _cast(self.embedding(mx.array(token_batch)), self._cd)
        out: List[Tuple[int, Array]] = []
        for i, layer in enumerate(self.layers):
            if not self.config.is_attention_layer(i):
                out.append((i, layer.mixing_matrix(h).mean(axis=1)))    # head-average -> (B,L,L)
            h = layer.forward_seq(h)
        return out

    def init_state(self, batch_size: int) -> State:
        c = self.config
        di, k = c.d_inner, c.d_conv
        H, P, N = c.n_heads, c.head_dim, c.d_state
        Ha, Dh = c.n_attn_heads_resolved, c.attn_head_dim
        # Per Mamba layer: (conv window (B,k-1,di), SSM state (B,H,P,N)).
        # Per attention layer: a zero-length KV cache (k,v), each (B,Ha,0,Dh), grown by step.
        def layer_state(i):
            if c.is_attention_layer(i):
                z = mx.zeros((batch_size, Ha, 0, Dh))
                return (z, z)
            return (mx.zeros((batch_size, k - 1, di)), mx.zeros((batch_size, H, P, N)))
        return [layer_state(i) for i in range(self.config.n_layers)]

    def get_state(self) -> State:
        return self._state

    def set_state(self, state: State) -> None:
        self._state = state

    def clone_state(self, state: State) -> State:
        # MLX arrays are immutable, so rebuilding the list/tuples (arrays shared) yields
        # an independent snapshot: later `step`s allocate new arrays rather than mutate.
        return [(conv, ssm) for (conv, ssm) in state]

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
