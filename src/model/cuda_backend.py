"""CUDA / PyTorch backend for the Mamba POC (scale-up milestone).

A faithful port of `mlx_backend.py`: a Mamba-2 / SSD block (SCALAR-A selective SSM,
multi-head with one shared B/C group). Written in plain PyTorch ops (no fused kernels)
so it runs on **CPU as well as CUDA** — forward/step parity (and later backend_parity
vs MLX) is therefore validatable on a Mac/Linux box before any CUDA hardware exists. The
optional `mamba-ssm` fused fast path is a separate follow-up (#40).

Two code paths must agree (forward_step_parity, fp32 ~1e-4 rel):

  * `parallel(x)`     : the SSD chunked-matmul scan over the full sequence (training).
  * `recurrence(x, h)`: one-step state update (inference).

This file imports `torch`, so it lives BELOW the seam — nothing portable imports it, and
it stays out of `tests/test_import_guard.py`'s portable set.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from .blocks import MambaConfig
from .interface import ModelInterface, State, Array


_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _silu(x: Array) -> Array:
    return F.silu(x)


def _softplus(x: Array) -> Array:
    # log(1 + exp(x)), numerically stable via logaddexp(x, 0) to match the MLX path.
    return torch.logaddexp(x, torch.zeros_like(x))


# --------------------------------------------------------------------------- #
# Mixed precision (mirrors mlx_backend): fp32 master weights + fp16/bf16 compute.
# Params stay fp32; cast to `compute_dtype` (cd) AT THE MATMUL SITE. For fp32 every
# cast is a no-op (returns the tensor unchanged), so the fp32 path stays verbatim and
# toy parity/conformance is bit-faithful.
# --------------------------------------------------------------------------- #
def _f32(t: Array) -> Array:
    return t if t.dtype == torch.float32 else t.to(torch.float32)


def _cast(t: Array, cd) -> Array:
    return t if t.dtype == cd else t.to(cd)


def _linear(layer: nn.Linear, x: Array, cd) -> Array:
    """nn.Linear with operands cast to `cd`. fp32 routes to the original call verbatim
    (fp16 @ fp32 would promote back to fp32, so BOTH operands must be cast)."""
    if cd == torch.float32:
        return layer(x)
    y = x.to(cd) @ layer.weight.to(cd).t()
    if layer.bias is not None:                       # in/x/out_proj are bias-free; dt_proj has bias
        y = y + layer.bias.to(cd)
    return y


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Array) -> Array:
        if x.dtype == torch.float32:                 # fp32 path: original op, bit-identical
            norm = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
            return self.weight * (x * norm)
        # Mixed precision: reduce in fp32 (weight is fp32), return in x's dtype.
        xf = x.to(torch.float32)
        norm = torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        return (self.weight * (xf * norm)).to(x.dtype)


def _segsum(x: Array) -> Array:
    """Lower-triangular segment-sum. out[..., i, j] = sum_{j < k <= i} x_k for i >= j,
    and -inf above the diagonal. `exp(_segsum(g))` is the within-window decay matrix of
    a scalar SSM (1-semiseparable); the -inf upper triangle exp's to 0 (causality)."""
    T = x.shape[-1]
    xc = torch.cumsum(x, dim=-1)
    seg = xc[..., :, None] - xc[..., None, :]
    mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device))
    return seg.masked_fill(~mask, float("-inf"))


def _chunk_seg_mask(seg_ids: Array, B: int, Q: int, nc: int, pad: int) -> Array:
    """Boundary mask for the SSD inter-chunk decay matrix (#68) — torch port of the MLX
    helper. Doc boundaries are chunk-aligned, so each length-Q chunk is single-document;
    this masks the inter-chunk state carry so SSM state can't bleed past a packed boundary.

    `seg_ids` (B, L) document ids -> (B, nc+1, nc+1) bool: entry [i, c] keeps the decay from
    state c (chunk c-1's final; c=0 is the zero initial state) into entering chunk i iff
    they share a document. Sentinels: -1 the zero-state column, -2 the dropped-output row."""
    seg = seg_ids
    if pad:
        seg = torch.cat([seg, seg[:, -1:].expand(B, pad)], dim=1)
    seg_chunks = seg.reshape(B, nc, Q)[:, :, 0]                  # (B, nc) doc id per chunk
    sentinel_a = torch.full((B, 1), -1, dtype=seg_chunks.dtype, device=seg_chunks.device)
    sentinel_b = torch.full((B, 1), -2, dtype=seg_chunks.dtype, device=seg_chunks.device)
    src = torch.cat([sentinel_a, seg_chunks], dim=1)            # state c axis
    out = torch.cat([seg_chunks, sentinel_b], dim=1)           # entering-chunk i axis
    return out[:, :, None] == src[:, None, :]                   # (B, nc+1, nc+1) bool


class SelectiveSSM(nn.Module):
    """Scalar-A Mamba-2 / SSD selective state space (see mlx_backend.SelectiveSSM)."""

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
        self.A_log = nn.Parameter(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(H))         # (H,) skip per head

        self._init_dt_bias()

    def _init_dt_bias(self) -> None:
        """LOAD-BEARING dt-projection bias init (inverse-softplus into a small positive
        range), per-head. Without this the model fails to learn recall.

            dt   = uniform(log(dt_min), log(dt_max)).exp().clamp(min=dt_init_floor)
            bias = dt + log(-expm1(-dt))          # inverse softplus
        """
        c = self.config
        u = torch.empty(c.n_heads).uniform_(math.log(c.dt_min), math.log(c.dt_max))
        dt = torch.clamp(torch.exp(u), min=c.dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # (H,)

    # --- shared projections --------------------------------------------------
    def _project(self, x: Array):
        """x: (..., d_inner) -> delta (..., H), a (H,), B (..., N), C (..., N).

        `x_proj` is the heavy SSM GEMM (runs in compute dtype); its outputs upcast to
        fp32 so the rest of the scan runs in fp32."""
        cd = _DTYPES[self.config.precision]
        dt_rank, d_state = self.config.dt_rank_resolved, self.config.d_state
        proj = _linear(self.x_proj, x, cd)
        dt_pre = _f32(proj[..., :dt_rank])
        B = _f32(proj[..., dt_rank:dt_rank + d_state])
        C = _f32(proj[..., dt_rank + d_state:])
        delta = _softplus(self.dt_proj(dt_pre))      # (..., H) per head, fp32
        a = -torch.exp(self.A_log)                   # (H,) scalar decay, fp32
        return delta, a, B, C

    def parallel(self, x: Array, seg_ids: Array = None) -> Array:
        """SSD chunked-matmul scan. x: (B, L, d_inner) -> (B, L, d_inner).

        Pads L up to a multiple of the chunk length Q (padded steps carry zero input and
        are trimmed). All decays are exp of non-positive sums, so it is overflow-safe.

        `seg_ids` (B, L) document ids (chunk-aligned boundaries) makes the scan
        packing-aware (#68): the inter-chunk state carry is masked so recurrent state can't
        bleed across documents. `None` is the original single-segment scan."""
        B_, L, d_inner = x.shape
        H, P, N = self.config.n_heads, self.config.head_dim, self.config.d_state
        Q = self.config.chunk_size or 64
        cd = _DTYPES[self.config.precision]
        delta, a, Bm, Cm = self._project(x)          # delta (B,L,H); Bm,Cm (B,L,N) — fp32
        X = _f32(x).reshape(B_, L, H, P)             # whole scan runs in fp32

        pad = (-L) % Q
        if pad:
            zc = lambda t, shp: torch.cat([t, t.new_zeros(shp)], dim=1)
            X, delta = zc(X, (B_, pad, H, P)), zc(delta, (B_, pad, H))
            Bm, Cm = zc(Bm, (B_, pad, N)), zc(Cm, (B_, pad, N))
        Lp, nc = L + pad, (L + pad) // Q
        g = delta * a                                # (B,Lp,H) log-decay (<= 0)
        Xin = delta[..., None] * X                   # (B,Lp,H,P) input = dt * X

        gc = g.reshape(B_, nc, Q, H).permute(0, 3, 1, 2)     # (B,H,nc,Q)
        Xc = Xin.reshape(B_, nc, Q, H, P)
        Bc = Bm.reshape(B_, nc, Q, N)
        Cc = Cm.reshape(B_, nc, Q, N)
        Acum = torch.cumsum(gc, dim=-1)                      # (B,H,nc,Q)

        # 1) intra-chunk diagonal block (attention-like, within each chunk)
        Lmask = torch.exp(_segsum(gc))                       # (B,H,nc,Q,Q)
        CB = torch.einsum("bcin,bcjn->bcij", Cc, Bc)         # (B,nc,Q,Q)
        Ydiag = torch.einsum("bhcij,bcij,bcjhp->bcihp", Lmask, CB, Xc)
        # 2) each chunk's final state, from that chunk's inputs only
        decay_end = torch.exp(Acum[..., -1:] - Acum)         # (B,H,nc,Q)
        states = torch.einsum("bhcj,bcjhp,bcjn->bchpn", decay_end, Xc, Bc)
        # 3) inter-chunk recurrence over the nc chunk-states, as a matmul against an
        # (nc+1, nc+1) decay matrix (Dao & Gu SSD form): O(nc^2) but nc is small.
        states = torch.cat(
            [states.new_zeros((B_, 1, H, P, N)), states], dim=1)
        chunk_tot = F.pad(Acum[..., -1], (1, 0))             # (B,H,nc+1)
        decay_chunk = torch.exp(_segsum(chunk_tot))          # (B,H,nc+1,nc+1)
        if seg_ids is not None:                              # #68: cross-doc reset
            seg_mask = _chunk_seg_mask(seg_ids, B_, Q, nc, pad)   # (B,nc+1,nc+1) bool
            decay_chunk = decay_chunk * seg_mask[:, None].to(decay_chunk.dtype)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        S_enter = new_states[:, :-1]                         # (B,nc,H,P,N)
        # 4) off-diagonal output: entering state decayed to each position
        out_decay = torch.exp(Acum)                          # (B,H,nc,Q)
        Yoff = torch.einsum("bcin,bchpn,bhci->bcihp", Cc, S_enter, out_decay)

        Y = (Ydiag + Yoff).reshape(B_, Lp, H, P)[:, :L]      # (B,L,H,P)
        Y = Y + X[:, :L] * self.D[None, None, :, None]       # skip
        return _cast(Y.reshape(B_, L, d_inner), cd)          # back to compute dtype

    def recurrence(self, x: Array, state: State) -> Tuple[Array, State]:
        """One timestep. x: (B, d_inner), state h: (B, H, P, N) -> y: (B, d_inner)."""
        B_ = x.shape[0]
        H, P = self.config.n_heads, self.config.head_dim
        cd = _DTYPES[self.config.precision]
        delta, a, Bm, Cm = self._project(x)          # delta (B,H); Bm,Cm (B,N) — fp32
        Xh = _f32(x).reshape(B_, H, P)               # scan + state stay fp32
        dA = torch.exp(delta * a)                    # (B,H)
        dBx = (delta[..., None] * Xh)[..., None] * Bm[:, None, None, :]   # (B,H,P,N)
        h = dA[:, :, None, None] * state + dBx        # (B,H,P,N) — fp32 state
        y = torch.sum(h * Cm[:, None, None, :], dim=-1) + Xh * self.D[None, :, None]
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
        """Causal depthwise conv in `cd`. torch Conv1d is channels-first, so transpose
        (B,L,di) <-> (B,di,L) around it; the caller slices to the first L outputs."""
        c = self.conv
        y = F.conv1d(x_main.transpose(1, 2).to(cd), c.weight.to(cd), c.bias.to(cd),
                     c.stride, c.padding, c.dilation, c.groups)
        return y.transpose(1, 2)

    def _conv_seq_seg(self, x_main: Array, cd, seg: Array) -> Array:
        """Boundary-aware causal depthwise conv (#68) — torch port. Taps reaching into a
        previous document are zeroed, so the conv window can't bleed across a packed
        boundary (the conv is part of the per-doc recurrent state). `seg` is (B, L).
        Returns length L, matching the full conv exactly within a single document."""
        c = self.conv
        K = self.config.d_conv
        B_, L, _ = x_main.shape
        x = x_main.to(cd)
        w = c.weight.to(cd)                         # (d_inner, 1, K) torch Conv1d layout
        acc = None
        for k in range(K):
            shift = K - 1 - k                       # how far this tap reaches into the past
            wk = w[:, 0, k]                          # (d_inner,)
            if shift == 0:
                xs = x
            elif shift >= L:
                continue                            # tap reaches entirely before the start
            else:
                xs = F.pad(x[:, :L - shift], (0, 0, shift, 0))   # x[t-shift]
                same = seg[:, shift:] == seg[:, :-shift]         # (B, L-shift)
                valid = F.pad(same, (shift, 0)).to(cd)           # 0 across boundary
                xs = xs * valid[..., None]
            term = xs * wk
            acc = term if acc is None else acc + term
        if acc is None:                             # K > L: all taps reach before the start
            acc = torch.zeros((B_, L, x.shape[-1]), dtype=cd, device=x.device)
        return acc + c.bias.to(cd)

    def forward_seq(self, x: Array, seg_ids: Array = None) -> Array:
        L = x.shape[1]
        cd = _DTYPES[self.config.precision]
        xn = self.norm(x)
        x_main, z = torch.chunk(_linear(self.in_proj, xn, cd), 2, dim=-1)   # (B,L,di) each
        # Causal depthwise conv: pad both sides (d_conv-1), keep the first L outputs. With
        # seg_ids the conv is boundary-aware so its window can't cross a packed doc boundary.
        if seg_ids is None:
            xc = self._conv_seq(x_main, cd)[:, :L]
        else:
            xc = self._conv_seq_seg(x_main, cd, seg_ids)
        xc = _silu(xc)
        y = self.ssm.parallel(xc, seg_ids)
        y = y * _silu(z)
        return x + _linear(self.out_proj, y, cd)

    def step(self, x: Array, state: State) -> Tuple[Array, State]:
        conv_state, ssm_state = state                        # (B,k-1,di), (B,H,P,N)
        cd = _DTYPES[self.config.precision]
        xn = self.norm(x)
        x_main, z = torch.chunk(_linear(self.in_proj, xn, cd), 2, dim=-1)   # (B,di) each
        window = torch.cat([conv_state, x_main[:, None, :]], dim=1)         # (B,k,di)
        # depthwise conv at this timestep: sum over kernel positions (in cd to match
        # forward_seq; cast no-ops for fp32). torch conv weight is (di, 1, k).
        wk = self.conv.weight[:, 0, :].t()                   # (k, di)
        conv_out = (torch.sum(window.to(cd) * wk.to(cd)[None], dim=1)
                    + self.conv.bias.to(cd))                 # (B, di)
        xc = _silu(conv_out)
        y, new_ssm = self.ssm.recurrence(xc, ssm_state)
        y = y * _silu(z)
        out = x + _linear(self.out_proj, y, cd)
        return out, (window[:, 1:], new_ssm)


# --------------------------------------------------------------------------- #
# Hybrid attention (#67): causal MHA with RoPE — torch port of the MLX block.
# --------------------------------------------------------------------------- #
def _rope_cos_sin(positions: Array, head_dim: int) -> Tuple[Array, Array]:
    """RoPE cos/sin for absolute `positions` (fp32). Computed on the fly (no buffer),
    so the parameter set matches MLX exactly. Returns (T, head_dim) each."""
    half = head_dim // 2
    device = positions.device
    inv_freq = torch.exp(-math.log(10000.0)
                         * torch.arange(0, half, dtype=torch.float32, device=device) / half)
    ang = positions.to(torch.float32)[:, None] * inv_freq[None, :]    # (T, half)
    cos = torch.cat([torch.cos(ang), torch.cos(ang)], dim=-1)         # (T, head_dim)
    sin = torch.cat([torch.sin(ang), torch.sin(ang)], dim=-1)
    return cos, sin


def _rotate_half(x: Array) -> Array:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    # x: (B, H, T, Dh); cos/sin: (T, Dh) -> broadcast over (B, H).
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


def _softmax_lastdim(scores: Array) -> Array:
    scores = scores - torch.amax(scores, dim=-1, keepdim=True)
    w = torch.exp(scores)
    return w / torch.sum(w, dim=-1, keepdim=True)


class AttentionBlock(nn.Module):
    """Pre-norm causal multi-head attention with RoPE (mirror of mlx_backend.AttentionBlock).

    Same `forward_seq` / `step` contract and the same submodule names/shapes
    (`norm`, `qkv_proj`, `o_proj`) so portable weights round-trip MLX<->torch. State is
    a (k_cache, v_cache) pair, each (B, H, T, Dh), grown one token per `step`."""

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
        T = xn.shape[1] if xn.dim() == 3 else 1
        qkv = _linear(self.qkv_proj, xn, cd)                 # (B,[T,]3*d_attn)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        def heads(t):                                        # -> (B,H,T,Dh) fp32
            return _f32(t).reshape(B, T, self.H, self.Dh).permute(0, 2, 1, 3)
        return heads(q), heads(k), heads(v)

    def forward_seq(self, x: Array, seg_ids: Array = None) -> Array:
        cd = _DTYPES[self.config.precision]
        L = x.shape[1]
        xn = self.norm(x)
        q, k, v = self._qkv(xn, cd)                          # (B,H,L,Dh) fp32
        cos, sin = _rope_cos_sin(torch.arange(L, device=x.device), self.Dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.Dh)      # (B,H,L,L)
        causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=x.device))
        if seg_ids is None:
            scores = scores.masked_fill(~causal, float("-inf"))
        else:
            # Block-diagonal: a token only attends within its own document (#68). Exact for
            # arbitrary boundaries (attention needs no chunk-alignment, unlike the SSM scan).
            same = seg_ids[:, :, None] == seg_ids[:, None, :]        # (B,L,L)
            allow = causal[None] & same                             # (B,L,L)
            scores = scores.masked_fill(~allow[:, None], float("-inf"))
        out = _softmax_lastdim(scores) @ v                   # (B,H,L,Dh)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], L, self.H * self.Dh)
        return x + _linear(self.o_proj, _cast(out, cd), cd)

    def step(self, x: Array, state: State) -> Tuple[Array, State]:
        cd = _DTYPES[self.config.precision]
        k_cache, v_cache = state                             # (B,H,T,Dh) each, fp32
        t = k_cache.shape[2]                                 # absolute position
        xn = self.norm(x)                                    # x: (B, d_model)
        q, k, v = self._qkv(xn, cd)                          # (B,H,1,Dh) fp32
        cos, sin = _rope_cos_sin(torch.arange(t, t + 1, device=x.device), self.Dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        k_cache = torch.cat([k_cache, k], dim=2)             # (B,H,t+1,Dh)
        v_cache = torch.cat([v_cache, v], dim=2)
        scores = (q @ k_cache.transpose(-1, -2)) / math.sqrt(self.Dh)   # (B,H,1,t+1)
        out = _softmax_lastdim(scores) @ v_cache             # (B,H,1,Dh)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], self.H * self.Dh)
        return x + _linear(self.o_proj, _cast(out, cd), cd), (k_cache, v_cache)


# --------------------------------------------------------------------------- #
# Top-level model implementing the seam
# --------------------------------------------------------------------------- #
class CUDAMambaModel(ModelInterface, nn.Module):
    def __init__(self, config: MambaConfig, device: str = "cpu"):
        nn.Module.__init__(self)
        config.validate()
        # Fail fast rather than silently build a DENSE model: the sparse-MoE block (#53) is
        # MLX-only for now, but MambaConfig.num_parameters()/parameter_breakdown() already
        # account for MoE capacity — so a `moe_every` config here would disagree with the
        # formula and skew any cross-backend sizing/comparison.
        if config.n_moe_layers > 0:
            raise NotImplementedError(
                "MoE-Mamba (#53) is not implemented in the CUDA backend; "
                "build a config without `moe_every` (it is an MLX-only experiment)."
            )
        self.config = config
        self._cd = _DTYPES[config.precision]         # compute dtype for the heavy GEMMs
        self._device = torch.device(device)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Hybrid (#67): attention blocks replace Mamba blocks at the gated positions.
        self.layers = nn.ModuleList(
            [AttentionBlock(config) if config.is_attention_layer(i) else MambaBlock(config)
             for i in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self._tie_embeddings = config.tie_embeddings
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._state = None
        self.to(self._device)

    def _head(self, h: Array) -> Array:
        # Logits + cross-entropy run in fp32 (wide-vocab softmax stability); h is upcast
        # so the head matmul is fp32 regardless of compute dtype.
        h = _f32(h)
        if self._tie_embeddings:
            return h @ self.embedding.weight.t()
        return self.lm_head(h)

    def _layer_forward(self, layer: "MambaBlock | AttentionBlock", h: Array,
                       seg_ids: Array = None) -> Array:
        # Gradient checkpointing: recompute the layer's forward in backward instead of
        # retaining its activations. Only meaningful under autograd; use_reentrant=False
        # runs normally in no-grad (eval/parity) contexts.
        if self.config.grad_checkpoint and torch.is_grad_enabled():
            if seg_ids is None:
                return _checkpoint(layer.forward_seq, h, use_reentrant=False)
            return _checkpoint(layer.forward_seq, h, seg_ids, use_reentrant=False)
        return layer.forward_seq(h, seg_ids)

    # --- ModelInterface ---
    def forward(self, token_batch: Array, seg_ids: Array = None) -> Array:
        ids = torch.as_tensor(np.asarray(token_batch), dtype=torch.long, device=self._device)
        h = _cast(self.embedding(ids), self._cd)     # activation stream in cd
        seg = None
        if seg_ids is not None:                      # (B, L) document ids -> boundary-aware (#68)
            seg = torch.as_tensor(np.asarray(seg_ids), dtype=torch.long, device=self._device)
        for layer in self.layers:
            h = self._layer_forward(layer, h, seg)
        return self._head(self.norm_f(h))

    def step(self, token: Array, state: State) -> Tuple[Array, State]:
        ids = torch.as_tensor(np.asarray(token), dtype=torch.long, device=self._device)
        h = _cast(self.embedding(ids), self._cd)
        new_state = []
        for layer, st in zip(self.layers, state):
            h, st2 = layer.step(h, st)
            new_state.append(st2)
        return self._head(self.norm_f(h)), new_state

    def init_state(self, batch_size: int) -> State:
        c = self.config
        di, k = c.d_inner, c.d_conv
        H, P, N = c.n_heads, c.head_dim, c.d_state
        Ha, Dh = c.n_attn_heads_resolved, c.attn_head_dim
        dev = self._device
        # Per Mamba layer: (conv window (B,k-1,di), SSM state (B,H,P,N)), fp32.
        # Per attention layer: a zero-length KV cache (k,v), each (B,Ha,0,Dh), grown by step.
        def layer_state(i):
            if c.is_attention_layer(i):
                z = torch.zeros((batch_size, Ha, 0, Dh), device=dev)
                return (z, z)
            return (torch.zeros((batch_size, k - 1, di), device=dev),
                    torch.zeros((batch_size, H, P, N), device=dev))
        return [layer_state(i) for i in range(self.config.n_layers)]

    def get_state(self) -> State:
        return self._state

    def set_state(self, state: State) -> None:
        self._state = state

    def clone_state(self, state: State) -> State:
        # torch `step` is not immutable, so deep-copy the buffers: the snapshot must not
        # be aliased by later steps.
        return [(conv.clone(), ssm.clone()) for (conv, ssm) in state]

    def save(self, path: str) -> None:
        from ..train.checkpoint import save_weights
        save_weights(self._portable_state_dict(), path, config=self.config)

    def load(self, path: str) -> None:
        from ..train.checkpoint import load_weights
        load_weights(self, path)

    # --- portable bridge: keep the MLX-canonical layout so MLX<->torch round-trips. ---
    def _portable_state_dict(self) -> dict:
        # {name: numpy}. The only layout difference vs MLX is the depthwise conv weight:
        # torch is (out, in/groups, k); MLX is (out, k, in/groups). Emit MLX layout.
        out = {}
        for k, v in self.named_parameters():
            arr = v.detach().to("cpu")
            if k.endswith(".conv.weight"):
                arr = arr.transpose(1, 2)            # (out,1,k) -> (out,k,1)
            out[k] = arr.numpy()
        return out

    def _load_portable(self, weights: dict) -> None:
        tensors = {}
        for k, v in weights.items():
            t = torch.as_tensor(np.asarray(v))
            if k.endswith(".conv.weight"):
                t = t.transpose(1, 2)                # (out,k,1) -> (out,1,k)
            tensors[k] = t
        self.load_state_dict(tensors, strict=True)
        self.to(self._device)
