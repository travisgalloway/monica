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

import contextlib
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


# --------------------------------------------------------------------------- #
# Optional fused fast paths (#40): mamba-ssm SSD kernel + causal-conv1d.
# Both degrade gracefully to the plain-PyTorch path when the package is absent
# or the tensor is on CPU (kernels require CUDA). mamba-ssm's top-level
# __init__.py has a stale transformers import; stub the package so only the ops
# submodule is loaded.
# --------------------------------------------------------------------------- #
_FUSED_SCAN_FN = None          # sentinel: None = not yet tried; False = unavailable


def _fused_scan():
    """Return mamba_chunk_scan_combined, or None if unavailable."""
    global _FUSED_SCAN_FN
    if _FUSED_SCAN_FN is None:
        try:
            import sys, types, importlib.metadata
            if "mamba_ssm" not in sys.modules:
                # Use the canonical PyPI name (hyphen); fall back to underscore for
                # environments where the dist was installed under the import name.
                for _distname in ("mamba-ssm", "mamba_ssm"):
                    try:
                        pkg_dir = str(importlib.metadata.distribution(_distname).locate_file("mamba_ssm"))
                        break
                    except importlib.metadata.PackageNotFoundError:
                        continue
                else:
                    _FUSED_SCAN_FN = False
                    return None
                stub = types.ModuleType("mamba_ssm")
                stub.__path__ = [pkg_dir]
                stub.__package__ = "mamba_ssm"
                sys.modules["mamba_ssm"] = stub
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            _FUSED_SCAN_FN = mamba_chunk_scan_combined
        except Exception:
            _FUSED_SCAN_FN = False
    return _FUSED_SCAN_FN if _FUSED_SCAN_FN else None


try:
    from causal_conv1d import causal_conv1d_fn as _CAUSAL_CONV1D
except Exception:
    _CAUSAL_CONV1D = None


_FAST_PATH_REPORTED = False


def fast_path_status() -> "tuple[bool, bool]":
    """(fused SSD scan available?, causal-conv1d available?) — the two #40 `cuda-fast`
    kernels. Both gate the fused paths; absent, the SSD scan + conv run in pure PyTorch."""
    return (_fused_scan() is not None, _CAUSAL_CONV1D is not None)


def _report_fast_path_once(device: "torch.device") -> None:
    """On the first CUDA model built, log whether the fused Mamba kernels are active. A
    silent pure-PyTorch fallback on a GPU is a (large) throughput trap during a sweep — make
    it loud so a missing `pip install -e .[cuda-fast]` is caught before a long run, not after."""
    global _FAST_PATH_REPORTED
    if _FAST_PATH_REPORTED or device.type != "cuda":
        return
    _FAST_PATH_REPORTED = True
    scan_ok, conv_ok = fast_path_status()
    if scan_ok and conv_ok:
        print("[cuda] fused Mamba kernels ACTIVE (mamba-ssm SSD scan + causal-conv1d).")
    else:
        import warnings
        missing = ", ".join(n for n, ok in
                            (("mamba-ssm (SSD scan)", scan_ok), ("causal-conv1d", conv_ok)) if not ok)
        warnings.warn(
            f"[cuda] running on GPU WITHOUT fused Mamba kernels — missing: {missing}. "
            "The SSD scan/conv fall back to pure PyTorch (much slower). Install the fast "
            "path with: pip install -e \".[dev,data,cuda-fast]\" (#40).",
            RuntimeWarning, stacklevel=2)


# --------------------------------------------------------------------------- #
# fp8 MoE-expert linears (#214/#240, live): NVIDIA Transformer Engine (TE) `te.Linear`
# on Hopper+ (sm_90) for the expert gate/up/down GEMMs only. Mirrors the
# `_fused_scan()` / `_report_fast_path_once()` lazy-probe + warn-once pattern above.
# `_Expert`/`MoEBlock` (below) build their linears via `_expert_linear` AND wrap the
# expert-compute region in `MoEBlock._fp8_ctx`'s `fp8_autocast`, so a config that sets
# `fp8_experts: true` on a Hopper+ box now actually runs the GEMMs in fp8 — not just
# reachable code. Un-verified without Hopper hardware (see `tests/test_cuda_fp8.py`'s
# GPU-gated acceptance tests and `docs/infrastructure.md`'s manual checklist).
# --------------------------------------------------------------------------- #
_TE_LINEAR_CLS = None      # sentinel: None = untried, False = unavailable/pre-Hopper


def _te_linear_cls():
    """Return `transformer_engine.pytorch.Linear`, or None if TE is unavailable or the
    device is pre-Hopper. Mirrors `_fused_scan()`. Hopper check uses
    `get_device_capability()[0] >= 9`: TE also runs on Blackwell (sm_100+), so `>= 9`
    is the correct forward-compatible lower bound, not an exact sm_90 match."""
    global _TE_LINEAR_CLS
    if _TE_LINEAR_CLS is None:
        try:
            if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
                _TE_LINEAR_CLS = False
                return None
            from transformer_engine.pytorch import Linear as _TELinear
            _TE_LINEAR_CLS = _TELinear
        except Exception:
            _TE_LINEAR_CLS = False
    return _TE_LINEAR_CLS if _TE_LINEAR_CLS else None


def fp8_status() -> bool:
    """Is the fp8 expert-linear path (TE + Hopper) available? Mirrors `fast_path_status()`."""
    return _te_linear_cls() is not None


_FP8_STATUS_REPORTED = False


def _report_fp8_status_once(device: "torch.device") -> None:
    """On the first CUDA model built with `fp8_experts=True` and at least one MoE
    layer, log whether TE fp8 is active. Like `_report_fast_path_once`, a silent bf16
    fallback on a GPU is a throughput trap during a sweep — make it loud. Called from
    `CUDAMambaModel.__init__` (#214)."""
    global _FP8_STATUS_REPORTED
    if _FP8_STATUS_REPORTED or device.type != "cuda":
        return
    _FP8_STATUS_REPORTED = True
    if fp8_status():
        print("[cuda] fp8 MoE experts ACTIVE (Transformer Engine, Hopper+).")
    else:
        import warnings
        warnings.warn(
            "[cuda] fp8_experts=True but Transformer Engine is unavailable or the "
            "device is pre-Hopper — MoE experts fall back to bf16/fp16 nn.Linear. "
            "Install the fp8 path with: pip install -e \".[dev,data,cuda-fp8]\" on a "
            "Hopper+ (sm_90) GPU (#240).",
            RuntimeWarning, stacklevel=2)


# === expert linear construction (#214, fully wired) =======================
# `_Expert.__init__` (below) builds its three linears (gate/up/down) via this helper:
# `te.Linear` when fp8 is available (TE keeps fp32 master weights and casts to fp8 at
# the GEMM, analogous to `_linear`'s cast-at-matmul above), else a plain bf16/fp16
# `nn.Linear`. `device`/`params_dtype=torch.float32` are passed through explicitly so
# TE builds the weight directly on the right device, in fp32 — the same
# fp32-master-weights invariant `_linear`'s cast-at-matmul documents everywhere else in
# this module (mixed precision never stores a low-precision master copy). A `te.Linear`
# called OUTSIDE an `fp8_autocast()` context still runs — but silently in bf16, not
# fp8 — so `MoEBlock._fp8_ctx` (below) must wrap every call site. `te.Linear` also
# graph-breaks `torch.compile` like `mamba-ssm` (safe/opaque, same as the fused scan
# above).
def _expert_linear(d_in: int, d_out: int, config: MambaConfig, device):
    cls = _te_linear_cls() if getattr(config, "fp8_experts", False) else None
    if cls is not None:
        return cls(d_in, d_out, bias=False, device=device, params_dtype=torch.float32)
    return nn.Linear(d_in, d_out, bias=False)       # bf16/fp16 fallback (no TE / pre-Hopper)
# ===========================================================================


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
        # Long-context extension (#54): mirror mlx_backend — divide the discretization
        # step so per-step decay moves toward 1, enlarging the receptive field at
        # inference. Guarded so factor 1.0 (default) leaves delta byte-identical.
        if self.config.long_ctx_factor != 1.0:
            delta = delta / self.config.long_ctx_factor
        a = -torch.exp(self.A_log)                   # (H,) scalar decay, fp32
        return delta, a, B, C

    def parallel(self, x: Array, seg_ids: Array = None, *,
                 return_state: bool = False) -> Array | Tuple[Array, Array]:
        """SSD chunked-matmul scan. x: (B, L, d_inner) -> (B, L, d_inner).

        Dispatches to `mamba_chunk_scan_combined` (#40) when the tensor is on CUDA and
        mamba-ssm is installed; otherwise falls back to the pure-PyTorch path.

        Pads L up to a multiple of the chunk length Q (padded steps carry zero input and
        are trimmed). All decays are exp of non-positive sums, so it is overflow-safe.

        `seg_ids` (B, L) document ids (chunk-aligned boundaries) makes the scan
        packing-aware (#68): the inter-chunk state carry is masked so recurrent state can't
        bleed across documents. `None` is the original single-segment scan.

        `return_state` (keyword-only, prefill #165) additionally returns the end-of-sequence
        SSM state (B, H, P, N) — the fused kernel's `return_final_states`, or the
        `new_states[:, -1]` entry the fallback normally discards. Exact under padding:
        padded steps have `delta = 0`, so they contribute no input and apply identity decay."""
        B_, L, d_inner = x.shape
        if return_state and seg_ids is not None:
            raise NotImplementedError(
                "parallel(..., return_state=True) does not support seg_ids: "
                "_chunk_seg_mask marks the entering-chunk axis's last row with the -2 "
                "sentinel, so row nc of decay_chunk is masked to all zeros — precisely "
                "because S_enter drops it. The carry-out would silently read as zeros. "
                "See ModelInterface.prefill.")
        H, P, N = self.config.n_heads, self.config.head_dim, self.config.d_state
        Q = self.config.chunk_size or 64
        cd = _DTYPES[self.config.precision]
        delta, a, Bm, Cm = self._project(x)          # delta (B,L,H); Bm,Cm (B,L,N) — fp32

        # --- fused CUDA fast path (#40) -------------------------------------------
        fn = _fused_scan()
        if fn is not None and x.device.type == "cuda":
            # mamba_chunk_scan_combined expects:
            #   x    (B,L,H,P) fp32    — raw head-shaped input (kernel handles dt*x)
            #   dt   (B,L,H)   fp32    — softplus(dt_proj(dt_pre)), already computed
            #   A    (H,)      fp32    — negative scalar decay (-exp(A_log))
            #   B    (B,L,G,N) fp32    — G=ngroups=1
            #   C    (B,L,G,N) fp32
            #   D    (H,)             — per-head skip (added inside kernel)
            #   seq_idx (B,L) int32   — maps each position to its document id (#68)
            X = _f32(x).reshape(B_, L, H, P)
            si = (seg_ids.to(torch.int32) if seg_ids is not None else None)
            if return_state:
                # `return_final_states=True` makes the kernel return (out, final_states)
                # with final_states (B,H,P,N) — the same carry-out the fallback reads off
                # new_states[:, -1]. Kept off the no-state call so that shape is unchanged.
                Y, final = fn(X, delta, a, Bm[:, :, None, :], Cm[:, :, None, :], Q,
                              D=self.D, seq_idx=si, dt_softplus=False,
                              return_final_states=True)
                final = _f32(final).reshape(B_, H, P, N)
                return _cast(Y.reshape(B_, L, d_inner), cd), final
            Y = fn(X, delta, a, Bm[:, :, None, :], Cm[:, :, None, :], Q,
                   D=self.D, seq_idx=si, dt_softplus=False)    # (B,L,H,P) fp32
            return _cast(Y.reshape(B_, L, d_inner), cd)

        # --- pure-PyTorch fallback ------------------------------------------------
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
        Y = _cast(Y.reshape(B_, L, d_inner), cd)             # back to compute dtype
        if return_state:
            return Y, new_states[:, -1]                      # (B,H,P,N) fp32 carry-out
        return Y

    def mixing_matrix(self, x: Array) -> Array:
        """Materialize the dense (B, H, L, L) 1-semiseparable mixing matrix M such that
        `einsum('bhij,bjhp->bihp', M, X) == parallel(x)` (X = head-split input) in the
        single-segment case — torch port of `mlx_backend.SelectiveSSM.mixing_matrix`. This is
        the matrix the SSD scan applies; the distillation `mixing-match` stage (#100) matches it
        against the teacher's attention. Folds in the per-step `delta` scaling and per-head `D`
        skip; does NOT model the `seg_ids` path. Dense O(L^2) — a training-time auxiliary."""
        B_, L, _ = x.shape
        delta, a, Bm, Cm = self._project(x)          # delta (B,L,H); Bm,Cm (B,L,N) — fp32
        gh = (delta * a).permute(0, 2, 1)            # (B,H,L) log-decay (<= 0)
        decay = torch.exp(_segsum(gh))               # (B,H,L,L): exp(cumA_i-cumA_j), causal
        CB = torch.einsum("bin,bjn->bij", Cm, Bm)    # (B,L,L) shared B/C group
        delta_col = delta.permute(0, 2, 1)[:, :, None, :]            # (B,H,1,L) delta_j
        M = decay * CB[:, None] * delta_col                          # (B,H,L,L)
        eye = torch.eye(L, dtype=M.dtype, device=M.device)
        return M + self.D[None, :, None, None] * eye[None, None]     # + D skip on the diagonal

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


def _conv_window(x_main: Array, k: int) -> Array:
    """The conv state after consuming `x_main` (B, L, d_inner) — torch port of
    `mlx_backend._conv_window` (#165).

    `step` keeps `window[:, 1:]`, so after L tokens the state is the LAST `k-1` rows of
    `x_main` — post-`in_proj`, PRE-conv, pre-SiLU. `L < k-1` LEFT-zero-pads (what an
    initially zeroed window plus L pushes leaves behind); `k == 1` needs an explicit empty
    window (`x_main[:, -0:]` is the whole array). Emitted fp32 to match a stepped window."""
    B_, L, di = x_main.shape
    w = k - 1
    if w <= 0:
        return x_main.new_zeros((B_, 0, di), dtype=torch.float32)
    tail = _f32(x_main[:, max(0, L - w):])                   # (B, min(L,w), di)
    missing = w - tail.shape[1]
    if missing:
        tail = torch.cat([tail.new_zeros((B_, missing, di)), tail], dim=1)
    return tail


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
        """Causal depthwise conv in `cd`. Uses causal_conv1d_fn (#40) on CUDA when
        available; falls back to F.conv1d. The caller slices to the first L outputs."""
        c = self.conv
        if _CAUSAL_CONV1D is not None and x_main.device.type == "cuda":
            # causal_conv1d_fn: (B, C, L) x (C, K) -> (B, C, L), no extra padding needed.
            # Our weight is (d_inner, 1, K) in torch Conv1d format; squeeze the group dim.
            w = c.weight.to(cd).squeeze(1)             # (d_inner, K)
            y = _CAUSAL_CONV1D(x_main.transpose(1, 2).to(cd), w,
                               c.bias.to(cd), activation=None)
            return y.transpose(1, 2)
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

    def forward_prefill(self, x: Array, seg_ids: Array = None) -> Tuple[Array, State]:
        """`forward_seq` plus the (conv_state, ssm_state) pair `step` would have left (#165)
        — torch port of `mlx_backend.MambaBlock.forward_prefill`."""
        L = x.shape[1]
        cd = _DTYPES[self.config.precision]
        xn = self.norm(x)
        x_main, z = torch.chunk(_linear(self.in_proj, xn, cd), 2, dim=-1)   # (B,L,di) each
        xc = _silu(self._conv_seq(x_main, cd)[:, :L])
        y, ssm_state = self.ssm.parallel(xc, seg_ids, return_state=True)
        y = y * _silu(z)
        out = x + _linear(self.out_proj, y, cd)
        return out, (_conv_window(x_main, self.config.d_conv), ssm_state)

    def mixing_matrix(self, x: Array) -> Array:
        """This block's head-split SSM mixing matrix (B, H, L, L), for distillation
        `mixing-match` (#100). Runs the block front-end (norm -> in_proj main -> causal conv
        -> SiLU) then materializes the SSM matrix on that input. Torch port of
        `mlx_backend.MambaBlock.mixing_matrix`."""
        L = x.shape[1]
        cd = _DTYPES[self.config.precision]
        xn = self.norm(x)
        x_main, _ = torch.chunk(_linear(self.in_proj, xn, cd), 2, dim=-1)
        xc = _silu(self._conv_seq(x_main, cd)[:, :L])
        return self.ssm.mixing_matrix(xc)

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
        # SDPA (#144): fuses QKᵀ/softmax/AV and never materializes the (B,H,L,L) score
        # tensor — the O(L²) memory hot spot at seq 8192. GEMMs run in the compute dtype
        # (cd); softmax stays fp32-stable inside SDPA. In an fp32 config cd==fp32, so this
        # matches the old eager path within reduction order (parity holds at ~1e-4); a
        # bf16 run becomes FlashAttention-eligible.
        q, k, v = q.to(cd), k.to(cd), v.to(cd)
        scale = 1.0 / math.sqrt(self.Dh)
        if seg_ids is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        else:
            # Block-diagonal: a token only attends within its own document (#68). Exact for
            # arbitrary boundaries (attention needs no chunk-alignment, unlike the SSM scan).
            # The causal diagonal keeps each row's self-key, so no row is fully masked (no NaN).
            causal = torch.tril(torch.ones((L, L), dtype=torch.bool, device=x.device))
            same = seg_ids[:, :, None] == seg_ids[:, None, :]        # (B,L,L)
            allow = causal[None] & same                             # (B,L,L)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allow[:, None], scale=scale)      # (B,H,L,Dh)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], L, self.H * self.Dh)
        return x + _linear(self.o_proj, _cast(out, cd), cd)

    def forward_prefill(self, x: Array, seg_ids: Array = None) -> Tuple[Array, State]:
        """`forward_seq` plus the (k_cache, v_cache) pair `step` would have built (#165).

        `k` is captured POST-RoPE but BEFORE the `to(cd)` SDPA cast, and `v` likewise, so
        both stay fp32 — the dtype `step` keeps its caches in. RoPE positions run from 0,
        so this is valid only from a fresh cache; see `ModelInterface.prefill`."""
        cd = _DTYPES[self.config.precision]
        L = x.shape[1]
        xn = self.norm(x)
        q, k, v = self._qkv(xn, cd)                          # (B,H,L,Dh) fp32
        cos, sin = _rope_cos_sin(torch.arange(L, device=x.device), self.Dh)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        k_cache, v_cache = k, v                              # fp32 caches, pre-cast
        out = F.scaled_dot_product_attention(
            q.to(cd), k.to(cd), v.to(cd), is_causal=True, scale=1.0 / math.sqrt(self.Dh))
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], L, self.H * self.Dh)
        return x + _linear(self.o_proj, _cast(out, cd), cd), (k_cache, v_cache)

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
# Sparse Mixture-of-Experts FFN block (#53, CUDA grouped-gather routing #214) — torch
# port of mlx_backend's MoE block, now with a real gathered-dispatch kernel alongside
# the dense evaluate-every-expert reference.
# --------------------------------------------------------------------------- #
_ROUTE_EPS = 1e-9      # #217: log-guard for the router-entropy diagnostic. Must be
                       # IDENTICAL to mlx_backend.py's — the two backends' entropies are
                       # compared at fp32 ~1e-4 by the parity test.


class _Expert(nn.Module):
    """A SwiGLU FFN expert: down(silu(gate(x)) * up(x)). Bias-free. Torch port of
    `mlx_backend._Expert`, built via `_expert_linear`: a plain `nn.Linear` unless
    `config.fp8_experts` resolves a Transformer Engine `te.Linear` (fp32 master
    weights, fp8 GEMM). The fp8_autocast context that actually makes a `te.Linear` run
    in fp8 (rather than silently in bf16) is entered by the CALLER — see
    `MoEBlock._fp8_ctx` — not here, so this class stays agnostic to it.

    `_linear` (this module's helper) reaches into `layer.weight` directly to cast
    operands to the compute dtype at the matmul site — correct for `nn.Linear`, but it
    would silently bypass a `te.Linear`'s own fp8 GEMM/amax bookkeeping entirely. `self._te`
    is cached at construction so `forward` can branch: a TE linear is always called
    directly (module `__call__`), never routed through `_linear`.
    """

    def __init__(self, d_model: int, d_ff: int, config: MambaConfig, device):
        super().__init__()
        cls = _te_linear_cls() if getattr(config, "fp8_experts", False) else None
        self._te = cls is not None
        self.gate = _expert_linear(d_model, d_ff, config, device)
        self.up = _expert_linear(d_model, d_ff, config, device)
        self.down = _expert_linear(d_ff, d_model, config, device)

    def forward(self, xn: Array, cd) -> Array:
        if self._te:
            # TE linears own their GEMM (fp8 under an fp8_autocast context — not yet
            # wired, #240 — or bf16/fp16 outside one); call the modules directly.
            return self.down(_silu(self.gate(xn)) * self.up(xn))
        return _linear(self.down,
                       _silu(_linear(self.gate, xn, cd)) * _linear(self.up, xn, cd), cd)


class MoEBlock(nn.Module):
    """Sparse Mixture-of-Experts FFN block (#53), interleaved like `AttentionBlock` —
    torch port of `mlx_backend.MoEBlock`, plus a real grouped-gather dispatch kernel
    (#214) alongside the dense evaluate-every-expert reference. Same `forward_seq(x)->x`
    / `step(x, state)->(x, state)` contract as the other blocks; stateless (pointwise
    over the sequence), so forward and step compute the same function by construction.

    `moe_impl` picks the compute strategy, resolved once at construction:
      * "dense"  — evaluate every expert, mask+renormalize the top_k gates. Exact and
        FLOP-accountable but not wall-clock-sparse. This is the MLX/Swift reference — the
        oracle every gather test below is checked against, so it must never be deleted.
      * "gather" — dispatch each token only to its top_k chosen experts.
      * "auto"   — dense when top_k >= n_experts or n_experts == 1 (gather buys nothing
        there), else gather.
    """

    _MOE_BIAS_PREFIX = "moe_route_bias."   # kept for symmetry with the model-level prefix

    def __init__(self, config: MambaConfig, device, ep_size: int = 1, ep_rank: int = 0):
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.d_model)
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        d_ff = config.moe_d_ff_resolved

        # Expert parallel (#271): each rank builds ONLY its shard of the global expert
        # set. `ep_size == 1` (the default, and every config before #271) gives every
        # rank the full `range(n_experts)` shard, in order — a ModuleDict keyed by the
        # STRINGIFIED global expert index produces the exact same `named_parameters()`
        # prefixes (`experts.0.gate.weight`, ...) a `ModuleList` did, so the portable
        # weight keys, `check_weight_keys`, `upcycle.py`, and the Swift parity fixtures
        # are all untouched at ep_size == 1.
        from ..train.parallel import expert_partition
        self.ep_size = int(ep_size)
        self.ep_rank = int(ep_rank)
        self.ep_group = None    # set post-construction by cuda_distributed once a PG exists
        self._local_expert_ids = expert_partition(config.n_experts, self.ep_size)[self.ep_rank]
        self.experts = nn.ModuleDict(
            {str(i): _Expert(config.d_model, d_ff, config, device)
             for i in self._local_expert_ids})
        # Shared experts (#214, DeepSeek-V2/V3 form): every token goes through every
        # shared expert unconditionally; combined ADDITIVELY in `_moe`, outside the
        # router softmax/top-k renormalization. Empty ModuleList for n_shared_experts=0.
        # NOT sharded by EP: every rank runs the full set on its own token shard, same
        # as the pre-#271 behavior (dispatch is only for the top-k ROUTED experts).
        self.shared_experts = nn.ModuleList(
            [_Expert(config.d_model, d_ff, config, device)
             for _ in range(config.n_shared_experts)])

        E = config.n_experts
        if self.ep_size > 1:
            if config.moe_impl == "dense":
                raise ValueError(
                    "expert parallel (ep_size > 1) requires moe_impl='gather' or "
                    "'auto' — 'dense' evaluates every expert locally, which is "
                    "incompatible with a partitioned expert set."
                )
            if config.top_k >= E:
                raise ValueError(
                    f"expert parallel (ep_size > 1) requires top_k ({config.top_k}) < "
                    f"n_experts ({E}) — top_k >= n_experts routes every token to every "
                    "expert (the dense path), which is incompatible with a partitioned "
                    "expert set."
                )
            self._use_gather = True     # EP dispatch IS the gather path (all-to-all)
        elif config.moe_impl == "dense":
            self._use_gather = False
        elif config.moe_impl == "gather":
            self._use_gather = True
        else:                                          # "auto"
            self._use_gather = not (config.top_k >= E or E == 1)

        # Loss-Free-Balancing (#213 D1), torch port. MLX excludes the bias/counts from
        # the parameter tree via a leading-underscore-attribute convention baked into
        # `nn.Module.valid_parameter_filter`; torch has no such convention, so
        # `register_buffer(..., persistent=False)` does the ACTUAL work here — verified:
        # absent from `named_parameters()` AND `state_dict()`, `load_state_dict(strict=
        # True)` does not demand it, and — unlike a plain attribute — it IS moved by
        # `.to()` (a plain attribute would silently stay on the wrong device the first
        # time a bias activates on GPU; `persistent=True` would make every load raise on
        # a missing key). The leading underscore on these names is therefore load-bearing
        # in MLX but purely COSMETIC here — kept only for cross-backend name parity.
        self.register_buffer("_route_bias", torch.zeros(E), persistent=False)
        self._bias_active = False
        self._count_loads = False
        self.register_buffer("_load_counts", torch.zeros(E), persistent=False)
        # #217 routing diagnostics, torch mirror of mlx_backend.MoEBlock. `persistent=False`
        # for the same reason as `_load_counts`: absent from `state_dict()`, moved by `.to()`.
        self.register_buffer("_entropy_sum", torch.zeros(()), persistent=False)
        self._n_routed = 0

    def set_route_bias(self, vec) -> None:
        """Set the per-expert selection bias (#213) — `vec` a `list[float]` of length
        `n_experts`. Mutates the buffer IN PLACE (`copy_`), never rebinds it, so it stays
        the exact object `.to()` already placed on the right device."""
        if len(vec) != self.config.n_experts:
            raise ValueError(
                f"route bias has {len(vec)} entries, expected "
                f"n_experts={self.config.n_experts}."
            )
        self._route_bias.copy_(
            torch.as_tensor(vec, dtype=torch.float32, device=self._route_bias.device))
        self._bias_active = True

    def route_bias(self) -> list:
        return [float(b) for b in self._route_bias.tolist()] if self._bias_active else []

    def set_load_counting(self, flag: bool) -> None:
        self._count_loads = bool(flag)

    def pop_load(self) -> list:
        """Per-expert token counts accumulated since the last pop, then reset."""
        counts = self._load_counts.clone()
        self._load_counts.zero_()
        return [float(c) for c in counts.tolist()]

    def pop_routing_stats(self) -> dict:
        """This block's routing diagnostics since the last pop, then reset. Torch mirror
        of `mlx_backend.MoEBlock.pop_routing_stats` — see there for the full contract.

        DRAINS the load accumulator too (calls `pop_load()`), so a caller must use either
        this or `pop_load()` for a given step, never both (#217)."""
        load = self.pop_load()
        n = self._n_routed
        total = float(self._entropy_sum.item()) if n else 0.0
        self._entropy_sum.zero_()
        self._n_routed = 0
        return {"load": load, "entropy": (total / n) if n else None, "n_tokens": int(n)}

    def _fp8_ctx(self):
        """Context manager for the expert-compute region (#214/#240). A `te.Linear`
        called OUTSIDE an `fp8_autocast` context silently runs in bf16, not fp8, so
        `_expert_linear` alone does not turn fp8 on — every expert GEMM call site in
        `_moe` must be wrapped in this. `contextlib.nullcontext()` when fp8 is off or
        unavailable, so the non-fp8 path is untouched byte-for-byte (no TE import even
        attempted). The router's logits/probs are computed by the CALLER, above this
        context — the router is a plain `nn.Linear` and always routes in fp32, and
        must stay bit-identical between fp8 and bf16 runs regardless (see the
        zero-tolerance routing-identity test in `tests/test_cuda_fp8.py`)."""
        if not (self.config.fp8_experts and fp8_status()):
            return contextlib.nullcontext()
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling
        return te.fp8_autocast(enabled=True, fp8_recipe=DelayedScaling())

    def _moe_dense(self, xn: Array, cd, gate: Array) -> Array:
        # `self.experts` is a ModuleDict (see __init__) — `.values()` in insertion
        # (== ascending global-id) order, matching `gate`'s expert axis.
        outs = torch.stack([e(xn, cd) for e in self.experts.values()], dim=-2)  # (...,E,d_model)
        return torch.sum(gate.unsqueeze(-1) * _f32(outs), dim=-2)       # combine in fp32

    def _moe_gather(self, xn: Array, cd, topk_ids: Array, gate_kept: Array) -> Array:
        """Grouped-gather routing (#214): dispatch each token only to its top_k chosen
        experts, instead of `_moe_dense`'s evaluate-every-expert-then-mask.

        Two-step gather — `expand` then `index_select` — is DELIBERATELY NOT the
        one-shot `flat.index_select(0, perm // k)`: that form's backward is `index_add_`
        with repeated indices, a CUDA atomic reduction whose summation order varies run
        to run and would break the bit-exact resume the smoke gate asserts (see the
        determinism note at `cuda_muon.py:25-27`). This form's backward is a
        deterministic `sum` (from the `expand`) composed with a permutation
        `index_select` — at the cost of one extra `(N*k, D)` buffer, noted for the FSDP
        follow-up.

        Loops over ALL E experts, deliberately with no `if count == 0: continue`: a
        zero-row GEMM still yields a real, exactly-zero, finite `weight.grad` (matching
        the dense path), whereas skipping the call leaves that expert's `.grad` as
        `None` and `Muon.step` (`cuda_muon.py:65-67`) then skips its momentum decay
        while the dense path decays it — a silent, impl-dependent training divergence.
        """
        E, k = self.config.n_experts, self.config.top_k
        D = xn.shape[-1]
        lead_shape = xn.shape[:-1]
        flat_x = xn.reshape(-1, D)
        N = flat_x.shape[0]
        flat_ids = topk_ids.reshape(N, k)
        flat_gate = gate_kept.reshape(N, k)

        # (N,D) -> (N,k,D) -> (N*k,D): each token's row repeated once per chosen expert
        # (its own dispatch copy), so the backward through this step is a deterministic
        # sum over the k repeats — never an atomic scatter-add.
        dispatch = flat_x.unsqueeze(1).expand(N, k, D).reshape(N * k, D)
        dispatch_expert = flat_ids.reshape(-1)              # (N*k,) expert id per slot

        perm = torch.argsort(dispatch_expert, stable=True)  # group dispatch slots by expert
        sorted_expert = dispatch_expert[perm]
        sorted_x = dispatch.index_select(0, perm)            # deterministic gather

        counts = torch.bincount(sorted_expert, minlength=E).tolist()   # one host sync
        chunks = []
        offset = 0
        for e in range(E):                     # ALL experts — see the no-`continue` note above
            c = counts[e]
            chunks.append(self.experts[str(e)](sorted_x[offset:offset + c], cd))
            offset += c
        out_sorted = torch.cat(chunks, dim=0)                 # (N*k, D)

        inv_perm = torch.argsort(perm)                        # undo the grouping sort
        out_dispatch = out_sorted.index_select(0, inv_perm).reshape(N, k, D)
        combined = torch.sum(_f32(out_dispatch) * flat_gate.unsqueeze(-1), dim=1)   # (N,D)
        return combined.reshape(*lead_shape, D)

    def _moe_gather_ep(self, xn: Array, cd, topk_ids: Array, gate_kept: Array) -> Array:
        """Expert-parallel twin of `_moe_gather` (#271): each token's top_k dispatch
        rows are routed via TWO `all_to_all_single` round trips through `self.ep_group`
        — once out to the rank that owns each chosen expert, once back with the
        result — instead of a purely-local grouped-gather. Both hops are permutations
        (index_select-based, same discipline as `_moe_gather`'s determinism note): the
        backward of an all_to_all is the transposed all_to_all, not an atomic
        scatter-add, so bit-exact resume still holds AT A FIXED world_size (not across
        a world_size change — that is a separate, documented limitation, see
        `cuda_distributed.load_resume_dcp`).

        `expert_partition` makes global expert ids CONTIGUOUS per EP rank, so sorting
        dispatch slots by expert id ascending (as `_moe_gather` already does) also
        groups them by destination rank — the existing `perm`/`sorted_expert` sort is
        reused as-is; only the per-expert local loop becomes a cross-rank exchange.
        """
        import torch.distributed as dist

        E, k = self.config.n_experts, self.config.top_k
        ep_size = self.ep_size
        per_rank = E // ep_size
        D = xn.shape[-1]
        lead_shape = xn.shape[:-1]
        flat_x = xn.reshape(-1, D)
        N = flat_x.shape[0]
        flat_ids = topk_ids.reshape(N, k)
        flat_gate = gate_kept.reshape(N, k)

        dispatch = flat_x.unsqueeze(1).expand(N, k, D).reshape(N * k, D)
        dispatch_expert = flat_ids.reshape(-1)

        perm = torch.argsort(dispatch_expert, stable=True)
        sorted_expert = dispatch_expert[perm]
        sorted_x = dispatch.index_select(0, perm)

        # How many dispatch slots go to each EP rank (contiguous blocks of `per_rank`
        # experts per rank — see `src.train.parallel.expert_partition`).
        counts_per_expert = torch.bincount(sorted_expert, minlength=E)
        send_counts = counts_per_expert.reshape(ep_size, per_rank).sum(dim=1).to(torch.long)
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        send_list, recv_list = send_counts.tolist(), recv_counts.tolist()

        recv_x = sorted_x.new_zeros((sum(recv_list), D))
        dist.all_to_all_single(recv_x, sorted_x, output_split_sizes=recv_list,
                               input_split_sizes=send_list, group=self.ep_group)
        recv_expert = sorted_expert.new_zeros(sum(recv_list))
        dist.all_to_all_single(recv_expert, sorted_expert, output_split_sizes=recv_list,
                               input_split_sizes=send_list, group=self.ep_group)

        # Group received rows by LOCAL expert (every received row's global id is one of
        # THIS rank's local ids by construction of the count exchange above).
        local_ids = self._local_expert_ids
        lo = local_ids[0] if local_ids else 0
        n_local = len(local_ids)
        local_perm = torch.argsort(recv_expert, stable=True)
        recv_expert_sorted = recv_expert[local_perm]
        recv_x_sorted = recv_x.index_select(0, local_perm)
        local_counts = (torch.bincount(recv_expert_sorted - lo, minlength=n_local)
                        if recv_expert_sorted.numel() else torch.zeros(n_local, dtype=torch.long))
        chunks = []
        offset = 0
        for i, gid in enumerate(local_ids):    # no `continue` on c==0 — see _moe_gather's note
            c = int(local_counts[i])
            chunks.append(self.experts[str(gid)](recv_x_sorted[offset:offset + c], cd))
            offset += c
        out_local_sorted = (torch.cat(chunks, dim=0) if chunks
                            else recv_x_sorted.new_zeros((0, D)))

        inv_local_perm = torch.argsort(local_perm)
        out_recv_order = out_local_sorted.index_select(0, inv_local_perm)

        out_sorted = sorted_x.new_zeros((sum(send_list), D))
        dist.all_to_all_single(out_sorted, out_recv_order, output_split_sizes=send_list,
                               input_split_sizes=recv_list, group=self.ep_group)

        inv_perm = torch.argsort(perm)
        out_dispatch = out_sorted.index_select(0, inv_perm).reshape(N, k, D)
        combined = torch.sum(_f32(out_dispatch) * flat_gate.unsqueeze(-1), dim=1)
        return combined.reshape(*lead_shape, D)

    def _moe(self, xn: Array) -> Array:
        cd = _DTYPES[self.config.precision]
        E, k = self.config.n_experts, self.config.top_k
        logits = _f32(_linear(self.router, xn, cd))          # (..., E) — route in fp32
        probs = F.softmax(logits, dim=-1)                    # UNBIASED — always the gate weight

        if self._count_loads:
            # #217 — see mlx_backend.MoEBlock._moe for the full rationale (outside the
            # k<E guard; gated by _count_loads; grad_checkpoint doubles numerator AND
            # denominator so the ratio is exact). Kept literally in step with MLX: the
            # two entropies are compared at fp32 ~1e-4 in tests.
            with torch.no_grad():
                ent = -(probs * torch.log(probs + _ROUTE_EPS)).sum(dim=-1)
                self._entropy_sum += ent.sum()
                self._n_routed += int(probs.numel() // E)

        if k < E:                                             # keep EXACTLY top_k per token
            # Loss-Free-Balancing (#213 D2): when active, rank by the BIASED selection
            # score `logits + route_bias` (LOGIT space); the gate weight below stays
            # `probs` (unbiased) regardless. Unbiased: rank by `probs`, NOT `logits` —
            # softmax can collide two distinct fp32 logits into one prob, so the two
            # tie-break differently; this must mirror `mlx_backend.py:694` literally.
            sel = (logits + self._route_bias) if self._bias_active else probs
            # `stable=True` is the WHOLE tie-break contract: verified to match MLX's
            # `argsort(argsort(-x)) < k` double-argsort on every tested tie pattern.
            # `torch.topk` is DISQUALIFIED — it diverges from that on ties (see the
            # explicit negative test in tests/test_cuda_moe.py).
            order = torch.argsort(-sel, dim=-1, stable=True)       # descending rank
            topk_ids, _ = torch.sort(order[..., :k], dim=-1)       # ascending: dense-order sum

            if self._count_loads:
                # Per-expert load bookkeeping for the balancer (#213), detached from the
                # graph (topk_ids already carries no grad) and accumulated; `pop_load()`
                # reads + resets it. Counted only here (k < E): with k == E every expert
                # takes every token and balancing is vacuous. Shared between dense/gather
                # (both derive from the same topk_ids), so `pop_moe_load()` agrees exactly
                # between impls.
                counts = torch.bincount(
                    topk_ids.reshape(-1), minlength=E).to(self._load_counts.dtype)
                self._load_counts += counts

            with self._fp8_ctx():                          # expert GEMMs only (#214/#240)
                if self._use_gather:
                    gate_kept = torch.gather(probs, -1, topk_ids)
                    gate_kept = gate_kept / gate_kept.sum(-1, keepdim=True)
                    if self.ep_size > 1:
                        y = self._moe_gather_ep(xn, cd, topk_ids, gate_kept)
                    else:
                        y = self._moe_gather(xn, cd, topk_ids, gate_kept)
                else:
                    mask = torch.zeros_like(probs, dtype=torch.bool)
                    mask.scatter_(-1, topk_ids, True)
                    gate = torch.where(mask, probs, torch.zeros_like(probs))
                    gate = gate / gate.sum(-1, keepdim=True)   # renormalize the kept gates
                    y = self._moe_dense(xn, cd, gate)
        else:
            with self._fp8_ctx():
                y = self._moe_dense(xn, cd, probs)            # softmax already sums to 1

        if len(self.shared_experts):
            # Additive, OUTSIDE the router softmax/top-k renormalization (DeepSeek-V2/V3
            # form). Guarded so n_shared_experts=0 takes the exact pre-#214 path.
            with self._fp8_ctx():
                y = y + sum(_f32(se(xn, cd)) for se in self.shared_experts)
        return _cast(y, cd)

    def forward_seq(self, x: Array, seg_ids: Array = None) -> Array:
        return x + self._moe(self.norm(x))                   # pointwise: seg_ids irrelevant

    def forward_prefill(self, x: Array, seg_ids: Array = None) -> Tuple[Array, State]:
        # Stateless: emit the SAME placeholder pair `init_state` builds for a MoE layer.
        B_ = x.shape[0]
        z = x.new_zeros((B_, 0))
        return x + self._moe(self.norm(x)), (z, z)

    def step(self, x: Array, state: State) -> Tuple[Array, State]:
        return x + self._moe(self.norm(x)), state            # stateless: pass state through


def _torch_ge_21() -> bool:
    # "2.3.1+cu121" -> (2, 3); robust to the +cuXXX / +cpu local suffix.
    parts = torch.__version__.split("+")[0].split(".")
    major, minor = int(parts[0]), int(parts[1])
    return (major, minor) >= (2, 1)


# --------------------------------------------------------------------------- #
# Top-level model implementing the seam
# --------------------------------------------------------------------------- #
class CUDAMambaModel(ModelInterface, nn.Module):
    # `moe_route_bias.{layer_index}` — the MoE route bias's key prefix in the portable
    # weights (#213 D3). Not a parameter (see MoEBlock.__init__), so it is added/popped
    # explicitly around the state_dict path rather than riding it.
    _MOE_BIAS_PREFIX = "moe_route_bias."

    def __init__(self, config: MambaConfig, device: str = "cpu",
                ep_size: int = 1, ep_rank: int = 0):
        nn.Module.__init__(self)
        config.validate()
        self.config = config
        self._cd = _DTYPES[config.precision]         # compute dtype for the heavy GEMMs
        self._device = torch.device(device)
        # Expert parallel (#271): threaded into every MoEBlock via `_make_layer`. Not a
        # `ModelInterface`/config concept — `scripts/train_dist.py` constructs the model
        # directly with these, bypassing `src.model.backend.get_backend`'s single-rank
        # factory closures (which have no notion of a process group).
        self._ep_size = int(ep_size)
        self._ep_rank = int(ep_rank)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Hybrid (#67): attention blocks replace Mamba blocks at the gated positions;
        # MoE (#53/#214): sparse-FFN blocks replace Mamba blocks at their gated positions
        # (attention takes precedence on a collision, matching `is_moe_layer`).
        self.layers = nn.ModuleList([self._make_layer(i) for i in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)
        self._tie_embeddings = config.tie_embeddings
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._state = None
        self.to(self._device)
        _report_fast_path_once(self._device)
        if config.fp8_experts and config.n_moe_layers:
            _report_fp8_status_once(self._device)
        # torch.compile (#145, #239): fuse the pure-tensor forward (layer loop + head) via
        # inductor. Tri-state config.torch_compile: None = AUTO (compile only on a real CUDA
        # device with torch>=2.1 — the real-run path), True/False = explicit override honored
        # on any device. NEVER auto-compile on CPU: CPU is the fp32 parity/conformance surface
        # (tests/test_cuda_parity.py builds on CPU) and must stay eager. We compile the inner
        # `_forward_compute` (not `forward`, which opens with an untraceable numpy->tensor
        # boundary). Bare torch.compile — no max-autotune (long autotune + fragile capture);
        # inductor's automatic dynamic-shape promotion handles varying (B, L). The optional
        # mamba-ssm/causal-conv1d kernels graph-break inductor around them (safe).
        # CUDA graphs: DEFERRED (out of scope) — follow-up is to profile launch-overhead at
        # the small M12 rung and add mode="reduce-overhead" only if a stall shows.
        if config.torch_compile is None:
            self._compiled = self._device.type == "cuda" and _torch_ge_21()
        else:
            self._compiled = bool(config.torch_compile)
        if self._compiled:
            self._forward_compute = torch.compile(self._forward_compute)

    def _make_layer(self, i: int):
        """Attention -> MoE -> Mamba precedence, mirroring `mlx_backend.MLXMambaModel.
        _make_layer` (and `MambaConfig.is_moe_layer`'s own attention-takes-precedence
        rule)."""
        if self.config.is_attention_layer(i):
            return AttentionBlock(self.config)
        if self.config.is_moe_layer(i):
            return MoEBlock(self.config, self._device,
                            ep_size=self._ep_size, ep_rank=self._ep_rank)
        return MambaBlock(self.config)

    @staticmethod
    def _fsdp_unshard(module) -> None:
        """FSDP2 (#271) hazard, discovered on this host: `fully_shard` unshards a
        wrapped module's params via a forward PRE-HOOK registered on `nn.Module.
        __call__`. This model's block API (`forward_seq`/`step`/`forward_prefill`) is
        invoked directly on the layer object, never through `layer(...)`/`__call__` —
        by design, so `forward` and `step` can share the exact same underlying compute
        without an nn.Module dispatch layer in between (see the module docstring's
        SSD/train-infer-parity note). That means FSDP2's unshard hook never fires, and
        a sharded DTensor param hits an op against a plain (unsharded) activation:
        `RuntimeError: aten.mul.Tensor got mixed torch.Tensor and DTensor` — reproduced
        in-session with a 2-rank gloo run (`tests/test_cuda_distributed.py`'s V6).

        Calling `.unshard()` explicitly, right before any DIRECT (non-`__call__`)
        access to a `fully_shard`-wrapped module's params, fixes it. Deliberately does
        NOT call the matching `.reshard()` afterward — under `grad_checkpoint`, the
        SAME direct call recomputes in backward, by which point a paired reshard would
        have already re-sharded the params out from under that recompute. Leaving
        params materialized (relying on the NEXT `unshard()` to be a cheap no-op, or on
        FSDP2's own end-of-backward reshard) trades peak memory for correctness here —
        the actual memory-reclaim timing under this codebase's direct-dispatch
        convention is UNRESOLVED and left as a CUDA-host follow-up (see
        `cuda_distributed.wrap_backbone`'s docstring), not claimed as solved by this
        method. A plain (non-FSDP) module has no `.unshard` attribute, so this is a
        strict no-op at world_size == 1 — unchanged from every pre-#271 call site.
        """
        unshard = getattr(module, "unshard", None)
        if unshard is not None:
            unshard()

    def _head(self, h: Array) -> Array:
        # Logits + cross-entropy run in fp32 (wide-vocab softmax stability); h is upcast
        # so the head matmul is fp32 regardless of compute dtype.
        h = _f32(h)
        if self._tie_embeddings:
            # `self.embedding(ids)` (the LOOKUP, in `forward`/`prefill`/`step` below) goes
            # through `__call__` and unshards itself fine; THIS raw `.weight` access does
            # not — see `_fsdp_unshard`'s docstring.
            self._fsdp_unshard(self.embedding)
            return h @ self.embedding.weight.t()
        return self.lm_head(h)

    def _layer_forward(self, layer: "MambaBlock | AttentionBlock", h: Array,
                       seg_ids: Array = None) -> Array:
        # Gradient checkpointing: recompute the layer's forward in backward instead of
        # retaining its activations. Only meaningful under autograd; use_reentrant=False
        # runs normally in no-grad (eval/parity) contexts.
        self._fsdp_unshard(layer)
        if self.config.grad_checkpoint and torch.is_grad_enabled():
            if self.config.fp8_experts and fp8_status() and isinstance(layer, MoEBlock):
                # `transformer_engine.pytorch.checkpoint`, NOT `torch.utils.checkpoint`
                # (#214/#240): plain checkpoint's recompute pass re-runs the fp8_autocast
                # region and double-updates the amax history TE uses to calibrate the fp8
                # scale, which `te.checkpoint` knows to skip. It has NO `use_reentrant`
                # kwarg — passing one is a TypeError, unlike the plain-checkpoint call
                # below. Gated on `fp8_status()` (not just the config flag) so this import
                # only fires when TE/Hopper is actually available — `fp8_experts=True` on
                # CPU/CI or non-Hopper CUDA must fall through to the plain-checkpoint path
                # below, matching `MoEBlock._fp8_ctx`'s own fp8_status() gate.
                import transformer_engine.pytorch as te
                if seg_ids is None:
                    return te.checkpoint(layer.forward_seq, h)
                return te.checkpoint(layer.forward_seq, h, seg_ids)
            if seg_ids is None:
                return _checkpoint(layer.forward_seq, h, use_reentrant=False)
            return _checkpoint(layer.forward_seq, h, seg_ids, use_reentrant=False)
        return layer.forward_seq(h, seg_ids)

    def _forward_compute(self, h: Array, seg: Array = None) -> Array:
        """Pure-tensor forward (layer loop + final norm + head) — the region torch.compile
        wraps (#145). Split out of `forward` so the numpy->tensor boundary stays eager."""
        for layer in self.layers:
            h = self._layer_forward(layer, h, seg)
        return self._head(self.norm_f(h))

    # --- ModelInterface ---
    def forward(self, token_batch: Array, seg_ids: Array = None) -> Array:
        ids = torch.as_tensor(np.asarray(token_batch), dtype=torch.long, device=self._device)
        h = _cast(self.embedding(ids), self._cd)     # activation stream in cd
        seg = None
        if seg_ids is not None:                      # (B, L) document ids -> boundary-aware (#68)
            seg = torch.as_tensor(np.asarray(seg_ids), dtype=torch.long, device=self._device)
        return self._forward_compute(h, seg)

    def prefill(self, token_batch: Array, seg_ids: Array = None, *,
                last_only: bool = False) -> Tuple[Array, State]:
        """One parallel scan over the whole prompt -> (logits, state). See
        `ModelInterface.prefill` (fresh-session only, no seg_ids)."""
        if seg_ids is not None:
            raise NotImplementedError(
                "prefill does not support seg_ids (#165): the SSD carry-out row is masked "
                "to zeros under the packing-aware scan, the conv window can straddle a "
                "document boundary, and AttentionBlock.step has no per-document masking. "
                "Use forward(token_batch, seg_ids) for packed training sequences.")
        ids = torch.as_tensor(np.asarray(token_batch), dtype=torch.long, device=self._device)
        h = _cast(self.embedding(ids), self._cd)
        state = []
        # Raw layers, NOT _forward_compute: that is the torch.compile region and returns
        # logits only; compiling a state-returning traversal is deliberate follow-up work.
        # Grad checkpointing is skipped too — prefill is inference.
        for layer in self.layers:
            self._fsdp_unshard(layer)     # see _fsdp_unshard's docstring (#271)
            h, st = layer.forward_prefill(h)
            state.append(st)
        h = self.norm_f(h)
        if last_only:
            h = h[:, -1]                                 # (B, d_model) -> logits (B, V)
        return self._head(h), state

    def step(self, token: Array, state: State) -> Tuple[Array, State]:
        ids = torch.as_tensor(np.asarray(token), dtype=torch.long, device=self._device)
        h = _cast(self.embedding(ids), self._cd)
        new_state = []
        for layer, st in zip(self.layers, state):
            self._fsdp_unshard(layer)     # see _fsdp_unshard's docstring (#271)
            h, st2 = layer.step(h, st)
            new_state.append(st2)
        return self._head(self.norm_f(h)), new_state

    # --- distillation matching accessors (#100) ------------------------------
    def hidden_states(self, token_batch: Array) -> Tuple[Array, ...]:
        """Per-layer hidden states for the `hidden-align` stage: the embedding output followed
        by each layer's output (length n_layers + 1) — the HF/teacher convention. Uses
        `_layer_forward` so grad_checkpoint is respected (torch port of
        `mlx_backend.MLXMambaModel.hidden_states`)."""
        ids = torch.as_tensor(np.asarray(token_batch), dtype=torch.long, device=self._device)
        h = _cast(self.embedding(ids), self._cd)
        hs = [h]
        for layer in self.layers:
            h = self._layer_forward(layer, h)
            hs.append(h)
        return tuple(hs)

    def mixing_matrices(self, token_batch: Array) -> List[Tuple[int, Array]]:
        """For the `mixing-match` stage: each Mamba layer's head-averaged mixing matrix
        `(B, L, L)` paired with its layer index. Attention AND MoE layers are skipped
        (`MoEBlock` has no `mixing_matrix` — it is pointwise, not a mixer). Torch port of
        `mlx_backend.MLXMambaModel.mixing_matrices`."""
        ids = torch.as_tensor(np.asarray(token_batch), dtype=torch.long, device=self._device)
        h = _cast(self.embedding(ids), self._cd)
        out: List[Tuple[int, Array]] = []
        for i, layer in enumerate(self.layers):
            if not self.config.is_attention_layer(i) and not self.config.is_moe_layer(i):
                out.append((i, layer.mixing_matrix(h).mean(dim=1)))     # head-average -> (B,L,L)
            h = self._layer_forward(layer, h)
        return out

    # --- Loss-Free-Balancing accessors (#213/#214), torch mirrors of the MLX ones ----
    def moe_blocks(self) -> List["MoEBlock"]:
        return [l for l in self.layers if isinstance(l, MoEBlock)]

    def set_moe_biases(self, biases: List[List[float]]) -> None:
        blocks = self.moe_blocks()
        if len(biases) != len(blocks):
            raise ValueError(
                f"got {len(biases)} bias vectors for {len(blocks)} MoE layers."
            )
        for block, vec in zip(blocks, biases):
            block.set_route_bias(vec)

    def moe_biases(self) -> List[List[float]]:
        return [block.route_bias() for block in self.moe_blocks()]

    def set_moe_load_counting(self, flag: bool) -> None:
        for block in self.moe_blocks():
            block.set_load_counting(flag)

    def pop_moe_load(self) -> List[List[float]]:
        return [block.pop_load() for block in self.moe_blocks()]

    def pop_moe_routing_stats(self) -> List[dict]:
        """Per-MoE-layer routing diagnostics since the last pop, in layer order (#217).
        Torch mirror of `MLXMambaModel.pop_moe_routing_stats` — see there for the
        contract. This is the ONLY pop the train step performs; `pop_moe_load()` must
        not also be called for the same step."""
        return [block.pop_routing_stats() for block in self.moe_blocks()]

    def set_ep_group(self, group) -> None:
        """Give every MoE block the expert-parallel process group its all-to-all
        dispatch (`MoEBlock._moe_gather_ep`, #271) uses. Called once by
        `cuda_distributed`/`scripts/train_dist.py` after the process group exists —
        model construction happens before that, so this can't be a constructor arg."""
        for block in self.moe_blocks():
            block.ep_group = group

    def init_state(self, batch_size: int) -> State:
        c = self.config
        di, k = c.d_inner, c.d_conv
        H, P, N = c.n_heads, c.head_dim, c.d_state
        Ha, Dh = c.n_attn_heads_resolved, c.attn_head_dim
        dev = self._device
        # Per Mamba layer: (conv window (B,k-1,di), SSM state (B,H,P,N)), fp32.
        # Per attention layer: a zero-length KV cache (k,v), each (B,Ha,0,Dh), grown by step.
        # Per MoE layer: a zero-length placeholder pair (stateless FFN).
        def layer_state(i):
            if c.is_attention_layer(i):
                z = torch.zeros((batch_size, Ha, 0, Dh), device=dev)
                return (z, z)
            if c.is_moe_layer(i):
                z = torch.zeros((batch_size, 0), device=dev)
                return (z, z)                     # keeps clone_state's (a, b) unpack valid
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
        #
        # Under FSDP2 (#271) `v` is a DTensor (a per-rank SHARD, global shape). Calling
        # `.full_tensor()` is a COLLECTIVE gather back to the full tensor — EVERY rank
        # must reach this line (see `cuda_distributed.gather_portable_state_dict`'s
        # docstring); a rank-0-only call here would deadlock the others. Plain
        # `nn.Parameter`s (the non-distributed / world_size==1 path) have no
        # `full_tensor` attribute and are untouched, so this is a strict no-op there.
        out = {}
        for k, v in self.named_parameters():
            arr = v.detach()
            if hasattr(arr, "full_tensor"):
                arr = arr.full_tensor()
            arr = arr.to("cpu")
            if k.endswith(".conv.weight"):
                arr = arr.transpose(1, 2)            # (out,1,k) -> (out,k,1)
            out[k] = arr.numpy()
        # Expert parallel (#271): this rank's `named_parameters()` above only yields ITS
        # OWN local expert shard (see MoEBlock.__init__'s ModuleDict). Merging every
        # rank's shard into one unsharded dict is `cuda_distributed.
        # gather_portable_state_dict`'s job (a `gather_object` to the EP group's rank 0,
        # called from the checkpoint-save path) — NOT this method's, so a plain
        # single-rank call here still returns exactly this rank's params, matching
        # every pre-#271 caller (sizing tests, `--init`, non-distributed saves).
        # Loss-Free-Balancing bias (#213 D3): rides in the PORTABLE weights, not the
        # resume bundle — it is routing state that changes the model's function at
        # inference. `_route_bias` is a non-persistent buffer (see MoEBlock.__init__),
        # so it never appears in `named_parameters()` above; emitted explicitly, and only
        # for a block whose bias is ACTIVE — an unconditional key would break
        # `sum(v.size) == cfg.num_parameters()` (tests/test_moe.py / test_sizing.py), and
        # the bias genuinely is not a parameter, so it does not belong in
        # `num_parameters()` either way. Mirrors `mlx_backend.py:986-1003`.
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MoEBlock) and layer._bias_active:
                out[f"{self._MOE_BIAS_PREFIX}{i}"] = layer._route_bias.detach().cpu().numpy()
        return out

    def _load_portable(self, weights: dict) -> None:
        # Pop the non-parameter route-bias keys FIRST — they must never reach
        # `load_state_dict`, which only knows about the real parameter/buffer tree and
        # would raise (strict=True) on an unexpected key. Absent keys (an old checkpoint,
        # balancing off, a dense model) simply leave every block inactive. Mirrors
        # `mlx_backend.py:1005-1028`.
        biases, tensors = {}, {}
        for k, v in weights.items():
            if k.startswith(self._MOE_BIAS_PREFIX):
                biases[int(k[len(self._MOE_BIAS_PREFIX):])] = v
                continue
            t = torch.as_tensor(np.asarray(v))
            if k.endswith(".conv.weight"):
                t = t.transpose(1, 2)                # (out,k,1) -> (out,1,k)
            tensors[k] = t
        if self._ep_size > 1:
            # Expert parallel (#271): a merged checkpoint carries EVERY rank's experts,
            # but this rank's module only declares its own shard (see
            # MoEBlock.__init__'s ModuleDict). Keep only the keys this rank's
            # state_dict actually has — silently dropping the rest is safe (every OTHER
            # rank keeps the keys IT needs); a genuinely missing local key still raises
            # via strict=True below, so this can't hide a real problem.
            local_keys = set(self.state_dict().keys())
            tensors = {k: v for k, v in tensors.items() if k in local_keys}
        # FSDP2 (#271): a DTensor parameter can't accept a plain full tensor via the
        # ordinary in-place copy `load_state_dict` does — reshard the incoming full
        # tensor to match each param's mesh/placement first, the inverse of
        # `_portable_state_dict`'s `.full_tensor()` gather. Plain (non-distributed)
        # params have no `device_mesh` and are untouched.
        current = dict(self.named_parameters())
        for k, t in list(tensors.items()):
            p = current.get(k)
            if p is not None and hasattr(p, "device_mesh"):
                from torch.distributed.tensor import distribute_tensor
                tensors[k] = distribute_tensor(t.to(p.dtype), p.device_mesh, p.placements)
        self.load_state_dict(tensors, strict=True)
        if not any(hasattr(p, "device_mesh") for p in self.parameters()):
            self.to(self._device)   # FSDP2 params already live on their mesh's device
        for i, vec in biases.items():
            # Check the key names a real MoE layer before indexing: a balanced checkpoint
            # loaded into a dense or differently-interleaved config would otherwise die on
            # an IndexError deep in the load, with nothing pointing at the real mismatch.
            if not (0 <= i < len(self.layers)) or not isinstance(self.layers[i], MoEBlock):
                raise ValueError(
                    f"checkpoint has {self._MOE_BIAS_PREFIX}{i}, but layer {i} of this "
                    "config is not an MoE layer — the weights and the config disagree "
                    "about the MoE interleave."
                )
            self.layers[i].set_route_bias(np.asarray(vec).reshape(-1).tolist())
