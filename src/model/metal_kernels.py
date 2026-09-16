"""Fused Metal kernels for Mamba-2 / SSD selective SSM on Apple Silicon (Issue #171).

Implements custom Metal kernels via `mx.fast.metal_kernel`:
  1. `fused_chunk_state`: intra-chunk state scan computing each chunk's final SSM state.
  2. `fused_chunk_out`: intra-chunk output scan computing Y from entering state and inputs.
  3. `fused_chunked_ssd_scan`: replaces the generic einsum chain (_segsum -> Lmask ->
     intra/inter-chunk einsums) with the 2-kernel fused chunked SSD scan.
  4. `fused_conv_step`: fused causal depthwise conv + SiLU + window update for decode.
  5. `fused_ssm_recurrence`: fused one-step SSM recurrence state update and output.

All kernels operate at fp32 precision and agree with `SelectiveSSM.parallel` /
`SelectiveSSM.recurrence` at ~1e-4 rel (well within the conformance seam).
"""

from __future__ import annotations

import os
from typing import Tuple, Optional
import mlx.core as mx

from .interface import Array, State


# --------------------------------------------------------------------------- #
# Availability & Caching
# --------------------------------------------------------------------------- #

_METAL_FAST_DISABLED = os.environ.get("MONICA_DISABLE_METAL_FAST", "0") in ("1", "true", "True")

_CHUNK_STATE_KERNEL = None
_CHUNK_OUT_KERNEL = None
_CONV_STEP_KERNEL = None
_SSM_REC_KERNEL = None


def is_metal_fast_available() -> bool:
    """True if running on Apple Silicon GPU with mx.fast.metal_kernel support."""
    if _METAL_FAST_DISABLED:
        return False
    try:
        return mx.default_device() == mx.gpu and hasattr(mx.fast, "metal_kernel")
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Metal Kernel Sources
# --------------------------------------------------------------------------- #

_SRC_CHUNK_STATE = """
    uint idx = thread_position_in_grid.x;
    uint total = B_val * nc_val * H_val * P_val;
    if (idx >= total) return;

    uint p = idx % P_val;
    uint tmp = idx / P_val;
    uint h = tmp % H_val;
    tmp = tmp / H_val;
    uint c = tmp % nc_val;
    uint b = tmp / nc_val;

    float s[256];
    for (uint n = 0; n < N_val; ++n) s[n] = 0.0f;

    uint gc_base = b * (H_val * nc_val * Q_val) + h * (nc_val * Q_val) + c * Q_val;
    uint xc_base = b * (nc_val * Q_val * H_val * P_val) + c * (Q_val * H_val * P_val);
    uint bc_base = b * (nc_val * Q_val * N_val) + c * (Q_val * N_val);

    for (uint i = 0; i < Q_val; ++i) {
        float da = metal::exp(gc[gc_base + i]);
        float x_val = Xc[xc_base + i * (H_val * P_val) + h * P_val + p];
        uint b_off = bc_base + i * N_val;
        for (uint n = 0; n < N_val; ++n) {
            s[n] = da * s[n] + x_val * Bc[b_off + n];
        }
    }

    uint out_base = b * (nc_val * H_val * P_val * N_val) + c * (H_val * P_val * N_val) + h * (P_val * N_val) + p * N_val;
    for (uint n = 0; n < N_val; ++n) {
        states[out_base + n] = s[n];
    }
"""

_SRC_CHUNK_OUT = """
    uint idx = thread_position_in_grid.x;
    uint total = B_val * nc_val * H_val * P_val;
    if (idx >= total) return;

    uint p = idx % P_val;
    uint tmp = idx / P_val;
    uint h = tmp % H_val;
    tmp = tmp / H_val;
    uint c = tmp % nc_val;
    uint b = tmp / nc_val;

    float d_val = D[h];
    float s[256];
    uint s_enter_base = b * (nc_val * H_val * P_val * N_val) + c * (H_val * P_val * N_val) + h * (P_val * N_val) + p * N_val;
    for (uint n = 0; n < N_val; ++n) {
        s[n] = S_enter[s_enter_base + n];
    }

    uint gc_base = b * (H_val * nc_val * Q_val) + h * (nc_val * Q_val) + c * Q_val;
    uint xc_base = b * (nc_val * Q_val * H_val * P_val) + c * (Q_val * H_val * P_val);
    uint bc_base = b * (nc_val * Q_val * N_val) + c * (Q_val * N_val);
    uint cc_base = b * (nc_val * Q_val * N_val) + c * (Q_val * N_val);
    uint x_base = b * (Lp_val * H_val * P_val) + c * (Q_val * H_val * P_val);

    for (uint i = 0; i < Q_val; ++i) {
        float da = metal::exp(gc[gc_base + i]);
        float x_val = Xc[xc_base + i * (H_val * P_val) + h * P_val + p];
        uint b_off = bc_base + i * N_val;
        uint c_off = cc_base + i * N_val;
        float y_acc = 0.0f;
        for (uint n = 0; n < N_val; ++n) {
            float sn = da * s[n] + x_val * Bc[b_off + n];
            s[n] = sn;
            y_acc += sn * Cc[c_off + n];
        }
        float raw_x = X[x_base + i * (H_val * P_val) + h * P_val + p];
        Y_out[x_base + i * (H_val * P_val) + h * P_val + p] = y_acc + raw_x * d_val;
    }
"""

_SRC_CONV_STEP = """
    uint idx = thread_position_in_grid.x;
    uint total = B_val * di_val;
    if (idx >= total) return;

    uint d = idx % di_val;
    uint b = idx / di_val;

    float acc = bias[d];
    uint w_minus_1 = K_val - 1;

    for (uint j = 0; j < w_minus_1; ++j) {
        float val = conv_state[b * (w_minus_1 * di_val) + j * di_val + d];
        acc += val * weight[j * di_val + d];
        if (j > 0) {
            new_conv_state[b * (w_minus_1 * di_val) + (j - 1) * di_val + d] = val;
        }
    }

    float x_val = x_main[b * di_val + d];
    acc += x_val * weight[w_minus_1 * di_val + d];
    if (w_minus_1 > 0) {
        new_conv_state[b * (w_minus_1 * di_val) + (w_minus_1 - 1) * di_val + d] = x_val;
    }

    float sig = 1.0f / (1.0f + metal::exp(-acc));
    xc[idx] = acc * sig;
"""

_SRC_SSM_REC = """
    uint idx = thread_position_in_grid.x;
    uint total = B_val * H_val * P_val;
    if (idx >= total) return;

    uint p = idx % P_val;
    uint tmp = idx / P_val;
    uint h = tmp % H_val;
    uint b = tmp / H_val;

    float dlt = delta[b * H_val + h];
    float da = metal::exp(dlt * a[h]);
    float x_val = Xh[idx];
    float d_val = D[h];

    float y_acc = 0.0f;
    uint state_offset = b * (H_val * P_val * N_val) + h * (P_val * N_val) + p * N_val;
    uint b_offset = b * N_val;

    for (uint n = 0; n < N_val; ++n) {
        float s = state[state_offset + n];
        float bm = Bm[b_offset + n];
        float cm = Cm[b_offset + n];
        float s_new = da * s + (dlt * x_val) * bm;
        new_state[state_offset + n] = s_new;
        y_acc += s_new * cm;
    }
    y_out[idx] = y_acc + x_val * d_val;
"""


# --------------------------------------------------------------------------- #
# Kernel Getters (Lazy Compiled)
# --------------------------------------------------------------------------- #

def get_chunk_state_kernel():
    global _CHUNK_STATE_KERNEL
    if _CHUNK_STATE_KERNEL is None:
        _CHUNK_STATE_KERNEL = mx.fast.metal_kernel(
            name="fused_chunk_state",
            input_names=["Xc", "Bc", "gc", "B_val", "nc_val", "H_val", "P_val", "Q_val", "N_val"],
            output_names=["states"],
            source=_SRC_CHUNK_STATE,
        )
    return _CHUNK_STATE_KERNEL


def get_chunk_out_kernel():
    global _CHUNK_OUT_KERNEL
    if _CHUNK_OUT_KERNEL is None:
        _CHUNK_OUT_KERNEL = mx.fast.metal_kernel(
            name="fused_chunk_out",
            input_names=[
                "S_enter", "Xc", "Bc", "Cc", "gc", "X", "D",
                "B_val", "nc_val", "H_val", "P_val", "Q_val", "Lp_val", "N_val",
            ],
            output_names=["Y_out"],
            source=_SRC_CHUNK_OUT,
        )
    return _CHUNK_OUT_KERNEL


def get_conv_step_kernel():
    global _CONV_STEP_KERNEL
    if _CONV_STEP_KERNEL is None:
        _CONV_STEP_KERNEL = mx.fast.metal_kernel(
            name="fused_conv_step",
            input_names=["conv_state", "x_main", "weight", "bias", "B_val", "di_val", "K_val"],
            output_names=["xc", "new_conv_state"],
            source=_SRC_CONV_STEP,
        )
    return _CONV_STEP_KERNEL


def get_ssm_recurrence_kernel():
    global _SSM_REC_KERNEL
    if _SSM_REC_KERNEL is None:
        _SSM_REC_KERNEL = mx.fast.metal_kernel(
            name="fused_ssm_rec",
            input_names=["state", "delta", "a", "Xh", "Bm", "Cm", "D", "B_val", "H_val", "P_val", "N_val"],
            output_names=["y_out", "new_state"],
            source=_SRC_SSM_REC,
        )
    return _SSM_REC_KERNEL


# --------------------------------------------------------------------------- #
# Fused Op Callables
# --------------------------------------------------------------------------- #

def fused_chunked_ssd_scan(
    Xc: Array,
    Bc: Array,
    Cc: Array,
    gc: Array,
    Acum: Array,
    X: Array,
    D: Array,
    B: int,
    L: int,
    Lp: int,
    nc: int,
    H: int,
    P: int,
    Q: int,
    N: int,
    d_inner: int,
    cd,
    seg_mask: Optional[Array] = None,
    return_state: bool = False,
    segsum_fn = None,
) -> Array | Tuple[Array, Array]:
    """Execute the fused chunked SSD scan replacing the generic einsum chain.

    Intra-chunk state accumulation and output generation are fused into two Metal
    kernels, while the tiny (nc+1, nc+1) inter-chunk recurrence connects them.
    """
    state_kernel = get_chunk_state_kernel()
    out_kernel = get_chunk_out_kernel()

    B_arr = mx.array(B, dtype=mx.uint32)
    nc_arr = mx.array(nc, dtype=mx.uint32)
    H_arr = mx.array(H, dtype=mx.uint32)
    P_arr = mx.array(P, dtype=mx.uint32)
    Q_arr = mx.array(Q, dtype=mx.uint32)
    Lp_arr = mx.array(Lp, dtype=mx.uint32)
    N_arr = mx.array(N, dtype=mx.uint32)

    num_threads = B * nc * H * P
    tg = min(256, num_threads)

    # 1. Intra-chunk state accumulation
    states_out = state_kernel(
        inputs=[Xc, Bc, gc, B_arr, nc_arr, H_arr, P_arr, Q_arr, N_arr],
        template=[("T", mx.float32)],
        grid=(num_threads, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, nc, H, P, N)],
        output_dtypes=[mx.float32],
    )[0]

    # 2. Inter-chunk recurrence over the nc chunk-states
    states_cat = mx.concatenate([mx.zeros((B, 1, H, P, N), dtype=states_out.dtype), states_out], axis=1)
    chunk_tot = mx.pad(Acum[..., -1], [(0, 0), (0, 0), (1, 0)])
    decay_chunk = mx.exp(segsum_fn(chunk_tot))
    if seg_mask is not None:
        decay_chunk = decay_chunk * seg_mask[:, None].astype(decay_chunk.dtype)
    new_states = mx.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_cat)
    S_enter = new_states[:, :-1]

    # 3. Intra-chunk output generation from entering state
    Y_out = out_kernel(
        inputs=[S_enter, Xc, Bc, Cc, gc, X, D, B_arr, nc_arr, H_arr, P_arr, Q_arr, Lp_arr, N_arr],
        template=[("T", mx.float32)],
        grid=(num_threads, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, Lp, H, P)],
        output_dtypes=[mx.float32],
    )[0]

    Y = Y_out[:, :L]
    Y = Y if cd == mx.float32 else Y.astype(cd)
    Y = Y.reshape(B, L, d_inner)

    if return_state:
        return Y, new_states[:, -1]
    return Y


def fused_conv_step(
    conv_state: Array,
    x_main: Array,
    weight: Array,
    bias: Array,
    B: int,
    di: int,
    K: int,
) -> Tuple[Array, Array]:
    """Fused causal depthwise conv + SiLU + window update for decode step."""
    kernel = get_conv_step_kernel()
    B_arr = mx.array(B, dtype=mx.uint32)
    di_arr = mx.array(di, dtype=mx.uint32)
    K_arr = mx.array(K, dtype=mx.uint32)

    total_threads = B * di
    tg = min(256, total_threads)

    w_minus_1 = K - 1
    new_conv_shape = (B, w_minus_1, di) if w_minus_1 > 0 else (B, 0, di)

    conv_st = conv_state if conv_state.dtype == mx.float32 else conv_state.astype(mx.float32)
    xm = x_main if x_main.dtype == mx.float32 else x_main.astype(mx.float32)
    w = weight if weight.dtype == mx.float32 else weight.astype(mx.float32)
    b = bias if bias.dtype == mx.float32 else bias.astype(mx.float32)

    xc, new_conv_st = kernel(
        inputs=[conv_st, xm, w, b, B_arr, di_arr, K_arr],
        template=[("T", mx.float32)],
        grid=(total_threads, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, di), new_conv_shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    return xc, new_conv_st


def fused_ssm_recurrence(
    state: State,
    delta: Array,
    a: Array,
    Xh: Array,
    Bm: Array,
    Cm: Array,
    D: Array,
    B: int,
    H: int,
    P: int,
    N: int,
    cd,
) -> Tuple[Array, State]:
    """Fused one-step SSM recurrence state update and output."""
    kernel = get_ssm_recurrence_kernel()
    B_arr = mx.array(B, dtype=mx.uint32)
    H_arr = mx.array(H, dtype=mx.uint32)
    P_arr = mx.array(P, dtype=mx.uint32)
    N_arr = mx.array(N, dtype=mx.uint32)

    total_threads = B * H * P
    tg = min(256, total_threads)

    st = state if state.dtype == mx.float32 else state.astype(mx.float32)
    dlt = delta if delta.dtype == mx.float32 else delta.astype(mx.float32)
    a_vec = a if a.dtype == mx.float32 else a.astype(mx.float32)
    x_h = Xh if Xh.dtype == mx.float32 else Xh.astype(mx.float32)
    bm = Bm if Bm.dtype == mx.float32 else Bm.astype(mx.float32)
    cm = Cm if Cm.dtype == mx.float32 else Cm.astype(mx.float32)
    d_vec = D if D.dtype == mx.float32 else D.astype(mx.float32)

    y_out, new_st = kernel(
        inputs=[st, dlt, a_vec, x_h, bm, cm, d_vec, B_arr, H_arr, P_arr, N_arr],
        template=[("T", mx.float32)],
        grid=(total_threads, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(B, H, P), (B, H, P, N)],
        output_dtypes=[mx.float32, mx.float32],
    )
    y = y_out if cd == mx.float32 else y_out.astype(cd)
    return y.reshape(B, -1), new_st
