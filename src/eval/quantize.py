"""Post-training quantization of portable weights (#51 / #65 Phase 4 — Quantize).

A MEASUREMENT spike, not a serving feature: it answers "what does W8/W4 weight
quantization cost in held-out perplexity, and what does it save in model size?" on
the portable safetensors produced by `src/train/checkpoint.py` (the cross-backend
bridge). The production inference path (serving #15) and CUDA (#16) are out of scope.

This module is the PORTABLE numeric core — pure numpy, importable on any host and
testable without a backend (it never imports `mlx`/`torch`). The MLX driver
`scripts/quantize.py` applies `quantize_state_dict` to a real checkpoint and runs
`src/eval/val_loss.evaluate` before/after.

Method: group-wise AFFINE quantization (the de-facto scheme behind MambaQuant /
LightMamba / MLX's own `mx.quantize`). Each weight matrix is grouped along its last
axis; within a group the values map linearly to `bits`-bit integers via a per-group
scale (and zero point, unless `symmetric`). "Fake quantization" — we quantize then
dequantize back to float — measures the *quality* cost exactly (the round-tripped
weight carries the true quantization error), while `packed_bytes` accounts the *size*
the genuinely-packed representation would occupy. Weight-only (the activations stay
float); true activation quant (the "A8" in W8A8) is a later step, not done here.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Group-wise affine quantization (the numeric core)
# --------------------------------------------------------------------------- #
def _effective_group_size(cols: int, group_size: int) -> int:
    """Group along the last axis; fall back to whole-row groups when `group_size`
    does not divide the row width (keeps the grouping exact rather than ragged)."""
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    return group_size if cols % group_size == 0 else cols


def quantize_dequantize(w: np.ndarray, bits: int, group_size: int,
                        symmetric: bool = False) -> np.ndarray:
    """Round-trip `w` through `bits`-bit group-wise affine quantization (fake quant).

    Returns a float32 array of `w`'s shape carrying the quantization error — feed it
    back into the model to measure the perplexity cost. Groups run along the last
    axis (`_effective_group_size`). `symmetric` centres the range on zero (no zero
    point), which is what weight-quant schemes typically use for signed weights.
    """
    if bits < 1 or bits > 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    w64 = np.asarray(w, dtype=np.float64)
    if w64.ndim < 1 or w64.size == 0:
        return w64.astype(np.float32)
    cols = w64.shape[-1]
    g = _effective_group_size(cols, group_size)
    groups = w64.reshape(-1, g)
    qmax = float(2 ** bits - 1)

    if symmetric:
        absmax = np.abs(groups).max(axis=1, keepdims=True)
        scale = np.where(absmax == 0.0, 1.0, absmax / (qmax / 2.0))
        q = np.clip(np.round(groups / scale), -(qmax // 2), qmax // 2)
        deq = q * scale
    else:
        lo = groups.min(axis=1, keepdims=True)
        hi = groups.max(axis=1, keepdims=True)
        scale = np.where(hi == lo, 1.0, (hi - lo) / qmax)
        q = np.clip(np.round((groups - lo) / scale), 0.0, qmax)
        deq = q * scale + lo

    return deq.reshape(w64.shape).astype(np.float32)


def quant_error(w: np.ndarray, w_hat: np.ndarray) -> Dict[str, float]:
    """Relative quantization error of a round-trip: rms and max-abs, each normalised
    by the rms magnitude of `w` (so it is scale-free and comparable across tensors)."""
    w = np.asarray(w, dtype=np.float64)
    w_hat = np.asarray(w_hat, dtype=np.float64)
    denom = float(np.sqrt(np.mean(w * w))) or 1.0
    err = w_hat - w
    return {
        "rel_rms": float(np.sqrt(np.mean(err * err)) / denom),
        "rel_max": float(np.abs(err).max() / denom),
    }


# --------------------------------------------------------------------------- #
# Size accounting
# --------------------------------------------------------------------------- #
def packed_bytes(shape: Tuple[int, ...], bits: int, group_size: int,
                 symmetric: bool = False, meta_bits: int = 16) -> float:
    """Bytes a genuinely-packed group affine tensor would occupy: the `bits`-bit
    codes plus per-group metadata (a `meta_bits` scale, and a zero point unless
    `symmetric`). The fake-quant float array is NOT this size — this is the figure a
    real packed export (e.g. `mx.quantize`) would write."""
    n = int(np.prod(shape)) if shape else 0
    cols = shape[-1] if shape else 1
    g = _effective_group_size(cols, group_size)
    n_groups = n // g if g else 0
    meta_per_group = (meta_bits if symmetric else 2 * meta_bits) / 8.0
    return n * bits / 8.0 + n_groups * meta_per_group


def original_bytes(shape: Tuple[int, ...], dtype_bytes: int = 2) -> float:
    """Reference size of an unquantized tensor. Defaults to fp16 (2 bytes) — the
    realistic deployment baseline for the portable weights, so W8/W4 deltas read as
    the genuine saving rather than an inflated one against fp32 masters."""
    return float(int(np.prod(shape)) if shape else 0) * dtype_bytes


# --------------------------------------------------------------------------- #
# Picking which params to quantize
# --------------------------------------------------------------------------- #
def is_quantizable(name: str, arr: np.ndarray, min_elems: int = 1) -> bool:
    """Quantize the heavy 2-D GEMM/embedding weights; leave 1-D params (RMSNorm
    weights, the SSM `A_log`/`D`, biases) and floating-point-sensitive tiny tensors in
    full precision — the standard weight-only PTQ rule. Non-float arrays are skipped.
    """
    if not np.issubdtype(np.asarray(arr).dtype, np.floating):
        return False
    if arr.ndim != 2:
        return False
    return int(np.prod(arr.shape)) >= min_elems


def quantize_state_dict(
    state_dict: Dict[str, np.ndarray], bits: int, group_size: int,
    symmetric: bool = False, min_elems: int = 1,
    baseline_dtype_bytes: int = 2,
    keys: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Fake-quantize the eligible tensors of a portable weight dict.

    Returns `(new_state_dict, report)`. `new_state_dict` is the input with each
    targeted tensor replaced by its round-tripped (error-carrying) version — load it
    into the model to measure perplexity. `report` carries per-tensor error/size and
    totals (original vs packed bytes over the *targeted* tensors, plus the whole-model
    size with untouched tensors counted at `baseline_dtype_bytes`).

    `keys`, if given, restricts quantization to those names (others pass through);
    otherwise `is_quantizable` decides.
    """
    selected = set(keys) if keys is not None else None
    out: Dict[str, np.ndarray] = {}
    per_tensor: List[dict] = []
    q_orig = q_packed = 0.0
    model_orig = model_after = 0.0

    for name, arr in state_dict.items():
        arr = np.asarray(arr)
        full = original_bytes(arr.shape, baseline_dtype_bytes)
        model_orig += full
        target = name in selected if selected is not None else is_quantizable(
            name, arr, min_elems)
        if not target:
            out[name] = arr
            model_after += full
            continue
        w_hat = quantize_dequantize(arr, bits, group_size, symmetric)
        out[name] = w_hat
        pk = packed_bytes(arr.shape, bits, group_size, symmetric)
        q_orig += full
        q_packed += pk
        model_after += pk
        per_tensor.append({
            "name": name, "shape": tuple(int(s) for s in arr.shape),
            "orig_bytes": full, "packed_bytes": pk,
            **quant_error(arr, w_hat),
        })

    report = {
        "bits": bits, "group_size": group_size, "symmetric": symmetric,
        "n_quantized": len(per_tensor), "n_total": len(state_dict),
        "quantized_orig_bytes": q_orig, "quantized_packed_bytes": q_packed,
        "quantized_compression": (q_orig / q_packed) if q_packed else 0.0,
        "model_orig_bytes": model_orig, "model_packed_bytes": model_after,
        "model_compression": (model_orig / model_after) if model_after else 0.0,
        "per_tensor": per_tensor,
    }
    return out, report
