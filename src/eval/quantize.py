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

    NOTE (#168): this is classic min/max affine (`scale = (hi-lo)/(2**bits-1) >= 0`,
    `bias = lo`) — a DIFFERENT variant from `mx.quantize`'s zero-preserving, edge-exact
    affine scheme (whose scale can be NEGATIVE). It is #51's measurement contract
    (`tests/test_quantize.py`) and stays exactly as it is; it is NOT the on-disk format
    a Swift/mlx-swift loader can consume. That format is `mlx_affine_quantize` /
    `mlx_affine_dequantize` below, verified byte-identical to `mx.quantize` in
    `tests/test_quantize_mlx_format.py`.
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
        # Signed range [-qhalf, qhalf] (e.g. ±127 at 8-bit). Scale and clip MUST use the
        # same qhalf, else a group's max-magnitude value can't be represented (it would
        # clip below its own level), inflating symmetric-mode error.
        qhalf = qmax // 2
        absmax = np.abs(groups).max(axis=1, keepdims=True)
        scale = np.where(absmax == 0.0, 1.0, absmax / qhalf)
        q = np.clip(np.round(groups / scale), -qhalf, qhalf)
        deq = q * scale
    else:
        lo = groups.min(axis=1, keepdims=True)
        hi = groups.max(axis=1, keepdims=True)
        scale = np.where(hi == lo, 1.0, (hi - lo) / qmax)
        q = np.clip(np.round((groups - lo) / scale), 0.0, qmax)
        deq = q * scale + lo

    return deq.reshape(w64.shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# MLX-compatible affine (the on-disk format — #168)
# --------------------------------------------------------------------------- #
# `quantize_dequantize` above is a DIFFERENT affine variant from `mx.quantize`
# (classic min/max affine: `scale = (hi-lo)/(2**bits-1)`, `bias = lo`, always
# `scale >= 0`). It stays exactly as-is — it is #51's measurement contract, covered
# by `tests/test_quantize.py` — but it is NOT byte-compatible with mlx-swift's
# quantized kernels and must never be used to produce a checkpoint Swift will load.
# The functions below are the format Swift actually consumes: verified against
# `mx.quantize`/`mx.dequantize` 0.31.2 to <1e-5 (scale/bias) and byte-identical
# (packed codes) across bits in {2,4,8} x group_size in {32,64,128}
# (`tests/test_quantize_mlx_format.py`). Two load-bearing differences from the
# classic scheme above:
#   - the scale is ZERO-PRESERVING and EDGE-EXACT (picks the larger-magnitude
#     group edge, then sets scale so that edge quantizes back exactly), not a
#     plain min/max span — and the scale CAN BE NEGATIVE;
#   - both `scale`/`bias` are always emitted (MLX's `QuantizationMode` is
#     `.affine`/`.mxfp4`; there is no separate symmetric mode to special-case).
def mlx_affine_quantize(
    w: np.ndarray, bits: int, group_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-wise affine quantize `w` the way `mx.quantize` does.

    `w` must be 2-D with `w.shape[-1] % group_size == 0` (MLX requires exact
    grouping; there is no ragged-group fallback on this path — see
    `_effective_group_size`'s docstring, which is for the OTHER scheme only).

    Returns `(codes, scales, biases)`:
      - `codes`: `uint32 (rows, cols)`, each entry in `[0, 2**bits - 1]` — the
        UNPACKED per-element code (packing into `uint32` words is a separate step,
        `pack_uint32_codes`, so this function stays easy to test element-by-element).
      - `scales`, `biases`: `float32 (rows, cols // group_size)`.

    Dequantize with `mlx_affine_dequantize`; `codes * scales[..., None] + biases[...,
    None]` per group reproduces the original weight to quantization error.
    """
    if bits < 1 or bits > 16:
        raise ValueError(f"bits must be in [1, 16], got {bits}")
    w64 = np.asarray(w, dtype=np.float64)
    if w64.ndim != 2:
        raise ValueError(f"mlx_affine_quantize requires a 2-D array, got shape {w64.shape}")
    rows, cols = w64.shape
    if group_size <= 0 or cols % group_size != 0:
        raise ValueError(
            f"group_size={group_size} must evenly divide the last axis ({cols}); "
            "MLX has no ragged-group fallback")
    n_groups = cols // group_size
    groups = w64.reshape(rows, n_groups, group_size)
    n = float(2 ** bits - 1)

    mn = groups.min(axis=2)
    mx_ = groups.max(axis=2)
    # Pick whichever edge has the larger magnitude — the scale is built to represent
    # THAT edge exactly, which is what makes the format zero-preserving (a group whose
    # true zero falls inside its range dequantizes back to exactly 0.0, not a rounded
    # near-zero value).
    mask = np.abs(mn) > np.abs(mx_)
    s0 = np.maximum(mx_ - mn, 1e-7) / n
    s0 = np.where(mask, s0, -s0)          # scale is SIGNED — do not assume scale > 0
    edge = np.where(mask, mn, mx_)
    q0 = np.round(edge / s0)
    # A group whose values are all equal (`mx_ == mn` -> `edge == 0` -> `q0 == 0`)
    # falls back to `scale = s0` (the 1e-7 floor) and `bias = 0.0`.
    scale = np.where(q0 == 0.0, s0, edge / q0)
    bias = np.where(q0 == 0.0, 0.0, edge)

    codes = np.clip(np.round((groups - bias[..., None]) / scale[..., None]), 0.0, n)
    codes = codes.reshape(rows, cols).astype(np.uint32)
    return codes, scale.astype(np.float32), bias.astype(np.float32)


def mlx_affine_dequantize(
    codes: np.ndarray, scales: np.ndarray, biases: np.ndarray, group_size: int
) -> np.ndarray:
    """Invert `mlx_affine_quantize`: `codes (rows, cols)` + per-group `scales`/`biases`
    `(rows, cols // group_size)` -> `float32 (rows, cols)`."""
    rows, cols = codes.shape
    n_groups = cols // group_size
    g = codes.reshape(rows, n_groups, group_size).astype(np.float64)
    deq = g * scales.astype(np.float64)[..., None] + biases.astype(np.float64)[..., None]
    return deq.reshape(rows, cols).astype(np.float32)


def pack_uint32_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack per-element `codes` (`uint32 (rows, cols)`, each in `[0, 2**bits-1]`) into
    `mx.quantize`'s on-disk layout: little-endian within each `uint32` word, `32 // bits`
    codes per word, groups running along the last axis. `packed[i, j // per_word] |=
    codes[i, j] << (bits * (j % per_word))`. Requires `cols % (32 // bits) == 0`
    (true whenever `bits` divides 32, which every supported bit-width does)."""
    rows, cols = codes.shape
    per_word = 32 // bits
    if cols % per_word != 0:
        raise ValueError(
            f"cols={cols} must be a multiple of 32/bits={per_word} to pack into uint32 words")
    n_words = cols // per_word
    codes = codes.astype(np.uint32).reshape(rows, n_words, per_word)
    shifts = (np.arange(per_word, dtype=np.uint32) * np.uint32(bits))
    packed = np.bitwise_or.reduce(codes << shifts, axis=-1)
    return packed


def unpack_uint32_codes(packed: np.ndarray, bits: int, cols: int) -> np.ndarray:
    """Invert `pack_uint32_codes`: `packed (rows, cols // per_word)` -> `codes (rows, cols)`."""
    rows = packed.shape[0]
    per_word = 32 // bits
    mask = np.uint32(2 ** bits - 1)
    codes = np.zeros((rows, cols), dtype=np.uint32)
    for j in range(cols):
        codes[:, j] = (packed[:, j // per_word] >> np.uint32(bits * (j % per_word))) & mask
    return codes


def quant_targets(
    state_dict: Dict[str, np.ndarray], group_size: int, bits: int,
    head_bits: Optional[int] = None,
    exclude: Tuple[str, ...] = ("dt_proj", "router"),
) -> Dict[str, int]:
    """Which module paths to quantize, and at what bit-width, for the MLX on-disk
    format. Starts from `is_quantizable` (2-D float, the heavy GEMM/embedding weights)
    and narrows it:

      - drop any name containing a token in `exclude` (default `dt_proj`, `router` —
        `dt_proj` is the load-bearing dt path, `router` decides argmax routing and is
        tiny; both would otherwise slip through `is_quantizable`'s shape-only test);
      - drop any tensor whose last axis is not evenly divisible by `group_size` (MLX
        has no ragged-group fallback on this path — unlike `_effective_group_size`,
        which is for the OTHER quant scheme and must not be reused here);
      - `head_bits`, if given, overrides the bit-width for `embedding.weight`
        specifically (the tied head shares that tensor — see module docstring note in
        the plan/design doc — so this is the lever for keeping the head at a higher
        precision than the rest of the model).

    Returns `{module_path: bits}` keyed by MODULE path (e.g. `"layers.0.mamba.in_proj"`),
    not parameter name (`"...in_proj.weight"`) — that is what the Swift
    `MLXNN.quantize(filter:)` callback matches against.
    """
    out: Dict[str, int] = {}
    for name, arr in state_dict.items():
        arr = np.asarray(arr)
        if not is_quantizable(name, arr):
            continue
        if any(tok in name for tok in exclude):
            continue
        cols = arr.shape[-1]
        if group_size <= 0 or cols % group_size != 0:
            continue
        assert name.endswith(".weight"), (
            f"quant_targets expects a '*.weight' parameter name, got {name!r}")
        module_path = name[: -len(".weight")]
        bits_for_this = (
            head_bits if (head_bits is not None and module_path == "embedding") else bits)
        out[module_path] = bits_for_this
    return out


def quantize_portable_state_dict(
    state_dict: Dict[str, np.ndarray], targets: Dict[str, int], group_size: int,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Apply `mlx_affine_quantize` to each targeted module's `.weight` in `state_dict`,
    replacing it with `{path}.weight` (packed uint32 codes), `{path}.scales`,
    `{path}.biases`. Every other key (including non-2-D params and any `.weight` not in
    `targets`) passes through unchanged.

    Returns `(new_state_dict, quant_block)` where `quant_block` is the JSON-able dict
    meant for the `save_weights(..., quant=quant_block)` sidecar: `{"mode": "affine",
    "group_size": group_size, "targets": targets}`. `targets` is recorded verbatim so a
    reader (Python or Swift) never has to re-derive which paths were quantized.
    """
    out: Dict[str, np.ndarray] = {}
    for name, arr in state_dict.items():
        arr = np.asarray(arr)
        module_path = name[: -len(".weight")] if name.endswith(".weight") else None
        if module_path is not None and module_path in targets:
            bits = targets[module_path]
            codes, scales, biases = mlx_affine_quantize(arr, bits, group_size)
            packed = pack_uint32_codes(codes, bits)
            out[name] = packed
            out[f"{module_path}.scales"] = scales
            out[f"{module_path}.biases"] = biases
        else:
            out[name] = arr
    quant_block = {"mode": "affine", "group_size": group_size, "targets": dict(targets)}
    return out, quant_block


def dequantize_portable_state_dict(
    state_dict: Dict[str, np.ndarray], quant_block: dict,
) -> Dict[str, np.ndarray]:
    """Invert `quantize_portable_state_dict`: reconstruct a plain fp32 state dict from a
    quantized one + its `quant` sidecar block. This is the FAKE-QUANT reference path —
    load the result into the unmodified `MLXMambaModel` (no change to `mlx_backend.py`)
    to get a Python reference whose weights are numerically identical to what Swift's
    true quantized GEMM operates on, isolating the comparison to
    accumulation-order/precision inside `quantizedMM` rather than the weights
    themselves."""
    group_size = quant_block["group_size"]
    targets: Dict[str, int] = quant_block["targets"]
    out: Dict[str, np.ndarray] = {}
    consumed = set()
    for module_path, bits in targets.items():
        w_key, s_key, b_key = (f"{module_path}.weight", f"{module_path}.scales",
                               f"{module_path}.biases")
        packed = np.asarray(state_dict[w_key])
        scales = np.asarray(state_dict[s_key])
        biases = np.asarray(state_dict[b_key])
        cols = scales.shape[-1] * group_size
        codes = unpack_uint32_codes(packed, bits, cols)
        out[w_key] = mlx_affine_dequantize(codes, scales, biases, group_size)
        consumed.update({w_key, s_key, b_key})
    for name, arr in state_dict.items():
        if name not in consumed and name not in out:
            out[name] = arr
    return out


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

    `keys`, if given, further restricts quantization to those names — but only tensors
    that ALSO satisfy `is_quantizable` are quantized (the allow-list narrows the eligible
    set, it never overrides the eligibility rules). Without `keys`, `is_quantizable`
    alone decides.
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
        # `keys` is an allow-list WITHIN the eligibility rules: a named tensor is
        # quantized only if it is also quantizable (a 2-D float weight). This prevents an
        # explicit key from silently casting a non-float/1-D tensor through fake-quant.
        target = is_quantizable(name, arr, min_elems) and (
            selected is None or name in selected)
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
