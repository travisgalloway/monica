"""Portable quantized-checkpoint driver (#168) — numpy + safetensors ONLY, no MLX/torch
import. Distinct from `scripts/quantize.py` (the #51 fake-quant MEASUREMENT driver,
which stays untouched): this script writes a REAL packed checkpoint — the format an
mlx-swift loader (`swift/engine/Sources/MonicaEngine/Quantization.swift`) consumes —
using the `mx.quantize`-compatible affine core (`src.eval.quantize.mlx_affine_quantize`).

    .venv/bin/python scripts/quantize_checkpoint.py \\
        --weights run/weights.safetensors --bits 8 --group-size 64 \\
        --out run/weights.q8.safetensors

    # int4 with the tied head kept at int8 (the accuracy lever the design doc flags):
    .venv/bin/python scripts/quantize_checkpoint.py \\
        --weights run/weights.safetensors --bits 4 --head-bits 8 \\
        --out run/weights.q4.safetensors

Prints a per-tensor size/error report and the whole-model packed/original byte totals
— the footprint evidence #168's acceptance criteria ask for (decode-speed and real
throughput are NOT measured here; see the design doc's note on that scope boundary).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.eval.quantize import quant_error, quant_targets, quantize_portable_state_dict
from src.train.checkpoint import load_config_sidecar, load_weights_dict, save_weights


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True,
                    help="portable weights.safetensors to quantize (its "
                         "<path>.config.json sidecar, if present, rides along)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: <weights>.q<bits>.safetensors)")
    ap.add_argument("--bits", type=int, default=8, choices=(2, 4, 8))
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--head-bits", type=int, default=None,
                    help="override bits for embedding.weight (the tied head); "
                         "defaults to 8 automatically when --bits 4 (see design doc)")
    args = ap.parse_args()
    if args.head_bits is None and args.bits == 4:
        args.head_bits = 8
    if args.out is None:
        args.out = args.weights.with_suffix(f".q{args.bits}.safetensors")
    return args


def main() -> None:
    args = _parse_args()

    sd = load_weights_dict(str(args.weights))
    config = load_config_sidecar(str(args.weights))  # None if no sidecar, or fp-only

    targets = quant_targets(sd, group_size=args.group_size, bits=args.bits,
                            head_bits=args.head_bits)
    if not targets:
        raise SystemExit(
            f"no quantizable tensors found for group_size={args.group_size} "
            f"(check the weights actually have 2-D GEMM/embedding weights whose last "
            f"axis divides {args.group_size})")

    qsd, quant_block = quantize_portable_state_dict(sd, targets, group_size=args.group_size)

    print(f"[quantize_checkpoint] {args.weights} -> {args.out}")
    print(f"[quantize_checkpoint] bits={args.bits} group_size={args.group_size} "
          f"head_bits={args.head_bits}  {len(targets)} targets")

    orig_bytes = 0
    packed_bytes_total = 0
    per_tensor = []
    for name, arr in sd.items():
        arr = np.asarray(arr)
        orig_bytes += arr.nbytes
        module_path = name[: -len(".weight")] if name.endswith(".weight") else None
        if module_path is not None and module_path in targets:
            w_key, s_key, b_key = (f"{module_path}.weight", f"{module_path}.scales",
                                   f"{module_path}.biases")
            packed_size = qsd[w_key].nbytes + qsd[s_key].nbytes + qsd[b_key].nbytes
            packed_bytes_total += packed_size
            from src.eval.quantize import mlx_affine_dequantize, unpack_uint32_codes
            codes = unpack_uint32_codes(qsd[w_key], targets[module_path], arr.shape[-1])
            deq = mlx_affine_dequantize(codes, qsd[s_key], qsd[b_key], args.group_size)
            err = quant_error(arr, deq)
            per_tensor.append({"name": name, "shape": arr.shape,
                               "orig_bytes": arr.nbytes, "packed_bytes": packed_size,
                               **err})
        else:
            packed_bytes_total += arr.nbytes

    for t in sorted(per_tensor, key=lambda t: -t["rel_rms"])[:10]:
        print(f"  {t['name']:<40} {str(t['shape']):<18} "
              f"{t['orig_bytes']/1024:8.1f} KiB -> {t['packed_bytes']/1024:8.1f} KiB   "
              f"rel_rms={t['rel_rms']:.4f} rel_max={t['rel_max']:.4f}")

    compression = orig_bytes / packed_bytes_total if packed_bytes_total else 0.0
    print(f"[quantize_checkpoint] model_orig_bytes={orig_bytes} "
          f"model_packed_bytes={packed_bytes_total} compression={compression:.2f}x")

    save_weights(qsd, str(args.out), config=config, quant=quant_block)
    print(f"[quantize_checkpoint] wrote {args.out} (+ .config.json with the quant block)")


if __name__ == "__main__":
    main()
