"""Portable tests for the PTQ numeric core (#51) — no backend required.

Validates the group-wise affine quant round-trip, the size accounting, and the
state-dict selection rules in `src.eval.quantize`. The MLX driver (`scripts/quantize.py`)
wires this to a real model + the eval path; the quality/size math is proven here.
"""

import numpy as np
import pytest

from src.eval.quantize import (
    is_quantizable,
    original_bytes,
    packed_bytes,
    quant_error,
    quantize_dequantize,
    quantize_state_dict,
)


def test_roundtrip_is_lossless_at_high_bits():
    # 16-bit affine over a small range reconstructs to ~float precision.
    rng = np.random.default_rng(0)
    w = rng.standard_normal((8, 64)).astype(np.float32)
    w_hat = quantize_dequantize(w, bits=16, group_size=64)
    assert quant_error(w, w_hat)["rel_rms"] < 1e-3


def test_error_decreases_monotonically_with_bits():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((16, 128)).astype(np.float32)
    errs = [quant_error(w, quantize_dequantize(w, b, 64))["rel_rms"]
            for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


def test_smaller_groups_reduce_error():
    # Finer grouping tracks local scale better, so error drops as the group shrinks.
    rng = np.random.default_rng(2)
    w = rng.standard_normal((4, 256)).astype(np.float32)
    e_coarse = quant_error(w, quantize_dequantize(w, 4, 256))["rel_rms"]
    e_fine = quant_error(w, quantize_dequantize(w, 4, 32))["rel_rms"]
    assert e_fine < e_coarse


def test_constant_group_has_zero_error():
    # A degenerate (hi == lo) group must not divide by zero and must reconstruct exactly.
    w = np.full((3, 32), 0.7, dtype=np.float32)
    w_hat = quantize_dequantize(w, bits=4, group_size=32)
    assert np.allclose(w_hat, w)


def test_symmetric_roundtrip_runs_and_is_reasonable():
    rng = np.random.default_rng(3)
    w = rng.standard_normal((8, 64)).astype(np.float32)
    e = quant_error(w, quantize_dequantize(w, 8, 64, symmetric=True))["rel_rms"]
    assert 0.0 < e < 0.1


def test_non_divisible_group_falls_back_to_whole_row():
    # 100 is not divisible by 64 -> whole-row groups; must still round-trip cleanly at 16 bits.
    rng = np.random.default_rng(4)
    w = rng.standard_normal((5, 100)).astype(np.float32)
    w_hat = quantize_dequantize(w, bits=16, group_size=64)
    assert w_hat.shape == w.shape
    assert quant_error(w, w_hat)["rel_rms"] < 1e-2


def test_packed_bytes_below_fp16_for_w8_and_w4():
    shape = (512, 512)
    fp16 = original_bytes(shape, dtype_bytes=2)
    assert packed_bytes(shape, bits=8, group_size=64) < fp16
    assert packed_bytes(shape, bits=4, group_size=64) < packed_bytes(shape, 8, 64)


def test_packed_bytes_accounts_group_metadata():
    # Two scale+zero fp16 per group on top of the bit-packed codes.
    shape = (64, 64)
    n, g = 64 * 64, 64
    expected = n * 8 / 8.0 + (n // g) * (2 * 16) / 8.0
    assert packed_bytes(shape, bits=8, group_size=64) == pytest.approx(expected)


def test_is_quantizable_targets_only_2d_float_weights():
    assert is_quantizable("layers.0.in_proj.weight", np.zeros((16, 16), np.float32))
    assert not is_quantizable("norm.weight", np.zeros((16,), np.float32))      # 1-D
    assert not is_quantizable("ssm.A_log", np.zeros((8,), np.float32))         # 1-D
    assert not is_quantizable("idx", np.zeros((16, 16), np.int64))             # non-float
    assert not is_quantizable("tiny", np.zeros((2, 2), np.float32), min_elems=100)


def test_quantize_state_dict_selects_and_reports():
    sd = {
        "embedding.weight": np.random.default_rng(5).standard_normal((128, 64)).astype(np.float32),
        "layers.0.in_proj.weight": np.random.default_rng(6).standard_normal((128, 64)).astype(np.float32),
        "norm.weight": np.ones((64,), np.float32),                # 1-D, untouched
        "layers.0.ssm.A_log": np.zeros((8,), np.float32),         # 1-D, untouched
    }
    qsd, report = quantize_state_dict(sd, bits=8, group_size=64)
    assert report["n_quantized"] == 2 and report["n_total"] == 4
    # untouched tensors pass through byte-identical
    assert np.array_equal(qsd["norm.weight"], sd["norm.weight"])
    # targeted tensors changed (carry quant error) but keep shape/dtype
    assert qsd["embedding.weight"].shape == sd["embedding.weight"].shape
    assert not np.array_equal(qsd["layers.0.in_proj.weight"], sd["layers.0.in_proj.weight"])
    assert report["model_compression"] > 1.0
    assert report["quantized_compression"] > 1.0


def test_quantize_state_dict_honours_explicit_keys():
    sd = {
        "a.weight": np.ones((16, 16), np.float32),
        "b.weight": np.ones((16, 16), np.float32),
    }
    qsd, report = quantize_state_dict(sd, 8, 16, keys=["a.weight"])
    assert report["n_quantized"] == 1
    assert "a.weight" in {t["name"] for t in report["per_tensor"]}
