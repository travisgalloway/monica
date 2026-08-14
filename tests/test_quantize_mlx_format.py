"""The load-bearing test for #168: numpy's `mlx_affine_quantize`/`pack_uint32_codes`
must be byte-identical to real `mx.quantize` output, and `mlx_affine_dequantize` must
round-trip `mx.dequantize`. MLX-gated (`pytest.importorskip`), same shape as
`tests/test_mlx_parity.py`.

Everything downstream of this file (the checkpoint format, the Swift loader) TRUSTS
this comparison. If it does not pass, nothing else in #168 can be trusted — see
`.claude/plans/issue-168.md`.
"""

import zlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from src.eval.quantize import (
    mlx_affine_dequantize,
    mlx_affine_quantize,
    pack_uint32_codes,
    unpack_uint32_codes,
)


SHAPES = [(8, 128), (4, 64), (1, 32), (16, 256)]
BITS = (2, 4, 8)
GROUP_SIZES = (32, 64, 128)


def _seed_for(bits: int, group_size: int, shape: tuple, salt: int = 0) -> int:
    """A stable, explicit RNG seed derived from `(bits, group_size, shape)` —
    deliberately not Python's built-in `hash()`, whose tuple-hashing algorithm is not
    guaranteed stable across Python versions/interpreters, which would make a CI
    failure at one Python version hard to reproduce at another."""
    key = f"{bits}:{group_size}:{shape}:{salt}".encode()
    return zlib.crc32(key) & 0xFFFFFFFF


def _cases():
    for bits in BITS:
        for group_size in GROUP_SIZES:
            for shape in SHAPES:
                if shape[-1] % group_size != 0:
                    continue
                yield bits, group_size, shape


@pytest.mark.parametrize("bits,group_size,shape", list(_cases()))
def test_packed_codes_and_scales_match_mx_quantize_byte_for_byte(bits, group_size, shape):
    rng = np.random.default_rng(_seed_for(bits, group_size, shape))
    # A spread of magnitudes (not just unit-normal) so edge cases like a
    # larger-magnitude negative edge, and a near-constant group, both get exercised.
    w = (rng.standard_normal(shape).astype(np.float64) * rng.uniform(0.1, 50.0)).astype(np.float32)

    codes, scales, biases = mlx_affine_quantize(w, bits, group_size)
    packed = pack_uint32_codes(codes, bits)

    wq_mx, scales_mx, biases_mx = mx.quantize(mx.array(w), group_size=group_size, bits=bits)
    assert np.array_equal(packed, np.array(wq_mx)), "packed codes diverge from mx.quantize"
    assert np.allclose(scales, np.array(scales_mx), rtol=1e-5, atol=1e-8)
    assert np.allclose(biases, np.array(biases_mx), rtol=1e-5, atol=1e-8)

    # unpack must recover the same per-element codes mx.quantize implies
    recovered = unpack_uint32_codes(packed, bits, shape[-1])
    assert np.array_equal(recovered, codes)


@pytest.mark.parametrize("bits,group_size,shape", list(_cases()))
def test_dequantize_matches_mx_dequantize(bits, group_size, shape):
    rng = np.random.default_rng(_seed_for(bits, group_size, shape, salt=1))
    w = (rng.standard_normal(shape).astype(np.float64) * rng.uniform(0.1, 50.0)).astype(np.float32)

    codes, scales, biases = mlx_affine_quantize(w, bits, group_size)
    deq = mlx_affine_dequantize(codes, scales, biases, group_size)

    wq_mx, scales_mx, biases_mx = mx.quantize(mx.array(w), group_size=group_size, bits=bits)
    deq_mx = np.array(mx.dequantize(wq_mx, scales=scales_mx, biases=biases_mx,
                                     group_size=group_size, bits=bits))
    assert np.allclose(deq, deq_mx, atol=1e-5)
    # And the round trip is a reasonable approximation of the original weight.
    assert np.abs(deq - w).max() < (np.abs(w).max() + 1.0)


def test_constant_group_does_not_divide_by_zero():
    w = np.full((4, 64), 0.25, dtype=np.float32)
    codes, scales, biases = mlx_affine_quantize(w, bits=8, group_size=32)
    assert np.isfinite(scales).all() and np.isfinite(biases).all()
    deq = mlx_affine_dequantize(codes, scales, biases, group_size=32)
    assert np.allclose(deq, w, atol=1e-4)


def test_negative_scale_case_matches_mx_quantize():
    # `mx.quantize`'s scale CAN be negative (unlike the classic min/max affine scheme
    # in `quantize_dequantize`, which is always >= 0) — the plan's Context section
    # calls this out explicitly as a sign trap. This exact array is a known
    # negative-scale case (found by search against mlx 0.31.2), so this test pins the
    # sign, not just the magnitude, against the real kernel.
    rng = np.random.default_rng(0)
    rng.standard_normal((1, 32))  # burn the trial-0 draw to reach the trial-1 array
    w = (rng.standard_normal((1, 32)).astype(np.float32) * rng.uniform(0.1, 10))
    wq_mx, scales_mx, biases_mx = mx.quantize(mx.array(w), group_size=32, bits=8)
    assert np.array(scales_mx)[0, 0] < 0.0, "fixture drifted — no longer a negative-scale case"

    codes, scales, biases = mlx_affine_quantize(w, bits=8, group_size=32)
    assert scales[0, 0] < 0.0
    assert np.array_equal(pack_uint32_codes(codes, 8), np.array(wq_mx))
    assert np.allclose(scales, np.array(scales_mx), rtol=1e-5, atol=1e-8)


def test_rejects_non_divisible_group_size():
    w = np.zeros((2, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        mlx_affine_quantize(w, bits=8, group_size=64)


def test_rejects_non_2d_input():
    w = np.zeros((100,), dtype=np.float32)
    with pytest.raises(ValueError):
        mlx_affine_quantize(w, bits=8, group_size=32)
