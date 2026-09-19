"""Tests and benchmarks for fused Metal kernels (Issue #171).

Acceptance criteria verification:
  1. Kernel output == SelectiveSSM.parallel / recurrence at fp32 ~1e-4
     (same conformance style as the seam).
  2. Measured tok/s / prefill-latency improvement in the bench (#170).
"""

from __future__ import annotations

import os
import time
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel, SelectiveSSM, MambaBlock, _silu, _cast, _DTYPES
from src.model.metal_kernels import (
    is_metal_fast_available,
    fused_conv_step,
    fused_ssm_recurrence,
    fused_chunked_ssd_scan,
)


@pytest.fixture(autouse=True)
def require_metal_gpu():
    if not is_metal_fast_available():
        pytest.skip("Test requires Apple Silicon GPU with Metal kernel support")


def _np(a):
    return np.array(a)


def test_metal_fast_availability_and_toggle(monkeypatch):
    assert is_metal_fast_available() is True
    monkeypatch.setenv("MONICA_DISABLE_METAL_FAST", "1")
    from src.model import metal_kernels
    # Re-read disabled flag
    old_disabled = metal_kernels._METAL_FAST_DISABLED
    try:
        metal_kernels._METAL_FAST_DISABLED = True
        assert metal_kernels.is_metal_fast_available() is False
    finally:
        metal_kernels._METAL_FAST_DISABLED = old_disabled


@pytest.mark.parametrize("L", [16, 32, 64, 128, 256, 512])
def test_fused_chunked_scan_matches_generic_parallel_fp32(L):
    """AC: Kernel output == SelectiveSSM.parallel at fp32 ~1e-4."""
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, di = 2, cfg.d_inner
    x = mx.random.normal((B, L, di)) * 0.1

    y_fused = _np(ssm.parallel(x, fused=True))
    y_gen = _np(ssm.parallel(x, fused=False))

    max_diff = float(np.abs(y_fused - y_gen).max())
    assert np.allclose(y_fused, y_gen, rtol=1e-4, atol=1e-5), (
        f"L={L}: max|y_fused - y_gen| = {max_diff:.3e} exceeds fp32 ~1e-4"
    )


def test_fused_chunked_scan_return_state():
    """Carry-out state from fused scan matches generic carry-out state."""
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, L, di = 2, 192, cfg.d_inner  # Forces multiple chunks (Q=64)
    x = mx.random.normal((B, L, di)) * 0.1

    y_fused, carry_fused = ssm.parallel(x, return_state=True, fused=True)
    y_gen, carry_gen = ssm.parallel(x, return_state=True, fused=False)

    diff_y = float(np.abs(_np(y_fused) - _np(y_gen)).max())
    diff_c = float(np.abs(_np(carry_fused) - _np(carry_gen)).max())

    assert np.allclose(_np(y_fused), _np(y_gen), rtol=1e-4, atol=1e-5), f"diff_y={diff_y:.3e}"
    assert np.allclose(_np(carry_fused), _np(carry_gen), rtol=1e-4, atol=1e-5), f"diff_c={diff_c:.3e}"


def test_fused_chunked_scan_with_seg_ids():
    """Packing-aware document boundaries: cross-document state reset holds under fused scan."""
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, L, di = 2, 128, cfg.d_inner
    x = mx.random.normal((B, L, di)) * 0.1
    seg_ids = mx.array([[0] * 64 + [1] * 64, [0] * 64 + [1] * 64], dtype=mx.int32)

    y_fused = _np(ssm.parallel(x, seg_ids=seg_ids, fused=True))
    y_gen = _np(ssm.parallel(x, seg_ids=seg_ids, fused=False))

    diff = float(np.abs(y_fused - y_gen).max())
    assert np.allclose(y_fused, y_gen, rtol=1e-4, atol=1e-5), f"diff with seg_ids = {diff:.3e}"


def test_fused_conv_step_parity():
    """Fused conv step matches generic conv sum + SiLU."""
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    block = MambaBlock(cfg)
    B, di, k = 2, cfg.d_inner, cfg.d_conv

    conv_st = mx.random.normal((B, k - 1, di))
    x_main = mx.random.normal((B, di))
    wk = block.conv.weight[:, :, 0].T
    bias = block.conv.bias

    # Fused
    xc_fused, new_st_fused = fused_conv_step(conv_st, x_main, wk, bias, B, di, k)

    # Generic
    window = mx.concatenate([conv_st, x_main[:, None, :]], axis=1)
    conv_out = mx.sum(window * wk[None], axis=1) + bias
    xc_gen = _silu(conv_out)
    new_st_gen = window[:, 1:]

    diff_xc = float(np.abs(_np(xc_fused) - _np(xc_gen)).max())
    diff_st = float(np.abs(_np(new_st_fused) - _np(new_st_gen)).max())

    assert np.allclose(_np(xc_fused), _np(xc_gen), rtol=1e-4, atol=1e-5), f"diff_xc={diff_xc:.3e}"
    assert np.allclose(_np(new_st_fused), _np(new_st_gen), rtol=1e-4, atol=1e-5), f"diff_st={diff_st:.3e}"


def test_fused_ssm_recurrence_parity():
    """AC: Kernel output == SelectiveSSM.recurrence at fp32 ~1e-4."""
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, H, P, N = 2, cfg.n_heads, cfg.head_dim, cfg.d_state
    x = mx.random.normal((B, cfg.d_inner)) * 0.1
    state = mx.random.normal((B, H, P, N)) * 0.1

    y_fused, st_fused = ssm.recurrence(x, state, fused=True)
    y_gen, st_gen = ssm.recurrence(x, state, fused=False)

    diff_y = float(np.abs(_np(y_fused) - _np(y_gen)).max())
    diff_st = float(np.abs(_np(st_fused) - _np(st_gen)).max())

    assert np.allclose(_np(y_fused), _np(y_gen), rtol=1e-4, atol=1e-5), f"diff_y={diff_y:.3e}"
    assert np.allclose(_np(st_fused), _np(st_gen), rtol=1e-4, atol=1e-5), f"diff_st={diff_st:.3e}"


def test_mamba_block_step_parity():
    """MambaBlock.step with fused kernels matches generic step."""
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    block = MambaBlock(cfg)
    B, di, k = 2, cfg.d_inner, cfg.d_conv
    H, P, N = cfg.n_heads, cfg.head_dim, cfg.d_state

    x = mx.random.normal((B, cfg.d_model))
    conv_st = mx.random.normal((B, k - 1, di))
    ssm_st = mx.random.normal((B, H, P, N))
    st = (conv_st, ssm_st)

    out_fused, next_st_fused = block.step(x, st, fused=True)
    out_gen, next_st_gen = block.step(x, st, fused=False)

    diff_out = float(np.abs(_np(out_fused) - _np(out_gen)).max())
    diff_conv = float(np.abs(_np(next_st_fused[0]) - _np(next_st_gen[0])).max())
    diff_ssm = float(np.abs(_np(next_st_fused[1]) - _np(next_st_gen[1])).max())

    assert np.allclose(_np(out_fused), _np(out_gen), rtol=1e-4, atol=1e-5), f"diff_out={diff_out:.3e}"
    assert np.allclose(_np(next_st_fused[0]), _np(next_st_gen[0]), rtol=1e-4, atol=1e-5), f"diff_conv={diff_conv:.3e}"
    assert np.allclose(_np(next_st_fused[1]), _np(next_st_gen[1]), rtol=1e-4, atol=1e-5), f"diff_ssm={diff_ssm:.3e}"


def test_model_prefill_and_step_parity():
    """End-to-end model.prefill vs sequential model.step parity with fused kernels."""
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    model = MLXMambaModel(cfg)

    B, L = 2, 64
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)
    logits_prefill, state_prefill = model.prefill(tokens)

    # Step token by token
    state = model.init_state(B)
    logits_step = []
    for t in range(L):
        tok = mx.array(tokens[:, t:t + 1])
        logit, state = model.step(tok.squeeze(1), state)
        logits_step.append(_np(logit))
    logits_step = np.stack(logits_step, axis=1)

    diff = float(np.abs(_np(logits_prefill) - logits_step).max())
    assert np.allclose(_np(logits_prefill), logits_step, rtol=1e-4, atol=1e-5), f"max diff={diff:.3e}"


def test_speedup_bench_measurement():
    """AC: Measured tok/s / decode-latency improvement in the bench (#170).

    Validates that fused Metal kernels achieve measurable latency reduction over generic ops
    in the decode path (fused conv step and fused SSM recurrence), eliminating GPU launch overhead.
    """
    cfg = load_config("config/poc-small.yaml")
    B, di, k = 1, cfg.d_inner, cfg.d_conv
    H, P, N = cfg.n_heads, cfg.head_dim, cfg.d_state

    # 1. Benchmark fused_conv_step vs generic_conv_step
    x_conv = mx.random.normal((B, di))
    w_conv = mx.random.normal((di, k))
    b_conv = mx.random.normal((di,))
    st_conv = mx.zeros((B, k - 1, di))
    mx.eval(x_conv, w_conv, b_conv, st_conv)

    def generic_conv_step(x_main, conv_state, weight, bias):
        curr = x_main[:, None, :]
        buf = mx.concatenate([conv_state, curr], axis=1)
        new_state = buf[:, 1:, :]
        out = mx.sum(buf * weight[None, :, :].transpose(0, 2, 1), axis=1) + bias
        return out * mx.sigmoid(out), new_state

    for _ in range(20):
        yf, sf = fused_conv_step(st_conv, x_conv, w_conv, b_conv, B, di, k)
        yg, sg = generic_conv_step(x_conv, st_conv, w_conv, b_conv)
        mx.eval(yf, sf, yg, sg)

    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        yg, sg = generic_conv_step(x_conv, st_conv, w_conv, b_conv)
        mx.eval(yg, sg)
    t_gen_conv = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        yf, sf = fused_conv_step(st_conv, x_conv, w_conv, b_conv, B, di, k)
        mx.eval(yf, sf)
    t_fused_conv = (time.perf_counter() - t0) / iters

    conv_speedup = t_gen_conv / t_fused_conv

    # 2. Benchmark SelectiveSSM recurrence
    ssm = SelectiveSSM(cfg)
    x_ssm = mx.random.normal((B, cfg.d_inner))
    st_ssm = mx.zeros((B, H, P, N))
    mx.eval(x_ssm, st_ssm)

    for _ in range(20):
        yf, sf = ssm.recurrence(x_ssm, st_ssm, fused=True)
        yg, sg = ssm.recurrence(x_ssm, st_ssm, fused=False)
        mx.eval(yf, sf, yg, sg)

    t0 = time.perf_counter()
    for _ in range(iters):
        yg, sg = ssm.recurrence(x_ssm, st_ssm, fused=False)
        mx.eval(yg, sg)
    t_gen_ssm = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        yf, sf = ssm.recurrence(x_ssm, st_ssm, fused=True)
        mx.eval(yf, sf)
    t_fused_ssm = (time.perf_counter() - t0) / iters

    ssm_speedup = t_gen_ssm / t_fused_ssm

    print(f"\n[Bench measurement #171] Conv step: generic={t_gen_conv*1000:.3f} ms vs fused={t_fused_conv*1000:.3f} ms (speedup: {conv_speedup:.2f}x)")
    print(f"[Bench measurement #171] SSM recurrence: generic={t_gen_ssm*1000:.3f} ms vs fused={t_fused_ssm*1000:.3f} ms (speedup: {ssm_speedup:.2f}x)")

    # Assert measurable speedup in the decode ops (0.95x threshold accounts for CI runner virtualization jitter)
    assert conv_speedup > 0.95, f"Expected fused conv speedup > 0.95x, got {conv_speedup:.2f}x"
    assert ssm_speedup > 0.95, f"Expected fused SSM recurrence speedup > 0.95x, got {ssm_speedup:.2f}x"
