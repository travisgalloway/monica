"""forward vs step parity verification suite (Issue #356, Milestone 10).

Validates single-step recurrent generation against parallel forward outputs within
fp32 relative tolerance 1e-4 for trunk models, hybrid layers, and MTP modules
on both MLX and CUDA/PyTorch backends.
"""

import numpy as np
import pytest

from src.conformance.forward_step_parity import check_forward_step_parity
from src.model.blocks import MambaConfig

try:
    import mlx.core as mx

    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_backend import MTPBlock as MLXMTPBlock
    HAVE_MLX = True
except ImportError:
    mx = None
    HAVE_MLX = False

try:
    import torch

    from src.model.cuda_backend import CUDAMambaModel
    from src.model.cuda_backend import MTPBlock as CUDAMTPBlock
    HAVE_TORCH = True
except ImportError:
    torch = None
    HAVE_TORCH = False


def _toy_config(**kwargs) -> MambaConfig:
    defaults = {
        "d_model": 32,
        "n_layers": 2,
        "head_dim": 16,
        "d_state": 8,
        "vocab_size": 64,
        "seq_len": 16,
        "precision": "fp32",
    }
    defaults.update(kwargs)
    cfg = MambaConfig(**defaults)
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------- #
# MLX Parity Tests
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_MLX, reason="requires mlx (Apple Silicon)")
def test_mlx_forward_step_parity_standard():
    cfg = _toy_config()
    mx.random.seed(0)
    model = MLXMambaModel(cfg)

    rng = np.random.default_rng(42)
    batch = rng.integers(0, cfg.vocab_size, size=(2, 12))
    res = check_forward_step_parity(model, batch, to_numpy=np.array, rtol=1e-4, atol=1e-5)
    assert res["ok"], f"MLX forward/step parity failed: max diff = {res['max_abs_diff']}"


@pytest.mark.skipif(not HAVE_MLX, reason="requires mlx (Apple Silicon)")
def test_mlx_forward_step_parity_hybrid():
    cfg = _toy_config(attn_every=2, n_attn_heads=2)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)

    rng = np.random.default_rng(42)
    batch = rng.integers(0, cfg.vocab_size, size=(2, 12))
    res = check_forward_step_parity(model, batch, to_numpy=np.array, rtol=1e-4, atol=1e-5)
    assert res["ok"], f"MLX hybrid forward/step parity failed: max diff = {res['max_abs_diff']}"


@pytest.mark.skipif(not HAVE_MLX, reason="requires mlx (Apple Silicon)")
def test_mlx_mtp_block_forward_step_parity():
    """Verify MLX MTPBlock single-step recurrence matches parallel forward output within 1e-4."""
    cfg = _toy_config(mtp_depth=1)
    mx.random.seed(42)
    mtp_block = MLXMTPBlock(cfg, depth=1)

    B, L, D = 2, 8, cfg.d_model
    rng = np.random.default_rng(123)
    h_seq = mx.array(rng.standard_normal((B, L, D), dtype=np.float32))
    e_seq = mx.array(rng.standard_normal((B, L, D), dtype=np.float32))

    # Parallel forward
    parallel_out = np.array(mtp_block.forward_seq(h_seq, e_seq))

    # Single-step recurrence
    # Initial state for inner Mamba block: conv_state (B, k-1, di), ssm_state (B, H, P, N)
    state = (
        mx.zeros((B, cfg.d_conv - 1, cfg.d_inner)),
        mx.zeros((B, cfg.n_heads, cfg.head_dim, cfg.d_state)),
    )
    step_outs = []
    for t in range(L):
        out_t, state = mtp_block.step(h_seq[:, t], e_seq[:, t], state)
        step_outs.append(np.array(out_t))
    step_out = np.stack(step_outs, axis=1)

    diff = np.abs(parallel_out - step_out)
    max_diff = float(diff.max())
    assert np.allclose(parallel_out, step_out, rtol=1e-4, atol=1e-5), (
        f"MLX MTPBlock step parity failed with max_diff={max_diff}"
    )


# --------------------------------------------------------------------------- #
# CUDA / PyTorch Parity Tests
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_TORCH, reason="requires torch")
def test_cuda_forward_step_parity_standard():
    cfg = _toy_config()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg, device="cpu")

    rng = np.random.default_rng(42)
    batch = rng.integers(0, cfg.vocab_size, size=(2, 12))
    to_np = lambda t: t.detach().cpu().numpy()
    res = check_forward_step_parity(model, batch, to_numpy=to_np, rtol=1e-4, atol=1e-5)
    assert res["ok"], f"CUDA forward/step parity failed: max diff = {res['max_abs_diff']}"


@pytest.mark.skipif(not HAVE_TORCH, reason="requires torch")
def test_cuda_forward_step_parity_hybrid():
    cfg = _toy_config(attn_every=2, n_attn_heads=2)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg, device="cpu")

    rng = np.random.default_rng(42)
    batch = rng.integers(0, cfg.vocab_size, size=(2, 12))
    to_np = lambda t: t.detach().cpu().numpy()
    res = check_forward_step_parity(model, batch, to_numpy=to_np, rtol=1e-4, atol=1e-5)
    assert res["ok"], f"CUDA hybrid forward/step parity failed: max diff = {res['max_abs_diff']}"


@pytest.mark.skipif(not HAVE_TORCH, reason="requires torch")
def test_cuda_mtp_block_forward_step_parity():
    """Verify PyTorch/CUDA MTPBlock single-step recurrence matches parallel forward output within 1e-4."""
    cfg = _toy_config(mtp_depth=1)
    torch.manual_seed(42)
    mtp_block = CUDAMTPBlock(cfg, depth=1)

    B, L, D = 2, 8, cfg.d_model
    rng = np.random.default_rng(123)
    h_seq = torch.as_tensor(rng.standard_normal((B, L, D), dtype=np.float32))
    e_seq = torch.as_tensor(rng.standard_normal((B, L, D), dtype=np.float32))

    # Parallel forward
    parallel_out = mtp_block(h_seq, e_seq).detach().cpu().numpy()

    # Single-step recurrence
    state = (
        torch.zeros((B, cfg.d_conv - 1, cfg.d_inner)),
        torch.zeros((B, cfg.n_heads, cfg.head_dim, cfg.d_state)),
    )
    step_outs = []
    for t in range(L):
        out_t, state = mtp_block.step(h_seq[:, t], e_seq[:, t], state)
        step_outs.append(out_t.detach().cpu().numpy())
    step_out = np.stack(step_outs, axis=1)

    diff = np.abs(parallel_out - step_out)
    max_diff = float(diff.max())
    assert np.allclose(parallel_out, step_out, rtol=1e-4, atol=1e-5), (
        f"CUDA MTPBlock step parity failed with max_diff={max_diff}"
    )
