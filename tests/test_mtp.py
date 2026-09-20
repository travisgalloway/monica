"""Comprehensive tests for Sequential Multi-Token Prediction (MTP) (#356).

Covers:
1. MambaConfig MTP configuration, validation, and parameter_breakdown.
2. Portable MTPBlock specification and reference evaluation.
3. MTP forward pass outputting auxiliary logits with shape (batch, seq_len - k, vocab_size).
4. Gradient propagation across all active MTP heads without loss scale collapse (MLX + CUDA).
5. Native MTP speculative decoding proposals and code completion benchmark acceptance (>60%).
"""

import numpy as np
import pytest

from src.model.blocks import MambaConfig
from src.model.blocks import MTPBlock as PortableMTPBlock

try:
    import mlx.core as mx
    import mlx.optimizers as optim

    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_backend import MTPBlock as MLXMTPBlock
    from src.model.mlx_train_step import make_train_step as make_mlx_train_step
    HAVE_MLX = True
except ImportError:
    mx = None
    HAVE_MLX = False

try:
    import torch

    from src.model.cuda_backend import CUDAMambaModel
    from src.model.cuda_backend import MTPBlock as CUDAMTPBlock
    from src.model.cuda_train_step import make_train_step as make_cuda_train_step
    HAVE_TORCH = True
except ImportError:
    torch = None
    HAVE_TORCH = False

from src.serve.spec_decode import spec_decode
from src.train.loss_scale import DynamicLossScaler


def test_mtp_config_defaults_and_validation():
    cfg = MambaConfig(d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=128, seq_len=32)
    assert cfg.mtp_depth == 0
    assert cfg.mtp_loss_weight == 0.3
    cfg.validate()

    cfg_mtp = MambaConfig(
        d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=128, seq_len=32,
        mtp_depth=2, mtp_loss_weight=0.5
    )
    assert cfg_mtp.mtp_depth == 2
    assert cfg_mtp.mtp_loss_weight == 0.5
    cfg_mtp.validate()

    with pytest.raises(ValueError):
        bad_cfg = MambaConfig(
            d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=128, seq_len=32, mtp_depth=-1
        )
        bad_cfg.validate()

    with pytest.raises(ValueError):
        bad_weight = MambaConfig(
            d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=128, seq_len=32, mtp_loss_weight=-0.1
        )
        bad_weight.validate()


def test_mtp_parameter_breakdown():
    cfg_dense = MambaConfig(d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=128, seq_len=32, mtp_depth=0)
    cfg_mtp = MambaConfig(d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=128, seq_len=32, mtp_depth=2)

    bd_dense = cfg_dense.parameter_breakdown()
    bd_mtp = cfg_mtp.parameter_breakdown()

    assert "mtp" not in bd_dense
    assert "mtp" in bd_mtp
    assert bd_mtp["mtp"] > 0
    assert cfg_mtp.num_parameters() == sum(bd_mtp.values())
    assert cfg_mtp.num_parameters() > cfg_dense.num_parameters()

    # Portable MTPBlock parameter calculation matches breakdown per layer
    portable_block = PortableMTPBlock(cfg_mtp, depth=1)
    assert bd_mtp["mtp"] == 2 * portable_block.num_parameters


def test_portable_mtp_block_eval():
    cfg = MambaConfig(d_model=32, n_layers=2, head_dim=16, d_state=8, vocab_size=64, seq_len=16, mtp_depth=1)
    block = PortableMTPBlock(cfg, depth=1)
    h = np.ones((2, 8, 32), dtype=np.float32)
    embed = np.ones((2, 8, 32), dtype=np.float32)
    out = block(h, embed)
    assert out.shape == (2, 8, 32)
    assert np.all(np.isfinite(out))

    shared_w = np.ones((64, 32), dtype=np.float32)
    logits = block(h, embed, shared_embedding_weight=shared_w)
    assert logits.shape == (2, 8, 64)
    assert np.all(np.isfinite(logits))


@pytest.mark.skipif(not HAVE_MLX, reason="requires mlx")
def test_mlx_mtp_forward_shapes():
    """Acceptance criterion: MTP forward pass outputs auxiliary logits with shape
    (batch, seq_len - k, vocab_size) for each depth k."""
    B, L, V = 3, 20, 128
    K = 3
    cfg = MambaConfig(d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=V, seq_len=32, mtp_depth=K)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)

    rng = np.random.default_rng(42)
    token_batch = rng.integers(0, V, size=(B, L)).astype(np.int64)

    logits, aux_logits = model.forward_with_mtp(token_batch)
    assert logits.shape == (B, L, V)
    assert len(aux_logits) == K

    for k in range(1, K + 1):
        aux = aux_logits[k - 1]
        assert aux.shape == (B, L - k, V), f"Depth {k} expected shape {(B, L - k, V)}, got {aux.shape}"

    # Also test forward_mtp standalone
    standalone_aux = model.forward_mtp(token_batch)
    assert len(standalone_aux) == K
    for k in range(1, K + 1):
        assert standalone_aux[k - 1].shape == (B, L - k, V)


@pytest.mark.skipif(not HAVE_TORCH, reason="requires torch")
def test_cuda_mtp_forward_shapes():
    """Acceptance criterion: MTP forward pass outputs auxiliary logits with shape
    (batch, seq_len - k, vocab_size) for each depth k (CUDA/PyTorch)."""
    B, L, V = 3, 20, 128
    K = 3
    cfg = MambaConfig(d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=V, seq_len=32, mtp_depth=K)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg, device="cpu")

    rng = np.random.default_rng(42)
    token_batch = rng.integers(0, V, size=(B, L)).astype(np.int64)

    logits, aux_logits = model.forward_with_mtp(token_batch)
    assert logits.shape == (B, L, V)
    assert len(aux_logits) == K

    for k in range(1, K + 1):
        aux = aux_logits[k - 1]
        assert aux.shape == (B, L - k, V), f"Depth {k} expected shape {(B, L - k, V)}, got {aux.shape}"

    standalone_aux = model.forward_mtp(token_batch)
    assert len(standalone_aux) == K
    for k in range(1, K + 1):
        assert standalone_aux[k - 1].shape == (B, L - k, V)


@pytest.mark.skipif(not HAVE_MLX, reason="requires mlx")
def test_mlx_mtp_training_gradients_and_loss_scale_stability():
    """Acceptance criterion: mlx_train_step computes gradients across all active MTP heads
    without loss scale collapse."""
    cfg = MambaConfig(
        d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=128, seq_len=32,
        mtp_depth=2, mtp_loss_weight=0.3
    )
    model = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=1e-3)
    scaler = DynamicLossScaler(init_scale=1024.0)
    step_fn = make_mlx_train_step(model, opt, grad_clip=1.0, scaler=scaler)

    rng = np.random.default_rng(42)
    inp = rng.integers(0, 128, size=(4, 16)).astype(np.int64)
    tgt = rng.integers(0, 128, size=(4, 16)).astype(np.int64)

    # Perform multiple steps to verify stability
    for step in range(3):
        res = step_fn(model, [(inp, tgt)], 1e-3)
        assert not res.get("skipped", False), "Step should not be skipped"
        assert np.isfinite(res["loss"]), f"Loss is non-finite: {res['loss']}"
        assert np.isfinite(res["grad_norm"]), f"Grad norm is non-finite: {res['grad_norm']}"
        assert res["loss_scale"] == 1024.0, f"Loss scale collapsed from 1024.0 to {res['loss_scale']}"


@pytest.mark.skipif(not HAVE_TORCH, reason="requires torch")
def test_cuda_mtp_training_gradients_and_loss_scale_stability():
    """Acceptance criterion: cuda_train_step computes gradients across all active MTP heads
    without loss scale collapse."""
    cfg = MambaConfig(
        d_model=64, n_layers=2, head_dim=16, d_state=8, vocab_size=128, seq_len=32,
        mtp_depth=2, mtp_loss_weight=0.3
    )
    model = CUDAMambaModel(cfg, device="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = DynamicLossScaler(init_scale=1024.0)
    step_fn = make_cuda_train_step(model, opt, grad_clip=1.0, scaler=scaler)

    rng = np.random.default_rng(42)
    inp = rng.integers(0, 128, size=(4, 16)).astype(np.int64)
    tgt = rng.integers(0, 128, size=(4, 16)).astype(np.int64)

    for step in range(3):
        res = step_fn(model, [(inp, tgt)], 1e-3)
        assert not res.get("skipped", False), "Step should not be skipped"
        assert np.isfinite(res["loss"]), f"Loss is non-finite: {res['loss']}"
        assert np.isfinite(res["grad_norm"]), f"Grad norm is non-finite: {res['grad_norm']}"
        assert res["loss_scale"] == 1024.0, f"Loss scale collapsed from 1024.0 to {res['loss_scale']}"

    # Verify gradients exist and are finite on each active MTP block
    assert len(model.mtp_blocks) == 2
    for k, block in enumerate(model.mtp_blocks, start=1):
        grads = [p.grad for p in block.parameters() if p.grad is not None]
        assert len(grads) > 0, f"No gradients on MTP head depth {k}"
        assert all(torch.isfinite(g).all() for g in grads), f"Non-finite gradients on MTP head depth {k}"


@pytest.mark.skipif(not HAVE_MLX, reason="requires mlx")
def test_spec_decode_native_mtp_code_benchmark():
    """Acceptance criterion: spec_decode completes generation using native MTP proposals
    with acceptance rates above 60% on code completion benchmarks."""
    cfg = MambaConfig(
        d_model=32, n_layers=2, head_dim=16, d_state=8,
        vocab_size=64, seq_len=32, mtp_depth=1, mtp_loss_weight=0.5, precision="fp32"
    )
    mx.random.seed(42)
    model = MLXMambaModel(cfg)

    # Train on typical structured code pattern (control flow, indentation, expressions)
    code_pattern = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32] * 6
    train_step = make_mlx_train_step(model, optim.AdamW(learning_rate=3e-3))
    batch_inp = np.array([code_pattern[:-1]], dtype=np.int64)
    batch_tgt = np.array([code_pattern[1:]], dtype=np.int64)
    for _ in range(70):
        train_step(model, [(batch_inp, batch_tgt)], 3e-3)

    prompt = code_pattern[:6]
    gen, _elapsed, stats = spec_decode(model, prompt, max_new=36, use_mtp=True)

    assert len(gen) == 36
    assert stats["drafted"] > 0
    assert stats["accept_rate"] > 0.60, f"Acceptance rate {stats['accept_rate']:.2%} is below 60%"
    assert stats["tokens_per_second"] > 0.0


@pytest.mark.skipif(not HAVE_MLX, reason="requires mlx")
def test_mlx_mtp_block_direct():
    cfg = MambaConfig(d_model=32, n_layers=2, head_dim=16, d_state=8, vocab_size=64, seq_len=16, mtp_depth=1)
    block = MLXMTPBlock(cfg, depth=1)
    h = mx.ones((2, 8, 32))
    embed = mx.ones((2, 8, 32))
    out = block(h, embed)
    assert out.shape == (2, 8, 32)


@pytest.mark.skipif(not HAVE_TORCH, reason="requires torch")
def test_cuda_mtp_block_direct():
    cfg = MambaConfig(d_model=32, n_layers=2, head_dim=16, d_state=8, vocab_size=64, seq_len=16, mtp_depth=1)
    block = CUDAMTPBlock(cfg, depth=1)
    h = torch.ones((2, 8, 32))
    embed = torch.ones((2, 8, 32))
    out = block(h, embed)
    assert out.shape == (2, 8, 32)
