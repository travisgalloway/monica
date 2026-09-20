"""Unit tests for Multi-Head Latent Attention (MLA #355).

Verifies all acceptance criteria for MLA:
1. MambaConfig.validate() validates MLA dimension bounds when enabled.
2. MLX and PyTorch backends match outputs within fp32 relative tolerance 1e-4.
3. Key-value cache memory on config/code-small-dense.yaml at 64k context drops below 0.35 GB.
4. MLA parameter breakdown matches built model parameters in both backends.
5. Inference state caches only compressed latent vector c^{KV} and decoupled rotary key k^R.
"""

import dataclasses
import pytest
import numpy as np

try:
    import mlx.core as mx
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from src.model.blocks import MambaConfig, load_config
from src.model.sizing import kv_cache_memory_bytes, kv_cache_elements_per_token
from src.serve.sessions import per_session_state_bytes


def test_mla_config_validation():
    """Verify MambaConfig.validate() enforces hardware alignment and bounds for MLA."""
    # Base valid hybrid config
    base = MambaConfig(
        d_model=64,
        n_layers=4,
        head_dim=16,
        attn_every=2,
        n_attn_heads=4,
        vocab_size=256,
        use_mla=True,
    )
    # Default resolved dimensions: dc = min(256, 32) = 32, dr = min(64, 16) = 16
    base.validate()
    assert base.mla_latent_dim_resolved == 32
    assert base.mla_rope_dim_resolved == 16

    # Explicit valid aligned dimensions
    valid_cfg = dataclasses.replace(base, mla_latent_dim=32, mla_rope_dim=16)
    valid_cfg.validate()

    # Latent dim not aligned to 8
    with pytest.raises(ValueError, match="must divide hardware alignment"):
        dataclasses.replace(base, mla_latent_dim=250).validate()

    with pytest.raises(ValueError, match="must divide hardware alignment"):
        dataclasses.replace(base, mla_latent_dim=15).validate()

    # Latent dim non-positive
    with pytest.raises(ValueError, match="must be a positive integer"):
        dataclasses.replace(base, mla_latent_dim=0).validate()

    with pytest.raises(ValueError, match="must be a positive integer"):
        dataclasses.replace(base, mla_latent_dim=-8).validate()

    # RoPE dim not aligned to 8
    with pytest.raises(ValueError, match="must divide hardware alignment"):
        dataclasses.replace(base, mla_rope_dim=7).validate()

    with pytest.raises(ValueError, match="must divide hardware alignment"):
        dataclasses.replace(base, mla_rope_dim=65).validate()

    # RoPE dim non-positive
    with pytest.raises(ValueError, match="must be a positive integer"):
        dataclasses.replace(base, mla_rope_dim=0).validate()

    # RoPE dim exceeds d_model
    with pytest.raises(ValueError, match="cannot exceed d_model"):
        dataclasses.replace(base, mla_rope_dim=128).validate()

    # Disabled MLA ignores invalid dimensions
    disabled = dataclasses.replace(base, use_mla=False, mla_latent_dim=255, mla_rope_dim=7)
    disabled.validate()


def test_mla_kv_cache_memory_code_small_dense():
    """Acceptance criterion: Key-value cache memory on config/code-small-dense.yaml
    at 64k context drops below 0.35 GB.
    """
    cfg = load_config("config/code-small-dense.yaml")
    assert cfg.n_attention_layers == 7

    # Standard MHA KV cache at 64k context:
    mha_bytes = kv_cache_memory_bytes(cfg, seq_len=64 * 1024, dtype="fp16")
    mha_gb = mha_bytes / 1e9
    assert mha_gb > 1.0, f"Expected standard MHA > 1 GB, got {mha_gb:.3f} GB"

    # With MLA enabled:
    cfg_mla = dataclasses.replace(cfg, use_mla=True)
    cfg_mla.validate()
    assert cfg_mla.mla_latent_dim_resolved == 256
    assert cfg_mla.mla_rope_dim_resolved == 64

    # Elements per token per layer: 256 (latent) + 64 (rotary) = 320 elements
    elements_per_token = kv_cache_elements_per_token(cfg_mla)
    assert elements_per_token == 7 * (256 + 64) == 2240

    mla_bytes = kv_cache_memory_bytes(cfg_mla, seq_len=64 * 1024, dtype="fp16")
    mla_gb = mla_bytes / 1e9

    # Acceptance criterion assertion:
    assert mla_gb < 0.35, f"Expected MLA KV cache < 0.35 GB, got {mla_gb:.4f} GB"
    assert mla_gb == pytest.approx(0.2936, rel=1e-2)

    # Memory reduction exceeds 70%
    reduction = (mha_bytes - mla_bytes) / mha_bytes
    assert reduction > 0.70, f"Expected >70% cache reduction, got {reduction:.1%}"


@pytest.mark.skipif(not (HAVE_MLX and HAVE_TORCH), reason="needs both mlx and torch")
def test_mla_backend_parity(tmp_path):
    """Acceptance criterion: MLX and PyTorch backends match outputs within fp32 relative
    tolerance 1e-4.
    """
    from src.model.mlx_backend import MLXMambaModel
    from src.model.cuda_backend import CUDAMambaModel
    from src.conformance.backend_parity import check_backend_parity

    cfg = load_config("config/toy-hybrid.yaml")
    cfg = dataclasses.replace(cfg, use_mla=True)
    cfg.validate()

    # 1. Initialize PyTorch model and save portable weights
    torch.manual_seed(42)
    torch_model = CUDAMambaModel(cfg)
    weights_path = str(tmp_path / "mla_weights.safetensors")
    torch_model.save(weights_path)

    # 2. Load into both models
    mlx_model = MLXMambaModel(cfg)
    mlx_model.load(weights_path)
    torch_model_loaded = CUDAMambaModel(cfg)
    torch_model_loaded.load(weights_path)

    # 3. Verify forward parity
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, cfg.vocab_size, size=(2, 24)).astype(np.int32)

    with torch.no_grad():
        res = check_backend_parity(
            mlx_model,
            torch_model_loaded,
            tokens,
            to_numpy_a=np.array,
            to_numpy_b=lambda t: t.detach().cpu().numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
    assert res["ok"], f"MLA forward parity failed: {res}"


@pytest.mark.skipif(not (HAVE_MLX and HAVE_TORCH), reason="needs both mlx and torch")
def test_mla_inference_cache_shape_and_prefill_decode_parity():
    """Verify that inference cache stores ONLY compressed latent vector c^{KV} and
    decoupled rotary key k^R, and that prefill/decode parity holds.
    """
    from src.model.mlx_backend import MLXMambaModel
    from src.model.cuda_backend import CUDAMambaModel
    from src.conformance.prefill_decode_parity import check_prefill_decode_parity

    cfg = load_config("config/toy-hybrid.yaml")
    cfg = dataclasses.replace(cfg, use_mla=True)
    cfg.validate()

    dc = cfg.mla_latent_dim_resolved
    dr = cfg.mla_rope_dim_resolved

    torch.manual_seed(0)
    torch_m = CUDAMambaModel(cfg)
    mlx_m = MLXMambaModel(cfg)

    # Check init_state shapes for attention layers
    for m, backend_name in [(mlx_m, "mlx"), (torch_m, "torch")]:
        state = m.init_state(batch_size=2)
        for i in range(cfg.n_layers):
            if cfg.is_attention_layer(i):
                c_kv_cache, k_r_cache = state[i]
                assert c_kv_cache.shape == (2, 1, 0, dc), (
                    f"{backend_name} layer {i} c_kv cache shape {c_kv_cache.shape} != (2, 1, 0, {dc})"
                )
                assert k_r_cache.shape == (2, 1, 0, dr), (
                    f"{backend_name} layer {i} k_r cache shape {k_r_cache.shape} != (2, 1, 0, {dr})"
                )

    tokens = np.random.default_rng(123).integers(0, cfg.vocab_size, size=(2, 16)).astype(np.int32)

    # Prefill and decode parity checks on both backends
    par_mlx = check_prefill_decode_parity(mlx_m, tokens, to_numpy=np.array, rtol=1e-4, atol=1e-5)
    assert par_mlx["ok"], f"MLX prefill/decode parity failed: {par_mlx}"

    with torch.no_grad():
        par_torch = check_prefill_decode_parity(
            torch_m, tokens, to_numpy=lambda t: t.detach().cpu().numpy(), rtol=1e-4, atol=1e-5
        )
    assert par_torch["ok"], f"PyTorch prefill/decode parity failed: {par_torch}"


def test_mla_parameter_breakdown():
    """Verify closed-form parameter breakdown matches built model tensors."""
    cfg = load_config("config/toy-hybrid.yaml")
    cfg_mla = dataclasses.replace(cfg, use_mla=True)
    cfg_mla.validate()

    bd = cfg_mla.parameter_breakdown()
    assert "attention" in bd
    assert cfg_mla.num_parameters() == sum(bd.values())

    if HAVE_MLX:
        from src.model.mlx_backend import MLXMambaModel
        m_mlx = MLXMambaModel(cfg_mla)
        actual_mlx = sum(int(v.size) for v in m_mlx._portable_state_dict().values())
        assert cfg_mla.num_parameters() == actual_mlx, (
            f"Formula {cfg_mla.num_parameters()} != MLX built {actual_mlx}"
        )

    if HAVE_TORCH:
        from src.model.cuda_backend import CUDAMambaModel
        m_torch = CUDAMambaModel(cfg_mla)
        actual_torch = sum(p.numel() for p in m_torch.parameters())
        assert cfg_mla.num_parameters() == actual_torch, (
            f"Formula {cfg_mla.num_parameters()} != PyTorch built {actual_torch}"
        )
