"""Unit tests for arbitrary-depth attention placement, d_state validation, and ablation sweep harness (#219)."""

from __future__ import annotations

import copy
import pytest

from src.model.blocks import MambaConfig
from src.eval.ablation_sweep import (
    AblationCandidate,
    AblationResult,
    AblationSweepRunner,
    compute_attention_layers,
    default_small_base_config,
    evaluate_candidate,
    format_sweep_table,
    generate_ablation_grid,
    mock_evaluator,
    rank_candidates,
    select_winner,
)


# ==============================================================================
# 1. Arbitrary-Depth Attention Placement Tests
# ==============================================================================

def test_attn_layers_explicit_placement():
    """Verify that attn_layers places attention blocks exactly at requested indices."""
    cfg = MambaConfig(
        d_model=64,
        n_layers=8,
        head_dim=16,
        attn_layers=[1, 4, 7],
    )
    cfg.validate()

    assert cfg.n_attention_layers == 3
    for i in range(8):
        if i in (1, 4, 7):
            assert cfg.is_attention_layer(i) is True
        else:
            assert cfg.is_attention_layer(i) is False


def test_attn_layers_equivalence_with_attn_every():
    """Verify that attn_layers equivalent to periodic attn_every yields identical parameters."""
    # In 8 layers, attn_every=2 means layers 1, 3, 5, 7 are attention (0-indexed)
    cfg_every = MambaConfig(
        d_model=64,
        n_layers=8,
        head_dim=16,
        attn_every=2,
    )
    cfg_every.validate()

    cfg_layers = MambaConfig(
        d_model=64,
        n_layers=8,
        head_dim=16,
        attn_layers=[1, 3, 5, 7],
    )
    cfg_layers.validate()

    assert cfg_every.n_attention_layers == cfg_layers.n_attention_layers == 4
    for i in range(8):
        assert cfg_every.is_attention_layer(i) == cfg_layers.is_attention_layer(i)

    assert cfg_every.parameter_breakdown() == cfg_layers.parameter_breakdown()
    assert cfg_every.num_parameters() == cfg_layers.num_parameters()
    assert cfg_every.active_num_parameters() == cfg_layers.active_num_parameters()


def test_attn_layers_precedence_over_moe():
    """Attention layers must take precedence over MoE layers when colliding."""
    # 12 layers: moe_every=4 selects 3, 7, 11.
    # attn_layers=[3] collides at layer 3.
    cfg = MambaConfig(
        d_model=64,
        n_layers=12,
        head_dim=16,
        moe_every=4,
        n_experts=4,
        top_k=2,
        attn_layers=[3],
    )
    cfg.validate()

    assert cfg.is_attention_layer(3) is True
    assert cfg.is_moe_layer(3) is False  # Attention takes precedence at layer 3!
    assert cfg.is_moe_layer(7) is True
    assert cfg.is_moe_layer(11) is True
    assert cfg.n_moe_layers == 2
    assert cfg.n_attention_layers == 1


def test_attn_layers_mutual_exclusion_with_attn_every():
    """Config must reject specifying both attn_every and attn_layers."""
    cfg = MambaConfig(
        d_model=64,
        n_layers=8,
        head_dim=16,
        attn_every=4,
        attn_layers=[3, 7],
    )
    with pytest.raises(ValueError, match="cannot specify both attn_every and attn_layers"):
        cfg.validate()


def test_attn_layers_out_of_range_rejected():
    """Indices outside [0, n_layers) must be rejected."""
    # Negative index
    cfg_neg = MambaConfig(d_model=64, n_layers=4, head_dim=16, attn_layers=[-1, 2])
    with pytest.raises(ValueError, match="out of range"):
        cfg_neg.validate()

    # Index >= n_layers
    cfg_high = MambaConfig(d_model=64, n_layers=4, head_dim=16, attn_layers=[1, 4])
    with pytest.raises(ValueError, match="out of range"):
        cfg_high.validate()


def test_attn_layers_duplicates_rejected():
    """Duplicate layer indices must be rejected."""
    cfg = MambaConfig(d_model=64, n_layers=4, head_dim=16, attn_layers=[1, 2, 2])
    with pytest.raises(ValueError, match="duplicate indices"):
        cfg.validate()


def test_attn_layers_non_int_rejected():
    """Non-integer layer entries must be rejected."""
    cfg_str = MambaConfig(d_model=64, n_layers=4, head_dim=16, attn_layers=["1", 2])
    with pytest.raises(ValueError, match="must be ints"):
        cfg_str.validate()

    cfg_bool = MambaConfig(d_model=64, n_layers=4, head_dim=16, attn_layers=[True, 2])
    with pytest.raises(ValueError, match="must be ints"):
        cfg_bool.validate()


def test_attn_layers_empty_is_pure_mamba():
    """attn_layers=[] yields 0 attention layers (pure Mamba)."""
    cfg = MambaConfig(d_model=64, n_layers=4, head_dim=16, attn_layers=[])
    cfg.validate()
    assert cfg.n_attention_layers == 0
    assert all(not cfg.is_attention_layer(i) for i in range(4))


# ==============================================================================
# 2. d_state Config Validation Bounds Tests
# ==============================================================================

@pytest.mark.parametrize("valid_d_state", [4, 8, 16, 32, 64, 128, 256])
def test_d_state_valid_bounds(valid_d_state):
    """Valid d_state values including 128 and 256 must pass validation."""
    cfg = MambaConfig(
        d_model=64,
        n_layers=2,
        head_dim=16,
        d_state=valid_d_state,
    )
    cfg.validate()
    assert cfg.d_state == valid_d_state


@pytest.mark.parametrize("invalid_bound", [0, -1, -16, 257, 300, 512, 1024])
def test_d_state_invalid_bounds_rejected(invalid_bound):
    """d_state <= 0 or > 256 must be rejected."""
    cfg = MambaConfig(
        d_model=64,
        n_layers=2,
        head_dim=16,
        d_state=invalid_bound,
    )
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize("unaligned_d_state", [3, 5, 7, 9, 13, 15, 17, 33, 65, 127, 129, 250, 255])
def test_d_state_unaligned_rejected(unaligned_d_state):
    """Unaligned d_state sizes must be rejected."""
    cfg = MambaConfig(
        d_model=64,
        n_layers=2,
        head_dim=16,
        d_state=unaligned_d_state,
    )
    with pytest.raises(ValueError, match="must be aligned"):
        cfg.validate()


def test_d_state_scaling_parameter_count():
    """Larger d_state increases x_proj dimensions and total parameters."""
    cfg128 = MambaConfig(d_model=768, n_layers=8, head_dim=64, d_state=128)
    cfg128.validate()

    cfg256 = MambaConfig(d_model=768, n_layers=8, head_dim=64, d_state=256)
    cfg256.validate()

    assert cfg256.num_parameters() > cfg128.num_parameters()
    # Delta per layer in x_proj: d_inner * (2 * (256 - 128)) = 1536 * 256 = 393,216 params/layer
    expected_delta_per_layer = 1536 * (2 * 128)
    actual_delta = (cfg256.parameter_breakdown()["layers"] - cfg128.parameter_breakdown()["layers"]) // 8
    assert actual_delta == expected_delta_per_layer


# ==============================================================================
# 3. Ablation Grid & Harness Tests
# ==============================================================================

def test_compute_attention_layers():
    """Verify compute_attention_layers for 8%, 12%, 16% ratios on 56 layers."""
    l8 = compute_attention_layers(56, 0.08)
    assert len(l8) == 4
    assert l8[-1] == 55  # Ends at top layer (FIM constraint)
    assert len(set(l8)) == 4

    l12 = compute_attention_layers(56, 0.12)
    assert len(l12) == 7
    assert l12[-1] == 55
    assert len(set(l12)) == 7

    l16 = compute_attention_layers(56, 0.16)
    assert len(l16) == 9
    assert l16[-1] == 55
    assert len(set(l16)) == 9


def test_generate_ablation_grid():
    """Grid generation must produce 12 valid candidate configurations."""
    base = default_small_base_config()
    candidates = generate_ablation_grid(base)

    assert len(candidates) == 12

    variants = {c.variant for c in candidates}
    assert variants == {"jamba", "routing_mamba"}

    d_states = {c.d_state for c in candidates}
    assert d_states == {128, 256}

    ratios = {c.attn_ratio for c in candidates}
    assert ratios == {0.08, 0.12, 0.16}

    for c in candidates:
        assert c.name.startswith(c.variant)
        assert c.config.n_layers == 56
        assert c.config.attn_layers is not None
        assert c.config.attn_every is None
        c.config.validate()  # Must all be valid!

        if c.variant == "jamba":
            assert c.config.n_shared_experts == 1
        else:
            assert c.config.n_shared_experts == 0


def test_runner_execution_and_kill_criterion():
    """Verify sweep execution, routing kill-check, and winner selection."""
    runner = AblationSweepRunner(
        attn_ratios=(0.08, 0.12),
        d_states=(128, 256),
        variants=("jamba", "routing_mamba"),
    )
    assert len(runner.candidates) == 8

    # Simulate kill for candidates named 'jamba_attn12_dstate256'
    killed_target = "jamba_attn12_dstate256"

    def _eval(cand):
        return mock_evaluator(cand, seed=123, simulate_kill_for=[killed_target])

    results, winner = runner.run(_eval)

    assert len(results) == 8
    # The killed candidate must be ranked at the end with rank=None
    killed_results = [r for r in results if r.candidate.name == killed_target]
    assert len(killed_results) == 1
    assert killed_results[0].killed is True
    assert killed_results[0].rank is None
    assert killed_results[0].kill_result["triggered"] is True

    # Winner must not be the killed candidate
    assert winner.candidate.name != killed_target
    assert winner.killed is False
    assert winner.rank == 1

    # Formatted table must include winner and killed candidate
    table = format_sweep_table(results, winner)
    assert "WINNER" in table
    assert "TRIGGERED" in table
    assert "PASSED" in table


def test_all_candidates_killed_raises():
    """If all candidates are killed, select_winner must raise RuntimeError."""
    cand = generate_ablation_grid()[0]
    res = AblationResult(
        candidate=cand,
        recall_score=0.8,
        killed=True,
        rank=None,
    )
    with pytest.raises(RuntimeError, match="No candidate survived"):
        select_winner([res])


# ==============================================================================
# 4. Backend Forward Execution Tests
# ==============================================================================

try:
    import mlx.core as mx
    from src.model.mlx_backend import MLXMambaModel
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

try:
    import torch
    from src.model.cuda_backend import CUDAMambaModel
    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False


@pytest.mark.skipif(not HAVE_MLX, reason="MLX not available")
def test_mlx_forward_with_arbitrary_attn_layers():
    """Verify MLXMambaModel forward and step with arbitrary attn_layers."""
    cfg = MambaConfig(
        d_model=32,
        n_layers=4,
        head_dim=8,
        d_state=16,
        vocab_size=64,
        seq_len=16,
        precision="fp32",
        attn_layers=[1, 3],
    )
    cfg.validate()

    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = model.forward(tokens)
    assert logits.shape == (1, 8, 64)

    # Step decode
    state = model.init_state(batch_size=1)
    tok = mx.array([1])
    next_logits, next_state = model.step(tok, state)
    assert next_logits.shape == (1, 64)


@pytest.mark.skipif(not HAVE_CUDA, reason="PyTorch/CUDA backend not available")
def test_cuda_cpu_forward_with_arbitrary_attn_layers():
    """Verify CUDAMambaModel forward and step with arbitrary attn_layers."""
    cfg = MambaConfig(
        d_model=32,
        n_layers=4,
        head_dim=8,
        d_state=16,
        vocab_size=64,
        seq_len=16,
        precision="fp32",
        attn_layers=[1, 3],
    )
    cfg.validate()

    torch.manual_seed(0)
    model = CUDAMambaModel(cfg, device=torch.device("cpu"))
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    logits = model.forward(tokens)
    assert logits.shape == (1, 8, 64)

    # Step decode
    state = model.init_state(batch_size=1)
    tok = torch.tensor([1], dtype=torch.long)
    next_logits, next_state = model.step(tok, state)
    assert next_logits.shape == (1, 64)
