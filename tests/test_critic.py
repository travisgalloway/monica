"""Tests for the auxiliary decision critic head and Go/No-Go evaluation gates.

Validates:
  1. Shape, configuration, and parameter budgeting for `noul`, `score`, and `choice` primitives.
  2. Mathematical correctness of forward passes (RMSNorm + MLP + GELU).
  3. Proper scoring rules: Brier score strictly penalizes overconfident wrong predictions.
  4. Calibration mechanics: Expected Calibration Error (ECE) calculation and temperature scaling.
  5. Go/No-Go Gate: Full architectural gate assessment for inclusion in Monica (latency,
     parameter overhead, calibration repair, discrimination gain, and seam invariance).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.critic import (
    CriticConfig,
    DecisionCriticHead,
    brier_score,
    expected_calibration_error,
    fit_temperature_scaling,
    evaluate_critic_gate,
)


def test_critic_config_and_param_budget():
    """Verify head parameter footprint at Monica's small/large d_model=768."""
    d_model = 768
    cfg = CriticConfig(d_model=d_model, hidden_dim=192, primitive="noul")
    head = DecisionCriticHead(cfg)

    # w1: 768 * 192 = 147,456
    # b1: 192
    # w2: 192 * 1 = 192
    # b2: 1
    # temperature: 1
    # total = 147,842 params
    assert head.param_count == (768 * 192 + 192 + 192 * 1 + 1 + 1)
    # M12-small is ~120M active params. Head must be < 0.2% of active parameters.
    overhead_pct = (head.param_count / 120_000_000) * 100.0
    assert overhead_pct < 0.2


def test_noul_primitive_prediction():
    """Test boolean decision primitive ('noul')."""
    cfg = CriticConfig(d_model=64, hidden_dim=16, primitive="noul")
    head = DecisionCriticHead(cfg, rng=np.random.default_rng(0))

    x = np.random.randn(64).astype(np.float32)
    res = head.predict_noul(x)

    assert "prob" in res
    assert "decision" in res
    assert "confidence" in res
    assert 0.0 <= res["prob"] <= 1.0
    assert isinstance(res["decision"], bool)
    assert 0.5 <= res["confidence"] <= 1.0

    # Batched input
    x_batch = np.random.randn(4, 64).astype(np.float32)
    logits = head.forward_logits(x_batch)
    assert logits.shape == (4, 1)


def test_score_primitive_ordinal_rubric():
    """Test ordinal score primitive ('score') for bug severity or PRM grading."""
    cfg = CriticConfig(d_model=64, hidden_dim=16, primitive="score", n_classes=4)
    head = DecisionCriticHead(cfg, rng=np.random.default_rng(1))

    x = np.random.randn(64).astype(np.float32)
    res = head.predict_score(x, rubric_values=[0.0, 1.0, 2.0, 3.0])

    assert "score" in res
    assert "best_level" in res
    assert "probs" in res
    assert len(res["probs"]) == 4
    assert 0.0 <= res["score"] <= 3.0
    assert 0 <= res["best_level"] < 4
    assert np.isclose(sum(res["probs"]), 1.0, atol=1e-5)


def test_choice_primitive_categorical():
    """Test categorical choice primitive ('choice') over named candidates."""
    choice_names = ["infra", "billing", "security", "general"]
    cfg = CriticConfig(
        d_model=64, hidden_dim=16, primitive="choice", n_classes=4, choice_names=choice_names
    )
    head = DecisionCriticHead(cfg, rng=np.random.default_rng(2))

    x = np.random.randn(64).astype(np.float32)
    res = head.predict_choice(x)

    assert res["choice"] in choice_names
    assert 0 <= res["choice_idx"] < 4
    assert 0.0 <= res["confidence"] <= 1.0
    assert np.isclose(sum(res["probs"]), 1.0, atol=1e-5)


def test_brier_score_properties():
    """Proper scoring rules: Brier score strictly penalizes overconfident wrong predictions."""
    targets = [1, 0, 1, 0]

    # Perfect predictions
    perfect_probs = [1.0, 0.0, 1.0, 0.0]
    assert brier_score(perfect_probs, targets) == 0.0

    # Uninformative / base rate predictions
    uninformative_probs = [0.5, 0.5, 0.5, 0.5]
    bs_uninformative = brier_score(uninformative_probs, targets)
    assert np.isclose(bs_uninformative, 0.25)

    # Confident wrong predictions (severely penalized)
    wrong_probs = [0.0, 1.0, 0.0, 1.0]
    bs_wrong = brier_score(wrong_probs, targets)
    assert bs_wrong == 1.0

    assert bs_wrong > bs_uninformative > 0.0


def test_ece_and_temperature_calibration():
    """Test Expected Calibration Error (ECE) and temperature scaling repair."""
    rng = np.random.default_rng(42)
    N = 1000

    # Generate synthetic true probabilities and outcomes
    true_probs = rng.uniform(0.1, 0.9, size=N)
    labels = (rng.uniform(size=N) < true_probs).astype(np.int64)

    # Distort logits to be sharply overconfident (ECE inflation)
    true_logits = np.log(true_probs / (1.0 - true_probs))
    overconfident_logits = true_logits * 3.5  # sharp overconfidence
    overconfident_probs = 1.0 / (1.0 + np.exp(-overconfident_logits))

    raw_ece = expected_calibration_error(overconfident_probs, labels, n_bins=10)
    assert raw_ece > 0.15, f"Expected uncalibrated ECE > 0.15, got {raw_ece}"

    # Temperature scaling optimization
    opt_t = fit_temperature_scaling(overconfident_logits, labels)
    # The temperature should soften overconfident logits (T > 1.0)
    assert opt_t > 1.5, f"Expected optimal T > 1.5, got {opt_t}"

    calibrated_probs = 1.0 / (1.0 + np.exp(-overconfident_logits / opt_t))
    calibrated_ece = expected_calibration_error(calibrated_probs, labels, n_bins=10)

    # Calibration error must drop substantially
    assert calibrated_ece < 0.08, f"Expected calibrated ECE < 0.08, got {calibrated_ece}"
    assert calibrated_ece < raw_ece / 2.0


def test_go_no_go_architectural_gate():
    """Evaluate the 5 architectural gates for inclusion in Monica."""
    result = evaluate_critic_gate(
        d_model=768,
        backbone_active_params=120_000_000,
        n_samples=500,
        rng_seed=42,
    )

    # All gates must be green
    assert result.verdict == "GO", f"Evaluation returned NO_GO:\n{result.summary}"
    assert result.parameter_overhead_pct < 0.5
    assert result.calibrated_ece < 0.08
    assert result.brier_improvement_pct >= 15.0
    assert result.gate_details["gate_seam_invariance (pure numpy)"] is True


try:
    import mlx.core  # noqa: F401
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def test_mamba_config_critic_heads_validation():
    """Verify MambaConfig schema validation for auxiliary critic heads (#386)."""
    from src.model.blocks import MambaConfig

    # 1. Valid configuration with CriticConfig object and dict
    cfg = MambaConfig(
        d_model=64,
        n_layers=2,
        critic_heads={
            "noul_gate": CriticConfig(d_model=64, hidden_dim=16, primitive="noul"),
            "score_rubric": {"primitive": "score", "hidden_dim": 16, "n_classes": 4},
        },
    )
    cfg.validate()
    assert isinstance(cfg.critic_heads["score_rubric"], CriticConfig)
    assert cfg.critic_heads["score_rubric"].d_model == 64
    bd = cfg.parameter_breakdown()
    assert "critic_heads" in bd
    assert bd["critic_heads"] > 0

    # 2. Rejection of hidden_dim that does not divide d_model cleanly
    with pytest.raises(ValueError, match="hidden_dim=25 must divide d_model=64 cleanly"):
        bad_cfg = MambaConfig(
            d_model=64,
            n_layers=2,
            critic_heads={"bad_dim": CriticConfig(d_model=64, hidden_dim=25, primitive="noul")},
        )
        bad_cfg.validate()

    # 3. Rejection of invalid primitive
    with pytest.raises(ValueError, match="primitive 'unsupported' must be one of"):
        bad_cfg = MambaConfig(
            d_model=64,
            n_layers=2,
            critic_heads={"bad_prim": CriticConfig(d_model=64, hidden_dim=16, primitive="unsupported")},
        )
        bad_cfg.validate()

    # 4. Rejection of d_model mismatch
    with pytest.raises(ValueError, match="d_model=128 must match model d_model=64"):
        bad_cfg = MambaConfig(
            d_model=64,
            n_layers=2,
            critic_heads={"bad_dmodel": CriticConfig(d_model=128, hidden_dim=16, primitive="noul")},
        )
        bad_cfg.validate()


@pytest.mark.skipif(not HAVE_MLX, reason="needs mlx")
def test_mlx_critic_head_matches_portable_reference():
    """Verify MLX DecisionCriticHead execution matches portable critic.py reference (#386)."""
    from src.model.mlx_backend import DecisionCriticHead as MLXCriticHead

    for prim, n_classes in [("noul", 1), ("score", 4), ("choice", 3)]:
        cfg = CriticConfig(d_model=64, hidden_dim=16, primitive=prim, n_classes=n_classes, temperature=1.2)
        ref = DecisionCriticHead(cfg, rng=np.random.default_rng(42))
        mlx_h = MLXCriticHead(cfg)
        mlx_h.copy_from_reference(ref)

        x = np.random.default_rng(0).normal(size=(5, 64)).astype(np.float32)
        y_ref = ref.forward_logits(x)
        y_mlx = np.array(mlx_h.forward_logits(x))
        assert np.allclose(y_ref, y_mlx, rtol=1e-4, atol=1e-5)

        # Single vector predictions
        x_single = x[0]
        if prim == "noul":
            res_ref = ref.predict_noul(x_single)
            res_mlx = mlx_h.predict_noul(x_single)
            assert np.isclose(res_ref["prob"], res_mlx["prob"], atol=1e-5)
            assert res_ref["decision"] == res_mlx["decision"]
        elif prim == "score":
            res_ref = ref.predict_score(x_single)
            res_mlx = mlx_h.predict_score(x_single)
            assert np.isclose(res_ref["score"], res_mlx["score"], atol=1e-5)
            assert np.allclose(res_ref["probs"], res_mlx["probs"], atol=1e-5)
        elif prim == "choice":
            res_ref = ref.predict_choice(x_single)
            res_mlx = mlx_h.predict_choice(x_single)
            assert res_ref["choice"] == res_mlx["choice"]
            assert np.allclose(res_ref["probs"], res_mlx["probs"], atol=1e-5)


@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_cuda_critic_head_matches_portable_reference():
    """Verify CUDA/PyTorch DecisionCriticHead execution matches portable critic.py reference (#386)."""
    from src.model.cuda_backend import DecisionCriticHead as TorchCriticHead

    for prim, n_classes in [("noul", 1), ("score", 4), ("choice", 3)]:
        cfg = CriticConfig(d_model=64, hidden_dim=16, primitive=prim, n_classes=n_classes, temperature=1.2)
        ref = DecisionCriticHead(cfg, rng=np.random.default_rng(42))
        th_h = TorchCriticHead(cfg)
        th_h.copy_from_reference(ref)

        x = np.random.default_rng(0).normal(size=(5, 64)).astype(np.float32)
        y_ref = ref.forward_logits(x)
        with torch.no_grad():
            y_th = th_h.forward_logits(torch.from_numpy(x)).detach().cpu().numpy()
        assert np.allclose(y_ref, y_th, rtol=1e-4, atol=1e-5)

        # Single vector predictions
        x_single = x[0]
        if prim == "noul":
            res_ref = ref.predict_noul(x_single)
            res_th = th_h.predict_noul(torch.from_numpy(x_single))
            assert np.isclose(res_ref["prob"], res_th["prob"], atol=1e-5)
            assert res_ref["decision"] == res_th["decision"]
        elif prim == "score":
            res_ref = ref.predict_score(x_single)
            res_th = th_h.predict_score(torch.from_numpy(x_single))
            assert np.isclose(res_ref["score"], res_th["score"], atol=1e-5)
            assert np.allclose(res_ref["probs"], res_th["probs"], atol=1e-5)
        elif prim == "choice":
            res_ref = ref.predict_choice(x_single)
            res_th = th_h.predict_choice(torch.from_numpy(x_single))
            assert res_ref["choice"] == res_th["choice"]
            assert np.allclose(res_ref["probs"], res_th["probs"], atol=1e-5)
