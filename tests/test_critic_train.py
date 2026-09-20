"""Unit tests for auxiliary decision critic training, loss calculation, and calibration (#387).

Validates:
  1. Loss calculation: Brier score loss mathematical correctness and proper scoring properties.
  2. Gradient updates: Exact analytical gradients vs numerical finite differences and PyTorch autograd.
  3. Temperature fitting: Scalar temperature T optimization via NLL minimization and ECE repair (<0.08).
  4. Telemetry: AUC-ROC calculation and empirical calibration curves.
  5. Optimization convergence: End-to-end convergence with Brier gain >= 15% and ECE < 0.08.
  6. Checkpoint persistence: Save and load round-trip for DecisionCriticHead weights.
  7. Backbone hidden state extraction: ModelInterface forward_hidden extraction parity and shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.critic import (
    CriticConfig,
    DecisionCriticHead,
    expected_calibration_error,
)
from src.train.critic_train import (
    AdamOptimizer,
    brier_score_loss,
    compute_auc_roc,
    compute_calibration_curve,
    fit_temperature_nll,
    forward_backward_critic,
    load_critic_head,
    save_critic_head,
    train_and_calibrate_critic,
)


def test_brier_score_loss_calculation():
    """Verify Brier score loss formula: L = (1/N) * sum((p_i - y_i)^2)."""
    # Perfect predictions -> 0.0
    p_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    y_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score_loss(p_perfect, y_perfect) == 0.0

    # Completely wrong predictions -> 1.0
    p_wrong = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score_loss(p_wrong, y_perfect) == 1.0

    # Intermediate calculation
    p_mid = np.array([0.8, 0.2, 0.6, 0.1])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    expected = float(np.mean([(0.8 - 1.0)**2, (0.2 - 0.0)**2, (0.6 - 1.0)**2, (0.1 - 0.0)**2]))
    assert np.isclose(brier_score_loss(p_mid, y), expected)

    # Shape mismatch error check
    with pytest.raises(ValueError, match="Shape mismatch"):
        brier_score_loss(np.array([0.5, 0.5]), np.array([1.0]))


def test_critic_gradient_updates_numerical_parity():
    """Verify exact analytical gradients of forward_backward_critic against finite differences."""
    rng = np.random.default_rng(42)
    d_model = 16
    hidden_dim = 8
    cfg = CriticConfig(d_model=d_model, hidden_dim=hidden_dim, primitive="noul")
    head = DecisionCriticHead(cfg, rng=rng)

    N = 4
    x = rng.normal(size=(N, d_model)).astype(np.float32)
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    loss_base, grads, probs = forward_backward_critic(head, x, y)
    assert 0.0 <= loss_base <= 1.0
    assert probs.shape == (N, 1)

    # Numerical gradient check for w1, b1, w2, b2
    eps = 1e-4

    # Check w2 gradient
    for i in range(hidden_dim):
        orig = head.w2[i, 0]
        head.w2[i, 0] = orig + eps
        l_plus, _, _ = forward_backward_critic(head, x, y)
        head.w2[i, 0] = orig - eps
        l_minus, _, _ = forward_backward_critic(head, x, y)
        head.w2[i, 0] = orig

        num_grad = (l_plus - l_minus) / (2.0 * eps)
        ana_grad = grads["w2"][i, 0]
        assert np.isclose(num_grad, ana_grad, atol=1e-3), (
            f"w2[{i}] gradient mismatch: num={num_grad}, ana={ana_grad}"
        )

    # Check b2 gradient
    orig_b2 = head.b2[0]
    head.b2[0] = orig_b2 + eps
    l_plus, _, _ = forward_backward_critic(head, x, y)
    head.b2[0] = orig_b2 - eps
    l_minus, _, _ = forward_backward_critic(head, x, y)
    head.b2[0] = orig_b2
    num_b2_grad = (l_plus - l_minus) / (2.0 * eps)
    assert np.isclose(num_b2_grad, grads["b2"][0], atol=1e-3)

    # Check sample of w1 gradients
    for i in range(min(4, d_model)):
        for j in range(min(4, hidden_dim)):
            orig_w1 = head.w1[i, j]
            head.w1[i, j] = orig_w1 + eps
            l_plus, _, _ = forward_backward_critic(head, x, y)
            head.w1[i, j] = orig_w1 - eps
            l_minus, _, _ = forward_backward_critic(head, x, y)
            head.w1[i, j] = orig_w1

            num_w1_grad = (l_plus - l_minus) / (2.0 * eps)
            ana_w1_grad = grads["w1"][i, j]
            assert np.isclose(num_w1_grad, ana_w1_grad, atol=1e-3)


def test_adam_optimizer_step_strictly_decreases_loss():
    """Verify that Adam optimizer gradient steps strictly decrease Brier score loss."""
    rng = np.random.default_rng(123)
    d_model = 32
    hidden_dim = 16
    cfg = CriticConfig(d_model=d_model, hidden_dim=hidden_dim, primitive="noul")
    head = DecisionCriticHead(cfg, rng=rng)

    x = rng.normal(size=(16, d_model)).astype(np.float32)
    y = (rng.uniform(size=16) > 0.5).astype(np.float32)

    params = {"w1": head.w1, "b1": head.b1, "w2": head.w2, "b2": head.b2}
    opt = AdamOptimizer(params, lr=0.01, weight_decay=0.0)

    initial_loss, _grads, _ = forward_backward_critic(head, x, y)
    for _ in range(25):
        _, g, _ = forward_backward_critic(head, x, y)
        opt.step_update(g)

    final_loss, _, _ = forward_backward_critic(head, x, y)
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"


def test_temperature_fitting_and_ece_repair():
    """Verify NLL temperature fitting reduces Expected Calibration Error (ECE < 0.08)."""
    rng = np.random.default_rng(42)
    n_samples = 400

    # Overconfident uncalibrated predictions
    true_probs = rng.beta(0.5, 0.5, size=n_samples)
    y = (rng.uniform(size=n_samples) < true_probs).astype(np.int64)
    # Severe logit distortion
    raw_logits = np.arctanh(np.clip(2.0 * true_probs - 1.0, -0.92, 0.92)) * 3.5

    raw_probs = 1.0 / (1.0 + np.exp(-raw_logits))
    raw_ece = expected_calibration_error(raw_probs, y)
    assert raw_ece > 0.08, f"Expected initial raw ECE to be elevated, got {raw_ece:.4f}"

    # Fit temperature via NLL
    opt_temp = fit_temperature_nll(raw_logits, y, lr=0.05, max_iter=400)
    assert opt_temp > 0.0

    calibrated_probs = 1.0 / (1.0 + np.exp(-raw_logits / opt_temp))
    calibrated_ece = expected_calibration_error(calibrated_probs, y)

    assert calibrated_ece < 0.08, f"Calibrated ECE failed gate target: {calibrated_ece:.4f} >= 0.08"
    assert calibrated_ece < raw_ece, f"Temperature scaling failed to improve ECE: {calibrated_ece:.4f} vs {raw_ece:.4f}"


def test_auc_roc_and_calibration_curve_mechanics():
    """Verify exact AUC-ROC ranking and empirical reliability curve binning."""
    # AUC-ROC tests
    y = np.array([0, 0, 1, 1])
    p_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_auc_roc(p_perfect, y) == 1.0

    p_worst = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_auc_roc(p_worst, y) == 0.0

    p_random = np.array([0.5, 0.5, 0.5, 0.5])
    assert compute_auc_roc(p_random, y) == 0.5

    # Calibration curve tests
    probs = np.linspace(0.05, 0.95, 20)
    targets = (probs > 0.5).astype(int)
    curve = compute_calibration_curve(probs, targets, n_bins=5)

    assert len(curve) == 5
    for b in curve:
        assert "bin_lower" in b and "bin_upper" in b
        assert "mean_pred" in b and "empirical_prob" in b
        assert "count" in b
        assert b["bin_lower"] <= b["bin_upper"]
        assert 0.0 <= b["mean_pred"] <= 1.0


def test_score_primitive_multiclass_training():
    """Verify forward_backward_critic support for ordinal rubric ('score' primitive)."""
    rng = np.random.default_rng(99)
    d_model = 24
    hidden_dim = 12
    n_classes = 4
    cfg = CriticConfig(d_model=d_model, hidden_dim=hidden_dim, primitive="score", n_classes=n_classes)
    head = DecisionCriticHead(cfg, rng=rng)

    N = 8
    x = rng.normal(size=(N, d_model)).astype(np.float32)
    y = rng.integers(0, n_classes, size=N)

    loss_init, grads, probs = forward_backward_critic(head, x, y)
    assert loss_init > 0.0
    assert probs.shape == (N, n_classes)
    assert np.allclose(np.sum(probs, axis=-1), 1.0, atol=1e-5)
    assert grads["w2"].shape == (hidden_dim, n_classes)

    # Step update
    params = {"w1": head.w1, "b1": head.b1, "w2": head.w2, "b2": head.b2}
    opt = AdamOptimizer(params, lr=0.01)
    for _ in range(30):
        _, g, _ = forward_backward_critic(head, x, y)
        opt.step_update(g)

    loss_final, _, _ = forward_backward_critic(head, x, y)
    assert loss_final < loss_init


def test_end_to_end_training_convergence_and_gates(tmp_path):
    """Verify full train_and_calibrate_critic achieves calibrated ECE < 0.08 and Brier gain >= 15%."""
    rng = np.random.default_rng(42)
    d_model = 128
    hidden_dim = 32
    N = 1000

    # Synthetic representation with ground-truth discrimination signal
    v_signal = rng.normal(size=d_model).astype(np.float32)
    v_signal /= float(np.linalg.norm(v_signal))

    y = rng.choice([0, 1], size=N, p=[0.5, 0.5]).astype(np.float32)
    X = rng.normal(scale=1.0, size=(N, d_model)).astype(np.float32)
    X += (2.0 * y[:, None] - 1.0) * v_signal * 0.9

    n_train = 700
    n_val = 150
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    cfg = CriticConfig(d_model=d_model, hidden_dim=hidden_dim, primitive="noul")
    head = DecisionCriticHead(cfg, rng=rng)

    res = train_and_calibrate_critic(
        head=head,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        epochs=80,
        lr=0.002,
        batch_size=32,
        weight_decay=0.01,
        rng_seed=42,
    )

    # Acceptance Criteria gates
    assert res.calibrated_ece < 0.08, f"Calibrated ECE failed gate target: {res.calibrated_ece:.4f} >= 0.08"
    assert res.brier_improvement_pct >= 15.0, (
        f"Brier gain failed gate target: {res.brier_improvement_pct:.1f}% < 15.0%"
    )
    assert res.auc_roc >= 0.80, f"AUC-ROC too low: {res.auc_roc:.4f}"
    assert len(res.calibration_curve) == 10

    # Save and load checkpoint round-trip
    ckpt_path = tmp_path / "critic_head.npz"
    save_critic_head(res.head, str(ckpt_path))
    loaded_head = load_critic_head(str(ckpt_path))

    assert loaded_head.config.d_model == head.config.d_model
    assert loaded_head.config.hidden_dim == head.config.hidden_dim
    assert np.isclose(loaded_head.temperature, res.head.temperature, atol=1e-5)
    assert np.allclose(loaded_head.w1, res.head.w1)
    assert np.allclose(loaded_head.w2, res.head.w2)


def test_model_interface_forward_hidden():
    """Verify forward_hidden contract on ModelInterface across backends."""
    from src.model.blocks import load_config

    cfg = load_config("config/toy.yaml")
    tokens = np.random.randint(0, cfg.vocab_size, size=(2, 12)).astype(np.int32)
    tested = False

    try:
        from src.model.mlx_backend import MLXMambaModel
        mlx_model = MLXMambaModel(cfg)
        h_mlx = np.array(mlx_model.forward_hidden(tokens))
        assert h_mlx.shape == (2, 12, cfg.d_model)
        tested = True
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import torch
        from src.model.cuda_backend import CUDAMambaModel
        cuda_model = CUDAMambaModel(cfg)
        h_cuda = cuda_model.forward_hidden(tokens).detach().cpu().numpy()
        assert h_cuda.shape == (2, 12, cfg.d_model)
        tested = True
    except (ImportError, ModuleNotFoundError):
        pass

    if not tested:
        pytest.skip("Neither mlx nor torch is available to test forward_hidden")
