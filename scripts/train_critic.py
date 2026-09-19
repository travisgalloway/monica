#!/usr/bin/env python3
"""Training and Calibration CLI for Auxiliary Decision Critic Head (#387).

Trains the lightweight MLP head parameters using Brier score loss:
  L_{Brier} = (1/N) * sum_{i=1}^N (p_i - y_i)^2
Fits scalar temperature T on held-out validation activations using negative log-likelihood minimization,
and evaluates Expected Calibration Error (ECE), Brier gain over base rate, and AUC-ROC on held-out test splits.

Outputs metrics to results/critic_training_metrics.json.

Usage:
  .venv/bin/python scripts/train_critic.py \
      --features data/critic_features.npz \
      --metrics-out results/critic_training_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model.critic import CriticConfig, DecisionCriticHead
from src.train.critic_train import (
    save_critic_head,
    train_and_calibrate_critic,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train and calibrate auxiliary DecisionCriticHead on cached hidden activations"
    )
    parser.add_argument("--features", type=Path, default=Path("data/critic_features.npz"),
                        help="Path to cached activations .npz file (default: data/critic_features.npz)")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/critic"),
                        help="Directory to save trained head checkpoint (default: runs/critic)")
    parser.add_argument("--metrics-out", type=Path, default=Path("results/critic_training_metrics.json"),
                        help="Path to output JSON metrics (default: results/critic_training_metrics.json)")
    parser.add_argument("--primitive", default="noul", choices=("noul", "score"),
                        help="Decision primitive to train (default: noul)")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Critic MLP hidden dimension (default: d_model // 4)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs (default: 150)")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Learning rate for Adam optimizer (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size (default: 64)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of dataset for validation temperature fitting (default: 0.15)")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Fraction of dataset for held-out test evaluation (default: 0.15)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="L2 regularization on weights (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 72)
    print("TRAINING AUXILIARY DECISION CRITIC HEAD (BRIER SCORE & TEMPERATURE CALIBRATION)")
    print("=" * 72)

    if not args.features.exists():
        print(f"Features file not found at: {args.features}", file=sys.stderr)
        print("Please run scripts/cache_critic_features.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading cached activations from: {args.features}")
    data = np.load(args.features, allow_pickle=True)
    X = data["features"].astype(np.float32)
    if args.primitive == "noul":
        y = data["labels_noul"].astype(np.float32)
    else:
        y = data["labels_score"].astype(np.int64)

    N, d_model = X.shape
    print(f"Loaded {N} samples with d_model={d_model}")
    print(f"Positive / clean rate: {np.mean(y):.2%}")

    # Stratified Split into Train / Val / Test
    rng = np.random.default_rng(args.seed)
    pos_idx = np.where(y == 1)[0] if args.primitive == "noul" else np.arange(N)
    neg_idx = np.where(y == 0)[0] if args.primitive == "noul" else np.array([], dtype=int)

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_val_pos = max(1, int(len(pos_idx) * args.val_split))
    n_test_pos = max(1, int(len(pos_idx) * args.test_split))
    n_train_pos = len(pos_idx) - n_val_pos - n_test_pos

    n_val_neg = max(1, int(len(neg_idx) * args.val_split)) if len(neg_idx) > 0 else 0
    n_test_neg = max(1, int(len(neg_idx) * args.test_split)) if len(neg_idx) > 0 else 0
    n_train_neg = len(neg_idx) - n_val_neg - n_test_neg

    train_idx = np.concatenate([pos_idx[:n_train_pos], neg_idx[:n_train_neg]])
    val_idx = np.concatenate([pos_idx[n_train_pos : n_train_pos + n_val_pos], neg_idx[n_train_neg : n_train_neg + n_val_neg]])
    test_idx = np.concatenate([pos_idx[n_train_pos + n_val_pos :], neg_idx[n_train_neg + n_val_neg :]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Data Splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Initialize DecisionCriticHead
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else max(16, d_model // 4)
    cfg = CriticConfig(
        d_model=d_model,
        hidden_dim=hidden_dim,
        primitive=args.primitive,
        n_classes=1 if args.primitive == "noul" else 4,
    )
    head = DecisionCriticHead(cfg, rng=rng)
    print(f"Critic Head Parameters: {head.param_count:,} (d_model={d_model}, hidden_dim={hidden_dim})")

    # Run Optimization Loop and Calibration
    print("\nStarting Brier score training loop...")
    res = train_and_calibrate_critic(
        head=head,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        rng_seed=args.seed,
    )

    t_elapsed = time.time() - t_start

    # Acceptance Criteria Verification
    gate_ece = res.calibrated_ece < 0.08
    gate_brier = res.brier_improvement_pct >= 15.0
    gate_time = t_elapsed < 300.0  # <5 minutes

    print("\n" + "=" * 72)
    print(res.summary)
    print(f"  - Training Elapsed Time: {t_elapsed:.2f}s (<5 minutes: {gate_time})")
    print(f"  - Gate ECE (< 0.08): {'PASS' if gate_ece else 'FAIL'} ({res.calibrated_ece:.4f})")
    print(f"  - Gate Brier Gain (>= 15%): {'PASS' if gate_brier else 'FAIL'} ({res.brier_improvement_pct:.1f}%)")
    print("=" * 72)

    # Save metrics JSON
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_config": {
            "d_model": d_model,
            "hidden_dim": hidden_dim,
            "primitive": args.primitive,
            "parameter_count": head.param_count,
        },
        "training": {
            "epochs": args.epochs,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "elapsed_seconds": round(t_elapsed, 2),
        },
        "calibration": {
            "optimal_temperature": round(res.temperature, 4),
            "raw_ece": round(res.raw_ece, 4),
            "calibrated_ece": round(res.calibrated_ece, 4),
            "ece_gate_passed": bool(gate_ece),
        },
        "scoring": {
            "brier_score": round(res.brier_score, 4),
            "base_rate_brier_score": round(res.base_rate_brier_score, 4),
            "brier_improvement_pct": round(res.brier_improvement_pct, 1),
            "brier_gate_passed": bool(gate_brier),
            "auc_roc": round(res.auc_roc, 4),
        },
        "calibration_curve": res.calibration_curve,
    }
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {args.metrics_out}")

    # Save trained head weights
    args.out_dir.mkdir(parents=True, exist_ok=True)
    head_ckpt_path = args.out_dir / "critic_head.npz"
    save_critic_head(head, str(head_ckpt_path))
    print(f"Saved trained critic head to: {head_ckpt_path}")

    all_passed = gate_ece and gate_brier and gate_time
    verdict = "GO" if all_passed else "NO_GO"
    print(f"\nFINAL VERDICT: {verdict}")
    print("=" * 72)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
