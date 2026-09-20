"""GRPO math (portable numpy — above the seam, no backend).

Group Relative Policy Optimization (Shao et al.) replaces PPO's learned value baseline with
a **group baseline**: sample K completions per prompt, score each with a verifier
(`train/verifiers.py`), and standardize the rewards *within the group* to advantages. The
policy-gradient objective is then `-mean(advantage * logp)` — REINFORCE with the group
mean/std baseline, no critic. The numeric core lives here so it is unit-testable anywhere;
the MLX GRPO step (`src/model/mlx_train_step.py`) mirrors the same loss on the autodiff
graph (advantages precomputed here, in the driver).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def group_advantages(rewards: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardize rewards within each group to advantages.

    `rewards` (n_groups, K) -> advantages (n_groups, K) = (r - mean_g) / (std_g + eps). A
    group whose rewards are all equal (e.g. all-correct or all-wrong) yields ~0 advantage,
    so it contributes no gradient — exactly the GRPO degenerate case.
    """
    r = np.asarray(rewards, dtype=np.float64)
    mean = r.mean(axis=-1, keepdims=True)
    std = r.std(axis=-1, keepdims=True)
    return (r - mean) / (std + eps)


def compute_kl_penalty(logp: np.ndarray, ref_logp: np.ndarray,
                       estimator: str = "schulman") -> np.ndarray:
    """Compute per-sequence KL penalty D_KL(policy || ref).

    Estimators:
    - 'schulman' (default; DeepSeek-R1 / Open-R1 unbiased non-negative estimator):
        diff = ref_logp - logp
        kl = exp(diff) - diff - 1
    - 'log_ratio' / 'k1':
        kl = logp - ref_logp
    """
    lp = np.asarray(logp, dtype=np.float64)
    ref_lp = np.asarray(ref_logp, dtype=np.float64)
    diff = ref_lp - lp
    if estimator == "schulman":
        # exp(ref - pol) - (ref - pol) - 1 >= 0
        clipped = np.clip(diff, -50.0, 50.0)
        return np.exp(clipped) - diff - 1.0
    elif estimator in ("log_ratio", "k1"):
        return -diff
    else:
        raise ValueError(f"unknown KL estimator {estimator!r}; expected 'schulman' or 'log_ratio'")


def grpo_loss_from_logprobs(logp: np.ndarray, advantages: np.ndarray,
                            ref_logp: Optional[np.ndarray] = None,
                            beta: float = 0.0,
                            *,
                            kl_estimator: str = "schulman") -> Tuple[float, float]:
    """GRPO policy-gradient loss `-mean(advantage * logp) + beta * kl` + the mean |advantage| diagnostic.

    `logp` and `advantages` are the same shape (per-sample sequence log-prob and its
    group-standardized advantage). If `ref_logp` is provided and `beta > 0.0`, regularizes policy
    drift away from the reference model via KL penalty. Returns `(loss, mean_abs_advantage)`.
    """
    logp = np.asarray(logp, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    pg_loss = float(-np.mean(adv * logp))
    if ref_logp is not None and beta > 0.0:
        kl = compute_kl_penalty(logp, ref_logp, estimator=kl_estimator)
        loss = pg_loss + float(beta * np.mean(kl))
    else:
        loss = pg_loss
    return loss, float(np.mean(np.abs(adv)))


def advantage_stats(advantages: np.ndarray) -> dict:
    """Group advantage diagnostics for logging and stability monitoring.

    Reports mean advantage (centered near 0), mean absolute advantage,
    standard deviation, and min/max bounds.
    """
    adv = np.asarray(advantages, dtype=np.float64)
    if adv.size == 0:
        return {"mean_adv": 0.0, "mean_abs_adv": 0.0, "std_adv": 0.0, "min_adv": 0.0, "max_adv": 0.0}
    return {
        "mean_adv": float(adv.mean()),
        "mean_abs_adv": float(np.mean(np.abs(adv))),
        "std_adv": float(adv.std()),
        "min_adv": float(adv.min()),
        "max_adv": float(adv.max()),
    }


def kl_stats(logp: np.ndarray, ref_logp: np.ndarray,
             estimator: str = "schulman") -> dict:
    """KL penalty diagnostics for monitoring policy drift against the reference model."""
    kl = compute_kl_penalty(logp, ref_logp, estimator=estimator)
    if kl.size == 0:
        return {"mean_kl": 0.0, "max_kl": 0.0, "min_kl": 0.0}
    return {
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "min_kl": float(kl.min()),
    }


def reward_stats(rewards: np.ndarray) -> dict:
    """Run diagnostics for logging: mean reward and fraction *fully solved* (reward >= 1.0,
    i.e. a perfect score — not merely the group max, so partial-credit groups with no
    perfect solution report 0, not 1). Empty input yields zeros, not NaN."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return {"mean_reward": 0.0, "frac_solved": 0.0}
    return {"mean_reward": float(r.mean()), "frac_solved": float(np.mean(r >= 1.0))}
