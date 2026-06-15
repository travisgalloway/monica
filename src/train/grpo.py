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

from typing import Tuple

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


def grpo_loss_from_logprobs(logp: np.ndarray, advantages: np.ndarray,
                            ) -> Tuple[float, float]:
    """GRPO policy-gradient loss `-mean(advantage * logp)` + the mean |advantage| diagnostic.

    `logp` and `advantages` are the same shape (per-sample sequence log-prob and its
    group-standardized advantage). Returns `(loss, mean_abs_advantage)`.
    """
    logp = np.asarray(logp, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    loss = float(-np.mean(adv * logp))
    return loss, float(np.mean(np.abs(adv)))


def reward_stats(rewards: np.ndarray) -> dict:
    """Run diagnostics for logging: mean reward and fraction *fully solved* (reward >= 1.0,
    i.e. a perfect score — not merely the group max, so partial-credit groups with no
    perfect solution report 0, not 1). Empty input yields zeros, not NaN."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return {"mean_reward": 0.0, "frac_solved": 0.0}
    return {"mean_reward": float(r.mean()), "frac_solved": float(np.mean(r >= 1.0))}
