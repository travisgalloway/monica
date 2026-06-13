"""DPO loss math (portable numpy — above the seam, no backend).

Direct Preference Optimization (Rafailov et al.) trains a policy to prefer the chosen
response over the rejected one, regularized toward a frozen reference by a KL term that
collapses into a simple log-ratio. The numeric core lives here so it is unit-testable
anywhere; the MLX DPO step (`src/model/mlx_train_step.py`) mirrors the same formula on
the autodiff graph and calls these only for the (no-grad) reward metrics.

`masked_sequence_logprob` is the shared primitive: the summed log-prob of a response's
tokens (mask = 1 on response positions). Sequence log-probs from policy and reference,
for chosen and rejected, feed `dpo_loss_from_logprobs`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def masked_sequence_logprob(logits: np.ndarray, target_ids: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
    """Per-sequence sum of `log p(target)` over masked (response) positions.

    `logits` (B, L, V), `target_ids` (B, L), `mask` (B, L). Returns (B,). Positions with
    mask 0 (prompt + padding) contribute nothing.
    """
    logits = np.asarray(logits, dtype=np.float64)
    target_ids = np.asarray(target_ids)
    mask = np.asarray(mask, dtype=np.float64)
    m = logits.max(axis=-1, keepdims=True)
    logZ = m[..., 0] + np.log(np.exp(logits - m).sum(axis=-1))      # (B, L)
    chosen = np.take_along_axis(logits, target_ids[..., None], axis=-1)[..., 0]
    logp = chosen - logZ                                            # (B, L)
    return (logp * mask).sum(axis=-1)                              # (B,)


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(x)) = -softplus(-x)."""
    return -np.logaddexp(0.0, -x)


def dpo_loss_from_logprobs(logp_pol_c, logp_ref_c, logp_pol_r, logp_ref_r,
                           beta: float = 0.1) -> Tuple[float, float, float]:
    """DPO loss + reward diagnostics from the four sequence log-probs.

    Each argument is a (B,) array (or scalar): policy/reference log-prob of the
    chosen / rejected response. With

        margin = beta*(logp_pol_c - logp_ref_c) - beta*(logp_pol_r - logp_ref_r)

    the loss is `-mean(log sigmoid(margin))`. Returns
    `(loss, reward_margin, reward_accuracy)` where reward_margin is the mean implicit
    reward gap (chosen minus rejected) and reward_accuracy is the fraction of pairs with
    a positive gap. When policy == reference, margin is 0 and loss is `ln 2`.
    """
    logp_pol_c = np.asarray(logp_pol_c, dtype=np.float64)
    logp_ref_c = np.asarray(logp_ref_c, dtype=np.float64)
    logp_pol_r = np.asarray(logp_pol_r, dtype=np.float64)
    logp_ref_r = np.asarray(logp_ref_r, dtype=np.float64)

    chosen_reward = beta * (logp_pol_c - logp_ref_c)
    rejected_reward = beta * (logp_pol_r - logp_ref_r)
    margin = chosen_reward - rejected_reward
    loss = float(-np.mean(_log_sigmoid(margin)))
    return loss, float(np.mean(margin)), float(np.mean(margin > 0))
