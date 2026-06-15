"""GRPO math (#78). Pure numpy reference for the group baseline + policy-gradient loss."""

import numpy as np
import pytest

from src.train.grpo import group_advantages, grpo_loss_from_logprobs, reward_stats


def test_group_advantages_all_equal_is_zero():
    # A group that's all-correct or all-wrong yields ~0 advantage -> no gradient.
    adv = group_advantages([[1.0, 1.0, 1.0]])
    assert np.allclose(adv, 0.0, atol=1e-5)


def test_group_advantages_standardized_per_group():
    adv = group_advantages([[0.0, 1.0], [5.0, 5.0]])
    assert np.isclose(adv[0].mean(), 0.0, atol=1e-9)
    assert adv[0, 1] > adv[0, 0]                 # higher reward -> higher advantage
    assert np.allclose(adv[1], 0.0, atol=1e-5)   # second group independent + degenerate


def test_grpo_loss_matches_formula():
    logp = np.array([-1.0, -2.0, -0.5])
    adv = np.array([1.0, -1.0, 0.0])
    loss, mabs = grpo_loss_from_logprobs(logp, adv)
    assert np.isclose(loss, float(-np.mean(adv * logp)))
    assert np.isclose(mabs, float(np.mean(np.abs(adv))))


def test_grpo_loss_zero_advantage_is_zero():
    loss, _ = grpo_loss_from_logprobs([-1.0, -2.0], [0.0, 0.0])
    assert loss == 0.0


def test_reward_stats():
    s = reward_stats([1.0, 0.0, 1.0])
    assert s["mean_reward"] == pytest.approx(2 / 3)
    assert s["frac_solved"] == pytest.approx(2 / 3)   # reward == max (1.0) for 2 of 3
