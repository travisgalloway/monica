"""DPO loss math (portable). Anchors: policy==reference gives loss ln 2 and zero margin;
the masked sequence log-prob sums only response positions; a better chosen response
lowers the loss and yields a positive reward margin."""

from __future__ import annotations

import numpy as np
import pytest

from src.train.dpo_math import dpo_loss_from_logprobs, masked_sequence_logprob


def test_policy_equals_reference_gives_ln2():
    lp = np.array([-3.0, -5.0, -2.0])  # arbitrary; policy and ref identical
    loss, margin, acc = dpo_loss_from_logprobs(lp, lp, lp - 1.0, lp - 1.0, beta=0.1)
    assert loss == pytest.approx(np.log(2.0), rel=1e-9)
    assert margin == pytest.approx(0.0, abs=1e-12)
    assert acc == 0.0  # a zero margin is not counted as a win


def test_chosen_preferred_lowers_loss_and_positive_margin():
    # Policy raises chosen vs ref and lowers rejected vs ref -> positive margin.
    logp_pol_c = np.array([-1.0]);  logp_ref_c = np.array([-2.0])
    logp_pol_r = np.array([-4.0]);  logp_ref_r = np.array([-3.0])
    loss, margin, acc = dpo_loss_from_logprobs(logp_pol_c, logp_ref_c,
                                               logp_pol_r, logp_ref_r, beta=0.1)
    assert margin > 0 and acc == 1.0
    assert loss < np.log(2.0)


def test_loss_matches_explicit_formula():
    a, b, c, d = (np.array([-1.0, -2.0]), np.array([-1.5, -1.0]),
                  np.array([-3.0, -2.5]), np.array([-2.0, -3.0]))
    beta = 0.1
    margin = beta * (a - b) - beta * (c - d)
    expected = float(-np.mean(-np.logaddexp(0.0, -margin)))
    loss, _, _ = dpo_loss_from_logprobs(a, b, c, d, beta=beta)
    assert loss == pytest.approx(expected, rel=1e-9)


def test_masked_sequence_logprob_sums_only_response_positions():
    # 1 sequence, length 3, vocab 4. Mask out position 0.
    logits = np.log(np.array([[[0.1, 0.2, 0.3, 0.4],
                               [0.4, 0.3, 0.2, 0.1],
                               [0.25, 0.25, 0.25, 0.25]]]))
    targets = np.array([[3, 0, 2]])
    mask = np.array([[0.0, 1.0, 1.0]])
    out = masked_sequence_logprob(logits, targets, mask)
    expected = np.log(0.4) + np.log(0.25)  # positions 1 (target 0) and 2 (target 2)
    assert out.shape == (1,)
    assert out[0] == pytest.approx(expected, rel=1e-9)


def test_masked_sequence_logprob_all_zero_mask_is_zero():
    logits = np.zeros((2, 3, 5))
    targets = np.zeros((2, 3), dtype=np.int64)
    out = masked_sequence_logprob(logits, targets, np.zeros((2, 3)))
    assert np.allclose(out, 0.0)
