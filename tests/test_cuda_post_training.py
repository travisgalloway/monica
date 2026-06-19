"""CUDA/torch post-training step factories (#110): SFT / DPO / GRPO.

Torch mirror of tests/test_sft_train_step.py, test_dpo_train_step.py, and
test_grpo_train_step.py. Runs on torch-CPU (no GPU), so the CUDA post-training path is
verified on the Mac before it ships to a CUDA host. Each objective is checked against the
same portable numpy reference the MLX suites use (masked_cross_entropy, dpo_math,
grpo loss), proving the two backends compute the same objective.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import load_config
from src.model.cuda_backend import CUDAMambaModel
from src.model.cuda_train_step import (_masked_seq_logprob, make_sft_train_step,
                                       make_dpo_train_step, make_grpo_train_step)
from src.eval.val_loss import masked_cross_entropy
from src.train.dpo_math import masked_sequence_logprob
from src.train.loss_scale import DynamicLossScaler

TOY_CFG = "config/toy.yaml"


def _adamw(model, lr, wd=0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _flat(model):
    return [p.detach().clone().cpu().numpy() for p in model.parameters()]


# --------------------------------------------------------------------------- #
# SFT
# --------------------------------------------------------------------------- #
def _rand_sft_batch(cfg, B=4, L=32, seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    tgt = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    mask = (rng.random((B, L)) > 0.5).astype(np.float32)
    return inp, tgt, mask


def test_sft_masked_loss_matches_numpy_reference():
    cfg = load_config(TOY_CFG)
    inp, tgt, mask = _rand_sft_batch(cfg)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    with torch.no_grad():
        logits = model.forward(inp).cpu().numpy()
    expected = masked_cross_entropy(logits, tgt, mask)
    step = make_sft_train_step(model, _adamw(model, 0.0), grad_clip=0.0)
    out = step(model, [(inp, tgt, mask)], 0.0)
    assert abs(out["loss"] - expected) < 1e-4


def test_sft_grad_accum_two_identical_microbatches_equal_single():
    cfg = load_config(TOY_CFG)
    inp, tgt, mask = _rand_sft_batch(cfg)

    torch.manual_seed(0)
    m1 = CUDAMambaModel(cfg)
    r1 = make_sft_train_step(m1, _adamw(m1, 1e-3), grad_clip=1.0)(m1, [(inp, tgt, mask)], 1e-3)

    torch.manual_seed(0)
    m2 = CUDAMambaModel(cfg)
    r2 = make_sft_train_step(m2, _adamw(m2, 1e-3), grad_clip=1.0)(
        m2, [(inp, tgt, mask), (inp, tgt, mask)], 1e-3)

    assert abs(r1["loss"] - r2["loss"]) < 1e-4
    assert abs(r1["grad_norm"] - r2["grad_norm"]) < 1e-4


def test_sft_fp16_overflow_skips_update_and_backs_off():
    cfg = load_config(TOY_CFG)
    inp, tgt, mask = _rand_sft_batch(cfg)

    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    scaler = DynamicLossScaler(init_scale=1e40, backoff=0.5, min_scale=1.0)
    step = make_sft_train_step(model, _adamw(model, 1e-3), grad_clip=1.0, scaler=scaler)

    before = _flat(model)
    out = step(model, [(inp, tgt, mask)], 1e-3)
    after = _flat(model)

    assert out["skipped"] is True
    assert scaler.scale == 0.5e40
    for b, a in zip(before, after):
        assert np.array_equal(b, a)


def test_sft_all_zero_mask_is_safe():
    cfg = load_config(TOY_CFG)
    inp, tgt, _ = _rand_sft_batch(cfg)
    mask = np.zeros_like(inp, dtype=np.float32)

    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    step = make_sft_train_step(model, _adamw(model, 1e-3), grad_clip=1.0)
    out = step(model, [(inp, tgt, mask)], 1e-3)

    assert out["loss"] == 0.0
    assert out["grad_norm"] == 0.0


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #
def _side(cfg, L, seed, B=3):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    tgt = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    mask = (rng.random((B, L)) > 0.4).astype(np.float32)
    return inp, tgt, mask


def _dpo_batch(cfg):
    return (*_side(cfg, 20, 0), *_side(cfg, 24, 100))


def test_dpo_policy_equals_reference_loss_is_ln2():
    cfg = load_config(TOY_CFG)
    batch = _dpo_batch(cfg)
    torch.manual_seed(0)
    policy = CUDAMambaModel(cfg)
    torch.manual_seed(0)
    ref = CUDAMambaModel(cfg)                  # identical init -> margin 0 everywhere
    step = make_dpo_train_step(policy, ref, _adamw(policy, 0.0), beta=0.1, grad_clip=0.0)
    out = step(policy, [batch], 0.0)
    assert abs(out["loss"] - math.log(2.0)) < 1e-4
    assert np.isfinite(out["grad_norm"]) and out["grad_norm"] > 0   # real policy gradient


def test_dpo_reference_frozen_policy_updates():
    cfg = load_config(TOY_CFG)
    batch = _dpo_batch(cfg)
    torch.manual_seed(0)
    policy = CUDAMambaModel(cfg)
    torch.manual_seed(1)
    ref = CUDAMambaModel(cfg)                  # different -> nonzero margin and gradient
    ref_before, pol_before = _flat(ref), _flat(policy)

    step = make_dpo_train_step(policy, ref, _adamw(policy, 1e-2), beta=0.1, grad_clip=1.0)
    step(policy, [batch], 1e-2)

    for b, a in zip(ref_before, _flat(ref)):
        assert np.array_equal(b, a)            # reference untouched
    assert any(not np.array_equal(b, a) for b, a in zip(pol_before, _flat(policy)))


def test_dpo_seq_logprob_matches_numpy_reference():
    cfg = load_config(TOY_CFG)
    inp, tgt, mask = _side(cfg, 24, 7)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    with torch.no_grad():
        torch_val = _masked_seq_logprob(model, inp, tgt, mask).cpu().numpy()
        np_val = masked_sequence_logprob(model.forward(inp).cpu().numpy(), tgt, mask)
    assert np.allclose(torch_val, np_val, atol=1e-3)


# --------------------------------------------------------------------------- #
# GRPO
# --------------------------------------------------------------------------- #
def _grpo_batch(cfg, B=4, L=16, seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    tgt = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    mask = (rng.random((B, L)) > 0.4).astype(np.float32)
    adv = rng.standard_normal(B).astype(np.float32)
    return inp, tgt, mask, adv


def test_grpo_loss_matches_reference_and_updates_policy():
    cfg = load_config(TOY_CFG)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    batch = _grpo_batch(cfg)

    with torch.no_grad():
        logp = _masked_seq_logprob(model, batch[0], batch[1], batch[2]).cpu().numpy()
    ref_loss = float(-np.mean(batch[3] * logp))

    before = _flat(model)
    step = make_grpo_train_step(model, _adamw(model, 1e-3))
    out = step(model, [batch], 1e-3)

    assert np.isfinite(out["loss"])
    assert np.isclose(out["loss"], ref_loss, rtol=1e-4, atol=1e-4)
    after = _flat(model)
    assert any(not np.allclose(a, b) for a, b in zip(before, after))   # policy updated


def test_grpo_zero_advantage_gives_no_update():
    cfg = load_config(TOY_CFG)
    torch.manual_seed(1)
    model = CUDAMambaModel(cfg)
    inp, tgt, mask, _ = _grpo_batch(cfg, seed=2)
    batch = (inp, tgt, mask, np.zeros(inp.shape[0], dtype=np.float32))   # all-zero advantage
    before = _flat(model)
    # weight_decay=0 so the decoupled decay can't move params when the gradient is zero.
    step = make_grpo_train_step(model, _adamw(model, 1e-3, wd=0.0))
    out = step(model, [batch], 1e-3)
    assert out["loss"] == 0.0
    after = _flat(model)
    assert all(np.allclose(a, b) for a, b in zip(before, after))        # no gradient -> no change
