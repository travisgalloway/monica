"""MLX DPO train_step (Apple Silicon only). Verifies: policy==reference gives the ln 2
loss with a finite policy gradient; the frozen reference is never updated while the
policy is; and the MLX masked sequence log-prob matches the portable numpy reference.
"""

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.model.mlx_train_step import _masked_seq_logprob, make_dpo_train_step
from src.train.dpo_math import masked_sequence_logprob

TOY_CFG = "config/toy.yaml"


def _side(cfg, L, seed, B=3):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    tgt = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    mask = (rng.random((B, L)) > 0.4).astype(np.float32)
    return inp, tgt, mask


def _dpo_batch(cfg):
    return (*_side(cfg, 20, 0), *_side(cfg, 24, 100))


def _flat(model):
    return [np.array(v) for _, v in tree_flatten(model.parameters())]


def test_policy_equals_reference_loss_is_ln2():
    cfg = load_config(TOY_CFG)
    batch = _dpo_batch(cfg)
    mx.random.seed(0)
    policy = MLXMambaModel(cfg)
    mx.random.seed(0)
    ref = MLXMambaModel(cfg)                  # identical init -> margin 0 everywhere
    step = make_dpo_train_step(policy, ref, optim.AdamW(learning_rate=0.0),
                               beta=0.1, grad_clip=0.0)
    out = step(policy, [batch], 0.0)
    assert abs(out["loss"] - math.log(2.0)) < 1e-4
    assert np.isfinite(out["grad_norm"]) and out["grad_norm"] > 0  # real policy gradient


def test_reference_frozen_policy_updates():
    cfg = load_config(TOY_CFG)
    batch = _dpo_batch(cfg)
    mx.random.seed(0)
    policy = MLXMambaModel(cfg)
    mx.random.seed(1)
    ref = MLXMambaModel(cfg)                  # different -> nonzero margin and gradient
    ref_before, pol_before = _flat(ref), _flat(policy)

    step = make_dpo_train_step(policy, ref, optim.AdamW(learning_rate=1e-2),
                               beta=0.1, grad_clip=1.0)
    step(policy, [batch], 1e-2)

    for b, a in zip(ref_before, _flat(ref)):
        assert np.array_equal(b, a)           # reference untouched
    assert any(not np.array_equal(b, a) for b, a in zip(pol_before, _flat(policy)))


def test_mlx_seq_logprob_matches_numpy_reference():
    cfg = load_config(TOY_CFG)
    inp, tgt, mask = _side(cfg, 24, 7)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    mlx_val = np.array(_masked_seq_logprob(model, inp, tgt, mask))
    np_val = masked_sequence_logprob(np.array(model.forward(inp)), tgt, mask)
    assert np.allclose(mlx_val, np_val, atol=1e-3)
