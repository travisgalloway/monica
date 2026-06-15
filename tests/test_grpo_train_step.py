"""MLX GRPO train_step (Apple Silicon only). Verifies the step's loss matches the portable
numpy reference `-mean(advantage * seq_logp)` and that the policy is updated.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.model.mlx_train_step import _masked_seq_logprob, make_grpo_train_step

TOY_CFG = "config/toy.yaml"


def _batch(cfg, B=4, L=16, seed=0):
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    tgt = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    mask = (rng.random((B, L)) > 0.4).astype(np.float32)
    adv = rng.standard_normal(B).astype(np.float32)
    return inp, tgt, mask, adv


def _flat(model):
    return [np.array(v) for _, v in tree_flatten(model.parameters())]


def test_grpo_loss_matches_reference_and_updates_policy():
    cfg = load_config(TOY_CFG)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    batch = _batch(cfg)

    # numpy reference loss at the initial params
    logp = np.array(_masked_seq_logprob(model, batch[0], batch[1], batch[2]))
    ref_loss = float(-np.mean(batch[3] * logp))

    before = _flat(model)
    opt = optim.AdamW(learning_rate=1e-3)
    step = make_grpo_train_step(model, opt)
    out = step(model, [batch], 1e-3)

    assert np.isfinite(out["loss"])
    assert np.isclose(out["loss"], ref_loss, rtol=1e-4, atol=1e-4)
    after = _flat(model)
    assert any(not np.allclose(a, b) for a, b in zip(before, after))   # policy updated


def test_zero_advantage_gives_no_update():
    cfg = load_config(TOY_CFG)
    mx.random.seed(1)
    model = MLXMambaModel(cfg)
    inp, tgt, mask, _ = _batch(cfg, seed=2)
    batch = (inp, tgt, mask, np.zeros(inp.shape[0], dtype=np.float32))   # all-zero advantage
    before = _flat(model)
    step = make_grpo_train_step(model, optim.AdamW(learning_rate=1e-3, weight_decay=0.0))
    out = step(model, [batch], 1e-3)
    assert out["loss"] == 0.0
    after = _flat(model)
    assert all(np.allclose(a, b) for a, b in zip(before, after))        # no gradient -> no change
