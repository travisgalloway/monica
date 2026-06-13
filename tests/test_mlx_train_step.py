"""MLX train_step: grad accumulation + dynamic fp16 loss scaling (Apple Silicon only).

Skipped where mlx is unavailable. Verifies the two new behaviors added for the M5
scale run:
  * accumulating N identical micro-batches equals a single micro-batch (the averaging
    is correct), and
  * a forced gradient overflow SKIPS the optimizer update and backs the loss scale off,
    leaving the model weights untouched.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.model.mlx_train_step import make_train_step
from src.train.loss_scale import DynamicLossScaler

TOY_CFG = "config/toy.yaml"


def _rand_batch(cfg, B=4, L=32, seed=0):
    """An in-range, in-memory batch (ids < vocab_size). Hermetic — no shared data
    path. Mirrors tests/test_cuda_train_step.py for cross-backend consistency."""
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    tgt = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    return inp, tgt


def test_grad_accum_two_identical_microbatches_equal_single():
    cfg = load_config(TOY_CFG)
    inp, tgt = _rand_batch(cfg)

    mx.random.seed(0)
    m1 = MLXMambaModel(cfg)
    s1 = make_train_step(m1, optim.AdamW(learning_rate=1e-3), grad_clip=1.0, scaler=None)
    r1 = s1(m1, [(inp, tgt)], 1e-3)

    mx.random.seed(0)
    m2 = MLXMambaModel(cfg)
    s2 = make_train_step(m2, optim.AdamW(learning_rate=1e-3), grad_clip=1.0, scaler=None)
    r2 = s2(m2, [(inp, tgt), (inp, tgt)], 1e-3)

    # Averaging two identical micro-batches == one micro-batch.
    assert abs(r1["loss"] - r2["loss"]) < 1e-4
    assert abs(r1["grad_norm"] - r2["grad_norm"]) < 1e-4


def test_fp16_overflow_skips_update_and_backs_off():
    cfg = load_config(TOY_CFG)
    inp, tgt = _rand_batch(cfg)

    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=1e-3)
    # A scale above fp32-max makes loss*scale -> inf, so the gradients are
    # non-finite regardless of the per-element grad magnitude (robust trigger).
    scaler = DynamicLossScaler(init_scale=1e40, backoff=0.5, min_scale=1.0)
    step_fn = make_train_step(model, opt, grad_clip=1.0, scaler=scaler)

    before = [np.array(v) for _, v in tree_flatten(model.parameters())]
    out = step_fn(model, [(inp, tgt)], 1e-3)
    after = [np.array(v) for _, v in tree_flatten(model.parameters())]

    assert out["skipped"] is True
    assert scaler.scale == 0.5e40                 # backed off on overflow
    for b, a in zip(before, after):
        assert np.array_equal(b, a)               # weights untouched on a skipped step


def test_fp16_clean_step_updates_and_reports_scale():
    cfg = load_config(TOY_CFG)
    inp, tgt = _rand_batch(cfg)

    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=1e-3)
    scaler = DynamicLossScaler(init_scale=1024.0)
    step_fn = make_train_step(model, opt, grad_clip=1.0, scaler=scaler)

    out = step_fn(model, [(inp, tgt)], 1e-3)
    assert out["skipped"] is False
    assert out["loss_scale"] == 1024.0
    assert np.isfinite(out["grad_norm"])
