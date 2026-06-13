"""MLX SFT train_step: masked CE numerics + shared accumulation tail (Apple Silicon).

Skipped where mlx is unavailable. Verifies that (1) the MLX masked loss matches the
portable `masked_cross_entropy` reference, (2) grad accumulation still averages
correctly through the refactored `_accumulate_and_step`, (3) the fp16 overflow-skip is
preserved for the SFT step, and (4) an all-padding (all-zero-mask) batch is safe.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.model.mlx_train_step import make_sft_train_step
from src.eval.val_loss import masked_cross_entropy
from src.train.loss_scale import DynamicLossScaler

TOY_CFG = "config/toy.yaml"


def _rand_sft_batch(cfg, B=4, L=32, seed=0):
    """In-range hermetic (inputs, targets, mask) — mask is a random response selection."""
    rng = np.random.default_rng(seed)
    inp = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    tgt = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int64)
    mask = (rng.random((B, L)) > 0.5).astype(np.float32)
    return inp, tgt, mask


def test_masked_loss_matches_numpy_reference():
    cfg = load_config(TOY_CFG)
    inp, tgt, mask = _rand_sft_batch(cfg)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    # Loss is computed on the pre-update weights, so capture logits first.
    logits = np.array(model.forward(inp))
    expected = masked_cross_entropy(logits, tgt, mask)
    # lr 0 / no clip leaves weights untouched; we only check the reported loss.
    step = make_sft_train_step(model, optim.AdamW(learning_rate=0.0), grad_clip=0.0)
    out = step(model, [(inp, tgt, mask)], 0.0)
    assert abs(out["loss"] - expected) < 1e-4


def test_grad_accum_two_identical_microbatches_equal_single():
    cfg = load_config(TOY_CFG)
    inp, tgt, mask = _rand_sft_batch(cfg)

    mx.random.seed(0)
    m1 = MLXMambaModel(cfg)
    r1 = make_sft_train_step(m1, optim.AdamW(learning_rate=1e-3), grad_clip=1.0)(
        m1, [(inp, tgt, mask)], 1e-3)

    mx.random.seed(0)
    m2 = MLXMambaModel(cfg)
    r2 = make_sft_train_step(m2, optim.AdamW(learning_rate=1e-3), grad_clip=1.0)(
        m2, [(inp, tgt, mask), (inp, tgt, mask)], 1e-3)

    assert abs(r1["loss"] - r2["loss"]) < 1e-4
    assert abs(r1["grad_norm"] - r2["grad_norm"]) < 1e-4


def test_fp16_overflow_skips_update_and_backs_off():
    cfg = load_config(TOY_CFG)
    inp, tgt, mask = _rand_sft_batch(cfg)

    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=1e-3)
    scaler = DynamicLossScaler(init_scale=1e40, backoff=0.5, min_scale=1.0)
    step = make_sft_train_step(model, opt, grad_clip=1.0, scaler=scaler)

    before = [np.array(v) for _, v in tree_flatten(model.parameters())]
    out = step(model, [(inp, tgt, mask)], 1e-3)
    after = [np.array(v) for _, v in tree_flatten(model.parameters())]

    assert out["skipped"] is True
    assert scaler.scale == 0.5e40
    for b, a in zip(before, after):
        assert np.array_equal(b, a)


def test_all_zero_mask_is_safe():
    cfg = load_config(TOY_CFG)
    inp, tgt, _ = _rand_sft_batch(cfg)
    mask = np.zeros_like(inp, dtype=np.float32)

    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    step = make_sft_train_step(model, optim.AdamW(learning_rate=1e-3), grad_clip=1.0)
    out = step(model, [(inp, tgt, mask)], 1e-3)

    # All padding -> zero loss and zero gradient (the denom guard avoids a div-by-zero).
    assert out["loss"] == 0.0
    assert out["grad_norm"] == 0.0
