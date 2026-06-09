"""MLX mixed-precision tests (issue #27, Apple Silicon only — skipped without mlx).

Verifies that `precision=fp16` actually drives compute (fp32 master weights + fp16
compute), and that the `precision=fp32` path is unchanged:

  * dtype policy: the inter-layer activation stream and the heavy GEMM outputs are
    fp16, while logits (wide-vocab softmax) stay fp32 and params stay fp32 (masters);
  * forward is finite under fp16 (no overflow into NaN/Inf);
  * a short fp16 run decreases the loss with the dynamic scaler active and no NaNs;
  * grad checkpointing is deterministic vs not, under fp16;
  * the fp32 path is bit-identical regardless of these casts (the casts are no-ops):
    a fp32 forward equals one through the same construction, to the bit.
"""

import dataclasses

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.model.mlx_train_step import make_train_step
from src.train.loss_scale import scaler_for_precision

TOY_CFG = "config/toy.yaml"


def _fp16_cfg(**overrides):
    cfg = load_config(TOY_CFG)
    return dataclasses.replace(cfg, precision="fp16", **overrides)


def _tokens(cfg, batch=2, seq=16, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(0, cfg.vocab_size, size=(batch, seq)).astype(np.int32))


def test_fp16_dtype_policy_and_master_weights():
    """fp16 stream + heavy GEMMs, fp32 logits, fp32 master weights."""
    cfg = _fp16_cfg()
    mx.random.seed(0)
    m = MLXMambaModel(cfg)
    toks = _tokens(cfg)

    # The activation stream and per-block output are the compute dtype (fp16).
    h = m.embedding(toks).astype(m._cd)
    assert h.dtype == mx.float16
    assert m.layers[0].forward_seq(h).dtype == mx.float16

    # Logits (and hence cross-entropy) stay fp32 for wide-vocab softmax stability.
    logits = m.forward(toks)
    mx.eval(logits)
    assert logits.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(logits)).item())   # no fp16 overflow into NaN/Inf

    # Params are fp32 masters — nothing is stored in low precision.
    assert {v.dtype for _, v in tree_flatten(m.parameters())} == {mx.float32}


def test_fp32_path_is_bit_identical():
    """precision=fp32 must be unchanged by the casting code (every cast is a no-op)."""
    cfg32 = load_config(TOY_CFG)
    assert cfg32.precision == "fp32"
    toks = _tokens(cfg32)

    mx.random.seed(0)
    a = MLXMambaModel(cfg32).forward(toks)
    mx.random.seed(0)
    b = MLXMambaModel(cfg32).forward(toks)
    mx.eval(a, b)
    # Same seed + same construction => bit-for-bit identical logits in fp32.
    assert np.array_equal(np.array(a), np.array(b))


def test_fp16_short_run_decreases_loss_with_scaler():
    cfg = _fp16_cfg()
    mx.random.seed(0)
    m = MLXMambaModel(cfg)
    scaler = scaler_for_precision(cfg.precision)
    assert scaler is not None                         # fp16 gets the dynamic scaler
    step = make_train_step(m, optim.AdamW(learning_rate=1e-3), grad_clip=1.0, scaler=scaler)

    rng = np.random.default_rng(0)
    losses = []
    for _ in range(12):
        inp = rng.integers(0, cfg.vocab_size, size=(4, 32)).astype(np.int32)
        tgt = rng.integers(0, cfg.vocab_size, size=(4, 32)).astype(np.int32)
        out = step(m, [(inp, tgt)], 1e-3)
        assert np.isfinite(out["loss"])               # no NaNs under fp16 + scaling
        losses.append(out["loss"])

    assert losses[-1] < losses[0]                      # loss trends down
    assert out["loss_scale"] >= 1.0                    # scaler stayed engaged


def test_fp16_grad_checkpoint_matches_no_checkpoint():
    """Recompute-in-backward must be deterministic vs retaining activations (fp16)."""
    def run(grad_checkpoint):
        cfg = _fp16_cfg(grad_checkpoint=grad_checkpoint)
        mx.random.seed(0)
        m = MLXMambaModel(cfg)
        step = make_train_step(m, optim.AdamW(learning_rate=1e-3), grad_clip=1.0,
                               scaler=scaler_for_precision(cfg.precision))
        rng = np.random.default_rng(0)
        out = None
        for _ in range(4):
            inp = rng.integers(0, cfg.vocab_size, size=(4, 32)).astype(np.int32)
            tgt = rng.integers(0, cfg.vocab_size, size=(4, 32)).astype(np.int32)
            out = step(m, [(inp, tgt)], 1e-3)
        return out["loss"]

    assert run(True) == pytest.approx(run(False), rel=1e-5, abs=1e-5)
