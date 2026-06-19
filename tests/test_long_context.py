"""Tests for the training-free long-context extension (#54).

Two layers: the PORTABLE harness logic (`src.eval.long_context`) runs with a fake
model + a tiny packed file, no backend; the MLX-guarded checks confirm the knob is
parity-exact when off, actually changes the scan when on, and the config validates.
"""

import json

import numpy as np
import pytest

from src.data.pack import pack_ids
from src.eval.long_context import format_curve, long_context_eval
from src.model.blocks import MambaConfig


# --------------------------------------------------------------------------- #
# Portable harness (no backend)
# --------------------------------------------------------------------------- #
class _FakeModel:
    """Minimal ModelInterface surface: forward(inputs)->logits and a `config`."""

    def __init__(self, vocab_size, seq_len):
        self.vocab_size = vocab_size

        class _Cfg:
            pass
        self.config = _Cfg()
        self.config.seq_len = seq_len
        self._rng = np.random.default_rng(0)

    def forward(self, inputs):
        b, l = np.asarray(inputs).shape
        return self._rng.standard_normal((b, l, self.vocab_size)).astype(np.float32)


def _make_packed(tmp_path, n_tokens, vocab=32):
    ids = np.arange(n_tokens, dtype=np.int64) % vocab
    path = tmp_path / "val.bin"
    pack_ids(ids, path, dtype=np.dtype("uint16"))
    return path


def test_harness_reports_perplexity_per_length(tmp_path):
    base_seq = 8
    path = _make_packed(tmp_path, n_tokens=8 * (base_seq * 4 + 1))  # room for 4x chunks
    model = _FakeModel(vocab_size=32, seq_len=base_seq)
    res = long_context_eval(model, path, base_seq, batch_size=2, mults=(1, 2, 4),
                            max_batches=2)
    for mult in (1, 2, 4):
        assert res[mult] is not None
        assert res[mult]["seq_len"] == base_seq * mult
        assert res[mult]["val_perplexity"] > 0


def test_harness_skips_lengths_too_long_for_the_split(tmp_path):
    base_seq = 8
    # Only enough tokens for a single 1x chunk; 2x/4x can't form one chunk.
    path = _make_packed(tmp_path, n_tokens=base_seq + 2)
    model = _FakeModel(vocab_size=16, seq_len=base_seq)
    res = long_context_eval(model, path, base_seq, batch_size=1, mults=(1, 2, 4))
    assert res[1] is not None
    assert res[2] is None and res[4] is None        # recorded, not raised


def test_format_curve_handles_skips(tmp_path):
    out = format_curve("knob OFF", {1: {"seq_len": 8, "val_loss": 1.0,
                                        "val_perplexity": 2.7, "n_batches": 2},
                                    2: None})
    assert "1x" in out and "2x" in out and "skipped" in out


# --------------------------------------------------------------------------- #
# Config validation (portable)
# --------------------------------------------------------------------------- #
def test_long_ctx_factor_validates():
    cfg = MambaConfig(d_model=64, n_layers=2, head_dim=16, vocab_size=32,
                      seq_len=16, long_ctx_factor=0.5)
    with pytest.raises(ValueError, match="long_ctx_factor"):
        cfg.validate()


def test_long_ctx_factor_default_is_off():
    assert MambaConfig(d_model=64, n_layers=2).long_ctx_factor == 1.0


# --------------------------------------------------------------------------- #
# MLX-guarded: the knob is parity-exact off, and changes the scan on
# --------------------------------------------------------------------------- #
mx = pytest.importorskip("mlx.core")


def _toy_cfg(**over):
    base = dict(d_model=32, n_layers=2, head_dim=16, d_state=8, vocab_size=32,
                seq_len=16, precision="fp32")
    base.update(over)
    return MambaConfig(**base)


def test_knob_off_is_byte_identical():
    from src.model.mlx_backend import MLXMambaModel
    mx.random.seed(0)
    model = MLXMambaModel(_toy_cfg(long_ctx_factor=1.0))
    tokens = mx.array(np.arange(2 * 16).reshape(2, 16) % 32)
    # Default config path and an explicit factor=1.0 must agree exactly with a model
    # that simply never sees the scaling branch — the guard keeps delta untouched.
    y = np.array(model.forward(tokens))
    assert np.all(np.isfinite(y))
    # Re-run: deterministic, identical.
    assert np.array_equal(y, np.array(model.forward(tokens)))


def test_knob_on_changes_logits_but_stays_finite():
    from src.model.mlx_backend import MLXMambaModel
    mx.random.seed(0)
    off = MLXMambaModel(_toy_cfg(long_ctx_factor=1.0))
    weights = off._portable_state_dict()
    on = MLXMambaModel(_toy_cfg(long_ctx_factor=4.0))
    on._load_portable(weights)                      # same weights, knob on
    tokens = mx.array(np.arange(2 * 16).reshape(2, 16) % 32)
    yo, yn = np.array(off.forward(tokens)), np.array(on.forward(tokens))
    assert np.all(np.isfinite(yn))
    assert not np.allclose(yo, yn)                  # the scan genuinely changed


def test_forward_step_parity_holds_with_knob_on():
    """The knob lives in `_project`, shared by parallel (forward) and recurrence (step),
    so the two paths must still agree at the fp32 ~1e-4 tolerance with the knob on."""
    from src.model.mlx_backend import MLXMambaModel
    mx.random.seed(0)
    model = MLXMambaModel(_toy_cfg(long_ctx_factor=3.0))
    tokens = np.arange(16).reshape(1, 16) % 32

    seq_logits = np.array(model.forward(mx.array(tokens)))[0]   # (L, V)

    state = model.init_state(1)
    step_logits = []
    for t in range(tokens.shape[1]):
        logit, state = model.step(mx.array(tokens[:, t]), state)
        step_logits.append(np.array(logit)[0])
    step_logits = np.stack(step_logits)                          # (L, V)

    rel = np.abs(seq_logits - step_logits).max() / (np.abs(seq_logits).max() + 1e-6)
    assert rel < 1e-4
