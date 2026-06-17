"""Distillation loss + train step (#100). MLX-only; skipped where mlx is unavailable.

Covers the materialized mixing matrix (vs the scan), the compound KL(top-k)+CE logit-distill
step, hidden-align (cached == on-the-fly), mixing-match, and the fp16 dynamic-scaler integration
(clean step + clean overflow-skip on the compound term). Tiny synthetic teacher + tiny hybrid
student, all offline.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn

from src.model.backend import get_backend
from src.model.blocks import MambaConfig
from src.model.mlx_backend import MLXMambaModel, SelectiveSSM
from src.model.mlx_teacher import MLXConversionTeacher
from src.model.teacher import TeacherConfig
from src.model.mlx_distill import make_distill_train_step, _kl_topk
from src.train.distill_manifest import DistillStage
from src.train.loss_scale import DynamicLossScaler

VOCAB = 256


def _teacher(n_layers=3):
    cfg = TeacherConfig(vocab_size=VOCAB, d_model=64, n_layers=n_layers, n_heads=4,
                        n_kv_heads=2, head_dim=16, intermediate_size=128)
    return MLXConversionTeacher.from_config(cfg, seed=0)


def _student(precision="fp32", seed=0):
    cfg = MambaConfig(d_model=48, n_layers=4, d_state=16, expand=2, d_conv=4, head_dim=16,
                      vocab_size=VOCAB, seq_len=16, attn_every=2, n_attn_heads=4,
                      precision=precision)
    get_backend("mlx").seed(seed)
    return MLXMambaModel(cfg)


def _tokens(B=2, L=8):
    rng = np.random.default_rng(0)
    return rng.integers(0, VOCAB, size=(B, L)).astype(np.int32)


def _opt(model, lr=1e-2):
    return get_backend("mlx").make_optimizer(model, lr)


# --- mixing matrix materialization vs the scan -------------------------------
def test_mixing_matrix_reproduces_parallel_scan():
    cfg = MambaConfig(d_model=32, n_layers=1, d_state=16, head_dim=16, vocab_size=64,
                      seq_len=12, precision="fp32")
    ssm = SelectiveSSM(cfg)
    mx.random.seed(0)
    x = mx.random.normal((2, 12, cfg.d_inner))
    Y = ssm.parallel(x)
    M = ssm.mixing_matrix(x)                                  # (B,H,L,L)
    Xh = x.reshape(2, 12, cfg.n_heads, cfg.head_dim)
    Ym = mx.einsum("bhij,bjhp->bihp", M, Xh).reshape(2, 12, cfg.d_inner)
    assert float(mx.max(mx.abs(Y - Ym))) < 1e-4


# --- KL on top-k -------------------------------------------------------------
def test_kl_topk_nonneg_and_zero_at_match():
    rng = np.random.default_rng(1)
    logits = mx.array(rng.normal(size=(2, 4, VOCAB)).astype(np.float32))
    k = 8
    idx = mx.argsort(-logits, axis=-1)[..., :k]
    vals = mx.take_along_axis(logits, idx, axis=-1)
    # student == teacher on the support -> KL == 0
    assert abs(float(_kl_topk(logits, vals, idx, 2.0))) < 1e-4
    # a mismatched teacher distribution -> KL > 0
    assert float(_kl_topk(logits, vals + 1.0 * mx.arange(k).astype(mx.float32), idx, 2.0)) > 0


# --- logit-distill -----------------------------------------------------------
def _logit_batch(teacher, tokens, k=16):
    vals, idx = teacher.topk_logits(tokens, k)
    targets = np.roll(tokens, -1, axis=1)
    return (tokens, targets, np.array(vals), np.array(idx))


def test_logit_distill_step_and_decreases():
    teacher, student = _teacher(), _student()
    step = make_distill_train_step(student, _opt(student), stage=DistillStage.LOGIT_DISTILL)
    mb = _logit_batch(teacher, _tokens())
    losses = [step(student, [mb], 1e-2)["loss"] for _ in range(4)]
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]                            # compound loss decreases


def test_logit_distill_accepts_string_stage():
    teacher, student = _teacher(), _student()
    step = make_distill_train_step(student, _opt(student), stage="logit-distill")
    out = step(student, [_logit_batch(teacher, _tokens())], 1e-2)
    assert "loss" in out and "grad_norm" in out


# --- hidden-align: cached == on-the-fly --------------------------------------
def test_hidden_align_cached_matches_on_the_fly():
    teacher = _teacher()
    tokens = _tokens()
    # two identically-seeded students so the first-step losses are comparable
    s1, s2 = _student(seed=7), _student(seed=7)
    on_fly = make_distill_train_step(s1, _opt(s1), stage=DistillStage.HIDDEN_ALIGN,
                                     teacher=teacher)
    cached_hs = [np.array(h) for h in teacher.forward(tokens, return_hidden=True).hidden_states]
    cached = make_distill_train_step(s2, _opt(s2), stage=DistillStage.HIDDEN_ALIGN)
    l_fly = on_fly(s1, [(tokens,)], 0.0)["loss"]             # lr 0 -> loss is pre-update
    l_cached = cached(s2, [(tokens, cached_hs)], 0.0)["loss"]
    assert l_fly >= 0 and abs(l_fly - l_cached) < 1e-4


def test_hidden_align_decreases():
    teacher, student = _teacher(), _student()
    step = make_distill_train_step(student, _opt(student), stage=DistillStage.HIDDEN_ALIGN,
                                   teacher=teacher)
    losses = [step(student, [(_tokens(),)], 1e-2)["loss"] for _ in range(4)]
    assert losses[-1] < losses[0]


# --- mixing-match ------------------------------------------------------------
def test_mixing_match_runs_and_decreases():
    teacher, student = _teacher(), _student()
    step = make_distill_train_step(student, _opt(student), stage=DistillStage.MIXING_MATCH,
                                   teacher=teacher)
    # the matrix-matching objective is stiff; a small lr decreases it smoothly
    losses = [step(student, [(_tokens(),)], 3e-3)["loss"] for _ in range(6)]
    assert losses[0] >= 0 and losses[-1] < losses[0]


def test_mixing_match_requires_teacher():
    student = _student()
    with pytest.raises(ValueError):
        make_distill_train_step(student, _opt(student), stage=DistillStage.MIXING_MATCH)


def test_mixing_match_rejects_pure_attention_layout():
    teacher = _teacher()
    # attn_every 1 -> every layer is attention, so there are no Mamba mixers to match
    cfg = MambaConfig(d_model=48, n_layers=2, d_state=16, head_dim=16, vocab_size=VOCAB,
                      seq_len=16, attn_every=1, n_attn_heads=4, precision="fp32")
    get_backend("mlx").seed(0)
    student = MLXMambaModel(cfg)
    with pytest.raises(ValueError):
        make_distill_train_step(student, _opt(student), stage=DistillStage.MIXING_MATCH,
                                teacher=teacher)


def test_align_maps_endpoints():
    from src.model.mlx_distill import _align
    assert [_align(i, 4, 28) for i in range(4)] == [0, 9, 18, 27]   # 0->0, last->last
    assert _align(0, 1, 28) == 0                                    # n_src <= 1 fallback


# --- fp16 dynamic-scaler integration (compound term) -------------------------
def test_compound_loss_with_scaler_clean_step():
    teacher, student = _teacher(), _student()
    scaler = DynamicLossScaler(init_scale=2.0 ** 12)
    step = make_distill_train_step(student, _opt(student), stage=DistillStage.LOGIT_DISTILL,
                                   scaler=scaler)
    out = step(student, [_logit_batch(teacher, _tokens())], 1e-3)
    assert out["skipped"] is False and "loss_scale" in out and np.isfinite(out["grad_norm"])


def test_compound_loss_overflow_skips_cleanly():
    teacher, student = _teacher(), _student()
    scaler = DynamicLossScaler(init_scale=2.0 ** 12)
    before = scaler.scale
    step = make_distill_train_step(student, _opt(student), stage=DistillStage.LOGIT_DISTILL,
                                   scaler=scaler)
    inputs, targets, vals, idx = _logit_batch(teacher, _tokens())
    vals = vals.copy()
    vals[0, 0, 0] = np.float32("nan")                        # poison -> non-finite gradient
    out = step(student, [(inputs, targets, vals, idx)], 1e-3)
    assert out["skipped"] is True
    assert scaler.scale < before                             # scale backed off, step dropped


# --- backend seam ------------------------------------------------------------
def test_make_distill_train_step_via_backend():
    teacher, student = _teacher(), _student()
    step = get_backend("mlx").make_distill_train_step(
        student, _opt(student), stage=DistillStage.LOGIT_DISTILL)
    out = step(student, [_logit_batch(teacher, _tokens())], 1e-3)
    assert "loss" in out
