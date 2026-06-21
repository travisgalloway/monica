"""CUDA/torch distillation loss + train step (#100). Torch CPU; parity vs MLX where available.

Covers the materialized mixing matrix (vs the scan), KL on top-k, the three stage steps (loss
decreases), the stage guards, and cross-backend loss parity at fp32 — the same standard as
src/conformance/backend_parity.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.backend import get_backend
from src.model.blocks import MambaConfig
from src.model.teacher import TeacherConfig
from src.model.cuda_backend import CUDAMambaModel, SelectiveSSM
from src.model.cuda_teacher import CUDATeacher
from src.model.cuda_distill import make_distill_train_step, _kl_topk
from src.train.distill_manifest import DistillStage, InitMethod

VOCAB = 256


def _have_mlx():
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _tcfg():
    return TeacherConfig(vocab_size=VOCAB, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                         head_dim=16, intermediate_size=128)


def _scfg(precision="fp32"):
    return MambaConfig(d_model=48, n_layers=4, d_state=16, expand=2, d_conv=4, head_dim=16,
                       vocab_size=VOCAB, seq_len=16, attn_every=2, n_attn_heads=4, precision=precision)


def _numpy_teacher_weights(cfg, seed=0):
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(cfg.d_model)
    f = lambda *s: (rng.standard_normal(s) * scale).astype(np.float32)
    w = {"embed": f(cfg.vocab_size, cfg.d_model)}
    for i in range(cfg.n_layers):
        p = f"layer.{i}."
        w[p + "input_ln"] = np.ones(cfg.d_model, np.float32)
        w[p + "q_w"] = f(cfg.q_dim, cfg.d_model); w[p + "q_b"] = f(cfg.q_dim)
        w[p + "k_w"] = f(cfg.kv_dim, cfg.d_model); w[p + "k_b"] = f(cfg.kv_dim)
        w[p + "v_w"] = f(cfg.kv_dim, cfg.d_model); w[p + "v_b"] = f(cfg.kv_dim)
        w[p + "o_w"] = f(cfg.d_model, cfg.q_dim)
        w[p + "post_ln"] = np.ones(cfg.d_model, np.float32)
        w[p + "gate_w"] = f(cfg.intermediate_size, cfg.d_model)
        w[p + "up_w"] = f(cfg.intermediate_size, cfg.d_model)
        w[p + "down_w"] = f(cfg.d_model, cfg.intermediate_size)
    w["final_ln"] = np.ones(cfg.d_model, np.float32)
    return w


def _teacher():
    return CUDATeacher.from_config(_tcfg(), seed=0)


def _student(precision="fp32"):
    get_backend("cuda").seed(0)
    return CUDAMambaModel(_scfg(precision))


def _tokens(B=2, L=8):
    return np.random.default_rng(0).integers(0, VOCAB, size=(B, L)).astype(np.int64)


def _opt(model, lr=1e-2):
    return get_backend("cuda").make_optimizer(model, lr)


# --- mixing matrix + KL ------------------------------------------------------
def test_mixing_matrix_reproduces_parallel_scan():
    cfg = MambaConfig(d_model=32, n_layers=1, d_state=16, head_dim=16, vocab_size=64,
                      seq_len=12, precision="fp32")
    ssm = SelectiveSSM(cfg)
    torch.manual_seed(0)
    with torch.no_grad():
        x = torch.randn(2, 12, cfg.d_inner)
        Y = ssm.parallel(x)
        M = ssm.mixing_matrix(x)
        Xh = x.reshape(2, 12, cfg.n_heads, cfg.head_dim)
        Ym = torch.einsum("bhij,bjhp->bihp", M, Xh).reshape(2, 12, cfg.d_inner)
        assert float((Y - Ym).abs().max()) < 1e-4


def test_kl_topk_nonneg_and_zero_at_match():
    rng = np.random.default_rng(1)
    logits = torch.tensor(rng.normal(size=(2, 4, VOCAB)).astype(np.float32))
    k = 8
    idx = torch.argsort(-logits, dim=-1)[..., :k]
    vals = torch.gather(logits, -1, idx)
    assert abs(float(_kl_topk(logits, vals, idx, 2.0))) < 1e-4               # student==teacher
    bumped = vals + torch.arange(k, dtype=torch.float32)
    assert float(_kl_topk(logits, bumped, idx, 2.0)) > 0


# --- stage steps -------------------------------------------------------------
def _logit_batch(teacher, tokens, k=16):
    vals, idx = teacher.topk_logits(tokens, k)
    targets = np.roll(tokens, -1, axis=1)
    return (tokens, targets, teacher.to_numpy(vals), teacher.to_numpy(idx))


def test_logit_distill_step_decreases():
    teacher, student = _teacher(), _student()
    step = make_distill_train_step(student, _opt(student), stage=DistillStage.LOGIT_DISTILL)
    mb = _logit_batch(teacher, _tokens())
    losses = [step(student, [mb], 1e-2)["loss"] for _ in range(4)]
    assert all(np.isfinite(losses)) and losses[-1] < losses[0]


def test_hidden_align_step_decreases():
    teacher = _teacher()
    student = _student()
    get_backend("cuda").init_student(student, teacher, InitMethod.MOHAWK)
    step = make_distill_train_step(student, _opt(student), stage=DistillStage.HIDDEN_ALIGN,
                                   teacher=teacher)
    losses = [step(student, [(_tokens(),)], 1e-2)["loss"] for _ in range(4)]
    assert all(np.isfinite(losses)) and losses[-1] < losses[0]


def test_mixing_match_step_decreases():
    teacher = _teacher()
    student = _student()
    get_backend("cuda").init_student(student, teacher, InitMethod.MOHAWK)
    step = make_distill_train_step(student, _opt(student), stage=DistillStage.MIXING_MATCH,
                                   teacher=teacher)
    losses = [step(student, [(_tokens(),)], 1e-2)["loss"] for _ in range(4)]
    assert all(np.isfinite(losses)) and losses[-1] < losses[0]


def test_mixing_match_requires_teacher():
    with pytest.raises(ValueError, match="requires a teacher"):
        make_distill_train_step(_student(), _opt(_student()), stage=DistillStage.MIXING_MATCH)


def test_mixing_match_rejects_pure_attention():
    cfg = MambaConfig(d_model=48, n_layers=2, d_state=16, head_dim=16, vocab_size=VOCAB,
                      seq_len=16, attn_every=1, n_attn_heads=4, precision="fp32")
    student = CUDAMambaModel(cfg)
    with pytest.raises(ValueError, match="no Mamba layers"):
        make_distill_train_step(student, _opt(student), stage=DistillStage.MIXING_MATCH,
                                teacher=_teacher())


# --- cross-backend loss parity ----------------------------------------------
@pytest.mark.skipif(not _have_mlx(), reason="mlx unavailable")
@pytest.mark.parametrize("stage", [DistillStage.LOGIT_DISTILL, DistillStage.HIDDEN_ALIGN,
                                   DistillStage.MIXING_MATCH])
def test_loss_parity_vs_mlx(stage):
    import mlx.core as mx
    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_teacher import MLXConversionTeacher
    from src.model.mlx_distill import make_distill_train_step as mlx_step

    tcfg, scfg = _tcfg(), _scfg()
    npw = _numpy_teacher_weights(tcfg, seed=7)
    ct = CUDATeacher(tcfg, {k: torch.tensor(v) for k, v in npw.items()})
    mt = MLXConversionTeacher(tcfg, {k: mx.array(v) for k, v in npw.items()})

    ms = MLXMambaModel(scfg)
    cs = CUDAMambaModel(scfg)
    cs._load_portable(ms._portable_state_dict())          # identical student weights

    tokens = _tokens()
    teacher_arg = None if stage == DistillStage.LOGIT_DISTILL else (ct, mt)
    if stage == DistillStage.LOGIT_DISTILL:
        # one cached top-k batch (from the shared teacher) feeds both backends.
        vals, idx = ct.topk_logits(tokens, 16)
        mb = (tokens, np.roll(tokens, -1, axis=1), ct.to_numpy(vals), ct.to_numpy(idx))
        c_step = make_distill_train_step(cs, _opt(cs), stage=stage)
        m_step = mlx_step(ms, get_backend("mlx").make_optimizer(ms, 1e-2), stage=stage)
    else:
        mb = (tokens,)
        c_step = make_distill_train_step(cs, _opt(cs), stage=stage, teacher=ct)
        m_step = mlx_step(ms, get_backend("mlx").make_optimizer(ms, 1e-2), stage=stage, teacher=mt)

    # lr=0 -> weights unchanged; compare the loss the two backends compute on identical inputs.
    c_loss = c_step(cs, [mb], 0.0)["loss"]
    m_loss = m_step(ms, [mb], 0.0)["loss"]
    assert abs(c_loss - m_loss) < 1e-3, f"{stage}: cuda {c_loss} vs mlx {m_loss}"
