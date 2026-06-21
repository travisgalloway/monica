"""CUDA/torch student init from a teacher (#99). Torch CPU; parity vs MLX where available.

Covers the adaptive `_fit`/`_expand_kv` helpers, embedding transfer, the Mamba-in-the-Llama
freeze (attention layers -> requires_grad False) vs MOHAWK (nothing frozen), the InitReport
counts, and full cross-backend parity (identical teacher + student start -> identical init).
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import MambaConfig
from src.model.teacher import TeacherConfig
from src.model.cuda_backend import CUDAMambaModel
from src.model.cuda_teacher import CUDATeacher
from src.model.cuda_student_init import init_student, _fit, _expand_kv
from src.train.distill_manifest import InitMethod

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


def _student_cfg():
    return MambaConfig(d_model=48, n_layers=4, d_state=16, expand=2, d_conv=4, head_dim=16,
                       vocab_size=VOCAB, seq_len=16, attn_every=2, n_attn_heads=4, precision="fp32")


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


# --- helpers -----------------------------------------------------------------
def test_fit_crop_and_pad():
    src = torch.arange(12).reshape(3, 4).float()
    out = _fit(src, (2, 6))                       # crop rows 3->2, pad cols 4->6
    assert out.shape == (2, 6)
    assert torch.equal(out[:, :4], src[:2])
    assert torch.all(out[:, 4:] == 0)
    assert torch.equal(_fit(src, (3, 4)), src)    # exact when shapes match


def test_expand_kv():
    w = torch.arange(2 * 16 * 5).reshape(2 * 16, 5).float()   # 2 kv heads, head_dim 16
    out = _expand_kv(w, n_heads=4, n_kv_heads=2, head_dim=16)
    assert out.shape == (4 * 16, 5)
    # each kv head's 16-row block is repeated `rep=2` times
    assert torch.equal(out[:16], w[:16]) and torch.equal(out[16:32], w[:16])


# --- init behavior -----------------------------------------------------------
def test_embedding_transfer():
    teacher = CUDATeacher.from_config(_tcfg(), seed=0)
    student = CUDAMambaModel(_student_cfg())
    init_student(student, teacher, InitMethod.MOHAWK)
    te = teacher.to_numpy(teacher.embedding_matrix())
    se = student.embedding.weight.detach().numpy()
    m = min(te.shape[1], se.shape[1])
    assert np.allclose(se[:, :m], te[:VOCAB, :m], atol=1e-5)


def test_mamba_in_the_llama_freezes_attention():
    cfg = _student_cfg()
    teacher = CUDATeacher.from_config(_tcfg(), seed=0)
    student = CUDAMambaModel(cfg)
    report = init_student(student, teacher, InitMethod.MAMBA_IN_THE_LLAMA)
    attn_layers = [i for i in range(cfg.n_layers) if cfg.is_attention_layer(i)]
    assert report.frozen_layers == attn_layers
    for i in attn_layers:
        assert all(not p.requires_grad for p in student.layers[i].parameters())
    for i in range(cfg.n_layers):
        if i not in attn_layers:
            assert all(p.requires_grad for p in student.layers[i].parameters())
    assert report.n_frozen_params + report.n_trainable_params == sum(
        p.numel() for p in student.parameters())
    assert report.n_frozen_params > 0


def test_mohawk_freezes_nothing():
    student = CUDAMambaModel(_student_cfg())
    report = init_student(student, CUDATeacher.from_config(_tcfg(), seed=0), InitMethod.MOHAWK)
    assert report.frozen_layers == []
    assert report.n_frozen_params == 0
    assert all(p.requires_grad for p in student.parameters())


@pytest.mark.skipif(not _have_mlx(), reason="mlx unavailable")
@pytest.mark.parametrize("method", [InitMethod.MAMBA_IN_THE_LLAMA, InitMethod.MOHAWK])
def test_parity_vs_mlx(method):
    import mlx.core as mx
    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_teacher import MLXConversionTeacher
    from src.model.mlx_student_init import init_student as mlx_init

    tcfg, scfg = _tcfg(), _student_cfg()
    npw = _numpy_teacher_weights(tcfg, seed=5)

    # Identical teachers from the shared numpy weights.
    ct = CUDATeacher(tcfg, {k: torch.tensor(v) for k, v in npw.items()})
    mt = MLXConversionTeacher(tcfg, {k: mx.array(v) for k, v in npw.items()})

    # Identical students: build both, then load the MLX student's portable weights into the torch one.
    ms = MLXMambaModel(scfg)
    cs = CUDAMambaModel(scfg)
    cs._load_portable(ms._portable_state_dict())

    mlx_init(ms, mt, method)
    init_student(cs, ct, method)

    pm, pc = ms._portable_state_dict(), cs._portable_state_dict()
    assert set(pm) == set(pc)
    for k in pm:
        assert np.max(np.abs(np.asarray(pm[k]) - np.asarray(pc[k]))) < 1e-4, f"mismatch at {k}"
