"""Student init from a frozen teacher (#99). MLX-only; skipped where mlx is unavailable.

Exercises Mamba-in-the-Llama + MOHAWK with a synthetic teacher and a tiny hybrid student:
exact Q/K/V/O copy when dims align, the adaptive-fit path when the student is wider, the
frozen/trainable split, and an end-to-end manifest -> config -> build -> init -> forward.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten

from src.model.mlx_backend import MLXMambaModel
from src.model.blocks import MambaConfig
from src.model.mlx_teacher import MLXConversionTeacher
from src.model.teacher import TeacherConfig
from src.model.mlx_student_init import init_student
from src.train.distill_manifest import InitMethod, InitReport


def _teacher(d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16):
    cfg = TeacherConfig(vocab_size=256, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        n_kv_heads=n_kv_heads, head_dim=head_dim, intermediate_size=128)
    return MLXConversionTeacher.from_config(cfg, seed=0)


def _student(d_model=64, n_attn_heads=4):
    cfg = MambaConfig(d_model=d_model, n_layers=4, d_state=16, expand=2, d_conv=4, head_dim=16,
                      vocab_size=256, seq_len=32, attn_every=2, n_attn_heads=n_attn_heads,
                      precision="fp32")
    return MLXMambaModel(cfg)


def _total(model):
    return sum(v.size for _, v in tree_flatten(model.parameters()))


# --- Mamba-in-the-Llama: exact copy on matched dims --------------------------
def test_mil_attention_exact_copy_when_dims_match():
    teacher, student = _teacher(), _student()        # student d_attn 64 == teacher q_dim 64
    rep = init_student(student, teacher, InitMethod.MAMBA_IN_THE_LLAMA)
    assert isinstance(rep, InitReport) and rep.method == "mamba-in-the-llama"
    # layer 1 is an attention layer (attn_every 2); teacher_layer_for(1,4,4) == 1.
    proj = teacher.attention_projection(1)
    qkv = student.layers[1].qkv_proj.weight          # (3*d_attn, d_model)
    assert mx.allclose(qkv[:64], proj.q).item()      # Q block exact
    assert mx.allclose(student.layers[1].o_proj.weight, proj.o).item()


def test_mil_mamba_layer_gets_teacher_projection():
    teacher, student = _teacher(), _student()
    proj = teacher.attention_projection(0)           # layer 0 is a Mamba layer
    init_student(student, teacher, InitMethod.MAMBA_IN_THE_LLAMA)
    cfg = student.config
    dt_rank, N = cfg.dt_rank_resolved, cfg.d_state
    xw = student.layers[0].ssm.x_proj.weight
    # C slice (rows dt_rank+N : dt_rank+2N) is Q fitted into (N, d_inner): top-left overlaps Q.
    C = xw[dt_rank + N:dt_rank + 2 * N]
    assert mx.allclose(C[:, :proj.q.shape[1]][:N], proj.q[:N]).item()


# --- adaptive fit when the student is wider than the teacher -----------------
def test_mil_adaptive_fit_wider_student_shapes_finite():
    teacher = _teacher(d_model=64)
    student = _student(d_model=128, n_attn_heads=8)   # wider: d_attn 128 > teacher q_dim 64
    before = {k: v.shape for k, v in tree_flatten(student.parameters())}
    init_student(student, teacher, InitMethod.MAMBA_IN_THE_LLAMA)
    after = {k: v.shape for k, v in tree_flatten(student.parameters())}
    assert before == after                            # every param keeps its correct shape
    for _, v in tree_flatten(student.parameters()):
        assert mx.all(mx.isfinite(v)).item()
    assert student.forward(mx.zeros((1, 8), dtype=mx.int32)).shape == (1, 8, 256)


# --- freeze gating -----------------------------------------------------------
def test_mil_freezes_kept_attention_layers():
    teacher, student = _teacher(), _student()
    rep = init_student(student, teacher, InitMethod.MAMBA_IN_THE_LLAMA)
    assert rep.frozen_layers == [1, 3]                # the attention layers (attn_every 2)
    trainable = {k for k, _ in tree_flatten(student.trainable_parameters())}
    assert not any(k.startswith("layers.1.") for k in trainable)   # frozen
    assert any(k.startswith("layers.0.") for k in trainable)       # Mamba trainable
    assert rep.n_frozen_params > 0
    assert rep.n_frozen_params + rep.n_trainable_params == _total(student)


def test_mil_no_grad_to_frozen_layers():
    teacher, student = _teacher(), _student()
    init_student(student, teacher, InitMethod.MAMBA_IN_THE_LLAMA)

    def loss(model):
        return model.forward(mx.zeros((1, 6), dtype=mx.int32)).sum()

    import mlx.nn as nn
    grads = nn.value_and_grad(student, loss)(student)[1]
    flat = dict(tree_flatten(grads))
    # frozen attention layers contribute no gradient entries; Mamba layers do.
    assert not any(k.startswith("layers.1.") for k in flat)
    assert any(k.startswith("layers.0.") for k in flat)


# --- MOHAWK ------------------------------------------------------------------
def test_mohawk_freezes_nothing():
    teacher, student = _teacher(), _student()
    rep = init_student(student, teacher, InitMethod.MOHAWK)
    assert rep.method == "mohawk" and rep.frozen_layers == []
    assert rep.n_frozen_params == 0
    assert rep.n_trainable_params == _total(student)
    # attention layers are still copied from the teacher
    proj = teacher.attention_projection(1)
    assert mx.allclose(student.layers[1].qkv_proj.weight[:64], proj.q).item()


# --- end to end via the manifest resolver ------------------------------------
def test_end_to_end_manifest_to_student():
    from src.train.distill_manifest import DistillManifest, manifest_to_config
    man = DistillManifest(
        student="tiny", conversion_teacher="t", tokenizer="qwen25", seq_len=32,
        init=InitMethod.MAMBA_IN_THE_LLAMA, stages=["mixing-match"],
        layout={"d_model": 64, "n_layers": 4, "attention_every": 2, "state_size": 16,
                "head_dim": 16, "n_attn_heads": 4},
    )
    cfg = manifest_to_config(man)
    cfg.vocab_size = 256                              # shrink vocab for the offline test
    cfg.precision, cfg.grad_checkpoint = "fp32", False
    student = MLXMambaModel(cfg)
    teacher = _teacher()
    rep = init_student(student, teacher, man.init)
    assert rep.n_layers_mapped == cfg.n_layers
    assert mx.all(mx.isfinite(student.forward(mx.zeros((1, 8), dtype=mx.int32)))).item()
