"""CUDA/torch conversion teacher (#94). Torch CPU; parity vs MLX where mlx is available.

The teacher runs on CPU (no CUDA needed), so forward shapes / top-k / the frozen contract are
testable anywhere torch is installed. The cross-backend parity test (built from one shared numpy
weight dict) guards that the torch port agrees with the MLX teacher at fp32 — the same standard
as src/conformance/backend_parity.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.teacher import TeacherConfig
from src.model.cuda_teacher import CUDATeacher


def _cfg():
    return TeacherConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                         head_dim=16, intermediate_size=128, tokenizer_vocab_size=200)


def _numpy_weights(cfg, seed=0):
    """Synthetic weights in the shared `mlx_teacher`/`cuda_teacher` dict layout, so both
    backends can be built from the SAME arrays for parity."""
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(cfg.d_model)
    f = lambda *s: (rng.standard_normal(s) * scale).astype(np.float32)
    w = {"embed": f(cfg.vocab_size, cfg.d_model)}
    for i in range(cfg.n_layers):
        p = f"layer.{i}."
        w[p + "input_ln"] = np.ones(cfg.d_model, np.float32)
        # Qwen3: no QKV bias; per-head Q/K RMSNorm weights (random, to exercise the norm).
        w[p + "q_w"] = f(cfg.q_dim, cfg.d_model); w[p + "q_norm"] = f(cfg.head_dim)
        w[p + "k_w"] = f(cfg.kv_dim, cfg.d_model); w[p + "k_norm"] = f(cfg.head_dim)
        w[p + "v_w"] = f(cfg.kv_dim, cfg.d_model)
        w[p + "o_w"] = f(cfg.d_model, cfg.q_dim)
        w[p + "post_ln"] = np.ones(cfg.d_model, np.float32)
        w[p + "gate_w"] = f(cfg.intermediate_size, cfg.d_model)
        w[p + "up_w"] = f(cfg.intermediate_size, cfg.d_model)
        w[p + "down_w"] = f(cfg.d_model, cfg.intermediate_size)
    w["final_ln"] = np.ones(cfg.d_model, np.float32)
    return w


def _tokens(B=2, L=8, vocab=200):
    return np.random.default_rng(1).integers(0, vocab, size=(B, L)).astype(np.int64)


def test_forward_shapes_and_effective_vocab():
    cfg = _cfg()
    teacher = CUDATeacher.from_config(cfg, seed=0)
    out = teacher.forward(_tokens(), return_hidden=True)
    assert out.logits.shape == (2, 8, cfg.effective_vocab_size)   # sliced to tokenizer vocab
    assert len(out.hidden_states) == cfg.n_layers + 1
    assert out.logits.requires_grad is False                      # frozen contract


def test_topk_descending_and_in_range():
    cfg = _cfg()
    teacher = CUDATeacher.from_config(cfg, seed=0)
    vals, idx = teacher.topk_logits(_tokens(), k=5)
    assert vals.shape == (2, 8, 5) and idx.shape == (2, 8, 5)
    v = teacher.to_numpy(vals)
    assert np.all(np.diff(v, axis=-1) <= 1e-5)                    # descending
    i = teacher.to_numpy(idx)
    assert i.min() >= 0 and i.max() < cfg.effective_vocab_size


def test_no_trainable_parameters():
    teacher = CUDATeacher.from_config(_cfg(), seed=0)
    assert teacher.trainable_parameters() == {}


def _have_mlx():
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_mlx(), reason="mlx unavailable")
def test_parity_vs_mlx_fp32():
    import mlx.core as mx
    from src.model.mlx_teacher import MLXConversionTeacher

    cfg = _cfg()
    npw = _numpy_weights(cfg, seed=3)
    cuda = CUDATeacher(cfg, {k: torch.tensor(v) for k, v in npw.items()})
    mlx_t = MLXConversionTeacher(cfg, {k: mx.array(v) for k, v in npw.items()})

    tokens = _tokens()
    lc = cuda.to_numpy(cuda.forward(tokens).logits)
    lm = np.array(mlx_t.forward(tokens).logits)
    assert lc.shape == lm.shape
    assert np.max(np.abs(lc - lm)) < 1e-3                          # fp32 cross-backend parity

    # top-k: same logits -> same indices and (close) values.
    vc, ic = cuda.topk_logits(tokens, 6)
    vm, im = mlx_t.topk_logits(tokens, 6)
    assert np.array_equal(cuda.to_numpy(ic), np.array(im))
    assert np.max(np.abs(cuda.to_numpy(vc) - np.array(vm))) < 1e-3
