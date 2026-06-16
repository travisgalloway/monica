"""Conversion-teacher loader (#93). MLX-only; skipped where mlx is unavailable.

Exercises the frozen, forward-only teacher behind the seam with the tiny synthetic
teacher (offline): output shapes (logits / hidden states / top-k), the Q/K/V/O
projection accessor the #99 init consumes, the frozen contract (no trainable params,
no gradient flow), determinism, and the real HF-safetensors loader path via an
offline synthetic checkpoint round-trip.
"""

import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from src.model.teacher import TeacherConfig
from src.model.mlx_teacher import MLXConversionTeacher


def _tiny():
    return MLXConversionTeacher.from_config(TeacherConfig.tiny(), seed=0)


def _tokens(B, L, vocab):
    rng = np.random.default_rng(0)
    return rng.integers(0, vocab, size=(B, L)).astype(np.int32)


# --- config ------------------------------------------------------------------
def test_qwen_1_5b_config_shape():
    c = TeacherConfig.qwen_1_5b()
    assert (c.vocab_size, c.d_model, c.n_layers) == (151936, 1536, 28)
    assert (c.n_heads, c.n_kv_heads, c.head_dim) == (12, 2, 128)
    assert c.intermediate_size == 8960 and c.tie_embeddings
    assert c.q_dim == 1536 and c.kv_dim == 256
    assert c.model_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    c.validate()


def test_config_validate_rejects_bad_gqa():
    with pytest.raises(ValueError):
        TeacherConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=4, n_kv_heads=3,
                      head_dim=4, intermediate_size=16).validate()


# --- forward / shapes --------------------------------------------------------
def test_forward_logits_shape():
    t = _tiny()
    c = t.config
    out = t.forward(_tokens(2, 5, c.vocab_size))
    assert out.logits.shape == (2, 5, c.vocab_size)
    assert out.hidden_states is None
    assert mx.all(mx.isfinite(out.logits)).item()


def test_forward_hidden_states():
    t = _tiny()
    c = t.config
    out = t.forward(_tokens(2, 5, c.vocab_size), return_hidden=True)
    assert len(out.hidden_states) == c.n_layers + 1        # embedding + each layer
    for h in out.hidden_states:
        assert h.shape == (2, 5, c.d_model)


def test_topk_logits():
    t = _tiny()
    c = t.config
    k = 8
    vals, idx = t.topk_logits(_tokens(2, 5, c.vocab_size), k)
    assert vals.shape == (2, 5, k) and idx.shape == (2, 5, k)
    # descending values, indices in range, and equal to a gather of the full logits
    diffs = vals[..., :-1] - vals[..., 1:]
    assert mx.all(diffs >= -1e-5).item()
    assert mx.all(idx >= 0).item() and mx.all(idx < c.vocab_size).item()
    full = t.forward(_tokens(2, 5, c.vocab_size)).logits
    gathered = mx.take_along_axis(full, idx, axis=-1)
    assert mx.allclose(gathered, vals, atol=1e-5).item()


# --- projection accessor (#99) ----------------------------------------------
def test_attention_projection_shapes_gqa():
    t = _tiny()
    c = t.config
    pr = t.attention_projection(0)
    assert pr.q.shape == (c.q_dim, c.d_model)
    assert pr.k.shape == (c.kv_dim, c.d_model)
    assert pr.v.shape == (c.kv_dim, c.d_model)
    assert pr.o.shape == (c.d_model, c.q_dim)
    assert c.n_kv_heads < c.n_heads                        # exercising the GQA path
    assert pr.q_bias.shape == (c.q_dim,)
    assert pr.k_bias.shape == (c.kv_dim,) and pr.v_bias.shape == (c.kv_dim,)


# --- frozen contract ---------------------------------------------------------
def test_no_trainable_parameters():
    assert _tiny().trainable_parameters() == {}


def test_no_gradient_flows_to_teacher():
    """A loss on the teacher's logits must produce zero gradient w.r.t. its weights:
    forward wraps its outputs in stop_gradient, so the teacher stays frozen even when
    composed into a student objective."""
    t = _tiny()
    toks = _tokens(1, 4, t.config.vocab_size)
    embed = t._w["embed"]

    def loss(w):
        t._w["embed"] = w
        return t.forward(toks).logits.sum()

    g = mx.grad(loss)(embed)
    t._w["embed"] = embed                                  # restore
    assert mx.all(g == 0).item()


def test_deterministic_forward():
    t = _tiny()
    toks = _tokens(2, 6, t.config.vocab_size)
    a = t.forward(toks).logits
    b = t.forward(toks).logits
    assert mx.array_equal(a, b).item()


# --- real HF loader path (offline synthetic checkpoint) ----------------------
def _write_hf_checkpoint(t: MLXConversionTeacher, path):
    """Serialize a teacher to an HF-named safetensors checkpoint + config.json, so the
    `from_pretrained` name-mapping is exercised without a multi-GB download."""
    c = t.config
    w = t._w
    hf = {"model.embed_tokens.weight": w["embed"], "model.norm.weight": w["final_ln"]}
    for i in range(c.n_layers):
        ip, p = f"model.layers.{i}.", f"layer.{i}."
        hf[ip + "input_layernorm.weight"] = w[p + "input_ln"]
        hf[ip + "post_attention_layernorm.weight"] = w[p + "post_ln"]
        for proj, src in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
            hf[ip + f"self_attn.{proj}.weight"] = w[p + src + "_w"]
            hf[ip + f"self_attn.{proj}.bias"] = w[p + src + "_b"]
        hf[ip + "self_attn.o_proj.weight"] = w[p + "o_w"]
        hf[ip + "mlp.gate_proj.weight"] = w[p + "gate_w"]
        hf[ip + "mlp.up_proj.weight"] = w[p + "up_w"]
        hf[ip + "mlp.down_proj.weight"] = w[p + "down_w"]
    path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path / "model.safetensors"), hf)
    hf_cfg = {
        "vocab_size": c.vocab_size, "hidden_size": c.d_model,
        "num_hidden_layers": c.n_layers, "num_attention_heads": c.n_heads,
        "num_key_value_heads": c.n_kv_heads, "head_dim": c.head_dim,
        "intermediate_size": c.intermediate_size, "rms_norm_eps": c.rms_norm_eps,
        "rope_theta": c.rope_theta, "tie_word_embeddings": c.tie_embeddings,
    }
    (path / "config.json").write_text(json.dumps(hf_cfg))


def test_from_pretrained_roundtrip(tmp_path):
    src = MLXConversionTeacher.from_config(TeacherConfig.tiny(), seed=3)
    _write_hf_checkpoint(src, tmp_path / "ckpt")
    loaded = MLXConversionTeacher.from_pretrained(tmp_path / "ckpt")
    # config recovered from config.json
    assert loaded.config.vocab_size == src.config.vocab_size
    assert loaded.config.n_layers == src.config.n_layers
    # identical logits => name-mapping + IO are correct
    toks = _tokens(2, 5, src.config.vocab_size)
    assert mx.allclose(loaded.forward(toks).logits, src.forward(toks).logits,
                       atol=1e-5).item()
