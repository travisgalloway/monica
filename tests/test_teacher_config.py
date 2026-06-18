"""Portable `TeacherConfig` tests (#93). NO backend import — `teacher.py` is above the
seam, so these run on non-Apple/CI hosts where the MLX-guarded `test_teacher.py` is skipped,
and they actually protect the conversion-teacher fixtures (`openr1_distill_7b`, `qwen_1_5b`)
and the cross-cutting `validate()` invariants.
"""

import pytest

from src.model.teacher import TeacherConfig


def test_openr1_distill_7b_config_shape():
    c = TeacherConfig.openr1_distill_7b()
    assert (c.vocab_size, c.d_model, c.n_layers) == (152064, 3584, 28)
    assert (c.n_heads, c.n_kv_heads, c.head_dim) == (28, 4, 128)
    assert c.intermediate_size == 18944 and not c.tie_embeddings
    assert c.rope_theta == 300000.0          # Open-R1 extends RoPE base for 32k context
    assert c.q_dim == 3584 and c.kv_dim == 512
    assert c.model_id == "open-r1/OpenR1-Distill-7B"
    assert c.tokenizer_vocab_size == 151646 and c.effective_vocab_size == 151646
    c.validate()


def test_qwen_1_5b_config_shape():
    c = TeacherConfig.qwen_1_5b()
    assert (c.vocab_size, c.d_model, c.n_layers) == (151936, 1536, 28)
    assert (c.n_heads, c.n_kv_heads, c.head_dim) == (12, 2, 128)
    assert c.intermediate_size == 8960 and c.tie_embeddings
    assert c.q_dim == 1536 and c.kv_dim == 256
    assert c.model_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    assert c.tokenizer_vocab_size == 151646 and c.effective_vocab_size == 151646
    c.validate()


def test_from_hf_dict_omits_tokenizer_vocab():
    # HF config.json has no tokenizer_vocab_size, so a config built from it alone emits over the
    # full (padded) model vocab — the gotcha openr1_distill_7b's docstring warns about.
    c = TeacherConfig.from_hf_dict({
        "vocab_size": 152064, "hidden_size": 3584, "num_hidden_layers": 28,
        "num_attention_heads": 28, "num_key_value_heads": 4, "head_dim": 128,
        "intermediate_size": 18944, "tie_word_embeddings": False,
    })
    assert c.tokenizer_vocab_size is None
    assert c.effective_vocab_size == c.vocab_size == 152064


def test_config_validate_rejects_bad_gqa():
    with pytest.raises(ValueError):
        TeacherConfig(vocab_size=8, d_model=16, n_layers=1, n_heads=4, n_kv_heads=3,
                      head_dim=4, intermediate_size=16).validate()


def test_config_rejects_tokenizer_vocab_above_model_vocab():
    with pytest.raises(ValueError):
        TeacherConfig(vocab_size=256, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                      head_dim=8, intermediate_size=16, tokenizer_vocab_size=300).validate()
