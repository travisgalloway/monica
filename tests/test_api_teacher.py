"""Offline tests for the top-k-logit-only endpoint teacher (`src.model.api_teacher`).

Portable (no mlx/torch). The HTTP call and the tokenizer are injected as fakes, so these run on
any host with no server and no HF download.
"""

import numpy as np
import pytest

from src.model.api_teacher import ApiTopkTeacher


class _FakeTokenizer:
    """Tiny char-keyed tokenizer: token strings 'a'..'z' map to ids 0..25; decode is a no-op."""
    unk_token_id = -1

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def convert_tokens_to_ids(self, tok):
        return ord(tok) - ord("a") if len(tok) == 1 and "a" <= tok <= "z" else -1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) - ord("a") for c in text if "a" <= c <= "z"]


def _fake_server(top_per_pos):
    """Build a `_post` returning an OpenAI completions response with the given top_logprobs."""
    def _post(url, payload, timeout):
        return {"choices": [{"logprobs": {"top_logprobs": top_per_pos}}]}
    return _post


def _teacher(top_per_pos, vocab_size=26):
    return ApiTopkTeacher(base_url="http://localhost:9/v1", vocab_size=vocab_size,
                          _post=_fake_server(top_per_pos), _tokenizer=_FakeTokenizer())


def test_topk_shapes_and_descending_values():
    top = [{"b": -0.1, "a": -2.0, "c": -3.0}, {"a": -0.5, "d": -1.5, "z": -4.0}]
    t = _teacher(top)
    vals, idx = t.topk_logits(np.array([[10, 11]]), k=3)
    assert vals.shape == (1, 2, 3) and idx.shape == (1, 2, 3)
    # position 0: b(1) > a(0) > c(2)
    assert list(idx[0, 0]) == [1, 0, 2]
    assert np.all(np.diff(vals[0, 0]) <= 0)        # descending
    # position 1: a(0) > d(3) > z(25)
    assert list(idx[0, 1]) == [0, 3, 25]


def test_capped_k_pads_with_neg_inf():
    top = [{"a": -0.1, "b": -1.0}]                  # only 2 entries, request k=4
    vals, idx = _teacher(top).topk_logits(np.array([[5]]), k=4)
    assert vals.shape == (1, 1, 4)
    assert vals[0, 0, 0] > vals[0, 0, 2]
    assert vals[0, 0, 2] == pytest.approx(np.finfo(np.float32).min)   # padding
    assert vals[0, 0, 3] == pytest.approx(np.finfo(np.float32).min)


def test_ids_outside_vocab_dropped():
    # vocab_size=2 -> only ids 0,1 are representable; 'c'(2) must be dropped.
    top = [{"a": -0.1, "c": -0.2, "b": -0.3}]
    vals, idx = _teacher(top, vocab_size=2).topk_logits(np.array([[0]]), k=3)
    assert list(idx[0, 0])[:2] == [0, 1]           # a, b kept; c dropped
    assert vals[0, 0, 2] == pytest.approx(np.finfo(np.float32).min)


def test_position_misalignment_pads_to_seq_len():
    top = [{"a": -0.1}]                              # server returns 1 position, seq_len 3
    vals, idx = _teacher(top).topk_logits(np.array([[1, 2, 3]]), k=2)
    assert vals.shape == (1, 3, 2)
    assert vals[0, 0, 0] > np.finfo(np.float32).min     # position 0 filled
    assert vals[0, 1, 0] == pytest.approx(np.finfo(np.float32).min)   # 1,2 padded
    assert vals[0, 2, 0] == pytest.approx(np.finfo(np.float32).min)


def test_white_box_methods_raise():
    t = _teacher([{"a": -0.1}])
    for call in (lambda: t.forward(np.array([[1]])),
                 lambda: t.attention_projection(0),
                 lambda: t.embedding_matrix(),
                 lambda: t.lm_head_matrix()):
        with pytest.raises(NotImplementedError):
            call()


def test_rejects_non_qwen3_tokenizer_name():
    with pytest.raises(ValueError):
        ApiTopkTeacher(base_url="http://x/v1", vocab_size=4, tokenizer="olmo")
