"""SFT record builder (portable). Verifies the response-only loss mask, next-token
shift, EOS training, vocab guard, and length skipping — all offline with ByteTokenizer
(1 byte = 1 token, so masks are exact char ranges)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.sft_data import build_sft_records
from src.data.tokenize import ByteTokenizer


class ByteTokEOS(ByteTokenizer):
    """ByteTokenizer with an EOS id, to exercise stop-token training."""
    eos_token_id = 255


def _records(rows, tok, **kw):
    stats = {}
    out = list(build_sft_records(rows, tok, stats=stats, **kw))
    return out, stats


def _decode_on(tok, rec, key="target_ids"):
    """Decode the tokens whose loss_mask is 1 (the trained targets)."""
    ids = [t for t, m in zip(rec[key], rec["loss_mask"]) if m]
    return tok.decode(ids)


def test_shapes_and_next_token_shift():
    rows = [{"messages": [{"role": "user", "content": "Hi"},
                          {"role": "assistant", "content": "Hello"}]}]
    (rec,), _ = _records(rows, ByteTokenizer())
    n = len(rec["input_ids"])
    assert len(rec["target_ids"]) == n == len(rec["loss_mask"])
    # target is input shifted by one (same underlying token sequence).
    assert rec["target_ids"][:-1] == rec["input_ids"][1:]


def test_mask_covers_only_assistant_response():
    rows = [{"messages": [{"role": "user", "content": "Hi"},
                          {"role": "assistant", "content": "Hello"}]}]
    (rec,), _ = _records(rows, ByteTokenizer())
    # The trained targets decode exactly to the response (ByteTokenizer has no EOS here).
    assert _decode_on(ByteTokenizer(), rec).strip() == "Hello"
    # The prompt marker/user text is never a trained target.
    assert "Hi" not in _decode_on(ByteTokenizer(), rec)


def test_eos_is_trained_as_final_target():
    rows = [{"messages": [{"role": "user", "content": "Hi"},
                          {"role": "assistant", "content": "Hello"}]}]
    (rec,), _ = _records(rows, ByteTokEOS())
    assert rec["target_ids"][-1] == 255          # last target is EOS
    assert rec["loss_mask"][-1] == 1             # and it is trained (model learns to stop)
    # The trained content tokens (everything before the EOS) decode to the response.
    content = [t for t, m in zip(rec["target_ids"], rec["loss_mask"]) if m and t != 255]
    assert ByteTokEOS().decode(content).strip() == "Hello"


def test_multi_turn_masks_both_assistant_turns_only():
    rows = [{"messages": [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "Bee"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "Dee"},
    ]}]
    (rec,), _ = _records(rows, ByteTokenizer())
    trained = _decode_on(ByteTokenizer(), rec)
    assert "Bee" in trained and "Dee" in trained
    assert "A" not in trained and "C" not in trained


def test_skips_examples_without_assistant():
    rows = [{"messages": [{"role": "user", "content": "Only a question"}]}]
    out, stats = _records(rows, ByteTokenizer())
    assert out == [] and stats["skipped"] == 1


def test_skips_over_length_examples():
    rows = [{"messages": [{"role": "user", "content": "Hi"},
                          {"role": "assistant", "content": "x" * 50}]}]
    out, stats = _records(rows, ByteTokenizer(), max_seq_len=10)
    assert out == [] and stats["skipped"] == 1


def test_dolly_fallback_fields():
    rows = [{"instruction": "Add", "response": "4", "context": "2+2"}]
    (rec,), _ = _records(rows, ByteTokenizer())
    assert _decode_on(ByteTokenizer(), rec).strip() == "4"


def test_vocab_guard_raises_on_out_of_range_token():
    class TinyVocab(ByteTokenizer):
        vocab_size = 32  # 'H'=72 exceeds this
    rows = [{"messages": [{"role": "user", "content": "Hi"},
                          {"role": "assistant", "content": "Hello"}]}]
    with pytest.raises(ValueError):
        list(build_sft_records(rows, TinyVocab()))
