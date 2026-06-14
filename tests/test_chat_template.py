"""Multi-turn chat template + assistant response spans (portable, no backend).

Byte-identity with the pretraining format is the load-bearing property: the SFT/chat
format must match what the base model already learned, or post-training fights the
prior. Response spans are verified with `ByteTokenizer` (1 byte = 1 token, so token
indices are exact) and, when available, against the real OLMo BPE to guard the
prefix-stability assumption.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.instruct_format import (
    format_example,
    format_prompt,
    render,
    response_spans,
)
from src.data.tokenize import ByteTokenizer


# --- render byte-identity with the pretraining format -------------------------------

def test_render_matches_format_example_single_turn():
    i, r = "Translate hello to French", "Bonjour"
    msgs = [{"role": "user", "content": i}, {"role": "assistant", "content": r}]
    assert render(msgs) == format_example(i, r)


def test_render_trailing_empty_assistant_is_format_prompt():
    i = "Translate hello to French"
    msgs = [{"role": "user", "content": i}, {"role": "assistant", "content": ""}]
    assert render(msgs) == format_prompt(i)


def test_render_system_turn_is_newline_free_and_ordered():
    s = render([
        {"role": "system", "content": "Be terse"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ])
    assert "\n" not in s
    assert s == "### System: Be terse ### Instruction: Hi ### Response: Hello"


def test_render_strips_content():
    msgs = [{"role": "user", "content": "  spaced  "},
            {"role": "assistant", "content": "  ok  "}]
    assert render(msgs) == "### Instruction: spaced ### Response: ok"


# --- response spans (ByteTokenizer: exact char offsets) -----------------------------

def _mask(full_ids, spans):
    m = np.zeros(len(full_ids), dtype=np.int64)
    for s, e in spans:
        m[s:e] = 1
    return m


def test_span_covers_only_assistant_content():
    tok = ByteTokenizer()
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    full_ids, spans = response_spans(msgs, tok)
    assert len(spans) == 1
    s, e = spans[0]
    assert tok.decode(full_ids[s:e]).strip() == "Hello"


def test_mask_is_one_only_on_assistant_tokens():
    tok = ByteTokenizer()
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    full_ids, spans = response_spans(msgs, tok)
    mask = _mask(full_ids, spans)
    on = [full_ids[i] for i in range(len(full_ids)) if mask[i]]
    off = [full_ids[i] for i in range(len(full_ids)) if not mask[i]]
    assert tok.decode(on).strip() == "Hello"
    assert "Hello" not in tok.decode(off)  # markers + user text only, never the answer


def test_full_ids_reconstruct_the_render():
    tok = ByteTokenizer()
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    full_ids, _ = response_spans(msgs, tok)
    assert tok.decode(full_ids) == render(msgs)


def test_multi_turn_yields_one_span_per_assistant_turn():
    tok = ByteTokenizer()
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "Bee"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "Dee"},
    ]
    full_ids, spans = response_spans(msgs, tok)
    assert len(spans) == 2
    assert tok.decode(full_ids[spans[0][0]:spans[0][1]]).strip() == "Bee"
    assert tok.decode(full_ids[spans[1][0]:spans[1][1]]).strip() == "Dee"


def test_trailing_empty_assistant_has_no_span():
    tok = ByteTokenizer()
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}]
    _, spans = response_spans(msgs, tok)
    assert spans == []


# --- OLMo BPE: guard prefix-stability when the tokenizer is reachable ----------------

def test_olmo_spans_reconstruct_and_are_prefix_stable():
    try:
        from src.data.tokenize import load_olmo_tokenizer
        tok = load_olmo_tokenizer()
    except Exception:  # offline / transformers missing
        pytest.skip("OLMo tokenizer unavailable")
    answer = "The capital of France is Paris."
    msgs = [{"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": answer}]
    full_ids, spans = response_spans(msgs, tok)
    assert len(spans) == 1
    s, e = spans[0]
    assert tok.decode(full_ids[s:e]).strip() == answer
    # The prompt's tokens must be an exact prefix of the full sequence's tokens — the
    # incremental-prefix span algorithm depends on this BPE property.
    prompt_ids = tok.encode(
        render([msgs[0], {"role": "assistant", "content": ""}]),
        add_special_tokens=False)
    assert full_ids[:len(prompt_ids)] == prompt_ids
