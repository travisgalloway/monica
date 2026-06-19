"""Qwen ChatML template (#95): render + response-span masking — the single source of truth for
the distillation student's chat format (distinct from the OLMo `instruct_format` covered by
tests/test_chat_template.py). Pure stdlib + the offline ByteTokenizer (no backend).
"""

from src.data import chat_template
from src.data.chat_template import IM_END, IM_START, render, response_spans
from src.data.tokenize import ByteTokenizer


def test_render_chatml_shape():
    msgs = [{"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    out = render(msgs)
    assert out == (f"{IM_START}system\nbe terse{IM_END}\n"
                   f"{IM_START}user\nhi{IM_END}\n"
                   f"{IM_START}assistant\nhello{IM_END}")


def test_render_generation_prompt():
    out = render([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert out.endswith(f"{IM_START}assistant\n")
    assert IM_END in out and out.count(f"{IM_START}assistant") == 1


def test_response_span_covers_content_and_im_end_only():
    tok = ByteTokenizer()
    msgs = [{"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"}]
    full_ids, spans = response_spans(msgs, tok)
    assert len(spans) == 1
    s, e = spans[0]
    # The span decodes to exactly the assistant content + its trailing <|im_end|> stop token —
    # nothing else (no <|im_start|>assistant header, no user text).
    assert tok.decode(full_ids[s:e]) == f"pong{IM_END}"
    assert e == len(full_ids)


def test_full_ids_reconstruct_the_render():
    tok = ByteTokenizer()
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    full_ids, _ = response_spans(msgs, tok)
    assert tok.decode(full_ids) == render(msgs)


def test_multi_turn_spans_line_up_and_exclude_user():
    tok = ByteTokenizer()
    msgs = [{"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"}]
    full_ids, spans = response_spans(msgs, tok)
    assert len(spans) == 2
    assert tok.decode(full_ids[spans[0][0]:spans[0][1]]) == f"b{IM_END}"
    assert tok.decode(full_ids[spans[1][0]:spans[1][1]]) == f"d{IM_END}"
    assert spans[0][1] <= spans[1][0]                 # disjoint; "c" falls between them
    # No user content is ever inside a span.
    off = [full_ids[i] for i in range(len(full_ids))
           if not any(s <= i < e for s, e in spans)]
    assert "c" in tok.decode(off) and "a" in tok.decode(off)


def test_empty_assistant_turn_yields_no_span():
    tok = ByteTokenizer()
    _, spans = response_spans([{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": ""}], tok)
    assert spans == []


def test_chat_eos_is_im_end():
    assert chat_template.CHAT_EOS == IM_END == "<|im_end|>"
