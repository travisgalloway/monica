"""Qwen ChatML template — the single source of truth for the distillation student's chat
format (#95), mirroring the role `instruct_format.py` plays for the OLMo POC.

The student shares the **Qwen3 tokenizer** with the conversion teacher, so its chat format is
Qwen **ChatML**:

    <|im_start|>system\\n{system}<|im_end|>\\n
    <|im_start|>user\\n{user}<|im_end|>\\n
    <|im_start|>assistant\\n{response}<|im_end|>\\n

**The detail that bites (docs/design/11-post-training.md):** the Qwen base defines `<|im_end|>`
as the chat EOS. It MUST be identical across SFT, RL, and serving — a mismatch degrades the model
at serving time. So this module is the one place the format is defined, both the data builders
(#95/#96) and (later) serving import it, and `response_spans` trains the assistant turn **up to
and including its trailing `<|im_end|>`** so the model learns to stop on it. No separate
`eos_token_id` is appended — `<|im_end|>` already plays that role (and for Qwen3, token-aligned
with Qwen2.5, it *is* the `eos_token_id`, id 151645).

Unlike `instruct_format` (newline-free, because the pretraining corpus is one-doc-per-line and
appends EOS per line), ChatML is multi-line — that is fine here because SFT records are stored as
token-id lists (`src/data/sft_loader.py`), not as corpus text lines, so internal newlines are
encoded into ids rather than read back as document boundaries.

Portable: pure string formatting + the injected tokenizer's `encode`; no `mlx`/`torch`, no HF
dependency (works with `tokenize.ByteTokenizer` offline).
"""

from __future__ import annotations

from typing import List, Tuple

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
#: The Qwen chat EOS — the cross-cutting SFT == RL == serving invariant (see module docstring).
CHAT_EOS = IM_END

_ROLES = ("system", "user", "assistant")


def _render_turn(role: str, content: str) -> str:
    """One ChatML turn: `<|im_start|>{role}\\n{content}<|im_end|>` (content stripped)."""
    if role not in _ROLES:
        raise ValueError(f"unknown chat role {role!r} (expected one of {_ROLES})")
    return f"{IM_START}{role}\n{content.strip()}{IM_END}"


def render(messages: List[dict], *, add_generation_prompt: bool = False) -> str:
    """Render a conversation to one ChatML string.

    `messages` is `[{"role": "system"|"user"|"assistant", "content": str}, ...]`; turns are joined
    by newlines. With `add_generation_prompt=True` the string ends at `<|im_start|>assistant\\n`
    (the open turn the model continues from at serving time) — the exact prefix that the matching
    `response_spans` assistant span begins after.
    """
    parts = [_render_turn(m["role"], m["content"]) for m in messages]
    text = "\n".join(parts)
    if add_generation_prompt:
        text = (text + "\n" if text else text) + f"{IM_START}assistant\n"
    return text


def _encode(tokenizer, text: str) -> List[int]:
    """Encode without auto-added specials (HF appends BOS/EOS otherwise); `ByteTokenizer.encode`
    takes no kwargs, hence the fallback. Literal `<|im_*|>` strings still map to their special
    token ids on a real Qwen tokenizer (that flag governs auto-added specials, not recognition)."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def response_spans(messages: List[dict], tokenizer) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Tokenize `render(messages)` and return `(full_ids, spans)` where each span is a half-open
    `[start, end)` token range covering one assistant turn's **content plus its trailing
    `<|im_end|>`** — so SFT trains the answer *and* the stop token, but never the
    `<|im_start|>assistant\\n` header or any user/system text.

    Spans are found by tokenizing growing prefixes of the *same* rendered string and diffing their
    lengths, so the indices line up with `full_ids` even though BPE boundaries do not coincide with
    character boundaries (the same technique as `instruct_format.response_spans`).
    """
    full = render(messages)
    full_ids = _encode(tokenizer, full)
    spans: List[Tuple[int, int]] = []
    prefix = ""
    for m in messages:
        turn = _render_turn(m["role"], m["content"])
        sep = "\n" if prefix else ""
        if m["role"] == "assistant" and m["content"].strip():
            # Span starts after the `<|im_start|>assistant\n` header, ends after `<|im_end|>`.
            header = prefix + sep + f"{IM_START}assistant\n"
            start = len(_encode(tokenizer, header))
            end = len(_encode(tokenizer, prefix + sep + turn))
            spans.append((start, end))
        prefix = prefix + sep + turn
    return full_ids, spans
