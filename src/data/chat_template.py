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

from typing import List, Optional, Sequence, Tuple

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
#: The Qwen chat EOS — the cross-cutting SFT == RL == serving invariant (see module docstring).
CHAT_EOS = IM_END

_ROLES = ("system", "user", "assistant")

# --- Dual-mode reasoning conventions (#362) -------------------------------------------
MODE_DIRECT = "direct"
MODE_REASONING = "reasoning"
VALID_MODES = (MODE_DIRECT, MODE_REASONING)

THINK_START = "<think>"
THINK_END = "</think>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_PAD = "<|fim_pad|>"

SYSTEM_PROMPT_DIRECT = (
    "You are a coding assistant. Provide direct code completions without chain-of-thought reasoning."
)
SYSTEM_PROMPT_REASONING = (
    "You are a reasoning coding assistant. Provide structured reasoning traces within <think>...</think> before outputting the final answer or code."
)


def is_fim_prompt(prompt: str | Sequence[int], *, fim_prefix_id: int = 1) -> bool:
    """Return True if prompt contains a FIM prefix sentinel."""
    if isinstance(prompt, str):
        return FIM_PREFIX in prompt
    if fim_prefix_id in prompt:
        return True
    fim_bytes = list(FIM_PREFIX.encode("utf-8"))
    if len(prompt) >= len(fim_bytes):
        for i in range(len(prompt) - len(fim_bytes) + 1):
            if list(prompt[i : i + len(fim_bytes)]) == fim_bytes:
                return True
    return False


def get_mode_for_prompt(
    prompt: str | Sequence[int],
    *,
    default_mode: str = MODE_DIRECT,
    fim_prefix_id: int = 1,
) -> str:
    """Determine whether a prompt indicates direct completion (e.g. FIM) or reasoning."""
    if is_fim_prompt(prompt, fim_prefix_id=fim_prefix_id):
        return MODE_DIRECT
    if isinstance(prompt, str):
        if IM_START in prompt or THINK_START in prompt or "### Instruction:" in prompt:
            return MODE_REASONING
    return default_mode


def format_mode_messages(
    messages: List[dict],
    *,
    mode: str = MODE_DIRECT,
    system_prompt: Optional[str] = None,
) -> List[dict]:
    """Ensure messages list includes a system prompt tailored for mode."""
    if mode not in VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {VALID_MODES}")
    has_system = any(m.get("role") == "system" for m in messages)
    if has_system:
        return list(messages)
    prompt = system_prompt or (
        SYSTEM_PROMPT_REASONING if mode == MODE_REASONING else SYSTEM_PROMPT_DIRECT
    )
    return [{"role": "system", "content": prompt}] + list(messages)


def _render_turn(role: str, content: str) -> str:
    """One ChatML turn: `<|im_start|>{role}\n{content}<|im_end|>` (content stripped)."""
    if role not in _ROLES:
        raise ValueError(f"unknown chat role {role!r} (expected one of {_ROLES})")
    return f"{IM_START}{role}\n{content.strip()}{IM_END}"


def render(
    messages: List[dict],
    *,
    add_generation_prompt: bool = False,
    mode: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Render a conversation to one ChatML string.

    `messages` is `[{"role": "system"|"user"|"assistant", "content": str}, ...]`; turns are joined
    by newlines. With `add_generation_prompt=True` the string ends at `<|im_start|>assistant\n`
    (the open turn the model continues from at serving time) — the exact prefix that the matching
    `response_spans` assistant span begins after.
    When `mode` is specified ("direct" or "reasoning"), adds mode-appropriate system instructions
    if none is present in `messages`.
    """
    if mode is not None:
        messages = format_mode_messages(messages, mode=mode, system_prompt=system_prompt)
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
