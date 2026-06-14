"""Single source of truth for the instruction template.

The model only learns a prompt->response behavior if the format it is *trained* on
(instruction examples baked into the corpus) is the same format it is *prompted* with
at inference (`scripts/generate.py --chat`). Both sides import from here so they can
never drift.

**Single line, deliberately.** The data pipeline is one-document-per-line and
`tokenize.py` appends EOS per line, so a training document cannot contain raw newlines
(each would become a spurious document boundary). The corpus builder also runs every
instruction doc through `download._normalize_doc` (whitespace-collapse). A multi-line
Alpaca block would therefore be flattened to spaces at train time while
`format_prompt` kept its newlines at inference — a silent train/inference mismatch.
Keeping the template newline-free makes `format_example` collapse-idempotent and
byte-identical to the prompt the model is later given.

`format_example` is the full instruction+response training document; `format_prompt`
is everything up to the response marker (the model continues from there).
`RESPONSE_MARKER` splits the response off the prompt; `INSTRUCTION_MARKER` is the
natural stop string for a single turn (the model emitting a new "### Instruction:"
means the turn is done — though with one example per training doc the primary stop is
EOS).

Portable: pure string formatting, no dependencies, lives above the seam.
"""

from __future__ import annotations

from typing import List, Tuple

INSTRUCTION_MARKER = "### Instruction:"
CONTEXT_MARKER = "### Input:"
RESPONSE_MARKER = "### Response:"
SYSTEM_MARKER = "### System:"

# Role -> marker. `user` reuses the instruction marker and `assistant` the response
# marker so a single user/assistant exchange renders byte-identically to
# `format_example` (the format the base model was pretrained on) — see `render`.
ROLE_MARKERS = {
    "system": SYSTEM_MARKER,
    "user": INSTRUCTION_MARKER,
    "assistant": RESPONSE_MARKER,
}


def format_prompt(instruction: str, context: str = "") -> str:
    """The instruction block up to the response marker (model continues from here).

    Ends exactly at the response marker with no trailing space: in a full example the
    answer follows as " {response}", so the model predicts that leading-space token —
    appending a trailing space here would tokenize the boundary differently than train.
    """
    parts = [f"{INSTRUCTION_MARKER} {instruction.strip()}"]
    if context and context.strip():
        parts.append(f"{CONTEXT_MARKER} {context.strip()}")
    parts.append(RESPONSE_MARKER)
    return " ".join(parts)


def format_example(instruction: str, response: str, context: str = "") -> str:
    """A full training document: the prompt block followed by the response text."""
    return f"{format_prompt(instruction, context)} {response.strip()}"


# --- multi-turn chat (SFT/DPO) ------------------------------------------------------

def _render_turn(role: str, content: str) -> str:
    """One turn as `<marker>[ <content>]` — marker alone when content is empty (a
    trailing empty assistant turn is exactly the prompt the model continues from)."""
    marker = ROLE_MARKERS[role]
    content = content.strip()
    return f"{marker} {content}" if content else marker


def render(messages: List[dict]) -> str:
    """Render a multi-turn conversation to one newline-free training/prompt string.

    `messages` is `[{"role": "system"|"user"|"assistant", "content": str}, ...]`, turns
    joined by single spaces. By construction this is byte-identical to the pretraining
    format on a single exchange: `render([{user:i},{assistant:r}]) == format_example(i, r)`
    and `render([{user:i},{assistant:""}]) == format_prompt(i)`, so the SFT/chat format
    never drifts from what the base model already knows. Content is only `.strip()`ed
    (matching `format_example`); collapsing internal whitespace to stay newline-free is
    the caller's job (the data builders run `download._normalize_doc` on each content).
    """
    return " ".join(_render_turn(m["role"], m["content"]) for m in messages)


def _encode(tokenizer, text: str) -> List[int]:
    """Encode without special tokens (HF appends EOS otherwise); ByteTokenizer.encode
    takes no kwargs, hence the fallback."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return tokenizer.encode(text)


def response_spans(messages: List[dict], tokenizer) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Tokenize `render(messages)` and return `(full_ids, spans)` where each span is a
    half-open `[start, end)` token range covering one assistant turn's *content* (the
    `### Response:` marker tokens are excluded — only the answer is trained on).

    Spans are found by tokenizing growing prefixes of the *same* rendered string and
    diffing their lengths, so the indices line up with `full_ids` even though BPE token
    boundaries do not coincide with character boundaries. The leading space before the
    content merges into the first content token (the token the model predicts after the
    marker), so it is included in the span — matching `format_example`'s " {response}".
    """
    full = render(messages)
    full_ids = _encode(tokenizer, full)
    spans: List[Tuple[int, int]] = []
    prefix = ""
    for m in messages:
        turn = _render_turn(m["role"], m["content"])
        sep = " " if prefix else ""
        if m["role"] == "assistant" and m["content"].strip():
            marker_prefix = prefix + sep + ROLE_MARKERS["assistant"]
            start = len(_encode(tokenizer, marker_prefix))
            end = len(_encode(tokenizer, prefix + sep + turn))
            spans.append((start, end))
        prefix = prefix + sep + turn
    return full_ids, spans
