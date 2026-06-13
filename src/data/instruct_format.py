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

INSTRUCTION_MARKER = "### Instruction:"
CONTEXT_MARKER = "### Input:"
RESPONSE_MARKER = "### Response:"


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
