"""Single source of truth for the instruction template.

The model only learns a prompt->response behavior if the format it is *trained* on
(instruction examples baked into the corpus) is the same format it is *prompted* with
at inference (`scripts/generate.py --chat`). Both sides import from here so they can
never drift.

The template is the classic Alpaca/Dolly-style block. `format_example` produces the
full instruction+response text used as a training document; `format_prompt` produces
everything up to and including the response marker, so the model continues from there
at inference. `RESPONSE_MARKER` is the boundary used to split the response off the
prompt, and `INSTRUCTION_MARKER` is the natural stop string for a single turn (the
model starting a new "### Instruction:" means the turn is done).

Portable: pure string formatting, no dependencies, lives above the seam.
"""

from __future__ import annotations

INSTRUCTION_MARKER = "### Instruction:\n"
CONTEXT_MARKER = "### Input:\n"
RESPONSE_MARKER = "### Response:\n"


def format_prompt(instruction: str, context: str = "") -> str:
    """The instruction block up to the response marker (model continues from here)."""
    parts = [f"{INSTRUCTION_MARKER}{instruction.strip()}\n"]
    if context and context.strip():
        parts.append(f"\n{CONTEXT_MARKER}{context.strip()}\n")
    parts.append(f"\n{RESPONSE_MARKER}")
    return "".join(parts)


def format_example(instruction: str, response: str, context: str = "") -> str:
    """A full training document: the prompt block followed by the response text."""
    return f"{format_prompt(instruction, context)}{response.strip()}"
