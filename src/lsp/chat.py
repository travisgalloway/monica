"""Chat-mode tool-call support (#199 follow-up) — the *fair* opponent for hard-ban.

Phase 0 measured the tool-call baseline against a **base** model, which cannot follow
instructions by construction, and found it barely moved (`no_progress_rate` up to
1.000). That is close to a tautology, and it makes "distribution-level feedback beats
diagnostics-as-tokens" an unearned conclusion: every production coding agent feeds
compiler errors back as text and it demonstrably works. This module builds the
comparison hard-ban actually has to beat — an **instruction-tuned** model, handed the
`tsc` diagnostic in a real chat turn.

Two jobs, both of which are where a chat-mode eval usually goes quietly wrong:

1. **Rendering** (`build_toolcall_messages`): the completion task has to survive the
   trip through a chat template. We ask for a bare continuation of a prefix, not a
   whole program, so the artifact stays comparable to the completion-mode strategies
   (same `prompt + completion` scoring, same `tsc` invocation).
2. **Extraction** (`extract_completion`): an instruct model answers with prose, fenced
   markdown, a re-statement of the prompt, or all three. Getting the code back out is
   not a formality — a sloppy extractor silently converts "the model fixed it" into
   "the model failed," or vice versa.

**Extraction failure is a RESULT, not a nuisance.** A model whose answer cannot be
parsed into a completion has failed the tool-call task in a way a real agent harness
would also have to handle, so `extract_completion` returns an explicit
`ExtractionResult` with an `ok` flag rather than silently returning `""`. The caller
counts those; they appear in the measurement table as `extraction_failure_rate`. The
one thing we must never do is drop them and report a clean-rate over the survivors.

ABOVE THE SEAM — stdlib only. No `mlx`/`torch` import anywhere in this module
(guarded by `tests/test_import_guard.py`). The chat *template* is applied by the
adapter below the seam (it belongs to the tokenizer); this module only decides what
the messages say and how to read the answer back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .diagnostics import Diagnostic, strip_suggestion

# A fenced block, optionally tagged (```ts / ```typescript / ```js / ```). Non-greedy
# so the first block wins; DOTALL so it spans newlines.
_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)(?:\n```|\Z)", re.DOTALL)
# A model that answers with a bare fence and no trailing newline before the close.
_INLINE_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\s*(.*?)\s*```", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a TypeScript code completion engine. You will be given an incomplete "
    "TypeScript snippet that ends mid-expression. Reply with ONLY the text that "
    "continues it — no explanation, no markdown fences, and do not repeat any of the "
    "code you were given. Your reply is concatenated directly onto the snippet, so it "
    "must start exactly where the snippet stops."
)

# Block-budget variant. Without this, the chat arm cannot be compared against the
# completion-mode strategies at all: an instruct model finishes the one expression it was
# asked for and stops at <|im_end|>, while `budget="block"` forces every other strategy to
# free-run to a fixed token budget. Measured live: 10 generated characters against
# baseline's 346 — and a *higher* clean-rate purely for writing almost nothing. That is
# the same length-mismatch artifact that faked the tool-call "win" in Phase 0's Table C,
# and it flatters whichever side happens to write less. Matching the requested output
# length is what makes the comparison about feedback rather than about verbosity.
BLOCK_SYSTEM_PROMPT = (
    "You are a TypeScript code completion engine. You will be given an incomplete "
    "TypeScript snippet that ends mid-expression. First finish that expression, then "
    "continue writing the next several statements of plausible TypeScript code that "
    "follow from it (aim for roughly 8-12 lines in total). Reply with ONLY code — no "
    "explanation, no markdown fences, and do not repeat any of the code you were given. "
    "Your reply is concatenated directly onto the snippet, so it must start exactly "
    "where the snippet stops."
)


_STRUCTURAL = "(){}[];=<>"
_WORD_RE = re.compile(r"[A-Za-z']+")


def _is_prose(body: str) -> bool:
    """True if `body` reads as an English sentence rather than code.

    Getting this boundary right decides whether the tool-call arm is scored fairly, and
    it is easy to get wrong in *both* directions:

    - Too lax (accept anything with a `.` or `'`) and "I'm sorry, I cannot complete that
      snippet" is scored as TypeScript — the tool-call path gets credit for an apology.
    - Too strict (demand structural punctuation like `(){};=`) and a perfectly good short
      completion — `.title`, `zx.x` — is thrown away as prose. Observed live: it cost the
      instruct model a 33% "extraction failure" rate that was entirely this function's
      fault, which would have handed hard-ban an unearned win in exactly the comparison
      this experiment exists to make fair.

    So: anything carrying structural punctuation is code, and anything *without* it is
    prose only if it also reads like a sentence (several alphabetic words in a row). A
    bare `.title` has one word and no sentence structure; an apology has seven.
    """
    if any(ch in body for ch in _STRUCTURAL):
        return False
    return len(_WORD_RE.findall(body)) >= 4


_JOIN_DELIMS = ".(,["


def _normalize_join(body: str, prompt: str) -> str:
    """Drop a delimiter the model repeated from the end of the prompt.

    The prompts end mid-expression (`const firstTitle = books[0].`), and an instruct
    model asked to "continue this" very often answers with the member access *including*
    the dot — `.title` — because that is how a human would say it. Concatenated naively
    that yields `books[0]..title` and a spurious `TS1003`.

    Observed live: this alone accounted for a large share of the chat tool-call's
    apparent failures. Left unfixed it would have understated the tool-call arm and
    handed hard-ban a win it did not earn — the precise bias this experiment exists to
    remove. Any real agent harness normalizes the join, so we do too.

    Deliberately narrow: only a *single* leading delimiter that exactly duplicates the
    prompt's trailing one is dropped. It never rewrites the model's actual tokens, so a
    genuinely bad completion (a method body where a member name belonged) still fails, as
    it should.
    """
    tail = prompt.rstrip()
    if not tail or not body:
        return body
    last = tail[-1]
    if last in _JOIN_DELIMS and body.lstrip()[:1] == last:
        stripped = body.lstrip()
        return stripped[1:]
    return body


@dataclass
class ExtractionResult:
    """`ok=False` means the response could not be read as a continuation.

    That is a real outcome of the tool-call path (the model answered with prose, or an
    empty string, or only a restatement of the prompt), and it is reported rather than
    hidden — see the module docstring.
    """
    completion: str
    ok: bool
    reason: str = ""


def build_toolcall_messages(
    prompt: str,
    diagnostic: Optional[Diagnostic] = None,
    previous_completion: Optional[str] = None,
    *,
    strip_suggestions: bool = False,
    budget: str = "stmt",
) -> List[dict]:
    """Chat messages for one tool-call round.

    Round 1 (`diagnostic is None`) asks for a completion. Round 2+ replays the model's
    own previous attempt and hands back the real `tsc` error, which is exactly what an
    agent harness does: *here is what you wrote, here is what the compiler said, try
    again.*

    `strip_suggestions` drops `tsc`'s "Did you mean 'x'?" clause — the same ablation the
    completion-mode path has, because otherwise a win may only show that the model can
    copy the compiler's own suggestion rather than reason from the error.
    """
    messages = [
        {"role": "system", "content": BLOCK_SYSTEM_PROMPT if budget == "block" else SYSTEM_PROMPT},
        {"role": "user", "content": f"Complete this TypeScript snippet:\n\n{prompt}"},
    ]
    if diagnostic is None:
        return messages

    diag = strip_suggestion(diagnostic) if strip_suggestions else diagnostic
    message = diag.message
    messages.append({"role": "assistant", "content": previous_completion or ""})
    messages.append({
        "role": "user",
        "content": (
            f"The TypeScript compiler rejected that completion:\n\n"
            f"    {diagnostic.code}: {message}\n\n"
            f"Reply with ONLY a corrected continuation of the original snippet."
        ),
    })
    return messages


def extract_completion(response: str, prompt: str) -> ExtractionResult:
    """Read a chat response back into a bare continuation of `prompt`.

    Handles, in order: a fenced code block (the overwhelmingly common instruct-model
    answer); a response that helpfully restates the whole snippet (strip the prompt back
    off, since the artifact is `prompt + completion` and we would otherwise double it);
    and a bare continuation. Leading blank lines are dropped, but **leading horizontal
    whitespace is preserved** — the prompt ends mid-expression (`console.log(u.`), so a
    space at the join can be load-bearing.

    Returns `ok=False` (with a reason) rather than an empty completion when the model
    answered with pure prose or nothing usable. See the module docstring: those are
    counted, never dropped.
    """
    if not response or not response.strip():
        return ExtractionResult("", False, "empty response")

    body = response
    fence = _FENCE_RE.search(response) or _INLINE_FENCE_RE.search(response)
    if fence:
        body = fence.group(1)

    # The model restated the snippet it was given: keep only what came after it.
    stripped_prompt = prompt.strip()
    if stripped_prompt and stripped_prompt in body:
        body = body.split(stripped_prompt, 1)[1]
    else:
        # Partial restatement: the model echoed the prompt's last line and continued it.
        tail = prompt.rstrip().rsplit("\n", 1)[-1].strip()
        if tail and tail in body:
            body = body.split(tail, 1)[1]

    body = body.lstrip("\n")
    if not body.strip():
        return ExtractionResult("", False, "no completion after removing prompt/fences")

    body = _normalize_join(body, prompt)

    if _is_prose(body):
        return ExtractionResult("", False, "response looks like prose, not code")

    return ExtractionResult(body, True)
