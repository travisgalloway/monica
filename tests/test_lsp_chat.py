"""Chat-mode tool-call: message construction and (mostly) response extraction.

Extraction is where a chat-mode eval quietly goes wrong. An instruct model answers with
prose, fenced markdown, a restatement of the prompt, or all three; a sloppy extractor
turns "the model fixed it" into "the model failed" (or worse, the reverse) and the whole
fair-opponent comparison becomes noise. These pin the rules.

ABOVE THE SEAM — no model, no node, pure stdlib.
"""

from __future__ import annotations

from src.lsp.chat import (SYSTEM_PROMPT, build_toolcall_messages,
                           extract_completion)
from src.lsp.diagnostics import Diagnostic

PROMPT = ('interface User { name: string; age: number; }\n'
          'const u: User = { name: "Ada", age: 32 };\n'
          'console.log(u.')

DIAG = Diagnostic(code="TS2339", line=3, col=15, offset=101,
                   message="Property 'gorblak' does not exist on type 'User'. Did you mean 'name'?")


# --- message construction --- #

def test_round_one_asks_for_a_completion():
    msgs = build_toolcall_messages(PROMPT)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert PROMPT in msgs[1]["content"]


def test_round_two_replays_the_attempt_and_the_real_error():
    """The agent-shaped turn: here's what you wrote, here's what the compiler said."""
    msgs = build_toolcall_messages(PROMPT, DIAG, previous_completion="gorblak);")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "gorblak);"
    assert "TS2339" in msgs[3]["content"]
    assert "does not exist on type 'User'" in msgs[3]["content"]


def test_strip_suggestions_removes_the_did_you_mean_clause():
    """Otherwise a 'win' may only show the model can copy tsc's own suggestion."""
    kept = build_toolcall_messages(PROMPT, DIAG, "gorblak);")[3]["content"]
    stripped = build_toolcall_messages(PROMPT, DIAG, "gorblak);",
                                        strip_suggestions=True)[3]["content"]
    assert "Did you mean" in kept
    assert "Did you mean" not in stripped
    assert "does not exist" in stripped      # the diagnostic itself survives


# --- extraction --- #

def test_extracts_from_a_fenced_block():
    r = extract_completion("Here you go:\n\n```typescript\nname);\n```\n", PROMPT)
    assert r.ok and r.completion.startswith("name);")


def test_extracts_a_bare_continuation():
    r = extract_completion("name);", PROMPT)
    assert r.ok and r.completion == "name);"


def test_strips_a_restated_prompt():
    """Instruct models love to echo the whole snippet back; the artifact is
    prompt+completion, so leaving the echo in would duplicate the prompt."""
    r = extract_completion(f"```ts\n{PROMPT}name);\n```", PROMPT)
    assert r.ok
    assert "interface User" not in r.completion
    assert r.completion.startswith("name);")


def test_strips_a_partially_restated_prompt():
    """The model echoes only the final line it was asked to continue."""
    r = extract_completion("```ts\nconsole.log(u.name);\n```", PROMPT)
    assert r.ok and r.completion.startswith("name);")


def test_prose_only_response_is_an_extraction_FAILURE_not_an_empty_completion():
    """The load-bearing one. A model that apologizes instead of coding has failed the
    tool-call task; scoring that as an empty completion would hide the failure."""
    r = extract_completion("I'm sorry, I cannot complete that snippet", PROMPT)
    assert not r.ok
    assert "prose" in r.reason


def test_empty_response_is_a_failure():
    assert not extract_completion("", PROMPT).ok
    assert not extract_completion("   \n  ", PROMPT).ok


def test_response_that_is_only_the_prompt_is_a_failure():
    """Echoing the prompt with nothing added is not a completion."""
    r = extract_completion(f"```ts\n{PROMPT}\n```", PROMPT)
    assert not r.ok


def test_short_member_expressions_are_code_not_prose():
    """Regression, caught live against the real instruct model.

    An earlier prose check demanded structural punctuation `(){};=`, which threw away
    perfectly good short completions like `.title` and `zx.x` as "prose" — a 33% bogus
    extraction-failure rate that was entirely the extractor's fault. Left in, it would
    have handed hard-ban an unearned win in the very comparison this experiment exists
    to make fair.
    """
    for completion in (".title", "zx.x", "name", "u.age"):
        r = extract_completion(completion, "const d = e")
        assert r.ok, f"{completion!r} is code, not prose (reason: {r.reason})"
        assert r.completion == completion


def test_apology_is_still_prose():
    """The other side of that boundary must not move: an apology is not a completion."""
    for prose in ("I'm sorry, I cannot complete that snippet",
                  "This snippet appears to be incomplete and I need more context"):
        assert not extract_completion(prose, PROMPT).ok


def test_leading_horizontal_whitespace_is_preserved():
    """The prompt ends mid-expression, so a space at the join can be load-bearing."""
    r = extract_completion('```ts\n "Ada");\n```', 'const msg = greet(')
    assert r.ok and r.completion.startswith(' "Ada");')
