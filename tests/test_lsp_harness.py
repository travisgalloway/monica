"""FakeLM tests for `src/lsp/harness.py` — the core repair-loop logic, provable
with no GPU and no node. `ScriptedFakeLM` is a deterministic single-character
tokenizer whose logits at each generation step are driven by a `script`: a dict
from the exact generated-token-prefix (a tuple) seen so far to a ranked list of
preferred next-token ids. Banning the top choice (as `harness.py`'s hard repair
does) deterministically reveals the next-ranked one — this is what makes the
"banning provably changes the greedy argmax" theorem directly observable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from src.lsp.diagnostics import Diagnostic
from src.lsp.harness import GenResult, generate_baseline, generate_slow_loop, generate_toolcall


class ScriptedFakeLM:
    """See module docstring. `alt_script`, if given, is used instead of `script`
    whenever `alt_trigger` appears in the string passed to `reset()` — this is
    what lets a test simulate "the model responds usefully to an injected
    diagnostic comment" (soft repair) vs. "the model can't read it at all"
    (no-progress) with the same simple mechanism.
    """

    def __init__(self, script: Dict[Tuple[int, ...], List[int]],
                 alt_script: Optional[Dict[Tuple[int, ...], List[int]]] = None,
                 alt_trigger: str = "// tsc:", vocab_size: int = 256,
                 default_token: int = ord(";")):
        self.script = script
        self.alt_script = alt_script
        self.alt_trigger = alt_trigger
        self.vocab_size = vocab_size
        self.default_token = default_token
        self.n_forward_tokens = 0
        self.n_forward_tokens_nocache = 0
        self._active_script = script
        self._gen_ids: List[int] = []

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in token_ids)

    def _logits_for(self, history: Tuple[int, ...]) -> np.ndarray:
        logits = np.full(self.vocab_size, -10.0, dtype=np.float64)
        prefs = self._active_script.get(history, [self.default_token])
        for rank, tok in enumerate(prefs):
            logits[tok] = float(len(prefs) - rank) * 10.0
        self.n_forward_tokens += 1
        self.n_forward_tokens_nocache += 1
        return logits

    def reset(self, context: str) -> np.ndarray:
        self._gen_ids = []
        if self.alt_script is not None and self.alt_trigger in context:
            self._active_script = self.alt_script
        else:
            self._active_script = self.script
        return self._logits_for(())

    def step(self, token_id: int) -> np.ndarray:
        self._gen_ids.append(token_id)
        return self._logits_for(tuple(self._gen_ids))

    def rollback(self, n_tokens: int) -> None:
        if n_tokens <= 0:
            return
        if n_tokens > len(self._gen_ids):
            raise ValueError("cannot roll back more tokens than generated")
        self._gen_ids = self._gen_ids[: len(self._gen_ids) - n_tokens]


def _linear_script(word: str, start: Tuple[int, ...] = ()) -> Dict[Tuple[int, ...], List[int]]:
    """A deterministic single-choice continuation: from `start`, emit `word` one
    character at a time with no branching."""
    script = {}
    history = start
    for ch in word:
        script[history] = [ord(ch)]
        history = history + (ord(ch),)
    return script


def _contains_diagnose(bad_substring: str, code: str = "TS9999") -> callable:
    """A fake `diagnose_fn`: flags `bad_substring`'s first occurrence in `source`,
    empty diagnostics otherwise. Stands in for a real `tsc` call in every test here.

    The diagnostic's `message` is deliberately generic (never echoes
    `bad_substring`): the harness's soft repair injects `message` verbatim into a
    `// tsc: ...` comment placed back into the context, and a message containing
    the very substring this fake searches for would make the fake re-flag its own
    injected comment on the next check -- a self-referential fixture bug, not a
    harness bug (a real `tsc` never flags comment text; this fake does substring
    search, so it must not be handed a substring that appears in its own output).
    """
    def _diagnose(source: str) -> List[Diagnostic]:
        idx = source.find(bad_substring)
        if idx == -1:
            return []
        return [Diagnostic(code=code, line=1, col=idx + 1,
                            message="undefined reference", offset=idx)]
    return _diagnose


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #

def test_baseline_stmt_stops_at_first_boundary():
    script = _linear_script("name);\nEXTRA")
    lm = ScriptedFakeLM(script)
    result = generate_baseline(lm, "console.log(u.", budget="stmt")
    assert result.completion == "name);"
    assert result.artifact == "console.log(u.name);"


def test_baseline_reports_forward_token_cost():
    script = _linear_script("x;\n")
    lm = ScriptedFakeLM(script)
    result = generate_baseline(lm, "const ", budget="stmt")
    # n_forward_tokens includes reset()'s prompt-prefill call plus one per
    # generated token -- strictly more than n_generated_tokens, never less.
    assert result.n_forward_tokens > result.n_generated_tokens
    assert result.n_generated_tokens > 0


# --------------------------------------------------------------------------- #
# hard repair: rolls back to the offending token, not the checkpoint
# --------------------------------------------------------------------------- #

def _gorblak_vs_name_lm() -> ScriptedFakeLM:
    base = "u."
    primary_tail = "gorblak);\n"
    fallback_tail = "name);\n"
    script: Dict[Tuple[int, ...], List[int]] = {}
    script.update(_linear_script(base))
    branch_prefix = tuple(ord(c) for c in base)
    script.update(_linear_script(primary_tail, start=branch_prefix))
    script.update(_linear_script(fallback_tail, start=branch_prefix))
    script[branch_prefix] = [ord(primary_tail[0]), ord(fallback_tail[0])]
    return ScriptedFakeLM(script)


def test_unrepaired_baseline_would_contain_the_error():
    # Sanity: absent repair, greedy really does walk into "gorblak".
    lm = _gorblak_vs_name_lm()
    result = generate_baseline(lm, "console.log(", budget="stmt")
    assert result.completion == "u.gorblak);"


def test_hard_repair_rolls_back_to_gorblak_not_console():
    lm = _gorblak_vs_name_lm()
    diagnose = _contains_diagnose("gorblak", code="TS2339")
    result = generate_slow_loop(lm, diagnose, "console.log(", repair="hard", budget="stmt")

    assert result.completion == "u.name);"
    assert result.artifact == "console.log(u.name);"
    assert result.n_rollbacks == 1
    # The rollback target was the 'g' token (index 2: 'u', '.', 'g', ...) -- it kept
    # the correctly-generated "u." prefix and did NOT roll back to the checkpoint
    # (index 0, which would have discarded "u." too, let alone "console" which was
    # never in gen_ids to begin with -- rollback structurally cannot reach the prompt).
    assert result.events[0]["kind"] == "hard_repair"
    assert result.events[0]["rolled_back_to"] == 2


def test_hard_repair_ban_is_keyed_to_the_exact_prefix():
    # Same fixture, direct evidence of the progress theorem: banning the
    # checkpoint-relative prefix's top choice deterministically flips the argmax
    # at that exact prefix (and nowhere else).
    lm = _gorblak_vs_name_lm()
    diagnose = _contains_diagnose("gorblak", code="TS2339")
    result = generate_slow_loop(lm, diagnose, "console.log(", repair="hard", budget="stmt")
    assert "gorblak" not in result.artifact
    assert result.unrepaired is False


def test_hard_repair_retry_cap_terminates_and_marks_unrepaired():
    # A diagnose_fn that is never satisfied: the loop must still terminate at
    # max_retries, not spin forever.
    script: Dict[Tuple[int, ...], List[int]] = {}
    lm = ScriptedFakeLM(script, default_token=ord(";"))
    always_bad = lambda source: [Diagnostic(code="TS9999", line=1, col=1,
                                             message="never satisfied", offset=0)]
    result = generate_slow_loop(lm, always_bad, "const x = ", repair="hard",
                                 budget="stmt", max_retries=3, max_gen_tokens=10)
    assert result.unrepaired is True
    assert result.n_retries == 3


def test_suppression_hack_with_no_diagnostic_does_not_crash():
    # A suppression hack (`as any`) makes `_is_clean` return False, but the diagnose
    # fn reports NO diagnostics -> `filtered` is empty. Hard repair is diagnostic-guided
    # and has nothing to roll back to; the loop must mark unrepaired and stop, not crash
    # on `filtered[0]` (regression: #201, exposed by --ignore-module-resolution making an
    # empty `filtered` common on real TS full of `as any`).
    lm = ScriptedFakeLM(_linear_script("v as any;\n"))
    no_diags = lambda source: []
    result = generate_slow_loop(lm, no_diags, "const x = ", repair="hard",
                                 budget="stmt", max_retries=3)
    assert result.unrepaired is True
    assert result.n_rollbacks == 0          # never rolled back correct code
    assert result.reward_hack_detected is True


# --------------------------------------------------------------------------- #
# soft repair
# --------------------------------------------------------------------------- #

def _bad_vs_good_lm() -> ScriptedFakeLM:
    primary = _linear_script("bad();")
    alt = _linear_script("good();")
    return ScriptedFakeLM(primary, alt_script=alt)


def test_soft_repair_injects_above_the_statement_and_yields_valid_artifact():
    lm = _bad_vs_good_lm()
    diagnose = _contains_diagnose("bad", code="TS2304")
    prompt = "interface X { ok(): void; }\nconst v = "

    result = generate_slow_loop(lm, diagnose, prompt, repair="soft", budget="stmt", max_retries=2)

    assert result.completion == "good();"
    assert result.artifact == prompt + "good();"
    assert "// tsc:" not in result.artifact          # the two-string invariant
    assert "// tsc:" in result.context                # but the model DID see it
    # Injected above the partial statement: before "const v = ", not inside it.
    assert result.context.index("// tsc:") < result.context.index("const v = ")
    assert result.n_soft_repairs == 1
    assert result.no_progress is False
    assert result.unrepaired is False


def test_soft_repair_no_progress_when_model_ignores_the_comment():
    # No alt_script: injecting the comment changes nothing, so the regenerated
    # statement is byte-identical to the first attempt.
    lm = ScriptedFakeLM(_linear_script("bad();"))
    diagnose = _contains_diagnose("bad", code="TS2304")
    result = generate_slow_loop(lm, diagnose, "const v = ", repair="soft",
                                 budget="stmt", max_retries=5)

    assert result.no_progress is True
    assert result.unrepaired is True
    # Must abort on the FIRST detected no-progress round, not burn all 5 retries.
    assert result.n_soft_repairs == 1


# --------------------------------------------------------------------------- #
# over-repair guard
# --------------------------------------------------------------------------- #

def test_clean_input_causes_zero_rollbacks():
    lm = ScriptedFakeLM(_linear_script("name);\n"))
    always_clean = lambda source: []
    result = generate_slow_loop(lm, always_clean, "console.log(u.", repair="hard", budget="stmt")

    assert result.n_rollbacks == 0
    assert result.n_retries == 0
    assert result.unrepaired is False
    assert result.completion == "name);"


# --------------------------------------------------------------------------- #
# tool-call baseline
# --------------------------------------------------------------------------- #

def test_toolcall_shares_soft_repair_machinery_under_its_own_label():
    lm = _bad_vs_good_lm()
    diagnose = _contains_diagnose("bad", code="TS2304")
    result = generate_toolcall(lm, diagnose, "const v = ", k=2)

    assert result.strategy == "toolcall-k2"
    assert result.completion == "good();"
    assert "// tsc:" not in result.artifact


def test_toolcall_respects_block_budget_not_hardcoded_to_stmt():
    # Regression test for a real bug: generate_toolcall hardcoded budget="stmt",
    # so under budget="block" it silently stopped at the first statement boundary
    # while every other strategy free-ran to block_size tokens -- an unfair
    # comparison where toolcall looked artificially cheap/clean purely because it
    # generated far less text, not because tool-call repair was actually working.
    script = _linear_script("a;" * 20)  # many trivial one-char statements
    lm = ScriptedFakeLM(script)
    always_clean = lambda source: []

    baseline = generate_baseline(lm, "", budget="block", block_size=20)
    toolcall = generate_toolcall(lm, always_clean, "", k=1, budget="block", block_size=20)

    assert toolcall.n_generated_tokens == 20
    assert toolcall.n_generated_tokens == baseline.n_generated_tokens


# --------------------------------------------------------------------------- #
# block budget: checkpoint stack across multiple statements
# --------------------------------------------------------------------------- #

def test_block_budget_commits_multiple_checkpoints():
    lm = ScriptedFakeLM(_linear_script("a;more"))  # each segment resets -> "a;" then boundary
    always_clean = lambda source: []
    result = generate_slow_loop(lm, always_clean, "", repair="hard",
                                 budget="block", block_size=4)

    assert result.completion == "a;a;"
    assert result.n_generated_tokens == 4
    assert len(result.checkpoints) == 2
    assert result.checkpoints == [0, 2]


# --------------------------------------------------------------------------- #
# stop_strings — real-code function-body generation (#199 F1)
# --------------------------------------------------------------------------- #

def test_baseline_block_stops_at_stop_string_and_truncates():
    """The body is `x;\\n` then the next construct `function `; stop_strings cuts it."""
    lm = ScriptedFakeLM(_linear_script("x;\nfunction y"))
    result = generate_baseline(lm, "", budget="block", block_size=40,
                                stop_strings=["\nfunction "])
    assert result.completion == "x;"          # truncated exactly at the stop, newline dropped with it


def test_baseline_block_without_stop_strings_is_unchanged():
    """Regression guard on the existing #194 block path: no stop_strings => old behaviour."""
    lm = ScriptedFakeLM(_linear_script("abcdefghij"))
    result = generate_baseline(lm, "", budget="block", block_size=5)
    assert result.completion == "abcde"        # exactly block_size tokens, no early stop


def test_slow_loop_block_stops_at_stop_string():
    """slow_loop checks the stop at segment-commit. Use a stop marker that sits WITHIN a
    segment (no statement boundary before it) so a single committed segment contains it."""
    lm = ScriptedFakeLM(_linear_script("aaaXbbbbbb"))   # no ; or \n before default ';' kicks in
    always_clean = lambda source: []
    result = generate_slow_loop(lm, always_clean, "", repair="hard",
                                 budget="block", block_size=40, stop_strings=["X"])
    assert result.completion == "aaa"                   # truncated at the stop marker
