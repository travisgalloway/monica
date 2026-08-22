"""End-to-end tests for the session-rewind entry point (`scripts/generate.py`, #305).

Portable — no backend, so these run in the Linux `portable` CI job. Every assertion goes
through `scripts.generate.interactive_repl` with scripted input lines, because the defect
this issue closes was a rewind layer with no caller: a test that reached the tree directly
would re-create that defect in a new costume.

**The model matters more than the plumbing.** `tests/test_serve.py`'s `FakeModel` returns
`eye(V)[token % V]` — logits that depend only on the last token fed, never on the
recurrent state — so a rewind test written against it passes with state restoration
completely broken. `StatefulFakeModel` below keys its logits off a rolling hash of every
token the session has consumed, which is what makes the identity assertion load-bearing.
`test_state_independent_model_makes_the_identity_vacuous` pins that reasoning in place by
running the same script against the state-independent model and watching the "differs from
the discarded branch" half fail.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serve import sampling
from src.serve.generate import generate
from src.serve.sessions import SessionHistory, SessionStore
from scripts.generate import _nonnegative_int, interactive_repl, main
from tests.test_serve import FakeModel

VOCAB = 16


class StatefulFakeModel(FakeModel):
    """A `FakeModel` whose logits are a function of the STATE, not just the last token.

    State is a rolling hash of every token consumed, and the logits one-hot at
    `state % vocab`, so any two distinct token histories almost surely decode to different
    continuations — a wrong restored state provably yields a wrong continuation.

    A plain running sum (the obvious choice) is *not* good enough here: greedy decoding on
    `logits = eye(V)[sum % V]` has absorbing fixed points, so distinct prefixes converge to
    the same repeated token within a few steps and the "differs from the discarded branch"
    assertion goes vacuous again. The hash has no such traps.
    """

    def step(self, token, state):
        new_state = (state * 131 + np.asarray(token) + 7) % 1_000_003
        logits = np.eye(self.config.vocab_size)[new_state % self.config.vocab_size]
        return logits, new_state


def _encode(text: str) -> list[int]:
    return [ord(ch) % VOCAB for ch in text]


def _decode(ids) -> str:
    return "".join(chr(ord("A") + int(i)) for i in ids)


class ScriptedRepl:
    """Drives `interactive_repl` with a fixed list of input lines and records everything.

    `run_turn` is the real thing — `src.serve.generate.generate` against a `SessionStore` —
    with a trivial character tokenizer standing in for the CLI's. Only stdin/stdout are
    faked, so the state handling under test is exactly the CLI's.
    """

    def __init__(self, lines, *, model=None, rewind_depth: int = 8,
                 max_new_tokens: int = 6):
        self.model = model if model is not None else StatefulFakeModel(vocab_size=VOCAB)
        self.store = SessionStore(self.model, max_concurrent=1)
        self.sid = "repl"
        self.store.create(self.sid)
        self.history = (SessionHistory(self.store, self.sid, max_depth=rewind_depth)
                        if rewind_depth > 0 else None)
        self.sampler = partial(sampling.sample, temperature=0.0,
                               rng=np.random.default_rng(0))
        self.max_new_tokens = max_new_tokens
        self._lines = list(lines)
        self.emitted: list[str] = []
        self.turns: list[tuple[str, str]] = []     # (input text, continuation)
        self.boundary_states: list = []            # store state at each turn boundary

    # --- the three injected seams ---
    def run_turn(self, text: str, *, prefill: bool = True) -> str:
        ids = _encode(text)
        out_ids = generate(self.store, self.sid, ids, sampler=self.sampler,
                           to_numpy=np.asarray, max_new_tokens=self.max_new_tokens,
                           pass_context=True, prefill=prefill)
        continuation = _decode(out_ids)
        self.turns.append((text, continuation))
        self.boundary_states.append(self.store.get_state(self.sid))
        return continuation

    def read_line(self):
        return self._lines.pop(0) if self._lines else None

    def emit(self, line: str) -> None:
        self.emitted.append(line)

    # --- helpers ---
    def run(self) -> "ScriptedRepl":
        interactive_repl(self.store, self.sid, self.history, run_turn=self.run_turn,
                         read_line=self.read_line, emit=self.emit,
                         rewind_enabled=self.history is not None)
        return self

    @property
    def chrome(self) -> str:
        return "\n".join(self.emitted)

    def continuation(self, index: int) -> str:
        return self.turns[index][1]


# The three lines that make the branch test sharp. `X` and `B` end with the SAME character
# but differ earlier, so a model that only looks at the last token cannot tell them apart
# (see test_state_independent_model_makes_the_identity_vacuous) while a state-dependent one
# must.
LINE_A, LINE_X, LINE_B = "hi", "ab", "cb"


# --- criterion 3: rewinding restores the recurrent state, proven by output ---------------

def test_rewound_branch_matches_direct_generation():
    """A -> B directly == A -> X -> rewind -> B, and != the discarded X continuation."""
    direct = ScriptedRepl([LINE_A, LINE_B]).run()
    rewound = ScriptedRepl([LINE_A, LINE_X, "/rewind 1", LINE_B]).run()

    out_b = direct.continuation(1)
    out_x = rewound.continuation(1)
    out_b_prime = rewound.continuation(2)

    assert out_b, "the fake model must actually produce a continuation"
    # State restored: the rewound branch regenerates B's continuation byte-for-byte.
    assert out_b_prime == out_b
    # A real branch, not a replay of the discarded one, and not a vacuous identity.
    assert out_b_prime != out_x


def test_state_independent_model_makes_the_identity_vacuous():
    """Why `StatefulFakeModel` exists: the base `FakeModel` cannot prove anything.

    With logits keyed off the last token alone, the discarded branch and the rewound one
    decode identically (both lines end in the same character), so the "differs from the
    discarded branch" half of the criterion FAILS — which is exactly the signal that a
    state-independent model would have let a broken rewind through.
    """
    rewound = ScriptedRepl([LINE_A, LINE_X, "/rewind 1", LINE_B],
                           model=FakeModel(vocab_size=VOCAB)).run()
    assert rewound.continuation(2) == rewound.continuation(1)


def test_restored_state_equals_the_snapshot():
    """The direct check: the store's state after `/rewind` == the boundary snapshot."""
    repl = ScriptedRepl([LINE_A, LINE_X, "/rewind 1"]).run()
    at_boundary = repl.boundary_states[0]           # state right after LINE_A
    after_rewind = repl.store.get_state(repl.sid)
    assert np.array_equal(np.asarray(at_boundary), np.asarray(after_rewind))
    # And it genuinely moved: the discarded branch's state differs.
    assert not np.array_equal(np.asarray(repl.boundary_states[1]),
                              np.asarray(after_rewind))


# --- criterion 6: every guard is watched rejecting ---------------------------------------

def test_over_deep_rewind_errors_and_leaves_the_session_untouched():
    repl = ScriptedRepl([LINE_A, "/rewind 5"]).run()
    before = np.asarray(repl.boundary_states[0])
    assert "cannot rewind 5 turn(s)" in repl.chrome
    assert "at most 1 rewind hop(s) are possible" in repl.chrome
    assert np.array_equal(before, np.asarray(repl.store.get_state(repl.sid)))


def test_rewind_with_nothing_committed_restores_the_root_and_says_so():
    repl = ScriptedRepl(["/rewind"]).run()
    assert "session is empty" in repl.chrome
    assert "Restored the root state." in repl.chrome
    # The root is the freshly-created (zero) state, and it really was reinstalled.
    assert np.array_equal(np.asarray(repl.store.get_state(repl.sid)),
                          np.asarray(repl.model.init_state(1)))


def test_rewind_multiple_at_root_reports_over_deep_not_empty():
    """`/rewind 2` at the root asks for a specific hop count that doesn't exist.

    It must get the precise "cannot rewind 2 turn(s) ... at most 0 rewind hop(s)" error
    (matching `test_over_deep_rewind_errors_and_leaves_the_session_untouched`'s pattern), not
    be collapsed into the generic "session is empty" message that only the bare `/rewind`
    (count 1) default gets.
    """
    repl = ScriptedRepl(["/rewind 2"]).run()
    assert "session is empty" not in repl.chrome
    assert "cannot rewind 2 turn(s)" in repl.chrome
    assert "at most 0 rewind hop(s) are possible" in repl.chrome
    # A rejected rewind must leave the session untouched, same as any other over-deep rewind.
    assert np.array_equal(np.asarray(repl.store.get_state(repl.sid)),
                          np.asarray(repl.model.init_state(1)))


def test_evicted_root_reports_the_hop_count_not_an_empty_session():
    """#318: `--rewind-depth 1` evicts turn 0, which is not the same as an empty session.

    Committing turn 1 under a depth-1 cap evicts turn 0 and reparents turn 1 onto None, so
    `history.depth()` collapses back to 1 — the predicate this branch used to key on. A real
    turn exists, so the bare `/rewind` must fall through to `rewind_turns` and report the
    precise hop count, never claim the session is empty.
    """
    repl = ScriptedRepl(["aa", "/rewind"], rewind_depth=1).run()
    assert "session is empty" not in repl.chrome
    assert "cannot rewind 1 turn(s)" in repl.chrome
    assert "at most 0 rewind hop(s) are possible" in repl.chrome
    # The rejected rewind touched nothing: `rewind_turns` raises before any `set_state`, so
    # the live state is still exactly what the committed turn left behind.
    assert np.array_equal(np.asarray(repl.boundary_states[0]),
                          np.asarray(repl.store.get_state(repl.sid)))


def test_rejected_rewind_leaves_generation_unchanged():
    """The behavioural half of #318's benign-state property: the next turn is unaffected."""
    with_rejected = ScriptedRepl(["aa", "/rewind", "bb"], rewind_depth=1).run()
    without = ScriptedRepl(["aa", "bb"], rewind_depth=1).run()
    assert with_rejected.continuation(1) == without.continuation(1)


def test_at_the_root_after_a_rewind_is_not_reported_as_empty():
    """#318: standing on turn 0 with turns behind you is not an empty session either."""
    repl = ScriptedRepl(["aa", "/rewind 1", "/rewind"]).run()
    assert "you are at the root (turn 0)" in repl.chrome
    assert "session is empty" not in repl.chrome
    assert "Restored the root state." in repl.chrome
    # The root snapshot really was reinstalled: it is the freshly-created (zero) state.
    assert np.array_equal(np.asarray(repl.store.get_state(repl.sid)),
                          np.asarray(repl.model.init_state(1)))


@pytest.mark.parametrize("argument", ["0", "-1"])
def test_rewind_non_positive_count_is_rejected(argument):
    repl = ScriptedRepl([LINE_A, LINE_X, f"/rewind {argument}"]).run()
    assert f"error: n must be >= 1 (got {argument})" in repl.chrome


def test_rewind_non_numeric_argument_is_rejected():
    repl = ScriptedRepl([LINE_A, "/rewind abc"]).run()
    assert "/rewind takes a turn count" in repl.chrome
    assert "got 'abc'" in repl.chrome


def test_rewind_too_many_arguments_is_rejected():
    repl = ScriptedRepl([LINE_A, "/rewind 1 2"]).run()
    assert "takes at most one argument" in repl.chrome


def test_rewind_to_unknown_node_id_names_the_retained_ids():
    repl = ScriptedRepl([LINE_A, "/rewind #999"]).run()
    assert "node 999 is not retained" in repl.chrome
    assert "retained ids: [0, 1]" in repl.chrome


def test_unknown_command_is_rejected_and_never_reaches_the_model():
    repl = ScriptedRepl(["/frobnicate"]).run()
    assert "unknown command '/frobnicate'" in repl.chrome
    assert "NOT sent to the model" in repl.chrome
    assert repl.turns == []          # the typo was never fed as prompt text


def test_rewind_depth_zero_disables_rewind_without_building_a_tree():
    repl = ScriptedRepl([LINE_A, "/rewind", "/tree"], rewind_depth=0).run()
    assert repl.history is None
    assert "rewind: DISABLED (--rewind-depth 0); no tree is built." in repl.chrome
    assert repl.chrome.count("rewind is disabled") == 2      # /rewind and /tree both say so
    assert len(repl.turns) == 1                              # the turn itself still works


def test_negative_rewind_depth_is_an_argparse_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv",
                        ["generate.py", "--interactive", "--rewind-depth", "-1"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
    assert "must be >= 0" in capsys.readouterr().err


def test_nonnegative_int_accepts_zero_and_rejects_negatives():
    assert _nonnegative_int("0") == 0
    with pytest.raises(Exception, match="must be >= 0"):
        _nonnegative_int("-3")


# --- criterion 6e / 6h: branching and eviction --------------------------------------------

def test_branch_twice_then_rewind_across_the_branch_point():
    """Two children off one boundary, reached by absolute id, with matching continuations."""
    repl = ScriptedRepl([LINE_A, LINE_X, "/rewind 1", LINE_B, "/tree"]).run()
    # Nodes: 0 root, 1 = after A, 2 = after X, 3 = after B (both children of 1).
    branch_point = [n for n in repl.history.nodes() if n[0] == 1][0]
    assert sorted(branch_point[2]) == [2, 3]
    assert "* #3" in repl.chrome

    # Rewind across the branch point to the discarded child and continue it: the
    # continuation must match what that branch produced the first time.
    resumed = ScriptedRepl([LINE_A, LINE_X, "/rewind 1", LINE_B, "/rewind #2", LINE_B]).run()
    from_branch_x = ScriptedRepl([LINE_A, LINE_X, LINE_B]).run()
    assert resumed.continuation(3) == from_branch_x.continuation(2)
    assert resumed.continuation(3) != resumed.continuation(2)   # the other branch differs


def test_eviction_bounds_the_tree_and_a_deeper_rewind_errors():
    lines = ["aa", "bb", "cc", "dd", "ee", "/rewind 4"]
    repl = ScriptedRepl(lines, rewind_depth=3).run()
    assert len(repl.history) == 3 == len(repl.history.tree)
    assert len(repl.history.retained_ids()) == 3
    assert "cannot rewind 4 turn(s)" in repl.chrome
    assert "at most 2 rewind hop(s) are possible" in repl.chrome


# --- criterion 7: the memory bound is stated ----------------------------------------------

def test_startup_states_the_retained_node_budget():
    repl = ScriptedRepl([], rewind_depth=4).run()
    per_node = repl.history.per_node_bytes()
    assert f"at most {4 * per_node} bytes" in repl.chrome
    assert f"4 x {per_node} B/snapshot" in repl.chrome


def test_help_lists_every_command():
    repl = ScriptedRepl(["/help"]).run()
    for command in ("/rewind [n]", "/rewind #<id>", "/tree", "/help", "/quit"):
        assert command in repl.chrome


def test_quit_stops_the_loop_before_later_lines():
    repl = ScriptedRepl([LINE_A, "/quit", LINE_B]).run()
    assert len(repl.turns) == 1
