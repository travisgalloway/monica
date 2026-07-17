"""Real-`typescript-language-server` tests for `src/lsp/ts_lsp.py`. Skipped
wholesale on a host without the toolchain installed (`npm i -D
typescript-language-server` in `eval_sets/ts_error_injection`), mirroring
`tests/test_lsp_tsc.py`'s `resolve_tsc() is None` idiom.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from src.lsp.diagnostics import filter_diagnostics, is_incomplete, is_source_balanced
from src.lsp.tsc import SET_DIR, TscRunner, resolve_tsc
from src.lsp.ts_lsp import TsLspOracle, resolve_ts_lsp

pytestmark = pytest.mark.skipif(resolve_ts_lsp() is None,
                                 reason="no typescript-language-server toolchain on this host")

_EVAL_SET_PATH = SET_DIR / "eval.jsonl"


def _load_record(rec_id: str) -> dict:
    with open(_EVAL_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["id"] == rec_id:
                return rec
    raise AssertionError(f"record {rec_id!r} not found in {_EVAL_SET_PATH}")


@pytest.fixture
def oracle():
    o = TsLspOracle(timeout_s=10.0)
    yield o
    o.close()


# --------------------------------------------------------------------------- #
# core correctness: right finding, right range, clean stays clean
# --------------------------------------------------------------------------- #

def test_type_error_yields_finding_at_correct_range(oracle: TsLspOracle):
    rec = _load_record("member-access-001")
    source = rec["prompt"] + rec["error_completion"]
    diags = oracle.diagnostics(source)
    ts2339 = [d for d in diags if d.code == "TS2339"]
    assert ts2339, f"expected a TS2339 among {[d.code for d in diags]}"
    assert ts2339[0].offset == source.index("gorblak")


def test_clean_code_yields_empty(oracle: TsLspOracle):
    rec = _load_record("member-access-001")
    source = rec["prompt"] + rec["gold_completion"]
    assert oracle.diagnostics(source) == []


# --------------------------------------------------------------------------- #
# Trap A: LSP's integer `code` must come back `TS`-prefixed, and `is_incomplete`
# must still recognize the TS1xxx family through this oracle.
# --------------------------------------------------------------------------- #

def test_codes_are_ts_prefixed_and_is_incomplete_still_fires(oracle: TsLspOracle):
    # An unterminated paren -- a genuine mid-generation "still typing" syntax
    # error, TS1005 ("')' expected").
    diags = oracle.diagnostics("const x = (1 + 2")
    assert diags, "expected at least one diagnostic for unterminated syntax"
    codes = [d.code for d in diags]
    assert all(c.startswith("TS") and c[2:].isdigit() for c in codes), codes
    assert "TS1005" in codes
    assert is_incomplete("TS1005")


# --------------------------------------------------------------------------- #
# Trap B: only severity == 1 (Error) survives; suggestion/hint severities
# (tsserver's unused-variable check is LSP severity 4, "Hint") must not leak in.
# --------------------------------------------------------------------------- #

def test_suggestion_severity_is_excluded(oracle: TsLspOracle):
    # `unused` triggers tsserver's TS6133 "declared but never read" -- a real
    # tsserver diagnostic (confirmed empirically: LSP severity 4), which `tsc`
    # itself never reports as an `error`. Must not appear here.
    source = "function f(): number { const unused = 5; return 1; }\n"
    diags = oracle.diagnostics(source)
    assert diags == [], f"suggestion/hint diagnostics leaked through: {diags}"


# --------------------------------------------------------------------------- #
# resilience: dead server restarts; no orphan process survives close()
# --------------------------------------------------------------------------- #

def test_killed_server_restarts_and_next_call_succeeds(oracle: TsLspOracle):
    rec = _load_record("member-access-001")
    source = rec["prompt"] + rec["error_completion"]

    assert oracle.diagnostics(source)  # server is up, sanity check
    assert oracle.n_restarts == 0

    oracle._proc.kill()
    oracle._proc.wait(timeout=5.0)

    diags = oracle.diagnostics(source)
    assert any(d.code == "TS2339" for d in diags), \
        "oracle did not recover a real finding after the server was killed"
    assert oracle.n_restarts == 1


def test_no_orphan_process_after_close():
    o = TsLspOracle(timeout_s=10.0)
    proc = o._proc
    pid = proc.pid
    o.close()
    # `Popen.wait()` inside `close()` already reaped it; `poll()` must reflect
    # that rather than leaving a zombie/orphan behind.
    assert proc.poll() is not None, "child process was not reaped by close()"
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# --------------------------------------------------------------------------- #
# Real-code (block-budget) divergence from tsc: a not-yet-finished function
# body, and an "unexpected EOF"-style error anchored one character differently
# than tsc anchors it -- both confirmed live, both real #199 Stage A findings
# (measured directly: an unguarded ts_lsp-backed slow-hard run hit 100%
# over-repair on otherwise-correct HumanEval-TS generation before these gates
# existed in diagnostics.py/harness.py). Regression coverage for both.
# --------------------------------------------------------------------------- #

def test_unclosed_function_reports_control_flow_completeness_raw(oracle: TsLspOracle):
    # tsserver's error-recovery parsing can still do semantic analysis on a
    # function whose closing brace hasn't been typed yet -- tsc's batch compile
    # never gets far enough to. Confirming the RAW (unfiltered) oracle output
    # here pins the fact `filter_diagnostics`'s source-balance gate depends on.
    source = "function f(): boolean {\n  for (let i = 0; i < 1; i++) {\n  }\n"
    assert not is_source_balanced(source)
    diags = oracle.diagnostics(source)
    codes = {d.code for d in diags}
    assert "TS2366" in codes or "TS2355" in codes, \
        f"expected a control-flow-completeness code among {codes}"


def test_unexpected_eof_anchors_one_character_before_tsc(oracle: TsLspOracle):
    # The exact scenario debugged live on HumanEval_0_has_close_elements: tsc
    # anchors "'}' expected" at len(source) (the true EOF); tsserver anchors the
    # SAME defect one character earlier (the last real character). A frontier
    # check of `offset < len(source)` therefore disagrees between the two
    # oracles for the identical missing-brace defect.
    source = "function f(): boolean {\n  for (let i = 0; i < 1; i++) {\n  }\n"
    ts2366 = [d for d in oracle.diagnostics(source) if d.code in ("TS1005", "TS2366", "TS2355")]
    assert ts2366
    assert all(d.offset < len(source) for d in ts2366 if d.code == "TS1005"), \
        "expected tsserver's EOF-style TS1005 to anchor before len(source), not at/past it"


def test_committed_incomplete_carveout_needs_balance_not_just_ts1xxx(oracle: TsLspOracle):
    """The exact fix: `filter_diagnostics` alone already drops TS2366 on
    unbalanced source (tested in test_lsp_diagnostics.py); this pins the OTHER
    half -- harness.py's `hit_boundary` "committed TS1xxx" carve-out must also
    be gated on `is_source_balanced`, or tsserver's off-by-one EOF anchor
    reinstates a "'}' expected" as a real defect for a function that's simply
    not finished yet (more segments legitimately still coming under
    budget="block") -- the exact mechanism the F1 --limit 5 smoke run exposed.
    """
    source = "function f(): boolean {\n  for (let i = 0; i < 1; i++) {\n  }\n"
    frontier = len(source)  # hit_boundary=True: this segment's generation is done
    raw = oracle.diagnostics(source)
    filtered = filter_diagnostics(raw, frontier=frontier, generation_start=0, source=source)
    assert filtered == [], "filter_diagnostics itself must already be clean here"

    committed_incomplete = [d for d in raw if is_incomplete(d.code) and d.offset < frontier]
    # tsserver's off-by-one EOF anchor means this list is often non-empty --
    # the bug is trusting it unconditionally, not that it's ever empty.
    if committed_incomplete:
        assert not is_source_balanced(source), (
            "a committed-incomplete TS1xxx only clears the offset<frontier bar "
            "when the source is still open -- reinstating it as real here would "
            "be exactly the false positive this test exists to catch")


# --------------------------------------------------------------------------- #
# parity: TsLspOracle finds the same #194 labeled errors TscRunner does
# --------------------------------------------------------------------------- #

def test_parity_with_tsc_on_194_set(oracle: TsLspOracle):
    if resolve_tsc() is None:
        pytest.skip("no node/tsc toolchain on this host")
    runner = TscRunner()
    try:
        for rec_id, expected_code in (
            ("member-access-001", "TS2339"),
            ("arity-mismatch-005", "TS2554"),
        ):
            rec = _load_record(rec_id)
            source = rec["prompt"] + rec["error_completion"]

            tsc_codes = set(runner.codes(source))
            lsp_codes = {d.code for d in oracle.diagnostics(source)}

            assert expected_code in tsc_codes, f"{rec_id}: tsc lost its own label"
            assert expected_code in lsp_codes, \
                f"{rec_id}: TS-LSP did not reproduce tsc's {expected_code} finding"
    finally:
        runner.close()
