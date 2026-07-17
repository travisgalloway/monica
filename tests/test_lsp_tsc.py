"""Real-`tsc` tests for `src/lsp/tsc.py` — pins the output format against the
actual pinned compiler (#199's finding #1: the issue's assumed `file:line:col:`
format never matches; only `file(line,col): error TSxxxx:` does).

Skipped wholesale (not just individual tests) on a node-less host, mirroring
`tests/test_ts_error_eval.py`'s `resolve_tsc() is None` pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.lsp.diagnostics import filter_diagnostics, is_incomplete
from src.lsp.tsc import DEFAULT_TSCONFIG_PATH, SET_DIR, TscRunner, resolve_tsc

pytestmark = pytest.mark.skipif(resolve_tsc() is None,
                                 reason="no node/tsc toolchain on this host")

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
def runner():
    r = TscRunner()
    yield r
    r.close()


def test_member_access_offset_pins_to_gorblak(runner: TscRunner):
    rec = _load_record("member-access-001")
    source = rec["prompt"] + rec["error_completion"]
    diags = runner.diagnostics(source)
    ts2339 = [d for d in diags if d.code == "TS2339"]
    assert ts2339, f"expected a TS2339 among {[d.code for d in diags]}"
    assert ts2339[0].offset == source.index("gorblak")


def test_arity_mismatch_offset_is_inside_prompt(runner: TscRunner):
    # Finding #3: arity-mismatch-005 anchors TS2554 at the call expression, which
    # lands INSIDE the prompt region (offset < len(prompt)) even though the error
    # is caused by the completion. filter_diagnostics must clamp this, not drop it.
    rec = _load_record("arity-mismatch-005")
    prompt = rec["prompt"]
    source = prompt + rec["error_completion"]
    diags = runner.diagnostics(source)
    ts2554 = [d for d in diags if d.code == "TS2554"]
    assert ts2554, f"expected a TS2554 among {[d.code for d in diags]}"
    assert ts2554[0].offset < len(prompt)

    # And filter_diagnostics correctly clamps it up to generation_start rather than
    # dropping it — it's a real, completion-caused diagnostic.
    clamped = filter_diagnostics(diags, frontier=len(source), generation_start=len(prompt))
    assert any(d.code == "TS2554" and d.offset == len(prompt) for d in clamped)


def test_bare_prompt_yields_only_ts1xxx(runner: TscRunner):
    rec = _load_record("member-access-001")
    diags = runner.diagnostics(rec["prompt"])
    assert diags, "expected the bare (incomplete) prompt to produce at least one diagnostic"
    assert all(is_incomplete(d.code) for d in diags), \
        f"expected only TS1xxx, got {[d.code for d in diags]}"
    assert filter_diagnostics(diags, frontier=len(rec["prompt"]), generation_start=0) == []


def test_gold_completion_is_diagnostic_clean(runner: TscRunner):
    rec = _load_record("member-access-001")
    source = rec["prompt"] + rec["gold_completion"]
    assert runner.diagnostics(source) == []


def test_runner_tracks_cumulative_cost(runner: TscRunner):
    rec = _load_record("member-access-001")
    source = rec["prompt"] + rec["gold_completion"]
    assert runner.n_calls == 0
    runner.diagnostics(source)
    runner.diagnostics(source)
    assert runner.n_calls == 2
    assert runner.wall_s > 0.0


def test_runner_scratch_dir_is_nested_under_set_dir(runner: TscRunner):
    assert SET_DIR in runner.scratch_dir.parents


def test_runner_close_cleans_up_scratch_dir(tmp_path: Path):
    r = TscRunner()
    scratch_dir = r.scratch_dir
    assert scratch_dir.exists()
    r.close()
    assert not scratch_dir.exists()
