"""End-to-end tests for the code eval suite driver (#221, `scripts/eval_code_suite.py`).

These pin the issue's acceptance criteria directly:

* **deterministic, per-instance records** — the driver is run twice at the same seed and the
  transcripts are compared byte-for-byte;
* **no pass@1 gating** — every scored record is teacher-forced (CE / accuracy / rank), which
  is checked structurally against the shared record schema;
* **decontam blocklist wired** — the blocklist's sha256 is echoed into the results JSON.

The driver runs with `--stub-model --byte-tokenizer`, so no backend, no checkpoint and no
network are involved.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/eval_code_suite.py"


def _script_module():
    """Import `scripts/eval_code_suite.py` as a module without running it --
    the `tests/test_build_domain_val_sets.py:22-29` idiom -- so `_run_tsc` can
    be called directly with a monkeypatched oracle, no subprocess."""
    spec = importlib.util.spec_from_file_location("eval_code_suite", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod

OFFLINE = ("--stub-model", "--byte-tokenizer", "--context-lens", "512",
           "--depths", "0.0,0.5,1.0", "--batch-size", "4")


def _run(tmp_path, tag, *extra):
    out = tmp_path / f"{tag}.json"
    transcript = tmp_path / f"{tag}.jsonl"
    res = subprocess.run(
        [sys.executable, str(SCRIPT), *OFFLINE, "--seed", "0",
         "--output", str(out), "--transcript", str(transcript), *extra],
        cwd=REPO_ROOT, capture_output=True, text=True)
    return res, out, transcript


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("code_suite")
    res, out, transcript = _run(tmp, "a", "--suites", "recall,needle,fim,external")
    assert res.returncode == 0, res.stderr
    return json.loads(out.read_text()), transcript, tmp


def test_the_offline_run_scores_every_requested_suite(offline_run):
    results, _, _ = offline_run
    assert set(results["summaries"]) == {"recall", "needle", "fim", "external"}
    assert results["suites_skipped"] == {}
    assert results["n_records"] > 0


def test_running_twice_at_the_same_seed_gives_byte_identical_records(offline_run):
    """THE acceptance check. A nondeterministic probe shows up here as a diff."""
    _, transcript_a, tmp = offline_run
    res, _, transcript_b = _run(tmp, "b", "--suites", "recall,needle,fim,external")
    assert res.returncode == 0, res.stderr
    assert transcript_a.read_bytes() == transcript_b.read_bytes()


def test_the_results_json_is_deterministic_apart_from_its_timing_block(offline_run):
    results_a, _, tmp = offline_run
    res, out_c, _ = _run(tmp, "c", "--suites", "recall,needle,fim,external")
    assert res.returncode == 0, res.stderr
    results_c = json.loads(out_c.read_text())
    # Wall-clock is quarantined in exactly one place, so a caller can diff the rest.
    results_a.pop("timing")
    results_c.pop("timing")
    assert results_a == results_c


def test_a_different_seed_moves_the_records(offline_run):
    """Anti-vacuity for the determinism test: identical output must not be because the seed
    is ignored."""
    _, transcript_a, tmp = offline_run
    res = subprocess.run(
        [sys.executable, str(SCRIPT), *OFFLINE, "--seed", "17",
         "--suites", "recall,needle,fim,external",
         "--transcript", str(tmp / "seed17.jsonl")],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (tmp / "seed17.jsonl").read_bytes() != transcript_a.read_bytes()


def test_every_record_uses_the_shared_schema_and_is_teacher_forced(offline_run):
    from src.eval.code_suite import RECORD_FIELDS

    _, transcript, _ = offline_run
    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert records
    for rec in records:
        assert tuple(sorted(rec)) == tuple(sorted(RECORD_FIELDS))
        # Teacher-forced: every scored record has a token span and a CE. Nothing generates,
        # so there is no pass@1 field anywhere.
        assert rec["n_scored_tokens"] > 0 and rec["ce_nats"] is not None
    assert {r["suite"] for r in records} >= {"code_recall", "code_needle", "fim"}


def test_the_transcript_is_ordered_by_suite_then_id(offline_run):
    _, transcript, _ = offline_run
    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    keys = [(r["suite"], r["id"]) for r in records]
    assert keys == sorted(keys)


def test_the_blocklist_hash_is_echoed_into_the_results(offline_run, tmp_path):
    from src.eval.code_suite import sha256_file

    results, _, _ = offline_run
    blocklist = Path(results["config"]["blocklist"])
    if blocklist.exists():
        assert results["config"]["blocklist_sha256"] == sha256_file(blocklist)
    else:
        # Not built in this tree — the driver must record null, not omit the field.
        assert results["config"]["blocklist_sha256"] is None


def test_the_config_echo_records_the_run_identity(offline_run):
    results, _, _ = offline_run
    cfg = results["config"]
    assert cfg["seed"] == 0 and cfg["temperature"] == 0.0
    assert cfg["tokenizer"]["kind"] == "byte"
    assert cfg["model"]["kind"] == "stub" and "NOT quality numbers" in cfg["model"]["warning"]
    # The external manifest (including every null revision) rides in the results, so an
    # unpinned run is visible in its own output.
    revisions = results["sources"]["external"]["sets"]
    assert set(revisions) and all(v["revision"] is None for v in revisions.values())


def test_fim_reports_both_keyings_from_one_run(offline_run):
    results, _, _ = offline_run
    assert set(results["summaries"]["fim"]["by_key"]) == {"prefix_len", "recall_distance"}


def test_domain_bpb_without_an_index_is_a_loud_skip(tmp_path):
    res, out, _ = _run(tmp_path, "skip", "--suites", "recall,domain-bpb")
    assert res.returncode == 0, res.stderr
    results = json.loads(out.read_text())
    assert "domain-bpb" in results["suites_skipped"]
    assert "--domains-json" in results["suites_skipped"]["domain-bpb"]
    assert "SKIPPED" in res.stdout
    # The suite that COULD run still ran — a skip must not take the run down.
    assert "recall" in results["summaries"]


def test_unknown_suite_is_rejected(tmp_path):
    res, _, _ = _run(tmp_path, "bad", "--suites", "not-a-suite")
    assert res.returncode != 0
    assert "unknown suite" in res.stderr


def test_a_tokenizer_must_be_chosen(tmp_path):
    res = subprocess.run([sys.executable, str(SCRIPT), "--stub-model"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode != 0
    assert "--byte-tokenizer" in res.stderr


# --------------------------------------------------------------------------------------- #
# Regression: `_run_tsc` must call the oracle by its REAL method name (#220).
#
# scripts/eval_code_suite.py:359 called `oracle.diagnose(artifact)` -- `CompositeOracle`
# (src/lsp/oracle.py) has never had a `diagnose` method, only `diagnostics`. That is the
# ONLY `.diagnose(` call site in the repo, so `--suites tsc` raised AttributeError on its
# very first row. This pins the fix by monkeypatching in a stub oracle with no `diagnose`
# method at all, so the pre-fix code fails loudly and the fixed code passes -- no node
# toolchain needed, so it runs in CI.
# --------------------------------------------------------------------------------------- #

class _StubOracle:
    """Minimal stand-in for `CompositeOracle`: only what `_run_tsc` actually
    uses (`diagnostics`, `close`, `sources_active`, `n_calls`, `wall_s`) --
    deliberately NO `diagnose` method, so calling that name raises
    `AttributeError` exactly as it would against the real class."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.sources_active = ["ts"]
        self.n_calls = 0
        self.wall_s = 0.0

    def diagnostics(self, source: str):
        from src.lsp.diagnostics import Diagnostic
        self.n_calls += 1
        if "gorblak" in source:
            return [Diagnostic(code="TS2339", line=1, col=1, message="stub finding", offset=0)]
        return []

    def close(self) -> None:
        pass


def test_run_tsc_calls_the_oracle_by_its_real_method_name(monkeypatch, tmp_path):
    mod = _script_module()
    monkeypatch.setattr("src.lsp.oracle.resolve_oracle", lambda kind: True)
    monkeypatch.setattr("src.lsp.oracle.CompositeOracle", _StubOracle)

    tsc_set = tmp_path / "tsc_set.jsonl"
    rows = [
        {"id": "clean-1", "prompt": "const x = 1;\n", "gold_completion": "",
         "error_class": "none", "expected_diagnostic": None},
        {"id": "error-1", "prompt": "console.log(u.", "gold_completion": "gorblak);\n",
         "error_class": "unfamiliar_member_access", "expected_diagnostic": "TS2339"},
    ]
    with open(tsc_set, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    args = argparse.Namespace(tsc_set=tsc_set, limit=None)
    result, src = mod._run_tsc(args)

    assert result["summary"]["n"] == 2
    assert result["summary"]["clean_rate"] == 0.5   # one clean, one flagged
    assert src["sources_active"] == ["ts"]


def test_composite_oracle_has_no_diagnose_method():
    from src.lsp.oracle import CompositeOracle

    assert not hasattr(CompositeOracle, "diagnose")
