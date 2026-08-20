"""Live, toolchain-gated tests for `src/lsp/ts_server_direct.py` (#279) --
including the **parity gate**: the direct-`tsserver` transport must report the
SAME diagnostics as the LSP client (`TsLspService`) over every candidate in the
#194 labeled error-injection set, in both directions.

Skipped wholesale on a host without BOTH toolchains (`npm install` +
`npm i -D typescript-language-server` in `eval_sets/ts_error_injection`),
mirroring `tests/test_ts_service.py`'s idiom. The binary-free mechanism tests
that gate this module in CI live in `tests/test_ts_server_direct_mechanism.py`.

**Parity against a silent oracle is not parity.** Two clients that both report
nothing agree perfectly and prove nothing -- the BLIND failure this repo guards
hardest. So the gate has two halves: the sets must match, AND the direct client
must independently report each labeled record's `expected_diagnostic`.

Cost note: the LSP half of the gate pays #278's ~350 ms client-side debounce on
every one of the 192 candidates (~70 s), against ~1.4 s for the direct half.
Both services are module-scoped and every candidate is evaluated ONCE, in a
fixture, so that cost is paid once for the whole file rather than per assertion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from src.lsp.diagnostics import Diagnostic
from src.lsp.ts_lsp import resolve_ts_lsp
from src.lsp.ts_server_direct import TsServerDirect, resolve_tsserver
from src.lsp.ts_service import TsLspService
from src.lsp.tsc import SET_DIR

pytestmark = pytest.mark.skipif(
    resolve_tsserver() is None or resolve_ts_lsp() is None,
    reason="need both a local `typescript` install (tsserver.js) and "
           "typescript-language-server in eval_sets/ts_error_injection")

EVAL_JSONL = SET_DIR / "eval.jsonl"
_CAND = "src/cand.ts"
_SEED_FILES = {_CAND: "export const _seed = 1;\n"}


def _load_records() -> List[dict]:
    return [json.loads(line) for line in
            EVAL_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(diags: List[Diagnostic]) -> List[Tuple[str, int, int]]:
    """The comparable projection: `(code, line, col)`, sorted. Messages are
    deliberately excluded -- both transports get them from the same TypeScript,
    but only the coordinates and codes are what downstream
    (`diagnostics.filter_diagnostics`, `is_incomplete`) actually consumes."""
    return sorted((d.code, d.line, d.col) for d in diags)


@pytest.fixture(scope="module")
def records() -> List[dict]:
    recs = _load_records()
    assert recs, f"{EVAL_JSONL} is empty -- the parity gate would pass vacuously"
    return recs


@pytest.fixture(scope="module")
def direct():
    svc = TsServerDirect(timeout_s=10.0)
    svc.open_project(dict(_SEED_FILES))
    yield svc
    svc.close()


@pytest.fixture(scope="module")
def lsp():
    svc = TsLspService(timeout_s=10.0)
    svc.open_project(dict(_SEED_FILES))
    yield svc
    svc.close()


@pytest.fixture(scope="module")
def parity_results(records, direct, lsp) -> List[dict]:
    """Every candidate, evaluated once through BOTH clients on the SAME text.

    Directions alternate error -> gold within a record, which keeps consecutive
    candidates on opposite sides of the clean/dirty line. That is deliberate:
    a clean -> clean transition is the one case the LSP client may legitimately
    never publish for (`n_no_publish`, `cli.mjs:20524`), which would make the
    LSP half of a comparison ambiguous rather than wrong.
    """
    out: List[dict] = []
    for rec in records:
        for direction in ("error", "gold"):
            completion = rec[f"{direction}_completion"]
            text = rec["prompt"] + completion
            direct.update(_CAND, text)
            direct_diags = direct.diagnostics(_CAND)
            lsp.update(_CAND, text)
            lsp_diags = lsp.diagnostics(_CAND)
            out.append({
                "id": rec["id"], "direction": direction,
                "expected": rec["expected_diagnostic"],
                "direct": direct_diags, "lsp": lsp_diags,
            })
    return out


# --------------------------------------------------------------------------- #
# The parity gate
# --------------------------------------------------------------------------- #

def test_parity_covers_every_record_both_directions(records, parity_results):
    assert len(parity_results) == 2 * len(records)
    assert {r["direction"] for r in parity_results} == {"error", "gold"}


def test_diagnostics_parity_with_ts_lsp_service(parity_results):
    """Sorted `(code, line, col)` sets must be equal on every candidate. A
    mismatch is reported with the candidate id, not just a count -- a bare
    "3 differed" is not actionable."""
    mismatches = [
        (r["id"], r["direction"], _key(r["direct"]), _key(r["lsp"]))
        for r in parity_results if _key(r["direct"]) != _key(r["lsp"])
    ]
    assert not mismatches, (
        f"{len(mismatches)}/{len(parity_results)} candidates disagreed between "
        f"the direct-tsserver and LSP clients: {mismatches[:5]}")


def test_direct_client_independently_reports_expected_diagnostic(parity_results):
    """The anti-BLIND half. Records whose `expected_diagnostic` is empty are the
    set's clean controls (no labeled error), skipped explicitly rather than
    silently passing."""
    labeled = [r for r in parity_results
               if r["direction"] == "error" and r["expected"]]
    assert len(labeled) >= 80, (
        f"only {len(labeled)} labeled error candidates -- the anti-vacuity "
        "guard itself would be near-vacuous")
    missing = [(r["id"], r["expected"], [d.code for d in r["direct"]])
               for r in labeled
               if r["expected"] not in [d.code for d in r["direct"]]]
    assert not missing, (
        f"{len(missing)} labeled records where the direct client did NOT report "
        f"the expected diagnostic: {missing[:5]}")


def test_no_timeouts_or_command_errors_during_the_gate(direct, parity_results):
    """A gate that ran entirely on timed-out `[]` results would report perfect
    parity. Assert the transport was actually healthy throughout."""
    assert direct.n_timeouts == 0
    assert direct.n_command_errors == 0
    assert direct.n_restarts == 0
    assert direct.op_counts["diagnostics"] >= len(parity_results)


def test_direct_is_substantially_faster_than_the_lsp_client(direct, lsp, parity_results):
    """#279's whole point. The bench (`scripts/bench_ts_lsp.py`) is the
    measurement of record; this is only a floor assertion that the ~350 ms
    client-side debounce (#278) is genuinely absent, with a wide margin so it
    cannot flake on a loaded runner."""
    direct_mean_ms = 1000.0 * direct.op_wall_s["diagnostics"] / direct.op_counts["diagnostics"]
    lsp_mean_ms = 1000.0 * lsp.op_wall_s["diagnostics"] / lsp.op_counts["diagnostics"]
    assert direct_mean_ms < 50.0, f"direct mean {direct_mean_ms:.1f}ms is not inside the 50ms bar"
    assert direct_mean_ms < lsp_mean_ms / 5.0, (
        f"direct {direct_mean_ms:.1f}ms vs lsp {lsp_mean_ms:.1f}ms -- expected the "
        "client-side debounce to be absent entirely")


# --------------------------------------------------------------------------- #
# Live mechanism checks that need a real tsserver
# --------------------------------------------------------------------------- #

def test_cold_load_is_bounded_and_recorded(direct):
    assert direct.cold_load_s is not None
    assert 0.0 < direct.cold_load_s < 10.0


def test_syntactic_and_semantic_are_both_reported():
    """Edge case 9 against the real compiler: a candidate carrying BOTH a parse
    error and a distinct type error yields both. `semanticDiagnosticsSync`
    alone would miss the parse error."""
    with TsServerDirect(timeout_s=10.0) as svc:
        svc.open_project({"a.ts": "export const seed = 1;\n"})
        svc.update("a.ts", "const n: number = 'a';\nfunction f( { return 1 }\n")
        codes = [d.code for d in svc.diagnostics("a.ts")]
    assert any(c == "TS2322" for c in codes), codes         # semantic
    assert any(c.startswith("TS1") for c in codes), codes    # syntactic


def test_clean_recheck_returns_empty_as_an_answer():
    """Edge case 15: two consecutive clean documents. On the LSP client this is
    the ambiguous `n_no_publish` case (#278); here `[]` is a real answer, and no
    counter moves."""
    with TsServerDirect(timeout_s=10.0) as svc:
        svc.open_project({"a.ts": "export const seed = 1;\n"})
        svc.update("a.ts", "export const a: number = 1;\n")
        assert svc.diagnostics("a.ts") == []
        svc.update("a.ts", "export const a: number = 2;\n")
        assert svc.diagnostics("a.ts") == []
        assert svc.n_timeouts == 0
        assert svc.n_command_errors == 0


def test_coordinates_match_the_lsp_client_exactly():
    """Edge case 12 with an explicit, human-checkable position rather than only
    the aggregate parity set."""
    text = "interface U { name: string }\nconst u: U = { name: 'ok' };\nconst q = u.gorblak;\n"
    with TsServerDirect(timeout_s=10.0) as svc:
        svc.open_project({"a.ts": "export const seed = 1;\n"})
        svc.update("a.ts", text)
        diags = svc.diagnostics("a.ts")
    assert len(diags) == 1
    d = diags[0]
    assert d.code == "TS2339"
    assert (d.line, d.col) == (3, 13)
    assert text[d.offset:d.offset + len("gorblak")] == "gorblak"


def test_restart_recovers_the_open_program_after_the_child_dies():
    with TsServerDirect(timeout_s=10.0) as svc:
        svc.open_project({"a.ts": "export const seed = 1;\n"})
        svc.update("a.ts", "const n: number = 'a';\n")
        assert [d.code for d in svc.diagnostics("a.ts")] == ["TS2322"]
        svc._proc.kill()
        svc._proc.wait(timeout=5.0)
        assert [d.code for d in svc.diagnostics("a.ts")] == ["TS2322"]
        assert svc.n_restarts == 1


def test_lazy_open_of_a_second_file_in_the_same_program():
    files: Dict[str, str] = {
        "src/hub.ts": "export interface Vec { x: number; y: number }\n",
        "src/uses.ts": ('import { Vec } from "./hub";\n'
                        "export function f(v: Vec): number { return v.x; }\n"),
    }
    with TsServerDirect(timeout_s=10.0) as svc:
        svc.open_project(dict(files), warm_path="src/hub.ts")
        assert svc.diagnostics("src/uses.ts") == []
        svc.update("src/uses.ts",
                   'import { Vec } from "./hub";\n'
                   "export function f(v: Vec): number { return v.nope; }\n")
        assert [d.code for d in svc.diagnostics("src/uses.ts")] == ["TS2339"]


def test_scratch_dir_is_cleaned_up_on_close():
    svc = TsServerDirect(timeout_s=10.0)
    scratch = Path(svc.scratch_dir)
    svc.open_project({"a.ts": "export const a = 1;\n"})
    assert scratch.exists()
    svc.close()
    assert not scratch.exists()
