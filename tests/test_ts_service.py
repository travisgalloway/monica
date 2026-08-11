"""Real-`typescript-language-server` tests for `src/lsp/ts_service.py`'s
`TsLspService` -- the warm, multi-op counterpart to `test_ts_lsp.py`. Skipped
wholesale on a host without the toolchain installed (`npm i -D
typescript-language-server` in `eval_sets/ts_error_injection`), mirroring
`tests/test_ts_lsp.py`'s `resolve_ts_lsp() is None` idiom. The binary-free
mechanism tests that gate this class in CI live in
`tests/test_ts_service_mechanism.py`.
"""

from __future__ import annotations

import os

import pytest

from src.lsp.diagnostics import Diagnostic
from src.lsp.ts_lsp import TsLspOracle, resolve_ts_lsp
from src.lsp.ts_service import Location, Symbol, TsLspService

pytestmark = pytest.mark.skipif(resolve_ts_lsp() is None,
                                reason="no typescript-language-server toolchain on this host")

_HUB_TS = (
    "export interface Vec { x: number; y: number }\n\n"
    "export function scale(v: Vec, k: number): Vec {\n"
    "  return { x: v.x * k, y: v.y * k };\n"
    "}\n"
)
_USES_HUB_TS = (
    'import { Vec, scale } from "./hub";\n\n'
    "export function double(v: Vec): Vec {\n"
    "  return scale(v, 2);\n"
    "}\n"
)
_OTHER_TS = "export function noop(): void {}\n"
_BROKEN_TS = (
    'import { Vec } from "./hub";\n\n'
    "export function touch(v: Vec): number {\n"
    "  return v.x;\n"
    "}\n"
)

_FIXTURE_FILES = {
    "src/hub.ts": _HUB_TS,
    "src/uses_hub.ts": _USES_HUB_TS,
    "src/other.ts": _OTHER_TS,
    "src/broken.ts": _BROKEN_TS,
}


@pytest.fixture(scope="module")
def service():
    svc = TsLspService(timeout_s=10.0)
    svc.open_project(dict(_FIXTURE_FILES), warm_path="src/hub.ts")
    yield svc
    svc.close()


# --------------------------------------------------------------------------- #
# cross-file definition / references / completions / quickinfo / symbols
# --------------------------------------------------------------------------- #

def test_cross_file_definition_lands_in_hub(service: TsLspService):
    call_offset = _USES_HUB_TS.index("scale(")
    defs = service.definition("src/uses_hub.ts", call_offset)
    assert defs, "expected at least one definition location"
    assert all(isinstance(d, Location) for d in defs)
    assert any(d.path == "src/hub.ts" for d in defs)
    hub_decl_line = _HUB_TS.splitlines().index(
        "export function scale(v: Vec, k: number): Vec {") + 1
    assert any(d.path == "src/hub.ts" and d.line == hub_decl_line for d in defs)


def test_references_on_hub_symbol_spans_at_least_two_files(service: TsLspService):
    scale_offset = _HUB_TS.index("function scale") + len("function ")
    refs = service.references("src/hub.ts", scale_offset, include_declaration=True)
    assert refs
    assert len({r.path for r in refs}) >= 2


def test_completions_after_dot_contains_x(service: TsLspService):
    dot_offset = _BROKEN_TS.index("v.x") + 2
    items = service.completions("src/broken.ts", dot_offset)
    assert any(item.label == "x" for item in items)


def test_quickinfo_on_hub_function_is_nonempty_and_names_it(service: TsLspService):
    call_offset = _USES_HUB_TS.index("scale(")
    info = service.quickinfo("src/uses_hub.ts", call_offset)
    assert info
    assert "scale" in info


def test_document_symbols_on_hub_contains_exported_names(service: TsLspService):
    symbols = service.document_symbols("src/hub.ts")
    assert symbols
    assert all(isinstance(s, Symbol) for s in symbols)
    names = {s.name for s in symbols}
    assert "Vec" in names and "scale" in names


# --------------------------------------------------------------------------- #
# version-aware diagnostics: edit -> error, revert -> clean
# --------------------------------------------------------------------------- #

def test_update_then_diagnostics_detects_and_clears_a_break(service: TsLspService):
    broken = _BROKEN_TS.replace("v.x", "v.gorblak", 1)
    v_after_break = service.update("src/broken.ts", broken)
    diags = service.diagnostics("src/broken.ts")
    assert any(d.code == "TS2339" for d in diags), \
        f"expected TS2339 among {[d.code for d in diags]}"

    v_after_revert = service.update("src/broken.ts", _BROKEN_TS)
    assert v_after_revert == v_after_break + 1
    clean = service.diagnostics("src/broken.ts")
    assert clean == [], f"expected clean after revert, got {clean}"


def test_benign_edit_to_clean_document_is_counted_as_no_publish():
    """#278: `cli.mjs:20524` returns early (never publishes) when a document's
    OLD and NEW diagnostic sets are both empty -- pinning that against the
    real server. A fresh, short-timeout `TsLspService` (not the module-scoped
    `service` fixture, whose counters are shared across tests and whose 10s
    timeout would stall the suite on the wait this exercises)."""
    svc = TsLspService(timeout_s=2.0)
    try:
        svc.open_project({"src/clean.ts": _OTHER_TS})
        assert svc.diagnostics("src/clean.ts") == []

        n_no_publish_before = svc.n_no_publish
        n_timeouts_before = svc.n_timeouts
        svc.update("src/clean.ts", _OTHER_TS + "// a benign, non-breaking comment\n")
        result = svc.diagnostics("src/clean.ts")

        # Edge case / decision rule (plan #278): if the live server DOES
        # publish here (e.g. a diagnostic kind was never populated for this
        # document, so cli.mjs's early-return does not fire), the counter
        # code is still correct -- record what actually happened rather than
        # assert a behaviour the server didn't exhibit.
        if svc.n_no_publish == n_no_publish_before + 1:
            assert result == []
            assert svc.n_timeouts == n_timeouts_before
        else:
            assert svc.n_timeouts == n_timeouts_before, (
                "expected either n_no_publish+1 (cli.mjs:20524 suppressed the "
                "publish) or a real answered push (n_no_publish unchanged) -- "
                f"got n_no_publish={svc.n_no_publish} (was {n_no_publish_before}), "
                f"n_timeouts={svc.n_timeouts} (was {n_timeouts_before}), result={result}")
    finally:
        svc.close()


def test_version_counter_is_strictly_increasing(service: TsLspService):
    v1 = service.update("src/other.ts", _OTHER_TS + "// edit 1\n")
    v2 = service.update("src/other.ts", _OTHER_TS + "// edit 2\n")
    v3 = service.update("src/other.ts", _OTHER_TS + "// edit 3\n")
    assert v1 < v2 < v3


# --------------------------------------------------------------------------- #
# resilience: dead server restarts and restores the warm program
# --------------------------------------------------------------------------- #

def test_killed_server_restarts_and_reopens_documents(service: TsLspService):
    assert service.n_restarts == 0

    service._proc.kill()
    service._proc.wait(timeout=5.0)

    call_offset = _USES_HUB_TS.index("scale(")
    defs = service.definition("src/uses_hub.ts", call_offset)
    assert any(d.path == "src/hub.ts" for d in defs), \
        "definition did not recover a real result after the server was killed"
    assert service.n_restarts == 1


def test_no_orphan_process_after_close():
    svc = TsLspService(timeout_s=10.0)
    proc = svc._proc
    pid = proc.pid
    svc.close()
    assert proc.poll() is not None, "child process was not reaped by close()"
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# --------------------------------------------------------------------------- #
# parity: TsLspService agrees with the validated TsLspOracle on a #194 record
# --------------------------------------------------------------------------- #

def test_parity_with_ts_lsp_oracle_on_194_member_access_001():
    prompt = "interface User { name: string; age: number; }\nconst u: User = { name: \"Ada\", age: 32 };\nconsole.log(u."
    error_completion = "gorblak);\n"
    source = prompt + error_completion

    oracle = TsLspOracle(timeout_s=10.0)
    svc = TsLspService(timeout_s=10.0)
    try:
        oracle_diags = oracle.diagnostics(source)
        svc.open_project({"src/single.ts": source})
        svc_diags = svc.diagnostics("src/single.ts")

        oracle_codes = {d.code for d in oracle_diags}
        svc_codes = {d.code for d in svc_diags}
        assert "TS2339" in oracle_codes, "TsLspOracle lost its own label"
        assert "TS2339" in svc_codes, \
            f"TsLspService did not reproduce TsLspOracle's TS2339 finding: {svc_codes}"
        for d in svc_diags:
            assert isinstance(d, Diagnostic)
    finally:
        oracle.close()
        svc.close()
