"""Persistent TypeScript language server oracle -- Stage A's TS-LSP arm, replacing
per-call batch `tsc` compiles with one long-lived, incremental
`typescript-language-server` process the harness's repair loop talks to over LSP
(via `src/lsp/jsonrpc.py`, built and fake-transport-tested first per the plan's
build order).

Why persistent-server, not batch-compile: a language server understands *open,
possibly-syntactically-incomplete* documents -- exactly what the repair loop's
mid-generation checks always are -- and re-checks incrementally, where a fresh
`tsc -p` invocation pays the whole-project cold-start cost on every single check
(the #199 F1 finding: `tsc_wall_s_total` 474s of a 1712s run).

Two failure modes this module exists to avoid (see `docs/design/12-lsp-in-the-loop.md`'s
Stage A plan):

  - **Trap A (formatting).** LSP reports a diagnostic's `code` as a bare **integer**
    (`2339`), not `tsc`'s `"TS2339"` string. `is_incomplete` matches on the `TS1xxx`
    string prefix, so every code from this module is reformatted as `f"TS{code}"`
    before it reaches `diagnostics.py` -- get this wrong and every partial-prefix
    syntax diagnostic looks "real," silently wrecking the clean-rate.
  - **Trap B (severity parity).** `tsc` only ever reports `error`. tsserver
    additionally publishes suggestion/hint diagnostics (LSP severity 3/4) `tsc`
    never would -- admitting them would silently inflate the diagnostic set and
    invalidate any LSP-vs-tsc comparison. Only `severity == 1` (Error) survives.

Not safe for concurrent use -- one oracle instance == one sequential
open/check/close stream, matching `TscRunner`'s per-run, single-threaded-per-record
contract.

ABOVE THE SEAM -- stdlib only. No `mlx`/`torch` import anywhere in this module
(guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .diagnostics import Diagnostic, line_col_to_offset
from .jsonrpc import JsonRpcEndpoint, spawn
from .tsc import DEFAULT_TSCONFIG_PATH, SET_DIR

TS_LSP_BIN = SET_DIR / "node_modules" / ".bin" / "typescript-language-server"

_DEFAULT_TIMEOUT_S = 10.0
_TEARDOWN_TIMEOUT_S = 5.0


def resolve_ts_lsp() -> Optional[List[str]]:
    """Return the argv prefix to invoke `typescript-language-server`, or `None` if
    no usable toolchain exists. Mirrors `tsc.resolve_tsc()` exactly: requires both
    the local bin (from `npm i -D typescript-language-server` in `SET_DIR`) and
    `node` on PATH (the bin's shebang needs it)."""
    if TS_LSP_BIN.exists() and shutil.which("node") is not None:
        return [str(TS_LSP_BIN)]
    return None


def _map_diagnostic(raw: dict, source: str) -> Diagnostic:
    start = raw["range"]["start"]
    line, col = start["line"] + 1, start["character"] + 1   # LSP is 0-indexed
    return Diagnostic(
        code=f"TS{raw['code']}", line=line, col=col, message=raw["message"],
        offset=line_col_to_offset(source, line, col),
        source="ts", severity=raw.get("severity", 1),
    )


class TsLspOracle:
    """One persistent `typescript-language-server --stdio` process, restarted at
    most once per call if it has died.

    Scratch dir is nested under `scratch_parent` (default `SET_DIR`) -- load-bearing
    exactly as `TscRunner`'s is: TypeScript's default `typeRoots` walk needs to find
    `SET_DIR/node_modules/@types/node` (ambient `console`, etc.) from wherever the
    scratch dir sits, confirmed empirically (a scratch dir elsewhere spuriously
    reports `Cannot find name 'console'`).

    Each `diagnostics()` call uses a **fresh, unique candidate uri**
    (`cand_{n}.ts`) -- reusing one uri risks a stale `publishDiagnostics` from the
    *previous* candidate's re-check landing after the new file is opened, which
    would silently attribute one candidate's errors to another.
    """

    def __init__(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S,
                 tsconfig: Path = DEFAULT_TSCONFIG_PATH,
                 scratch_parent: Path = SET_DIR,
                 argv: Optional[List[str]] = None) -> None:
        self.argv = argv if argv is not None else resolve_ts_lsp()
        if self.argv is None:
            raise RuntimeError("no typescript-language-server toolchain resolvable "
                                f"(run `npm i -D typescript-language-server` in {scratch_parent})")
        self.timeout_s = timeout_s
        self.tsconfig = tsconfig

        self.n_calls = 0
        self.wall_s = 0.0
        self.n_timeouts = 0
        self.n_restarts = 0

        self._tmpdir_obj = tempfile.TemporaryDirectory(dir=str(scratch_parent), prefix="lsp_scratch_")
        self.scratch_dir = Path(self._tmpdir_obj.name)
        (self.scratch_dir / "tsconfig.json").write_text(
            tsconfig.read_text(encoding="utf-8"), encoding="utf-8")

        self._next_cand = 0
        self._diag_lock = threading.Lock()
        self._diag_events: Dict[str, threading.Event] = {}
        self._diag_payloads: Dict[str, List[dict]] = {}

        self._proc: Optional[subprocess.Popen] = None
        self._endpoint: Optional[JsonRpcEndpoint] = None
        self._start_server()

    # ----------------------------------------------------------------- #
    # server lifecycle
    # ----------------------------------------------------------------- #

    def _on_notification(self, msg: dict) -> None:
        if msg.get("method") != "textDocument/publishDiagnostics":
            return
        params = msg.get("params") or {}
        uri = params.get("uri")
        if uri is None:
            return
        with self._diag_lock:
            event = self._diag_events.get(uri)
            if event is None:
                return  # not (or no longer) awaited -- drop it; no stale bleed
            self._diag_payloads[uri] = params.get("diagnostics", [])
            event.set()

    def _start_server(self) -> None:
        self._proc, self._endpoint = spawn(
            self.argv + ["--stdio"], cwd=str(self.scratch_dir),
            on_notification=self._on_notification)
        self._endpoint.request("initialize", {
            "processId": os.getpid(),
            "rootUri": self.scratch_dir.as_uri(),
            "capabilities": {"textDocument": {"publishDiagnostics": {}}},
            "initializationOptions": {},
        }, timeout=self.timeout_s)
        self._endpoint.notify("initialized", {})

    def _ensure_alive(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self.n_restarts += 1
        self._teardown_process()
        self._start_server()

    def _teardown_process(self) -> None:
        if self._endpoint is not None:
            self._endpoint.close()
            self._endpoint = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=_TEARDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=_TEARDOWN_TIMEOUT_S)
            except OSError:
                pass  # already dead
            self._proc = None

    # ----------------------------------------------------------------- #
    # the oracle contract
    # ----------------------------------------------------------------- #

    def _next_uri(self) -> Tuple[str, Path]:
        with self._diag_lock:
            n = self._next_cand
            self._next_cand += 1
        path = self.scratch_dir / f"cand_{n}.ts"
        return path.as_uri(), path

    def diagnostics(self, source: str) -> List[Diagnostic]:
        """Open `source` as a fresh candidate document, wait for its
        `publishDiagnostics`, and return every `severity == 1` finding, UNFILTERED
        by `is_incomplete`/frontier (the caller applies
        `diagnostics.filter_diagnostics`, same as `TscRunner`). Never raises on
        timeout or a dead server -- returns `[]` and counts it."""
        self._ensure_alive()
        uri, path = self._next_uri()
        event = threading.Event()
        with self._diag_lock:
            self._diag_events[uri] = event
            self._diag_payloads.pop(uri, None)

        t0 = time.monotonic()
        path.write_text(source, encoding="utf-8")
        self._endpoint.notify("textDocument/didOpen", {
            "textDocument": {"uri": uri, "languageId": "typescript", "version": 1, "text": source},
        })
        got = event.wait(self.timeout_s)

        with self._diag_lock:
            payload = self._diag_payloads.pop(uri, None)
            self._diag_events.pop(uri, None)

        self._close_candidate(uri, path)
        self.wall_s += time.monotonic() - t0
        self.n_calls += 1

        if not got or payload is None:
            self.n_timeouts += 1
            return []

        return [_map_diagnostic(d, source) for d in payload if d.get("severity", 1) == 1]

    def _close_candidate(self, uri: str, path: Path) -> None:
        if self._endpoint is not None:
            try:
                self._endpoint.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
            except (OSError, ValueError):
                pass  # server already gone -- nothing left to tell it
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def codes(self, source: str) -> List[str]:
        """`[d.code for d in self.diagnostics(source)]` -- mirrors `TscRunner.codes`."""
        return [d.code for d in self.diagnostics(source)]

    def close(self) -> None:
        self._teardown_process()
        self._tmpdir_obj.cleanup()

    def __enter__(self) -> "TsLspOracle":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
