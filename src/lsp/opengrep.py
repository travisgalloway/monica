"""opengrep oracle arm -- Stage A's second oracle, carrying the custom, frozen
`eval_sets/opengrep_rules/correctness.yaml` ruleset over `opengrep lsp` (an
undocumented, unlisted subcommand -- absent from `opengrep --help`'s subcommand
descriptions in some builds, present as a bare `lsp` entry; there is no public
protocol doc, so everything below was verified against the real pinned `v1.25.0`
binary, not assumed from `tsc`/tsserver conventions).

Three real, confirmed quirks this module exists to work around (see
`eval_sets/opengrep_rules/README.md` for the install/protocol-shape notes; this
docstring covers the two runtime bugs found while getting a *repeated*
single-file rescan loop working, which is exactly this harness's access
pattern):

  - **`initializationOptions` nesting / `onlyGitDirty`** -- see
    `eval_sets/opengrep_rules/README.md`.
  - **A workspace-target-cache bug on brand-new files.** opengrep's LSP
    `scan_file` handler (`Scan_helpers.ml`) only scans a uri if it is a member
    of the workspace's *target cache*, built once by walking `rootUri` at
    `initialize` time. If a candidate file didn't exist yet at that moment, the
    handler tries to recompute the cache -- but then re-checks membership
    against the STALE pre-recompute list, so a freshly-`didOpen`ed file is
    silently treated as having zero scan targets (an empty `publishDiagnostics`
    with no error, indistinguishable from "no findings"). Confirmed by tracing
    the actual OCaml source and reproducing it directly. Fix: use **one
    candidate file, created on disk BEFORE `initialize`** (so it's in the
    cache from the start) and reused for the oracle's whole lifetime --
    exactly `TscRunner`'s "one `snippet.ts` overwritten per call" pattern, not
    `TsLspOracle`'s fresh-uri-per-call pattern (which is what surfaces this bug
    in the first place).
  - **No protocol-level scan-readiness signal, and naive "wait for the next
    notification" races a still-in-flight prior scan.** Neither the
    `initialize` response nor `semgrep/rulesRefreshed` reliably means "the next
    scan will see the loaded ruleset" (confirmed by direct repeated
    measurement: an immediate post-refresh scan came back empty on an
    unambiguous positive in multiple trials). And once the same file is
    rescanned repeatedly, waiting for "any new `publishDiagnostics`" can
    resolve on a **stale, still-in-flight response to the PREVIOUS scan**,
    silently misattributing old findings to new content (also confirmed
    directly, and NOT fully fixed by acting on the first newer generation
    alone -- a second, even-newer notification can still arrive a few hundred
    ms later correcting the first, so `_rescan` also **settles**: after seeing
    one new generation it keeps listening for `_SETTLE_S` more and takes the
    LAST one observed, not the first). `_warmup()` reuses this same mechanism
    with a known-positive canary, retried (via repeated `didSave`) until it
    actually fires, before the server is considered ready. Bounded by
    `_WARMUP_TIMEOUT_S` -- a startup cost paid once per process (construction
    + each restart), never a hang.

**Known residual limitation (measured, not theoretical):** even with both
fixes above, a stress test of ~80 sequential rescans against one long-lived
`opengrep lsp` process saw a **~10% rate of a genuine, full-timeout stall**
(no response at all, confirmed not just "slow" -- raising the per-call timeout
to 25s did not recover it) that appears to build up under sustained repeated
single-file rescanning of the same process. `diagnostics()` still never hangs
past its own `timeout_s` (a stalled call is counted in `n_timeouts` and returns
`[]`, exactly like any other timeout), but this means the opengrep arm is
**not yet reliable enough to trust unconditionally at full-run scale** --
Stage A's shipped default (`--oracle ts` in both drivers, and
`CompositeOracle(kind="ts")` by default) does not depend on it. `--oracle
opengrep`/`--oracle both` are implemented and functionally correct when they
respond, but should be treated as experimental until this is root-caused
further or fixed upstream.

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
from typing import List, Optional, Tuple

from .diagnostics import Diagnostic, line_col_to_offset
from .jsonrpc import JsonRpcEndpoint, spawn
from .tsc import SET_DIR

RULES_DIR = SET_DIR.parent / "opengrep_rules"

_DEFAULT_TIMEOUT_S = 10.0
_TEARDOWN_TIMEOUT_S = 5.0
_WARMUP_TIMEOUT_S = 15.0
_WARMUP_POLL_S = 1.5
_POLL_INTERVAL_S = 0.5   # how often _rescan re-checks its own deadline
_SETTLE_S = 0.3          # grace window to absorb a still-arriving, newer generation

# Self-verifying warmup canary: this ruleset's own `loop-bound-off-by-one` positive
# fixture (`tests/test_opengrep.py`) -- a known-positive input from the same
# pre-registered taxonomy, unrelated to any eval prompt (a startup health-check,
# not a measurement).
_WARMUP_CANARY_SOURCE = (
    "function sumAll(arr: number[]): number {\n"
    "  let total = 0;\n"
    "  for (let i = 0; i <= arr.length; i++) {\n"
    "    total += arr[i];\n"
    "  }\n"
    "  return total;\n"
    "}\n"
)


def resolve_opengrep() -> Optional[List[str]]:
    """Return the argv prefix to invoke `opengrep`, or `None` if no usable
    toolchain exists. Unlike `resolve_tsc`/`resolve_ts_lsp` (which check a local
    `node_modules/.bin` shim), opengrep is a standalone binary expected on
    `PATH` -- pinned `v1.25.0`, cosign-verified at install time (see
    `eval_sets/opengrep_rules/README.md`)."""
    path = shutil.which("opengrep")
    if path is None:
        return None
    return [path]


def _strip_rules_dir_prefix(code: str) -> str:
    """A finding's `code` comes back as the bare rule id when `RULES_DIR` sits
    inside a recognized git project (confirmed for this repo's layout); an
    early probe against a rules dir *outside* any git repo instead saw a
    `"<rules-dir-basename>.<rule-id>"` form. Strip that prefix defensively so
    either shape maps to the same bare id."""
    prefix = f"{RULES_DIR.name}."
    return code[len(prefix):] if code.startswith(prefix) else code


def _map_finding(raw: dict, source: str) -> Diagnostic:
    start = raw["range"]["start"]
    line, col = start["line"] + 1, start["character"] + 1   # LSP is 0-indexed
    code = _strip_rules_dir_prefix(raw["code"])
    return Diagnostic(
        code=code, line=line, col=col, message=raw["message"],
        offset=line_col_to_offset(source, line, col),
        source="opengrep", severity=raw.get("severity", 1),
    )


class OpengrepOracle:
    """One persistent `opengrep lsp` process carrying the frozen custom
    ruleset. Same contract/attrs as `TsLspOracle`: `diagnostics`, `codes`,
    `close`, `__enter__`/`__exit__`, `n_calls`, `wall_s`, `n_timeouts`,
    `n_restarts`. Not safe for concurrent use.

    Unlike `TsLspOracle`, this oracle reuses **one** candidate file for its
    whole lifetime (overwritten per call, `TscRunner`-style) rather than a
    fresh uri per call -- see the module docstring for why a fresh-uri
    approach silently breaks against this specific server.
    """

    def __init__(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S,
                 rules_dir: Path = RULES_DIR,
                 scratch_parent: Path = SET_DIR,
                 argv: Optional[List[str]] = None) -> None:
        self.argv = argv if argv is not None else resolve_opengrep()
        if self.argv is None:
            raise RuntimeError("no opengrep toolchain resolvable (install opengrep "
                                "and put it on PATH -- see eval_sets/opengrep_rules/README.md)")
        self.timeout_s = timeout_s
        self.rules_dir = rules_dir

        self.n_calls = 0
        self.wall_s = 0.0
        self.n_timeouts = 0
        self.n_restarts = 0

        self._tmpdir_obj = tempfile.TemporaryDirectory(dir=str(scratch_parent), prefix="lsp_scratch_")
        self.scratch_dir = Path(self._tmpdir_obj.name)
        # Created BEFORE `initialize` -- see module docstring's target-cache note.
        self._cand_path = self.scratch_dir / "cand.ts"
        self._cand_path.write_text("", encoding="utf-8")
        self._cand_uri = self._cand_path.as_uri()

        self._diag_lock = threading.Lock()
        self._diag_event = threading.Event()
        self._diag_seq = 0
        self._diag_payload: List[dict] = []

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
        if params.get("uri") != self._cand_uri:
            return  # not our (only) tracked document -- ignore
        with self._diag_lock:
            self._diag_payload = params.get("diagnostics", [])
            self._diag_seq += 1
            self._diag_event.set()

    def _start_server(self) -> None:
        self._proc, self._endpoint = spawn(
            self.argv + ["lsp"], cwd=str(self.scratch_dir),
            on_notification=self._on_notification)
        self._endpoint.request("initialize", {
            "processId": os.getpid(),
            "rootUri": self.scratch_dir.as_uri(),
            "capabilities": {},
            "initializationOptions": {
                "scan": {"configuration": [str(self.rules_dir)], "onlyGitDirty": False},
            },
        }, timeout=self.timeout_s)
        self._endpoint.notify("initialized", {})
        self._endpoint.notify("textDocument/didOpen", {
            "textDocument": {"uri": self._cand_uri, "languageId": "typescript",
                              "version": 1, "text": ""},
        })
        self._warmup()

    def _warmup(self) -> None:
        """Block (bounded by `_WARMUP_TIMEOUT_S`) until a known-positive canary
        actually produces a finding -- see the module docstring for why neither
        the `initialize` response nor `semgrep/rulesRefreshed` is a reliable
        readiness signal on their own."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < _WARMUP_TIMEOUT_S:
            got, payload = self._rescan(_WARMUP_CANARY_SOURCE, _WARMUP_POLL_S)
            if got and payload:
                return
        raise RuntimeError(
            f"opengrep lsp never became ready within {_WARMUP_TIMEOUT_S}s "
            "(warmup canary never fired) -- see eval_sets/opengrep_rules/README.md")

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

    def _rescan(self, source: str, timeout_s: float) -> Tuple[bool, List[dict]]:
        """Overwrite the persistent candidate file with `source`, trigger a
        rescan via `didSave`, and wait for `publishDiagnostics` generations
        to **settle** (not just "the first new one" -- see module docstring: a
        second, newer generation can still arrive `_SETTLE_S` later correcting
        the first). Bounded by `timeout_s` total; returns `(False, [])` rather
        than hanging past it -- including the measured residual case where the
        server simply never responds at all (see module docstring's Known
        residual limitation).
        """
        with self._diag_lock:
            before_seq = self._diag_seq
            self._diag_event.clear()
        self._cand_path.write_text(source, encoding="utf-8")
        self._endpoint.notify("textDocument/didSave", {
            "textDocument": {"uri": self._cand_uri}, "text": source,
        })

        deadline = time.monotonic() + timeout_s
        got_once = False
        payload: List[dict] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return got_once, payload
            wait_for = _SETTLE_S if got_once else min(remaining, _POLL_INTERVAL_S)
            if self._diag_event.wait(min(remaining, wait_for)):
                with self._diag_lock:
                    if self._diag_seq > before_seq:
                        got_once = True
                        payload = list(self._diag_payload)
                        before_seq = self._diag_seq  # absorb further bumps within the settle window
                    self._diag_event.clear()
            elif got_once:
                return True, payload

    def diagnostics(self, source: str) -> List[Diagnostic]:
        """Rescan the persistent candidate document with `source` and return
        every finding as a `Diagnostic` (`source="opengrep"`). Never raises on
        timeout or a dead server -- returns `[]` and counts it."""
        self._ensure_alive()
        t0 = time.monotonic()
        got, payload = self._rescan(source, self.timeout_s)
        self.wall_s += time.monotonic() - t0
        self.n_calls += 1

        if not got:
            self.n_timeouts += 1
            return []
        return [_map_finding(d, source) for d in payload]

    def codes(self, source: str) -> List[str]:
        """`[d.code for d in self.diagnostics(source)]` -- mirrors `TscRunner.codes`."""
        return [d.code for d in self.diagnostics(source)]

    def close(self) -> None:
        self._teardown_process()
        self._tmpdir_obj.cleanup()

    def __enter__(self) -> "OpengrepOracle":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
