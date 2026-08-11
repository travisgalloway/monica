"""Warm, multi-op `typescript-language-server` client -- the counterpart to
`ts_lsp.py`'s `TsLspOracle` for two M12 consumers that need the OPPOSITE shape:

  - **#226 completion-list logit masking** needs a per-decode-step
    `textDocument/completion` against a warm, multi-file program -- one edit
    plus one query per generated token.
  - **The code eval suite's cross-file / type-aware arm** needs
    `definition`/`references`/`documentSymbol` over a real multi-file project.

`TsLspOracle` is deliberately **one-shot**: a fresh, never-reused `cand_{n}.ts`
URI per call, single-file, no `didChange`, no version counter, no cross-file
program (`ts_lsp.py:85-88`) -- that contract is load-bearing for the #194/#211
harness path and every `CompositeOracle` consumer (`oracle.py`). `TsLspService`
does not share it: it opens ONE project once, keeps documents open with
monotonically versioned `didChange` edits, and answers five more LSP ops in
addition to diagnostics. `ts_lsp.py` and `oracle.py` are UNTOUCHED by this
module -- both new and old share only the already fully op-agnostic
`jsonrpc.py`.

**No settle window on `diagnostics()`.** #211 established
(`docs/design/12-lsp-in-the-loop.md:664-700`) that on the pinned
`typescript-language-server` **5.3.0** the FIRST `publishDiagnostics` push is
semantically complete -- 0/97 #194 candidates disagreed between their first
push and the union of all pushes, and a crafted candidate carrying both a
syntactic and a distinct semantic defect coalesced into **one** push. The
server does publish up to 3 idempotent re-publishes ~1.2s apart, but the first
is already the answer. A settle/quiescence window (`opengrep.py`'s
`_SETTLE_S`) puts a floor of *seconds* on every call, which makes >=200
calls/s structurally unreachable -- so `diagnostics()` below is first-push-wins,
same as `TsLspOracle`. `opengrep.py:288-323` settles deliberately, for a
*different* server with a confirmed stale-generation race (its own module
docstring) -- that divergence is intentional. **Do not "fix" this module to
settle like opengrep.py; it would defeat the reason this class exists.**

5.3.0 also advertises **no** `capabilities.diagnosticProvider` in its
`initialize` result, so LSP 3.17 *pull* diagnostics (`textDocument/diagnostic`)
are not available on this pin -- push-and-wait, gated by version/seq (see
`diagnostics()`), is the only option.

**Most of the measured `diagnostics` latency is a client-side timer, not
type-checking (#278).** Two hardcoded timers in the pinned server, neither
with a configuration gate: `cli.mjs:17868`'s
`min(max(ceil(lineCount / 20), 300), 800)` ms delay on every `didChange`
(our short synthetic files hit the 300 ms floor), plus `cli.mjs:20516`'s
`pDebounce(..., 50)`. So a warm re-check on this pin costs >=350 ms of
*client* wait before tsserver is asked anything, independent of project
size -- a control probe sweeping project size (roughly flat ~350-380ms
across 10..5000 files) and edit-file length (tracking the clamp's own
formula from 350ms to 850ms) confirms this; the genuine recheck cost once
the debounce is subtracted out is ~11-31ms, inside a 50ms bar. See
`scripts/probe_ts_lsp_debounce.py` and #279 (a direct-`tsserver`
`semanticDiagnosticsSync` bypass that would sidestep the client-side
debounce entirely).

Not safe for concurrent use -- one service instance == one sequential
open/edit/query stream, matching `TsLspOracle`'s contract.

ABOVE THE SEAM -- stdlib only. No `mlx`/`torch` import anywhere in this module
(guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Set
from urllib.parse import unquote, urlparse

from .diagnostics import Diagnostic, line_col_to_offset, offset_to_lsp_position
from .jsonrpc import JsonRpcEndpoint, spawn
from .ts_lsp import resolve_ts_lsp
from .tsc import DEFAULT_TSCONFIG_PATH, SET_DIR

_DEFAULT_TIMEOUT_S = 10.0
_TEARDOWN_TIMEOUT_S = 5.0


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Location:
    uri: str
    path: str          # project-relative, for readable assertions/reports
    line: int           # 1-indexed, matching Diagnostic's convention
    col: int             # 1-indexed UTF-16 column, matching Diagnostic's convention
    end_line: int
    end_col: int


@dataclass(frozen=True)
class CompletionItem:
    label: str
    kind: Optional[int] = None
    detail: Optional[str] = None
    insert_text: Optional[str] = None
    sort_text: Optional[str] = None


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: int
    line: int
    col: int
    container: Optional[str] = None


# --------------------------------------------------------------------------- #
# Pure helpers -- no I/O, this is what the mechanism tests exercise directly
# --------------------------------------------------------------------------- #

def _content_changes(text: str, change_range: Optional[dict] = None) -> List[dict]:
    """Full-text change by default (`contentChanges: [{"text": text}]`, no
    `range` key) -- valid under sync kind Full (1) *and* Incremental (2) alike
    (a change event with no `range` means whole-document replacement).
    Structured so a *ranged* change is a drop-in later."""
    if change_range is None:
        return [{"text": text}]
    return [{"range": change_range, "text": text}]


def _did_change_params(uri: str, version: int, text: str) -> dict:
    return {"textDocument": {"uri": uri, "version": version},
            "contentChanges": _content_changes(text)}


def _text_document_position(uri: str, position: dict) -> dict:
    return {"textDocument": {"uri": uri}, "position": position}


def _references_params(uri: str, position: dict, include_declaration: bool) -> dict:
    params = _text_document_position(uri, position)
    params["context"] = {"includeDeclaration": include_declaration}
    return params


def _uri_to_relpath(uri: str, root: Path) -> str:
    """Inverse of `Path.as_uri()`, resolved relative to `root` (the scratch
    dir) for readable assertions/reports. Falls back to the absolute path if
    `uri` doesn't sit under `root` (shouldn't happen for this class -- every
    file lives in the scratch dir -- but never raise over it)."""
    parsed = urlparse(uri)
    p = Path(unquote(parsed.path))
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _map_location(raw, root: Path) -> List[Location]:
    """Normalize a `textDocument/definition` or `textDocument/references`
    result -- `null`, a single `Location` (`uri` + `range`), a single
    `LocationLink` (`targetUri` + `targetSelectionRange`/`targetRange`), or a
    list of either -- into `List[Location]`. We request `linkSupport: False`
    so a plain `Location` is expected, but map defensively regardless."""
    if raw is None:
        items = []
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]

    out: List[Location] = []
    for item in items:
        if "targetUri" in item:   # LocationLink
            uri = item["targetUri"]
            rng = item.get("targetSelectionRange") or item["targetRange"]
        else:                      # Location
            uri = item["uri"]
            rng = item["range"]
        start, end = rng["start"], rng["end"]
        out.append(Location(
            uri=uri, path=_uri_to_relpath(uri, root),
            line=start["line"] + 1, col=start["character"] + 1,
            end_line=end["line"] + 1, end_col=end["character"] + 1,
        ))
    return out


def _map_completion_items(raw) -> List[CompletionItem]:
    """`textDocument/completion` result is `CompletionItem[] | CompletionList
    {isIncomplete, items} | null` -- normalize all three. `isIncomplete` is not
    representable in a bare list, so the caller (`TsLspService.completions`)
    records it separately on `self.last_completion_incomplete` rather than
    silently treating a truncated list as complete."""
    if raw is None:
        return []
    items = raw.get("items", []) if isinstance(raw, dict) else raw
    out = []
    for it in items:
        out.append(CompletionItem(
            label=it.get("label", ""), kind=it.get("kind"),
            detail=it.get("detail"), insert_text=it.get("insertText"),
            sort_text=it.get("sortText"),
        ))
    return out


def _map_hover(contents) -> Optional[str]:
    """Normalize an LSP `Hover.contents` value -- `MarkupContent`
    (`{"kind","value"}`), a bare string, `MarkedString` (`{"language","value"}`),
    or a list of any of those -- to one string (list entries joined with
    `"\\n"`). `None` in (a `null` hover, or a hover with no contents) -> `None`
    out."""
    if contents is None:
        return None
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return contents.get("value")
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("value"):
                parts.append(item["value"])
        return "\n".join(parts) if parts else None
    return None


def _map_symbols(raw) -> List[Symbol]:
    """`textDocument/documentSymbol` result is `DocumentSymbol[]` (hierarchical
    -- has `children` + `selectionRange`) *or* `SymbolInformation[]` (flat --
    has `location` + `containerName`), or `null`. `DocumentSymbol` trees are
    flattened depth-first, carrying the parent's `name` as `container`."""
    if not raw:
        return []
    out: List[Symbol] = []

    def _flatten(items, container: Optional[str]) -> None:
        for it in items:
            if "selectionRange" in it:   # DocumentSymbol
                pos = it["selectionRange"]["start"]
                out.append(Symbol(name=it["name"], kind=it["kind"],
                                   line=pos["line"] + 1, col=pos["character"] + 1,
                                   container=container))
                children = it.get("children") or []
                if children:
                    _flatten(children, it["name"])
            else:                          # SymbolInformation
                loc = it["location"]
                pos = loc["range"]["start"]
                out.append(Symbol(name=it["name"], kind=it["kind"],
                                   line=pos["line"] + 1, col=pos["character"] + 1,
                                   container=it.get("containerName")))

    _flatten(raw, None)
    return out


# --------------------------------------------------------------------------- #
# Client capabilities
# --------------------------------------------------------------------------- #

_CLIENT_CAPABILITIES = {
    "textDocument": {
        # Sync KIND is a SERVER capability (reported in the initialize result); the
        # client only declares these flags. didChange payloads below are full-text,
        # which is valid under Full (1) and Incremental (2) sync alike.
        "synchronization": {"dynamicRegistration": False, "willSave": False,
                            "willSaveWaitUntil": False, "didSave": True},
        # versionSupport asks the server to stamp publishDiagnostics with the
        # document version -- how diagnostics() tells a post-edit push from a
        # stale pre-edit one. Optional in LSP 3.16; there is a seq-based fallback.
        "publishDiagnostics": {"relatedInformation": False, "versionSupport": True},
        "definition": {"dynamicRegistration": False, "linkSupport": False},
        "references": {"dynamicRegistration": False},
        "completion": {
            "dynamicRegistration": False,
            "contextSupport": True,
            "completionItem": {"snippetSupport": False,
                               "documentationFormat": ["plaintext"]},
        },
        "hover": {"dynamicRegistration": False,
                  "contentFormat": ["plaintext", "markdown"]},
        "documentSymbol": {"dynamicRegistration": False,
                           "hierarchicalDocumentSymbolSupport": True},
    },
    "workspace": {"workspaceFolders": True},
}


# --------------------------------------------------------------------------- #
# TsLspService
# --------------------------------------------------------------------------- #

class TsLspService:
    """One persistent `typescript-language-server --stdio` process carrying a
    warm, multi-file program: stable URIs, monotonically versioned
    `didChange` edits, and `definition`/`references`/`completions`/
    `quickinfo`/`document_symbols` in addition to `diagnostics`. See the
    module docstring for why this is a NEW class rather than an extension of
    `TsLspOracle`.

    Lazy `didOpen`: `open_project` materializes every file on disk (so the
    tsconfig's whole-directory `include` puts every file in the *program*,
    letting cross-file `definition`/`references` resolve without any file
    being open) but only `didOpen`s the warm-up file. Every other document is
    opened lazily by `_ensure_open` the first time an op touches it -- only
    *diagnostics* need a document to be open at all (servers publish
    diagnostics for open documents only), so this is the difference between a
    bench that measures a warm program and one that measures 5000 didOpens.

    Not safe for concurrent use.
    """

    def __init__(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S,
                 tsconfig: Path = DEFAULT_TSCONFIG_PATH,
                 scratch_parent: Path = SET_DIR,
                 initialization_options: Optional[dict] = None,
                 argv: Optional[List[str]] = None) -> None:
        self.argv = argv if argv is not None else resolve_ts_lsp()
        if self.argv is None:
            raise RuntimeError("no typescript-language-server toolchain resolvable "
                                f"(run `npm i -D typescript-language-server` in {scratch_parent})")
        self.timeout_s = timeout_s
        self.tsconfig = tsconfig
        self.initialization_options = initialization_options or {}

        self.n_calls = 0
        self.wall_s = 0.0
        self.n_timeouts = 0
        self.n_no_publish = 0
        self.n_restarts = 0
        self.op_counts: Dict[str, int] = {}
        self.op_wall_s: Dict[str, float] = {}
        self.cold_load_s: Optional[float] = None
        self.last_completion_incomplete: Optional[bool] = None

        # Nesting under `scratch_parent` (default `SET_DIR`) is load-bearing --
        # see ts_lsp.py:79-83: TypeScript's default `typeRoots` walk needs to
        # find `SET_DIR/node_modules/@types/node` from wherever the scratch dir
        # sits. `.gitignore`'s `lsp_scratch_*` already covers the prefix.
        self._tmpdir_obj = tempfile.TemporaryDirectory(dir=str(scratch_parent), prefix="lsp_scratch_")
        self.scratch_dir = Path(self._tmpdir_obj.name)
        # The pinned tsconfig has no include/files, so it picks up every .ts
        # under the scratch dir recursively -- what makes a warm multi-file
        # program work with zero extra config.
        (self.scratch_dir / "tsconfig.json").write_text(
            tsconfig.read_text(encoding="utf-8"), encoding="utf-8")

        self._project_opened = False
        self._files: Dict[str, str] = {}          # project-relative path -> text
        self._versions: Dict[str, int] = {}        # project-relative path -> version
        self._open: Set[str] = set()               # project-relative paths, didOpen'd

        self._diag_lock = threading.Lock()
        self._diag_events: Dict[str, threading.Event] = {}
        self._diag_state: Dict[str, dict] = {}      # uri -> {"seq", "version", "payload"}
        self._diag_seq_counter = 0
        self._update_marker: Dict[str, int] = {}    # uri -> seq counter as-of last edit/open

        self.server_sync_kind: Optional[int] = None
        self.server_capabilities: dict = {}

        self._proc: Optional[subprocess.Popen] = None
        self._endpoint: Optional[JsonRpcEndpoint] = None
        self._start_server()

    # ----------------------------------------------------------------- #
    # path/uri helpers
    # ----------------------------------------------------------------- #

    def _abs_path(self, path: str) -> Path:
        return self.scratch_dir / path

    def _uri(self, path: str) -> str:
        return self._abs_path(path).as_uri()

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
            self._diag_seq_counter += 1
            self._diag_state[uri] = {
                "seq": self._diag_seq_counter,
                "version": params.get("version"),
                "payload": params.get("diagnostics", []),
            }
            event = self._diag_events.get(uri)
            if event is not None:
                event.set()

    def _start_server(self) -> None:
        self._proc, self._endpoint = spawn(
            self.argv + ["--stdio"], cwd=str(self.scratch_dir),
            on_notification=self._on_notification)
        result = self._endpoint.request("initialize", {
            "processId": os.getpid(),
            "rootUri": self.scratch_dir.as_uri(),
            "capabilities": _CLIENT_CAPABILITIES,
            "initializationOptions": self.initialization_options,
        }, timeout=self.timeout_s)
        self._endpoint.notify("initialized", {})

        self.server_capabilities = (result or {}).get("capabilities", {})
        sync = self.server_capabilities.get("textDocumentSync")
        if isinstance(sync, dict):
            self.server_sync_kind = sync.get("change")
        elif isinstance(sync, int):
            self.server_sync_kind = sync
        else:
            self.server_sync_kind = None

    def _ensure_alive(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._restart()

    def _restart(self) -> None:
        """Restart lost the warm program (every open document + the server's
        project state); files are already on disk (both `open_project` and
        `update` write through), so re-`didOpen` every previously-open path at
        its CURRENT version -- version counters stay monotonic across restarts
        (LSP requires per-document monotonicity; reusing 1 after an edit would
        let a stale push masquerade as current). `_diag_state` is cleared so no
        pre-restart push is mistaken for a post-restart one."""
        self.n_restarts += 1
        open_paths = list(self._open)
        self._teardown_process()
        with self._diag_lock:
            self._diag_state.clear()
            self._diag_events.clear()
            self._update_marker.clear()
        self._open.clear()
        self._start_server()
        for path in open_paths:
            self._ensure_open(path)

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
    # project / document lifecycle
    # ----------------------------------------------------------------- #

    def open_project(self, files: Mapping[str, str], *, warm_path: Optional[str] = None) -> None:
        """Materialize `files` (project-relative path -> text) on disk, then
        `didOpen` exactly one (the first key, or `warm_path`) to force
        tsserver to load the configured project, blocking on its first
        `publishDiagnostics` (bounded by `timeout_s`, counted as a timeout,
        never a hang). Records `self.cold_load_s`. One service instance ==
        one project -- calling this twice is a `RuntimeError`."""
        if self._project_opened:
            raise RuntimeError("open_project called twice -- one TsLspService "
                                "instance == one project")
        if not files:
            raise RuntimeError("open_project requires at least one file")
        self._ensure_alive()

        for rel, text in files.items():
            abs_path = self._abs_path(rel)
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(text, encoding="utf-8")
            self._files[rel] = text
            self._versions[rel] = 1
        self._project_opened = True

        warm = warm_path if warm_path is not None else next(iter(files))
        uri = self._uri(warm)
        t0 = time.monotonic()
        with self._diag_lock:
            event = threading.Event()
            self._diag_events[uri] = event
        self._ensure_open(warm)
        # Still n_timeouts, not n_no_publish (#278): a didOpen with no prior
        # entry for this URI cannot hit cli.mjs:20524's "both sets empty"
        # suppression -- there IS no previous set -- so no push here is a
        # real cold-load failure, not the clean-document ambiguity.
        got = event.wait(self.timeout_s)
        with self._diag_lock:
            self._diag_events.pop(uri, None)
        self.cold_load_s = time.monotonic() - t0
        if not got:
            self.n_timeouts += 1

    def _notify_retrying(self, method: str, params: dict) -> None:
        """`self._endpoint.notify(...)`, retried once after `_restart()` if the
        write itself raises `OSError` (e.g. `BrokenPipeError` -- the server died
        in the narrow window between `_ensure_alive()`'s `poll()` and this
        write). Without this, that race would surface as an unhandled
        exception from `_ensure_open`/`update`, breaking `diagnostics()`'s
        documented "never raises on a dead server" contract two frames up."""
        try:
            self._endpoint.notify(method, params)
        except OSError:
            self._restart()
            self._endpoint.notify(method, params)

    def _ensure_open(self, path: str) -> None:
        if path in self._open:
            return
        uri = self._uri(path)
        text = self._files[path]
        with self._diag_lock:
            self._update_marker.setdefault(uri, self._diag_seq_counter)
        self._notify_retrying("textDocument/didOpen", {
            "textDocument": {"uri": uri, "languageId": "typescript",
                              "version": self._versions[path], "text": text},
        })
        self._open.add(path)

    def update(self, path: str, text: str) -> int:
        """`_ensure_open(path)`, write `text` through to disk (so a restart
        can recover it and tsserver's on-disk view stays consistent), bump the
        version, and send `textDocument/didChange` (full-text). Returns the
        new version."""
        self._ensure_alive()
        self._ensure_open(path)
        uri = self._uri(path)
        self._abs_path(path).write_text(text, encoding="utf-8")
        self._files[path] = text
        self._versions[path] += 1
        version = self._versions[path]
        with self._diag_lock:
            self._update_marker[uri] = self._diag_seq_counter
            # A push ALREADY RECORDED before this edit is not an answer to it.
            # On the pinned 5.3.0, `publishDiagnostics` carries no `version`
            # (cli.mjs:20535 publishes {uri, diagnostics} only), so
            # `_push_is_current` falls through to seq-vs-marker -- and the
            # marker is taken here, an instant before the didChange below.
            # Without this pop, a pre-edit push satisfies `seq > marker` and is
            # returned as the post-edit answer in ~0.3ms (#278: the min 0.292ms
            # mode, ~28 of 200 samples in #277's run).
            self._diag_state.pop(uri, None)
        self._notify_retrying("textDocument/didChange", _did_change_params(uri, version, text))
        return version

    # ----------------------------------------------------------------- #
    # diagnostics -- version-aware first-push-wins
    # ----------------------------------------------------------------- #

    def _push_is_current(self, state: dict, want_version: int, marker: int) -> bool:
        version = state.get("version")
        if version is not None:
            return version == want_version
        # Server omits `version` (optional even with versionSupport: True) --
        # fall back to seq-vs-last-edit comparison. Never assume a push is
        # current just because it's the latest one seen.
        #
        # On the pinned typescript-language-server 5.3.0 this fallback is the
        # ONLY live path: `version` is never populated (cli.mjs:20535 publishes
        # `{uri, diagnostics}` only, no `version` field). The branch above is
        # kept for a future server that does populate it, but its correctness
        # on THIS pin rests entirely on `update()` popping `_diag_state[uri]`
        # before bumping `_update_marker` (#278) -- otherwise a push recorded
        # before the marker-taking instant still satisfies `seq > marker`.
        return state["seq"] > marker

    def diagnostics(self, path: str) -> List[Diagnostic]:
        """Return every `severity == 1` diagnostic for `path`'s CURRENT
        version. **Version-aware first-push-wins, no settle window** (see the
        module docstring's #211 evidence). If a push already recorded for this
        document matches the current version (or, when the server omits
        `version`, post-dates the last `update()`/open), it is returned
        immediately from a cache -- the honest fast path for "diagnose the
        same unchanged document again", NOT a re-check. Otherwise waits up to
        `timeout_s` for a matching push.

        `[]` is returned for TWO distinct situations that are **not
        distinguishable** for an already-clean document on this pin: a
        genuinely clean re-check, and "nothing has been published since the
        edit" (`n_no_publish`, #278) -- `cli.mjs:20524` returns early and
        never publishes when both the previous and new diagnostic sets are
        empty, so a benign edit to an already-clean document may legitimately
        never get a push. A stale, version-tagged push that fails
        `_push_is_current` is a real timeout (`n_timeouts`) -- that case IS
        distinguishable, because something was published, it just wasn't
        current. Never raises on timeout or a dead server, `TsLspOracle`'s
        contract."""
        self._ensure_alive()
        self._ensure_open(path)
        uri = self._uri(path)
        want_version = self._versions[path]

        t0 = time.monotonic()
        with self._diag_lock:
            state = self._diag_state.get(uri)
            marker = self._update_marker.get(uri, 0)
            if state is not None and self._push_is_current(state, want_version, marker):
                payload = list(state["payload"])
                self._record_op("diagnostics", time.monotonic() - t0)
                return self._filter_diagnostics_payload(payload, self._files[path])
            event = threading.Event()
            self._diag_events[uri] = event

        # No settle window -- see module docstring. The pinned
        # typescript-language-server 5.3.0's FIRST publishDiagnostics push for
        # a document is already semantically complete; waiting for quiescence
        # here (opengrep.py's _SETTLE_S idiom) would put a floor of seconds on
        # every call and make >=200 calls/s structurally unreachable. Do not
        # "fix" this to settle like opengrep.py -- that divergence is
        # intentional (see the module docstring).
        event.wait(self.timeout_s)
        with self._diag_lock:
            self._diag_events.pop(uri, None)
            state = self._diag_state.get(uri)
            marker = self._update_marker.get(uri, 0)

        self._record_op("diagnostics", time.monotonic() - t0)

        if state is None:
            # NOTHING was published for this document within timeout_s. On this
            # pin that is AMBIGUOUS by construction: cli.mjs:20524 returns early
            # when the previous and new diagnostic sets are BOTH empty, so a
            # benign edit to an already-clean document legitimately never gets a
            # push -- indistinguishable from a server that never answered.
            # Counted separately from n_timeouts so the ambiguity is visible in
            # the stats instead of silently folded into either a clean result or
            # a failure.
            self.n_no_publish += 1
            return []
        if not self._push_is_current(state, want_version, marker):
            self.n_timeouts += 1
            return []
        return self._filter_diagnostics_payload(list(state["payload"]), self._files[path])

    def _filter_diagnostics_payload(self, payload: List[dict], text: str) -> List[Diagnostic]:
        # Re-expressed (NOT imported) from ts_lsp.py:65's `_map_diagnostic` --
        # that helper is private and takes the source text positionally.
        # Trap A: LSP's integer `code` must become f"TS{code}" or
        # `is_incomplete` breaks. Trap B: only severity == 1 (Error) survives,
        # or tsserver's hints inflate the set (ts_lsp.py:16-24). Pinned
        # independently for this class in tests/test_ts_service_mechanism.py.
        out = []
        for d in payload:
            if d.get("severity", 1) != 1:
                continue
            start = d["range"]["start"]
            line, col = start["line"] + 1, start["character"] + 1
            out.append(Diagnostic(
                code=f"TS{d['code']}", line=line, col=col, message=d["message"],
                offset=line_col_to_offset(text, line, col),
                source="ts", severity=d.get("severity", 1),
            ))
        return out

    # ----------------------------------------------------------------- #
    # the other four ops
    # ----------------------------------------------------------------- #

    def _record_op(self, op_name: str, elapsed: float) -> None:
        self.wall_s += elapsed
        self.n_calls += 1
        self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
        self.op_wall_s[op_name] = self.op_wall_s.get(op_name, 0.0) + elapsed

    def _timed_request(self, op_name: str, fn: Callable[[], object], *, empty):
        """`_ensure_alive()`, run `fn()`, record cost. A `TimeoutError` or
        `ConnectionError` is counted (`n_timeouts`) and returns `empty` --
        never raises. `ConnectionError` already covers the write-side dead-
        server case (`BrokenPipeError` etc. are `ConnectionError` subclasses
        in the stdlib, unlike the bare `notify()` calls in `_ensure_open`/
        `update` -- see `_notify_retrying` -- those had no handler at all). A
        JSON-RPC `error` reply (`RuntimeError` from `jsonrpc.py:210`) is
        RE-RAISED, not swallowed: a server that rejects our params is a bug in
        us, and hiding it as "no completions" would be exactly the BLIND
        failure mode this repo guards hardest."""
        self._ensure_alive()
        t0 = time.monotonic()
        try:
            result = fn()
        except (TimeoutError, ConnectionError):
            self.n_timeouts += 1
            result = empty
        finally:
            self._record_op(op_name, time.monotonic() - t0)
        return result

    def definition(self, path: str, offset: int) -> List[Location]:
        def _op():
            self._ensure_open(path)
            uri = self._uri(path)
            pos = offset_to_lsp_position(self._files[path], offset)
            raw = self._endpoint.request("textDocument/definition",
                                         _text_document_position(uri, pos),
                                         timeout=self.timeout_s)
            return _map_location(raw, self.scratch_dir)
        return self._timed_request("definition", _op, empty=[])

    def references(self, path: str, offset: int, *,
                    include_declaration: bool = False) -> List[Location]:
        def _op():
            self._ensure_open(path)
            uri = self._uri(path)
            pos = offset_to_lsp_position(self._files[path], offset)
            raw = self._endpoint.request(
                "textDocument/references",
                _references_params(uri, pos, include_declaration),
                timeout=self.timeout_s)
            return _map_location(raw, self.scratch_dir)
        return self._timed_request("references", _op, empty=[])

    def completions(self, path: str, offset: int) -> List[CompletionItem]:
        def _op():
            self._ensure_open(path)
            uri = self._uri(path)
            pos = offset_to_lsp_position(self._files[path], offset)
            raw = self._endpoint.request("textDocument/completion",
                                         _text_document_position(uri, pos),
                                         timeout=self.timeout_s)
            self.last_completion_incomplete = (
                bool(raw.get("isIncomplete")) if isinstance(raw, dict) else False)
            return _map_completion_items(raw)
        return self._timed_request("completions", _op, empty=[])

    def quickinfo(self, path: str, offset: int) -> Optional[str]:
        def _op():
            self._ensure_open(path)
            uri = self._uri(path)
            pos = offset_to_lsp_position(self._files[path], offset)
            raw = self._endpoint.request("textDocument/hover",
                                         _text_document_position(uri, pos),
                                         timeout=self.timeout_s)
            contents = raw.get("contents") if raw is not None else None
            return _map_hover(contents)
        return self._timed_request("quickinfo", _op, empty=None)

    def document_symbols(self, path: str) -> List[Symbol]:
        def _op():
            self._ensure_open(path)
            uri = self._uri(path)
            raw = self._endpoint.request("textDocument/documentSymbol",
                                         {"textDocument": {"uri": uri}},
                                         timeout=self.timeout_s)
            return _map_symbols(raw)
        return self._timed_request("document_symbols", _op, empty=[])

    # ----------------------------------------------------------------- #
    # misc
    # ----------------------------------------------------------------- #

    @property
    def calls_per_s(self) -> float:
        return self.n_calls / self.wall_s if self.wall_s else 0.0

    def close(self) -> None:
        self._teardown_process()
        self._tmpdir_obj.cleanup()

    def __enter__(self) -> "TsLspService":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
