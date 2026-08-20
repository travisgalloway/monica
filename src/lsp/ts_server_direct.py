"""Direct-`tsserver` diagnostics client (#279) -- a SECOND transport to the same
pinned TypeScript, bypassing `typescript-language-server` entirely.

#278 established that most of `ts_service.py`'s measured `diagnostics` latency
is not type-checking. Two hardcoded, non-configurable client-side timers inside
the pinned `typescript-language-server` **5.3.0** -- `cli.mjs:17868`'s
`min(max(ceil(lineCount / 20), 300), 800)` ms delay on every `didChange`, plus
`cli.mjs:20516`'s 50 ms `pDebounce` -- are spent *before tsserver is asked
anything*. The genuine recheck cost underneath is ~11-31 ms. This module talks
to `node_modules/typescript/lib/tsserver.js` in **tsserver's own protocol** and
asks for diagnostics **synchronously** (`syntacticDiagnosticsSync` +
`semanticDiagnosticsSync`), so neither timer exists on the path at all.

**Why this is a separate module and not a flag on `jsonrpc.py`.** tsserver's
wire protocol is asymmetric and its demux key differs:

  1. **Framing is asymmetric.** tsserver reads **newline-delimited JSON** on
     stdin but writes **`Content-Length: N\\r\\n\\r\\n` + JSON** on stdout. The
     read side is byte-identical to LSP, so `jsonrpc.read_message` is imported
     and reused unchanged -- a real reuse, not a copy. The write side is NOT:
     sending `encode_message`'s `Content-Length` header to tsserver's stdin
     hangs the child with no error. Do not "unify" `encode_command` with
     `jsonrpc.encode_message`; that asymmetry is the whole reason for the split.
  2. **Correlation key is `request_seq`, not `id`.** Responses are
     `{"seq":0,"type":"response","command":...,"request_seq":N,"success":bool,
     "body":...}`. `JsonRpcEndpoint`'s demux keys on `id` and cannot be reused.
  3. **Unsolicited events interleave** (`projectLoadingStart`,
     `projectLoadingFinish`, `telemetry`, `configFileDiag`). They carry
     `"type":"event"` and must never resolve a waiter.
  4. **Payload shape differs from LSP.** tsserver reports
     `{"start":{"line":2,"offset":7},"end":{...},"text":...,"code":2322,
     "category":"error"}` -- **1-based** `line`/`offset` (LSP is 0-based),
     integer `code`, and a string `category` where LSP has an integer
     `severity`. `_map_diagnostics` handles all three; the two traps
     `ts_lsp.py:16-24` names apply here identically (Trap A: `code` must become
     `f"TS{code}"` or `is_incomplete`'s `TS1xxx` prefix match breaks; Trap B:
     only `category == "error"` survives, or tsserver's suggestions inflate the
     set and invalidate any comparison against `tsc`).

**The second, quieter win: an unambiguous clean result.** `TsLspService`
cannot distinguish "the recheck came back clean" from "nothing was published"
(`n_no_publish`, #278 -- `cli.mjs:20524` returns early and never publishes when
the previous and new diagnostic sets are both empty). The `*Sync` commands
always answer, so `[]` here is an **answer**, not an absence. There is
deliberately no `n_no_publish` analogue on this class.

`open_project` / `update` / `diagnostics` are signature-compatible with
`TsLspService`'s so the bench and the parity gate can drive either without a
shim (pinned by
`tests/test_ts_server_direct_mechanism.py::test_signature_compatibility_with_ts_lsp_service`).
The other five LSP ops (`definition`/`references`/`completions`/`quickinfo`/
`document_symbols`) are deliberately NOT ported -- #279 is a diagnostics-latency
question, and `TsLspService` remains the LSP-conformant client of record.

Not safe for concurrent use -- one instance == one sequential open/edit/check
stream, matching `TsLspService`'s contract.

ABOVE THE SEAM -- stdlib only. No `mlx`/`torch` import anywhere in this module
(guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, BinaryIO, Callable, Deque, Dict, List, Mapping, Optional, Set, Tuple

from .diagnostics import Diagnostic, line_col_to_offset
from .jsonrpc import Waiter, read_message
from .tsc import DEFAULT_TSCONFIG_PATH, SET_DIR

TSSERVER_JS = SET_DIR / "node_modules" / "typescript" / "lib" / "tsserver.js"

_DEFAULT_TIMEOUT_S = 10.0
_TEARDOWN_TIMEOUT_S = 5.0
_STDERR_TAIL_LINES = 200

OnEvent = Callable[[dict], None]


def resolve_tsserver() -> Optional[List[str]]:
    """Return the argv prefix to invoke `tsserver`, or `None` if no usable
    toolchain exists. Mirrors `tsc.resolve_tsc()` / `ts_lsp.resolve_ts_lsp()`
    exactly: requires both the pinned local `typescript` package (from
    `npm install` in `SET_DIR`) and `node` on PATH, and **never raises** -- a
    node-less or uninstalled host gets a clean "unavailable" signal, not a
    `FileNotFoundError` out of `subprocess`.

    Unlike the `.bin/` shims `resolve_tsc`/`resolve_ts_lsp` return,
    `tsserver.js` has no shebang, so `node` is part of the argv rather than
    just a PATH precondition.
    """
    node = shutil.which("node")
    if TSSERVER_JS.exists() and node is not None:
        return [node, str(TSSERVER_JS)]
    return None


# --------------------------------------------------------------------------- #
# Pure helpers -- no I/O, this is what the mechanism tests exercise directly
# --------------------------------------------------------------------------- #

def encode_command(obj: dict) -> bytes:
    """Frame `obj` for tsserver's **stdin**: one line of JSON, newline
    terminated. Deliberately NOT `jsonrpc.encode_message` -- see the module
    docstring's point 1; a `Content-Length` header here hangs the child."""
    return json.dumps(obj).encode("utf-8") + b"\n"


def _request_frame(seq: int, command: str, arguments: Optional[dict] = None) -> dict:
    frame: dict = {"seq": seq, "type": "request", "command": command}
    if arguments is not None:
        frame["arguments"] = arguments
    return frame


def _map_diagnostics(raw: Optional[List[dict]], text: str) -> List[Diagnostic]:
    """tsserver diagnostic bodies -> `Diagnostic`s, in wire order.

    `raw` may be `None` (a response with no `body`, e.g. `updateOpen`'s or a
    command that answers with nothing) -- treated as an empty result, never a
    `KeyError`.

    Trap A: integer `code` 2339 becomes `"TS2339"`. Trap B: only
    `category == "error"` survives -- tsserver's `"suggestion"`/`"warning"`
    entries are dropped, exactly as `ts_lsp.py` drops LSP severity 2/3/4.
    tsserver's `{line, offset}` is already **1-based on both axes**, which is
    `Diagnostic`'s convention, so unlike the LSP mapping there is no +1 here;
    the parity gate against `TsLspService` is what pins that there is no
    off-by-one either way.
    """
    out: List[Diagnostic] = []
    for d in raw or []:
        if d.get("category") != "error":
            continue
        start = d["start"]
        line, col = start["line"], start["offset"]
        out.append(Diagnostic(
            code=f"TS{d['code']}", line=line, col=col, message=d.get("text", ""),
            offset=line_col_to_offset(text, line, col),
            source="ts", severity=1,
        ))
    return out


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #

class TsServerEndpoint:
    """A live tsserver-protocol endpoint over any `(reader, writer)` binary-
    stream pair.

    `reader`/`writer` need only `.read(n)` / `.write(bytes)` / `.flush()`, so an
    `os.pipe()` pair satisfies this with no subprocess at all -- which is what
    `tests/test_ts_server_direct_mechanism.py` drives against, mirroring
    `tests/test_jsonrpc.py`'s build order (framing/demux first, process later).

    Demux, on every message read:

      - `type == "response"` -> resolves the `request_seq` waiter.
      - `type == "event"` -> handed to `on_event` for observation only. An
        event NEVER resolves a waiter; tsserver interleaves
        `projectLoadingStart`/`projectLoadingFinish`/`telemetry`/`configFileDiag`
        freely with responses.
      - anything else -> dropped (raising out of the reader thread would kill
        the demux for everyone).

    EOF or a dead child fails every pending waiter rather than leaving it to
    hang; `request()` always has a timeout on top of that.
    """

    def __init__(self, reader: BinaryIO, writer: BinaryIO,
                 on_event: Optional[OnEvent] = None) -> None:
        self._reader = reader
        self._writer = writer
        self._on_event = on_event
        self._next_seq = 1
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: Dict[int, Waiter] = {}
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def start(self) -> None:
        """Start the daemon reader thread. Must be called once before
        `request`/`notify`."""
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while True:
            try:
                # Read framing is byte-identical to LSP -- reused, not copied.
                msg = read_message(self._reader)
            except (ValueError, OSError):
                msg = None
            if msg is None:
                self._fail_all_pending(ConnectionError("tsserver endpoint: EOF or dead process"))
                return
            self._dispatch(msg)

    def _dispatch(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "response":
            with self._lock:
                waiter = self._pending.pop(msg.get("request_seq"), None)
            if waiter is not None:
                waiter.set(msg)
            return
        if kind == "event":
            if self._on_event is not None:
                try:
                    self._on_event(msg)
                except Exception:
                    pass
            return
        # Unrecognized shape -- drop it rather than raise out of the reader
        # thread.

    def request(self, command: str, arguments: Optional[dict] = None, *,
                timeout: float) -> Any:
        """Send `command`, block up to `timeout` seconds for its response, and
        return its `body` (`None` when the response carries none -- `updateOpen`
        and `configure` both do). Raises `TimeoutError` (never hangs) or
        `RuntimeError` when tsserver answers `success: false`.

        Mirrors `jsonrpc.py`'s policy on a rejected command deliberately: a
        command tsserver refuses is a bug in us, and silently returning "no
        diagnostics" would be exactly the BLIND failure this repo guards
        hardest. `diagnostics()` is the one caller that converts it to a
        counted `[]`, and it counts it in its OWN bucket."""
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            waiter = Waiter()
            self._pending[seq] = waiter

        self._write(_request_frame(seq, command, arguments))

        got = waiter.wait(timeout)
        if got is None:
            with self._lock:
                self._pending.pop(seq, None)
            raise TimeoutError(f"timed out after {timeout}s waiting for {command!r} (seq={seq})")
        if isinstance(got, BaseException):
            raise got
        if not got.get("success", False):
            raise RuntimeError(f"{command} failed: {got.get('message')!r}")
        return got.get("body")

    def notify(self, command: str, arguments: Optional[dict] = None) -> None:
        """Fire-and-forget command -- no response awaited. Only `exit` uses
        this (tsserver never answers it; it just goes away)."""
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
        self._write(_request_frame(seq, command, arguments))

    def _write(self, obj: dict) -> None:
        data = encode_command(obj)
        with self._write_lock:
            self._writer.write(data)
            self._writer.flush()

    def _fail_all_pending(self, exc: BaseException) -> None:
        with self._lock:
            waiters = list(self._pending.values())
            self._pending.clear()
        for w in waiters:
            w.set(exc)

    def close(self) -> None:
        """Idempotent. Fails any still-pending waiter rather than leaving it to
        time out on its own."""
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.close()
        except (OSError, ValueError):
            pass
        self._fail_all_pending(ConnectionError("tsserver endpoint closed"))


def spawn_tsserver(argv: List[str], cwd: Optional[str] = None,
                   on_event: Optional[OnEvent] = None,
                   ) -> Tuple[subprocess.Popen, TsServerEndpoint]:
    """Start `argv` as a child process and wrap its stdio in a started
    `TsServerEndpoint`. Drains stderr in a background thread for the same
    deadlock reason `jsonrpc.spawn` does (an undrained stderr pipe fills and
    blocks the child's next write) and keeps a bounded tail for debugging."""
    proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

    stderr_tail: Deque[bytes] = deque(maxlen=_STDERR_TAIL_LINES)

    def _drain_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_tail.append(line)

    threading.Thread(target=_drain_stderr, daemon=True).start()

    endpoint = TsServerEndpoint(proc.stdout, proc.stdin, on_event=on_event)
    endpoint.stderr_tail = stderr_tail  # type: ignore[attr-defined]
    endpoint.start()
    return proc, endpoint


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #

class TsServerDirect:
    """One persistent `node tsserver.js` process carrying a warm, multi-file
    program, answering diagnostics through the synchronous commands.

    Child flags: `--disableAutomaticTypingAcquisition` (no npm typings
    installer -- it would fetch over the network mid-bench) and
    `--suppressDiagnosticEvents` (this transport only ever uses the `*Sync`
    commands, so the push events are pure noise; the sync commands themselves
    are unaffected by the flag).

    Scratch dir nesting under `scratch_parent` (default `SET_DIR`) is
    load-bearing, exactly as in `ts_lsp.py:79-83`: TypeScript's default
    `typeRoots` walk must find `SET_DIR/node_modules/@types/node` from wherever
    the scratch dir sits, or ambient `console` spuriously fails to resolve.
    `.gitignore`'s `lsp_scratch_*` already covers the prefix.

    Lazy `open`, mirroring `TsLspService`'s deliberate choice: `open_project`
    materializes every file on disk (the pinned tsconfig has no `include`, so
    the whole scratch tree enters the *program*) but `open`s only the warm
    file; every other document is opened by `_ensure_open` the first time an op
    touches it.

    Not safe for concurrent use.
    """

    def __init__(self, *, timeout_s: float = _DEFAULT_TIMEOUT_S,
                 tsconfig: Path = DEFAULT_TSCONFIG_PATH,
                 scratch_parent: Path = SET_DIR,
                 argv: Optional[List[str]] = None) -> None:
        self.argv = argv if argv is not None else resolve_tsserver()
        if self.argv is None:
            raise RuntimeError("no tsserver toolchain resolvable "
                               f"(run `npm install` in {scratch_parent})")
        self.timeout_s = timeout_s
        self.tsconfig = tsconfig

        self.n_calls = 0
        self.wall_s = 0.0
        self.n_timeouts = 0
        self.n_restarts = 0
        # A refused command (`success: false`) is counted in its OWN bucket:
        # folding it into n_timeouts would hide a protocol bug behind a
        # latency story. There is deliberately no `n_no_publish` analogue --
        # see the module docstring.
        self.n_command_errors = 0
        self.op_counts: Dict[str, int] = {}
        self.op_wall_s: Dict[str, float] = {}
        self.cold_load_s: Optional[float] = None

        self._tmpdir_obj = tempfile.TemporaryDirectory(dir=str(scratch_parent),
                                                       prefix="lsp_scratch_")
        self.scratch_dir = Path(self._tmpdir_obj.name)
        (self.scratch_dir / "tsconfig.json").write_text(
            tsconfig.read_text(encoding="utf-8"), encoding="utf-8")

        self._project_opened = False
        self._files: Dict[str, str] = {}       # project-relative path -> text
        self._versions: Dict[str, int] = {}     # project-relative path -> version
        self._open: Set[str] = set()            # project-relative paths, opened
        self._events: List[dict] = []           # observation only, bounded below

        self._proc: Optional[subprocess.Popen] = None
        self._endpoint: Optional[TsServerEndpoint] = None
        self._start_server()

    # ----------------------------------------------------------------- #
    # path helpers -- tsserver addresses files by absolute PATH, not uri
    # ----------------------------------------------------------------- #

    def _abs_path(self, path: str) -> Path:
        return self.scratch_dir / path

    def _file(self, path: str) -> str:
        return str(self._abs_path(path))

    # ----------------------------------------------------------------- #
    # server lifecycle
    # ----------------------------------------------------------------- #

    def _on_event(self, msg: dict) -> None:
        # Observation only. Bounded so a chatty server can't grow this
        # without limit over a 5000-file bench.
        if len(self._events) < _STDERR_TAIL_LINES:
            self._events.append(msg)

    def _start_server(self) -> None:
        self._proc, self._endpoint = spawn_tsserver(
            self.argv + ["--disableAutomaticTypingAcquisition",
                         "--suppressDiagnosticEvents"],
            cwd=str(self.scratch_dir), on_event=self._on_event)
        self._endpoint.request("configure", {
            "hostInfo": "monica-ts-server-direct",
            "preferences": {},
        }, timeout=self.timeout_s)

    def _ensure_alive(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._restart()

    def _restart(self) -> None:
        """Restart lost the warm program. Files are already on disk (both
        `open_project` and `update` write through), so re-`open` every
        previously-open path at its CURRENT text. Version counters stay
        monotonic across restarts for the same reason `TsLspService`'s do --
        they are the caller-visible edit ordinal, not a server-side token."""
        self.n_restarts += 1
        open_paths = list(self._open)
        self._teardown_process()
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
        `open` exactly one (the first key, or `warm_path`) to force tsserver to
        load the configured project. The `open` RESPONSE arrives only after
        `projectLoadingFinish`, so blocking on it makes cold load bounded and
        attributable rather than smeared into the first recheck -- and unlike
        the LSP client there is no first-push ambiguity to wait through.
        Records `self.cold_load_s`. One instance == one project; calling this
        twice is a `RuntimeError`."""
        if self._project_opened:
            raise RuntimeError("open_project called twice -- one TsServerDirect "
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
        t0 = time.monotonic()
        try:
            self._ensure_open(warm)
        except (TimeoutError, ConnectionError, OSError):
            # A cold load that never answers is a real failure, counted --
            # never a silent "the project is empty".
            self.n_timeouts += 1
        except RuntimeError:
            self.n_command_errors += 1
        self.cold_load_s = time.monotonic() - t0

    def _ensure_open(self, path: str) -> None:
        if path in self._open:
            return
        self._endpoint.request("open", {
            "file": self._file(path),
            "fileContent": self._files[path],
            "scriptKindName": "TS",
            "projectRootPath": str(self.scratch_dir),
        }, timeout=self.timeout_s)
        self._open.add(path)

    def _request_retrying(self, command: str, arguments: dict) -> Any:
        """`self._endpoint.request(...)`, retried once after `_restart()` if the
        transport itself failed (the child died in the narrow window between
        `_ensure_alive()`'s `poll()` and this write). Without it that race
        surfaces as an unhandled exception two frames up, breaking the
        "never raises on a dead server" contract."""
        try:
            return self._endpoint.request(command, arguments, timeout=self.timeout_s)
        except (ConnectionError, OSError):
            self._restart()
            return self._endpoint.request(command, arguments, timeout=self.timeout_s)

    def update(self, path: str, text: str) -> int:
        """`_ensure_open(path)`, write `text` through to disk (so a restart can
        recover it and tsserver's on-disk view stays consistent), bump the
        version, and send `updateOpen` carrying the file's FULL `fileContent`
        -- a whole-file replace. (`textChanges` expresses only ranges, and this
        client has no reason to compute a diff; `TsLspService` sends full-text
        `didChange` for the same reason.) Returns the new version, kept for API
        symmetry with `TsLspService.update`; tsserver has no version concept of
        its own on this path.

        Never raises -- unlike `TsLspService.update()` (a fire-and-forget
        `didChange` notify), `updateOpen` is a real request/response round
        trip through `_request_retrying()`, so a timeout or a refused command
        is counted (`n_timeouts` / `n_command_errors`, same buckets
        `diagnostics()` uses) instead of propagating out of a caller that
        expects `TsLspService.update()`'s non-raising contract."""
        self._ensure_alive()
        self._ensure_open(path)
        self._abs_path(path).write_text(text, encoding="utf-8")
        self._files[path] = text
        self._versions[path] += 1
        try:
            self._request_retrying("updateOpen", {"openFiles": [{
                "file": self._file(path),
                "fileContent": text,
                "scriptKindName": "TS",
                "projectRootPath": str(self.scratch_dir),
            }]})
        except (TimeoutError, ConnectionError, OSError):
            self.n_timeouts += 1
        except RuntimeError:
            self.n_command_errors += 1
        return self._versions[path]

    # ----------------------------------------------------------------- #
    # diagnostics -- synchronous, no debounce, no push ambiguity
    # ----------------------------------------------------------------- #

    def _diagnostics_once(self, path: str) -> List[dict]:
        self._ensure_open(path)
        args = {"file": self._file(path), "includeLinePosition": False}
        syntactic = self._endpoint.request("syntacticDiagnosticsSync", args,
                                           timeout=self.timeout_s)
        semantic = self._endpoint.request("semanticDiagnosticsSync", args,
                                          timeout=self.timeout_s)
        # Syntactic first: a parse error and a distinct type error must BOTH
        # come back, and `semanticDiagnosticsSync` alone would miss the first.
        return list(syntactic or []) + list(semantic or [])

    def diagnostics(self, path: str) -> List[Diagnostic]:
        """Return every `category == "error"` diagnostic for `path`'s current
        text, from `syntacticDiagnosticsSync` + `semanticDiagnosticsSync`.

        Unlike `TsLspService.diagnostics`, `[]` here is an **answer**: the sync
        commands always reply, so a clean re-check is not confusable with "the
        server never published" (`n_no_publish`, #278). There is no cache and
        no fast path -- every call is a real recheck against tsserver.

        Never raises. A timeout counts `n_timeouts`; a dead child triggers one
        `_restart()` and one retry (`n_restarts`); a refused command counts
        `n_command_errors`. Each returns `[]`."""
        self._ensure_alive()
        t0 = time.monotonic()
        raw: Optional[List[dict]] = None
        try:
            raw = self._diagnostics_once(path)
        except TimeoutError:
            self.n_timeouts += 1
        except (ConnectionError, OSError):
            try:
                self._restart()
                raw = self._diagnostics_once(path)
            except TimeoutError:
                self.n_timeouts += 1
            except (ConnectionError, OSError):
                self.n_timeouts += 1
            except RuntimeError:
                self.n_command_errors += 1
        except RuntimeError:
            self.n_command_errors += 1
        self._record_op("diagnostics", time.monotonic() - t0)
        return _map_diagnostics(raw, self._files[path])

    def codes(self, path: str) -> List[str]:
        """`[d.code for d in self.diagnostics(path)]` -- mirrors
        `TscRunner.codes` / `TsLspOracle.codes`."""
        return [d.code for d in self.diagnostics(path)]

    # ----------------------------------------------------------------- #
    # misc
    # ----------------------------------------------------------------- #

    def _record_op(self, op_name: str, elapsed: float) -> None:
        self.wall_s += elapsed
        self.n_calls += 1
        self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
        self.op_wall_s[op_name] = self.op_wall_s.get(op_name, 0.0) + elapsed

    @property
    def calls_per_s(self) -> float:
        return self.n_calls / self.wall_s if self.wall_s else 0.0

    def close(self) -> None:
        if self._endpoint is not None:
            try:
                self._endpoint.notify("exit")
            except (OSError, ValueError):
                pass  # already gone -- nothing left to tell it
        self._teardown_process()
        self._tmpdir_obj.cleanup()

    def __enter__(self) -> "TsServerDirect":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
