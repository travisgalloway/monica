"""Binary-free mechanism tests for `src/lsp/ts_server_direct.py` (#279) -- the
CI gate for the direct-`tsserver` transport (no `node`, no `typescript`
install needed; the live parity gate is in `tests/test_ts_server_direct.py`,
skipped without the toolchain).

Built on `tests/test_jsonrpc.py`'s `os.pipe()` fake-transport pattern and
`tests/test_ts_service_mechanism.py`'s `_ScriptedServer` + "subclass and bypass
`__init__`" precedent, adapted to tsserver's ASYMMETRIC wire protocol: the
scripted server reads **newline-delimited JSON** from the client and writes
**`Content-Length`-framed** responses back. Getting that backwards hangs the
real child silently, which is exactly why the framing has its own test here.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

import pytest

from src.lsp.diagnostics import Diagnostic
from src.lsp.jsonrpc import encode_message, read_message
from src.lsp.ts_service import TsLspService
from src.lsp.ts_server_direct import (
    TsServerDirect, TsServerEndpoint, _map_diagnostics, _request_frame,
    encode_command, resolve_tsserver,
)

_TIMEOUT = 5.0


def _pipe_files() -> Tuple[BinaryIO, BinaryIO]:
    r_fd, w_fd = os.pipe()
    return os.fdopen(r_fd, "rb", buffering=0), os.fdopen(w_fd, "wb", buffering=0)


def _raw_diag(code: int, line: int = 1, offset: int = 1, text: str = "msg",
              category: str = "error") -> dict:
    """One tsserver diagnostic body entry -- 1-based `line`/`offset`, integer
    `code`, string `category` (all three differ from LSP; see the module
    docstring of `ts_server_direct.py`)."""
    return {"start": {"line": line, "offset": offset},
            "end": {"line": line, "offset": offset + 1},
            "text": text, "code": code, "category": category}


class _TsError:
    """Sentinel telling `_ScriptedTsServer` to answer `success: false`."""

    def __init__(self, message: str = "scripted failure") -> None:
        self.message = message


class _ScriptedTsServer:
    """The "other side" of tsserver's wire: two `os.pipe()` pairs plus a daemon
    thread that reads **newline-delimited** client frames and replies
    **`Content-Length`-framed** per `responses` (command -> body, or a
    `body(request) -> body` callable, or a `_TsError`), recording every frame it
    saw. Commands listed in `pending` are deliberately never answered
    (exercises the timeout path); commands listed in `no_body` answer
    `success: true` with no `body` key at all."""

    def __init__(self, responses=None, pending=(), no_body=()) -> None:
        self.to_client_r, self.to_client_w = _pipe_files()   # server writes here
        self.to_server_r, self.to_server_w = _pipe_files()   # endpoint writes here
        self.responses: Dict[str, object] = dict(responses or {})
        self.pending = set(pending)
        self.no_body = set(no_body)
        self.seen: List[dict] = []
        self.raw_seen: List[bytes] = []
        self.auto_reply = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            line = self.to_server_r.readline()
            if not line:
                return
            with self._lock:
                self.raw_seen.append(line)
            try:
                msg = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            with self._lock:
                self.seen.append(msg)
            if not self.auto_reply:
                continue
            command = msg.get("command")
            if command in self.pending:
                continue   # deliberately unanswered
            self.reply(msg)

    def reply(self, msg: dict) -> None:
        command = msg.get("command")
        value = self.responses.get(command, [])
        if callable(value):
            value = value(msg)
        frame = {"seq": 0, "type": "response", "command": command,
                 "request_seq": msg.get("seq")}
        if isinstance(value, _TsError):
            frame["success"] = False
            frame["message"] = value.message
        else:
            frame["success"] = True
            if command not in self.no_body:
                frame["body"] = value
        self.send(frame)

    def send(self, obj: dict) -> None:
        self.to_client_w.write(encode_message(obj))

    def push_event(self, event: str, body: Optional[dict] = None) -> None:
        self.send({"seq": 0, "type": "event", "event": event, "body": body or {}})

    def frames_for(self, command: str) -> List[dict]:
        with self._lock:
            return [m for m in self.seen if m.get("command") == command]

    def close(self) -> None:
        for f in (self.to_client_r, self.to_client_w, self.to_server_r, self.to_server_w):
            try:
                f.close()
            except (OSError, ValueError):
                pass


class _FakeDirect(TsServerDirect):
    """A `TsServerDirect` whose subprocess-touching pieces are bypassed -- no
    spawn, no `configure` handshake, no tempdir -- wired instead to a
    `_ScriptedTsServer`'s pipes. `_ensure_alive` is a no-op (there is no real
    process to poll); the restart path is driven explicitly by
    `test_diagnostics_dead_child_restarts_once`, which swaps the endpoint."""

    def __init__(self, scripted: _ScriptedTsServer, *, timeout_s: float,
                 scratch_dir: Path) -> None:
        self.argv = ["node", "tsserver.js"]
        self.timeout_s = timeout_s
        self.tsconfig = None

        self.n_calls = 0
        self.wall_s = 0.0
        self.n_timeouts = 0
        self.n_restarts = 0
        self.n_command_errors = 0
        self.op_counts: Dict[str, int] = {}
        self.op_wall_s: Dict[str, float] = {}
        self.cold_load_s = None

        self.scratch_dir = scratch_dir
        self._project_opened = False
        self._files: Dict[str, str] = {}
        self._versions: Dict[str, int] = {}
        self._open: set = set()
        self._events: List[dict] = []

        self._proc = None
        self._endpoint = TsServerEndpoint(scripted.to_client_r, scripted.to_server_w,
                                          on_event=self._on_event)
        self._endpoint.start()

    def _ensure_alive(self) -> None:
        pass  # no real process in the fake


@pytest.fixture
def scripted():
    s = _ScriptedTsServer(responses={"open": None, "configure": None,
                                     "updateOpen": True,
                                     "syntacticDiagnosticsSync": [],
                                     "semanticDiagnosticsSync": []})
    yield s
    s.close()


@pytest.fixture
def fake(scripted, tmp_path):
    svc = _FakeDirect(scripted, timeout_s=0.3, scratch_dir=tmp_path)
    yield svc, scripted
    svc._endpoint.close()


def _seed_file(svc: TsServerDirect, path: str, text: str, *, version: int = 1) -> None:
    """Register `path` as if `open_project` had materialized it, WITHOUT
    sending a real `open` -- the test decides when that happens."""
    (svc.scratch_dir / path).parent.mkdir(parents=True, exist_ok=True)
    (svc.scratch_dir / path).write_text(text, encoding="utf-8")
    svc._files[path] = text
    svc._versions[path] = version


# --------------------------------------------------------------------------- #
# 1. Framing -- the asymmetry that makes this a separate module (edge cases 1-2)
# --------------------------------------------------------------------------- #

def test_encode_command_is_newline_delimited_never_content_length():
    frame = {"seq": 1, "type": "request", "command": "open"}
    raw = encode_command(frame)
    assert raw.endswith(b"\n")
    assert b"Content-Length" not in raw          # would hang the real child
    assert b"\r\n\r\n" not in raw
    assert raw == json.dumps(frame).encode("utf-8") + b"\n"
    assert json.loads(raw.decode("utf-8")) == frame


def test_request_frame_shape_and_optional_arguments():
    assert _request_frame(7, "exit") == {"seq": 7, "type": "request", "command": "exit"}
    with_args = _request_frame(8, "open", {"file": "/a.ts"})
    assert with_args["arguments"] == {"file": "/a.ts"}


def test_endpoint_writes_newline_frames_and_reads_content_length(scripted):
    endpoint = TsServerEndpoint(scripted.to_client_r, scripted.to_server_w)
    endpoint.start()
    try:
        body = endpoint.request("semanticDiagnosticsSync", {"file": "/a.ts"},
                                timeout=_TIMEOUT)
    finally:
        endpoint.close()
    assert body == []
    # The exact bytes that went over the wire: one line of JSON, no header.
    assert len(scripted.raw_seen) == 1
    assert scripted.raw_seen[0].endswith(b"\n")
    assert b"Content-Length" not in scripted.raw_seen[0]
    sent = scripted.frames_for("semanticDiagnosticsSync")[0]
    assert sent["type"] == "request" and sent["seq"] == 1


# --------------------------------------------------------------------------- #
# 2. Demux -- events, correlation, failures (edge cases 3-7)
# --------------------------------------------------------------------------- #

def test_events_never_resolve_a_waiter(scripted):
    """`projectLoadingStart`/`telemetry`/`configFileDiag` interleave freely with
    responses; the waiter must still get its own body."""
    scripted.auto_reply = False
    endpoint = TsServerEndpoint(scripted.to_client_r, scripted.to_server_w)
    seen_events: List[str] = []
    endpoint._on_event = lambda msg: seen_events.append(msg["event"])
    endpoint.start()

    def _drip():
        while not scripted.frames_for("semanticDiagnosticsSync"):
            time.sleep(0.005)
        scripted.push_event("projectLoadingStart", {"projectName": "p"})
        scripted.push_event("telemetry", {"telemetryEventName": "projectInfo"})
        scripted.push_event("configFileDiag", {"triggerFile": "/a.ts"})
        scripted.push_event("projectLoadingFinish", {"projectName": "p"})
        scripted.reply(scripted.frames_for("semanticDiagnosticsSync")[0])

    threading.Thread(target=_drip, daemon=True).start()
    try:
        scripted.responses["semanticDiagnosticsSync"] = [_raw_diag(2339)]
        body = endpoint.request("semanticDiagnosticsSync", {"file": "/a.ts"},
                                timeout=_TIMEOUT)
    finally:
        endpoint.close()
    assert [d["code"] for d in body] == [2339]
    assert seen_events == ["projectLoadingStart", "telemetry", "configFileDiag",
                           "projectLoadingFinish"]


def test_out_of_order_responses_resolve_by_request_seq(scripted):
    """Correlation is `request_seq`, not `id` -- and a LATER seq answered FIRST
    must not hand its body to the earlier waiter."""
    scripted.auto_reply = False
    endpoint = TsServerEndpoint(scripted.to_client_r, scripted.to_server_w)
    endpoint.start()
    results: Dict[str, object] = {}

    def _call(name: str, command: str):
        results[name] = endpoint.request(command, {}, timeout=_TIMEOUT)

    t1 = threading.Thread(target=_call, args=("first", "syntacticDiagnosticsSync"))
    t1.start()
    while not scripted.frames_for("syntacticDiagnosticsSync"):
        time.sleep(0.005)
    t2 = threading.Thread(target=_call, args=("second", "semanticDiagnosticsSync"))
    t2.start()
    while not scripted.frames_for("semanticDiagnosticsSync"):
        time.sleep(0.005)

    seq_first = scripted.frames_for("syntacticDiagnosticsSync")[0]["seq"]
    seq_second = scripted.frames_for("semanticDiagnosticsSync")[0]["seq"]
    assert seq_second == seq_first + 1
    # Answer the SECOND request first.
    scripted.send({"seq": 0, "type": "response", "command": "semanticDiagnosticsSync",
                   "request_seq": seq_second, "success": True, "body": [_raw_diag(2322)]})
    scripted.send({"seq": 0, "type": "response", "command": "syntacticDiagnosticsSync",
                   "request_seq": seq_first, "success": True, "body": [_raw_diag(1005)]})
    t1.join(timeout=_TIMEOUT)
    t2.join(timeout=_TIMEOUT)
    endpoint.close()
    assert [d["code"] for d in results["first"]] == [1005]
    assert [d["code"] for d in results["second"]] == [2322]


def test_request_raises_runtime_error_on_unsuccessful_command(scripted):
    scripted.responses["bogusCommand"] = _TsError("Unrecognized JSON command: bogusCommand")
    endpoint = TsServerEndpoint(scripted.to_client_r, scripted.to_server_w)
    endpoint.start()
    try:
        with pytest.raises(RuntimeError, match="Unrecognized JSON command"):
            endpoint.request("bogusCommand", {}, timeout=_TIMEOUT)
    finally:
        endpoint.close()


def test_request_times_out_rather_than_hanging(scripted):
    scripted.pending.add("semanticDiagnosticsSync")
    endpoint = TsServerEndpoint(scripted.to_client_r, scripted.to_server_w)
    endpoint.start()
    try:
        with pytest.raises(TimeoutError):
            endpoint.request("semanticDiagnosticsSync", {}, timeout=0.2)
    finally:
        endpoint.close()


def test_response_with_no_body_is_none_not_key_error(scripted):
    scripted.no_body.add("updateOpen")
    endpoint = TsServerEndpoint(scripted.to_client_r, scripted.to_server_w)
    endpoint.start()
    try:
        assert endpoint.request("updateOpen", {}, timeout=_TIMEOUT) is None
    finally:
        endpoint.close()
    assert _map_diagnostics(None, "x = 1\n") == []


def test_eof_fails_every_pending_waiter(scripted):
    scripted.pending.update({"semanticDiagnosticsSync", "syntacticDiagnosticsSync"})
    endpoint = TsServerEndpoint(scripted.to_client_r, scripted.to_server_w)
    endpoint.start()
    errors: List[BaseException] = []

    def _call(command: str):
        try:
            endpoint.request(command, {}, timeout=_TIMEOUT)
        except BaseException as exc:   # noqa: BLE001 -- recording it is the assertion
            errors.append(exc)

    threads = [threading.Thread(target=_call, args=(c,))
               for c in ("semanticDiagnosticsSync", "syntacticDiagnosticsSync")]
    for t in threads:
        t.start()
    while len(scripted.seen) < 2:
        time.sleep(0.005)
    scripted.to_client_w.close()   # server side goes away mid-request
    for t in threads:
        t.join(timeout=_TIMEOUT)
        assert not t.is_alive()
    endpoint.close()
    assert len(errors) == 2
    assert all(isinstance(e, ConnectionError) for e in errors)


def test_malformed_message_is_dropped_not_fatal(scripted):
    endpoint = TsServerEndpoint(scripted.to_client_r, scripted.to_server_w)
    endpoint.start()
    try:
        scripted.send({"seq": 0, "type": "wat"})            # unrecognized shape
        scripted.send({"seq": 0, "type": "response", "request_seq": 999,
                       "success": True, "body": []})         # nobody is waiting
        assert endpoint.request("semanticDiagnosticsSync", {}, timeout=_TIMEOUT) == []
    finally:
        endpoint.close()


# --------------------------------------------------------------------------- #
# 3. Diagnostic mapping (edge cases 9-12)
# --------------------------------------------------------------------------- #

def test_map_diagnostics_formats_code_with_ts_prefix():
    """Trap A: `is_incomplete` matches on the `TS1xxx` STRING prefix, so a bare
    integer `2339` here would silently break frontier filtering."""
    out = _map_diagnostics([_raw_diag(2339)], "const a = 1;\n")
    assert len(out) == 1 and isinstance(out[0], Diagnostic)
    assert out[0].code == "TS2339"


def test_map_diagnostics_drops_non_error_categories():
    """Trap B: tsserver publishes suggestion/warning entries `tsc` never
    would; admitting them would inflate the set and invalidate any
    LSP-vs-direct comparison."""
    raw = [_raw_diag(2339, category="error"),
           _raw_diag(80001, category="suggestion"),
           _raw_diag(7027, category="warning"),
           _raw_diag(6133, category="message")]
    out = _map_diagnostics(raw, "const a = 1;\n")
    assert [d.code for d in out] == ["TS2339"]
    assert all(d.severity == 1 for d in out)


def test_map_diagnostics_coordinates_are_already_one_based():
    """tsserver's `{line, offset}` is 1-based on BOTH axes, unlike LSP's
    0-based `{line, character}` -- so there is no +1 here. Confusing the two is
    a silent one-column/one-line shift the parity gate would catch live; this
    pins it without a binary."""
    text = "const a = 1;\nconst b = a.nope;\n"
    out = _map_diagnostics([_raw_diag(2339, line=2, offset=13)], text)
    assert (out[0].line, out[0].col) == (2, 13)
    assert out[0].offset == text.index("nope")


def test_map_diagnostics_preserves_order_for_merged_syntactic_then_semantic():
    """Edge case 9: a candidate carrying both a parse error and a distinct type
    error must yield BOTH, syntactic first -- `semanticDiagnosticsSync` alone
    would miss the parse error entirely."""
    merged = [_raw_diag(1005, line=1, offset=5, text="',' expected."),
              _raw_diag(2322, line=2, offset=7, text="Type error.")]
    out = _map_diagnostics(merged, "let x y;\nconst z: number = 'a';\n")
    assert [d.code for d in out] == ["TS1005", "TS2322"]


def test_map_diagnostics_missing_text_is_empty_message():
    out = _map_diagnostics([{"start": {"line": 1, "offset": 1},
                             "end": {"line": 1, "offset": 2},
                             "code": 2339, "category": "error"}], "a\n")
    assert out[0].message == ""


# --------------------------------------------------------------------------- #
# 4. Toolchain resolution (criterion 2, edge case 14)
# --------------------------------------------------------------------------- #

def test_resolve_tsserver_missing_toolchain_returns_none(monkeypatch, tmp_path):
    """Mirrors `resolve_tsc()`: returns `None`, never raises, when either half
    of the toolchain is missing. Both halves are exercised independently."""
    import src.lsp.ts_server_direct as mod

    monkeypatch.setattr(mod, "TSSERVER_JS", tmp_path / "definitely-absent.js")
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/node")
    assert resolve_tsserver() is None

    present = tmp_path / "tsserver.js"
    present.write_text("// stub\n", encoding="utf-8")
    monkeypatch.setattr(mod, "TSSERVER_JS", present)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    assert resolve_tsserver() is None

    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/node")
    assert resolve_tsserver() == ["/usr/bin/node", str(present)]


def test_constructor_raises_cleanly_with_no_toolchain(monkeypatch, tmp_path):
    import src.lsp.ts_server_direct as mod
    monkeypatch.setattr(mod, "TSSERVER_JS", tmp_path / "absent.js")
    with pytest.raises(RuntimeError, match="no tsserver toolchain resolvable"):
        TsServerDirect(scratch_parent=tmp_path)


# --------------------------------------------------------------------------- #
# 5. The service surface (criteria 3-4, edge cases 8, 13, 15)
# --------------------------------------------------------------------------- #

def test_signature_compatibility_with_ts_lsp_service():
    """The bench and the parity gate drive either client through the same three
    calls; a drifted signature would force a shim and make the two columns
    quietly incomparable."""
    for name in ("open_project", "update", "diagnostics"):
        assert (inspect.signature(getattr(TsServerDirect, name))
                == inspect.signature(getattr(TsLspService, name))), name


def test_open_project_materializes_all_files_but_opens_only_the_warm_one(fake):
    svc, scripted = fake
    svc.open_project({"src/a.ts": "export const a = 1;\n",
                      "src/b.ts": "export const b = 2;\n"})
    assert (svc.scratch_dir / "src/a.ts").read_text() == "export const a = 1;\n"
    assert (svc.scratch_dir / "src/b.ts").read_text() == "export const b = 2;\n"
    opens = scripted.frames_for("open")
    assert len(opens) == 1
    assert opens[0]["arguments"]["file"] == str(svc.scratch_dir / "src/a.ts")
    assert opens[0]["arguments"]["fileContent"] == "export const a = 1;\n"


def test_open_project_honors_warm_path(fake):
    svc, scripted = fake
    svc.open_project({"src/a.ts": "a", "src/b.ts": "b"}, warm_path="src/b.ts")
    assert scripted.frames_for("open")[0]["arguments"]["file"] == str(svc.scratch_dir / "src/b.ts")


def test_open_project_records_cold_load_outside_the_timed_ops(fake):
    """Edge case 8: cold project load is measured into `cold_load_s` and is NOT
    folded into the diagnostics op stats."""
    svc, _ = fake
    svc.open_project({"src/a.ts": "export const a = 1;\n"})
    assert svc.cold_load_s is not None and svc.cold_load_s >= 0.0
    assert svc.n_calls == 0 and svc.op_counts == {}


def test_open_project_twice_is_an_error(fake):
    svc, _ = fake
    svc.open_project({"a.ts": "x"})
    with pytest.raises(RuntimeError, match="open_project called twice"):
        svc.open_project({"b.ts": "y"})


def test_open_project_requires_at_least_one_file(fake):
    svc, _ = fake
    with pytest.raises(RuntimeError, match="at least one file"):
        svc.open_project({})


def test_update_sends_whole_file_content_and_writes_through(fake):
    """Edge case 13: `updateOpen` carries the file's FULL new text (a
    whole-file replace), and the same text lands on disk so a restart can
    recover it."""
    svc, scripted = fake
    svc.open_project({"a.ts": "const a = 1;\n"})
    version = svc.update("a.ts", "const a = 2;\nconst b = 3;\n")
    assert version == 2
    frames = scripted.frames_for("updateOpen")
    assert len(frames) == 1
    open_files = frames[0]["arguments"]["openFiles"]
    assert len(open_files) == 1
    assert open_files[0]["fileContent"] == "const a = 2;\nconst b = 3;\n"
    assert open_files[0]["file"] == str(svc.scratch_dir / "a.ts")
    assert (svc.scratch_dir / "a.ts").read_text() == "const a = 2;\nconst b = 3;\n"


def test_update_timeout_never_raises_and_counts(fake):
    """`update()`'s `updateOpen` is a real request/response round trip
    (unlike `TsLspService.update()`'s fire-and-forget `didChange` notify), so
    a timeout must be counted rather than propagate out of `update()` and
    crash a caller that expects `TsLspService.update()`'s non-raising
    contract."""
    svc, scripted = fake
    svc.open_project({"a.ts": "const a = 1;\n"})
    scripted.pending.add("updateOpen")
    version = svc.update("a.ts", "const a = 2;\n")
    assert version == 2
    assert svc.n_timeouts == 1
    assert svc.n_command_errors == 0


def test_update_unsuccessful_command_never_raises_and_counts(fake):
    svc, scripted = fake
    svc.open_project({"a.ts": "const a = 1;\n"})
    scripted.responses["updateOpen"] = _TsError("No Project.")
    version = svc.update("a.ts", "const a = 2;\n")
    assert version == 2
    assert svc.n_command_errors == 1
    assert svc.n_timeouts == 0


def test_diagnostics_lazily_opens_an_untouched_document(fake):
    svc, scripted = fake
    svc.open_project({"a.ts": "const a = 1;\n", "b.ts": "const b = 1;\n"})
    assert len(scripted.frames_for("open")) == 1
    svc.diagnostics("b.ts")
    assert [f["arguments"]["file"] for f in scripted.frames_for("open")] == [
        str(svc.scratch_dir / "a.ts"), str(svc.scratch_dir / "b.ts")]


def test_diagnostics_merges_syntactic_then_semantic(fake):
    svc, scripted = fake
    scripted.responses["syntacticDiagnosticsSync"] = [_raw_diag(1005, line=1, offset=5)]
    scripted.responses["semanticDiagnosticsSync"] = [_raw_diag(2322, line=2, offset=7)]
    svc.open_project({"a.ts": "let x y;\nconst z: number = 'a';\n"})
    out = svc.diagnostics("a.ts")
    assert [d.code for d in out] == ["TS1005", "TS2322"]
    args = scripted.frames_for("semanticDiagnosticsSync")[0]["arguments"]
    assert args["includeLinePosition"] is False


def test_clean_recheck_is_an_answer_not_an_absence(fake):
    """Edge case 15 -- the second, quieter #279 win. `TsLspService` cannot tell
    a clean re-check from "nothing was published" (`n_no_publish`, #278);
    the sync path always answers, so there is no such counter here at all."""
    svc, _ = fake
    svc.open_project({"a.ts": "const a = 1;\n"})
    svc.update("a.ts", "const a = 2;\n")
    assert svc.diagnostics("a.ts") == []
    assert svc.n_timeouts == 0 and svc.n_command_errors == 0
    assert not hasattr(svc, "n_no_publish")
    assert svc.op_counts["diagnostics"] == 1


def test_diagnostics_timeout_returns_empty_and_counts(fake):
    svc, scripted = fake
    svc.open_project({"a.ts": "const a = 1;\n"})
    scripted.pending.add("semanticDiagnosticsSync")
    assert svc.diagnostics("a.ts") == []
    assert svc.n_timeouts == 1
    assert svc.op_counts["diagnostics"] == 1


def test_diagnostics_unsuccessful_command_returns_empty_and_counts(fake):
    svc, scripted = fake
    svc.open_project({"a.ts": "const a = 1;\n"})
    scripted.responses["semanticDiagnosticsSync"] = _TsError("No Project.")
    assert svc.diagnostics("a.ts") == []
    assert svc.n_command_errors == 1
    assert svc.n_timeouts == 0


def test_diagnostics_dead_child_restarts_once_and_retries(tmp_path):
    """A dead child mid-request surfaces as `ConnectionError` from the demux;
    `diagnostics()` restarts once and retries rather than raising."""
    first = _ScriptedTsServer(responses={"open": None, "syntacticDiagnosticsSync": [],
                                         "semanticDiagnosticsSync": []},
                              pending=("semanticDiagnosticsSync",))
    second = _ScriptedTsServer(responses={"open": None, "syntacticDiagnosticsSync": [],
                                          "semanticDiagnosticsSync": [_raw_diag(2339)]})
    svc = _FakeDirect(first, timeout_s=_TIMEOUT, scratch_dir=tmp_path)

    def _restart() -> None:
        svc.n_restarts += 1
        svc._endpoint.close()
        svc._open.clear()
        svc._endpoint = TsServerEndpoint(second.to_client_r, second.to_server_w,
                                         on_event=svc._on_event)
        svc._endpoint.start()
        for path in ("a.ts",):
            svc._ensure_open(path)

    svc._restart = _restart   # type: ignore[method-assign]
    _seed_file(svc, "a.ts", "const a = 1;\n")
    svc._open.add("a.ts")

    def _die():
        while not first.frames_for("semanticDiagnosticsSync"):
            time.sleep(0.005)
        first.to_client_w.close()

    threading.Thread(target=_die, daemon=True).start()
    out = svc.diagnostics("a.ts")
    assert [d.code for d in out] == ["TS2339"]
    assert svc.n_restarts == 1
    assert svc.n_timeouts == 0 and svc.n_command_errors == 0
    svc._endpoint.close()
    first.close()
    second.close()


def test_codes_mirrors_diagnostics(fake):
    svc, scripted = fake
    scripted.responses["semanticDiagnosticsSync"] = [_raw_diag(2339), _raw_diag(2322)]
    svc.open_project({"a.ts": "const a = 1;\n"})
    assert svc.codes("a.ts") == ["TS2339", "TS2322"]


def test_calls_per_s_is_zero_before_any_op(fake):
    svc, _ = fake
    assert svc.calls_per_s == 0.0
    svc.open_project({"a.ts": "const a = 1;\n"})
    svc.diagnostics("a.ts")
    assert svc.calls_per_s > 0.0


def test_module_is_stdlib_only():
    """The seam guard (`tests/test_import_guard.py`) covers backend leakage;
    this pins the narrower promise in the module docstring -- no third-party
    import at all, so the CI gate runs on the `portable` job."""
    source = (Path(__file__).resolve().parents[1] / "src/lsp/ts_server_direct.py").read_text()
    for banned in ("import mlx", "import torch", "import numpy", "import requests"):
        assert banned not in source
