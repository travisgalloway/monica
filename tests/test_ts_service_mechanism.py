"""Binary-free mechanism tests for `src/lsp/ts_service.py`'s `TsLspService` --
the CI gate for this class (no `typescript-language-server` needed; the live
correctness tests are in `tests/test_ts_service.py`, skipped without the
toolchain).

Built on `tests/test_jsonrpc.py`'s `os.pipe()` fake-transport pattern (a
`_ScriptedServer` here plays the "other side of the wire" and auto-replies
per a canned table) plus `tests/test_opengrep_oracle.py`'s "subclass and
bypass `__init__`" precedent (`_FakeService`), so the class's own logic --
version-aware diagnostics gating, lazy `didOpen`, timeout/error handling,
restart bookkeeping, and every result-mapping helper -- runs unchanged
against a scripted server, with no subprocess anywhere.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

import pytest

from src.lsp.jsonrpc import JsonRpcEndpoint, encode_message, read_message
from src.lsp.ts_service import (
    CompletionItem, Location, Symbol, TsLspService,
    _content_changes, _did_change_params, _map_completion_items, _map_hover,
    _map_location, _map_symbols, _references_params, _text_document_position,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
_TIMEOUT = 5.0


def _pipe_files() -> Tuple[BinaryIO, BinaryIO]:
    r_fd, w_fd = os.pipe()
    return os.fdopen(r_fd, "rb", buffering=0), os.fdopen(w_fd, "wb", buffering=0)


def _wait_until(cond, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


class _RpcError:
    """Sentinel telling `_ScriptedServer` to reply with a JSON-RPC `error`
    instead of a `result`."""

    def __init__(self, message: str = "scripted failure") -> None:
        self.message = message


def _raw_diag(code: int, line0: int = 0, char0: int = 0, message: str = "msg",
              severity: int = 1) -> dict:
    return {"range": {"start": {"line": line0, "character": char0},
                       "end": {"line": line0, "character": char0 + 1}},
            "code": code, "message": message, "severity": severity}


class _ScriptedServer:
    """The "other side" of the wire: two `os.pipe()` pairs plus a daemon
    thread that reads client->server frames and replies per `responses`
    (method -> value, or a `value(request) -> value` callable, or an
    `_RpcError`), recording every request/notification it saw. Methods listed
    in `pending` are deliberately never answered (exercises the timeout
    path)."""

    def __init__(self, responses=None, pending=()) -> None:
        self.to_client_r, self.to_client_w = _pipe_files()   # server writes here
        self.to_server_r, self.to_server_w = _pipe_files()   # endpoint writes here
        self.responses: Dict[str, object] = dict(responses or {})
        self.pending = set(pending)
        self.seen: List[dict] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            msg = read_message(self.to_server_r)
            if msg is None:
                return
            with self._lock:
                self.seen.append(msg)
            if "id" not in msg:
                continue   # a notification -- no reply
            method = msg.get("method")
            if method in self.pending:
                continue   # deliberately unanswered
            value = self.responses.get(method, {})
            if callable(value):
                value = value(msg)
            if isinstance(value, _RpcError):
                self._send({"jsonrpc": "2.0", "id": msg["id"],
                           "error": {"code": -32000, "message": value.message}})
            else:
                self._send({"jsonrpc": "2.0", "id": msg["id"], "result": value})

    def _send(self, obj: dict) -> None:
        self.to_client_w.write(encode_message(obj))

    def push_diagnostics(self, uri: str, diagnostics: List[dict], version=None) -> None:
        params = {"uri": uri, "diagnostics": diagnostics}
        if version is not None:
            params["version"] = version
        self._send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                   "params": params})

    def requests_for(self, method: str) -> List[dict]:
        with self._lock:
            return [m for m in self.seen if m.get("method") == method]

    def notifications_for(self, method: str) -> List[dict]:
        return [m for m in self.requests_for(method) if "id" not in m]

    def close(self) -> None:
        for f in (self.to_client_r, self.to_client_w, self.to_server_r, self.to_server_w):
            try:
                f.close()
            except (OSError, ValueError):
                pass


class _FakeService(TsLspService):
    """A `TsLspService` whose subprocess-touching pieces are bypassed -- no
    spawn, no `initialize` handshake -- wired instead to a `_ScriptedServer`'s
    pipes. `_ensure_alive` is stubbed to a no-op (mirrors
    `test_opengrep_oracle.py`'s `_FakeOracle`): there is no real process to
    poll, and none of these tests exercise restart (the plan's restart-
    recovery acceptance test needs a real process and lives in
    `tests/test_ts_service.py`)."""

    def __init__(self, scripted: _ScriptedServer, *, timeout_s: float, scratch_dir: Path) -> None:
        # Deliberately bypass TsLspService.__init__ (no binary, no spawn, no
        # tempdir).
        self.timeout_s = timeout_s
        self.tsconfig = None
        self.initialization_options: dict = {}
        self.n_calls = 0
        self.wall_s = 0.0
        self.n_timeouts = 0
        self.n_restarts = 0
        self.op_counts: Dict[str, int] = {}
        self.op_wall_s: Dict[str, float] = {}
        self.cold_load_s = None
        self.last_completion_incomplete = None

        self.scratch_dir = scratch_dir
        self._project_opened = False
        self._files: Dict[str, str] = {}
        self._versions: Dict[str, int] = {}
        self._open: set = set()

        self._diag_lock = threading.Lock()
        self._diag_events: Dict[str, threading.Event] = {}
        self._diag_state: Dict[str, dict] = {}
        self._diag_seq_counter = 0
        self._update_marker: Dict[str, int] = {}

        self.server_sync_kind = None
        self.server_capabilities: dict = {}

        self._proc = None
        self._endpoint = JsonRpcEndpoint(scripted.to_client_r, scripted.to_server_w,
                                         on_notification=self._on_notification)
        self._endpoint.start()

    def _ensure_alive(self) -> None:
        pass  # no real process in the fake


@pytest.fixture
def scripted():
    s = _ScriptedServer()
    yield s
    s.close()


@pytest.fixture
def fake(scripted, tmp_path):
    svc = _FakeService(scripted, timeout_s=0.3, scratch_dir=tmp_path)
    yield svc, scripted
    svc._endpoint.close()


def _seed_open_file(svc: TsLspService, path: str, text: str, *, version: int = 1) -> str:
    """Register `path` as if `open_project` had materialized it, WITHOUT
    sending a real `didOpen` -- the test decides when that happens."""
    (svc.scratch_dir / path).parent.mkdir(parents=True, exist_ok=True)
    (svc.scratch_dir / path).write_text(text, encoding="utf-8")
    svc._files[path] = text
    svc._versions[path] = version
    return svc._uri(path)


# --------------------------------------------------------------------------- #
# 1. Pure helpers, no transport at all
# --------------------------------------------------------------------------- #

def test_did_change_params_shape():
    params = _did_change_params("file:///a.ts", 3, "hello world")
    assert params["textDocument"] == {"uri": "file:///a.ts", "version": 3}
    assert params["contentChanges"] == [{"text": "hello world"}]
    assert "range" not in params["contentChanges"][0]


def test_content_changes_ranged_variant():
    rng = {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}
    out = _content_changes("x", change_range=rng)
    assert out == [{"range": rng, "text": "x"}]


def test_text_document_position_shape():
    pos = {"line": 2, "character": 5}
    params = _text_document_position("file:///a.ts", pos)
    assert params == {"textDocument": {"uri": "file:///a.ts"}, "position": pos}


def test_references_params_carries_include_declaration():
    pos = {"line": 0, "character": 0}
    params = _references_params("file:///a.ts", pos, True)
    assert params["context"] == {"includeDeclaration": True}
    params_false = _references_params("file:///a.ts", pos, False)
    assert params_false["context"] == {"includeDeclaration": False}


# --------------------------------------------------------------------------- #
# 2. Result mapping variants
# --------------------------------------------------------------------------- #

def test_map_location_plain_location(tmp_path):
    raw = {"uri": (tmp_path / "src/a.ts").as_uri(),
           "range": {"start": {"line": 1, "character": 2}, "end": {"line": 1, "character": 5}}}
    out = _map_location(raw, tmp_path)
    assert len(out) == 1
    loc = out[0]
    assert isinstance(loc, Location)
    assert loc.path == "src/a.ts"
    assert loc.line == 2 and loc.col == 3
    assert loc.end_line == 2 and loc.end_col == 6


def test_map_location_location_link(tmp_path):
    raw = {"targetUri": (tmp_path / "src/b.ts").as_uri(),
           "targetRange": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
           "targetSelectionRange": {"start": {"line": 3, "character": 4},
                                    "end": {"line": 3, "character": 7}}}
    out = _map_location(raw, tmp_path)
    assert len(out) == 1
    # targetSelectionRange preferred over targetRange.
    assert out[0].line == 4 and out[0].col == 5


def test_map_location_none_is_empty_list(tmp_path):
    assert _map_location(None, tmp_path) == []


def test_map_location_bare_dict_vs_list(tmp_path):
    raw_single = {"uri": (tmp_path / "a.ts").as_uri(),
                  "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}}
    single = _map_location(raw_single, tmp_path)
    assert len(single) == 1
    as_list = _map_location([raw_single, raw_single], tmp_path)
    assert len(as_list) == 2


def test_map_completion_items_bare_array():
    raw = [{"label": "x"}, {"label": "y", "kind": 5, "insertText": "y()"}]
    items = _map_completion_items(raw)
    assert [i.label for i in items] == ["x", "y"]
    assert items[1].kind == 5 and items[1].insert_text == "y()"
    assert isinstance(items[0], CompletionItem)


def test_map_completion_items_completion_list():
    raw = {"isIncomplete": True, "items": [{"label": "z"}]}
    items = _map_completion_items(raw)
    assert [i.label for i in items] == ["z"]


def test_map_completion_items_none():
    assert _map_completion_items(None) == []


def test_map_hover_markup_content():
    assert _map_hover({"kind": "markdown", "value": "**bold**"}) == "**bold**"


def test_map_hover_bare_string():
    assert _map_hover("plain hover text") == "plain hover text"


def test_map_hover_list_of_marked_strings():
    contents = ["first", {"language": "ts", "value": "const x = 1;"}]
    assert _map_hover(contents) == "first\nconst x = 1;"


def test_map_hover_none():
    assert _map_hover(None) is None


def test_map_symbols_nested_document_symbol():
    raw = [{
        "name": "Outer", "kind": 5,
        "range": {"start": {"line": 0, "character": 0}, "end": {"line": 5, "character": 0}},
        "selectionRange": {"start": {"line": 0, "character": 5}, "end": {"line": 0, "character": 10}},
        "children": [{
            "name": "inner", "kind": 6,
            "range": {"start": {"line": 1, "character": 0}, "end": {"line": 2, "character": 0}},
            "selectionRange": {"start": {"line": 1, "character": 2}, "end": {"line": 1, "character": 7}},
        }],
    }]
    out = _map_symbols(raw)
    assert [s.name for s in out] == ["Outer", "inner"]
    assert out[0].container is None
    assert out[1].container == "Outer"
    assert out[1].line == 2 and out[1].col == 3


def test_map_symbols_flat_symbol_information():
    raw = [{
        "name": "foo", "kind": 12, "containerName": "MyModule",
        "location": {"uri": "file:///a.ts",
                     "range": {"start": {"line": 2, "character": 0}, "end": {"line": 2, "character": 3}}},
    }]
    out = _map_symbols(raw)
    assert len(out) == 1
    assert out[0].name == "foo" and out[0].container == "MyModule"
    assert out[0].line == 3


def test_map_symbols_none():
    assert _map_symbols(None) == []


# --------------------------------------------------------------------------- #
# 3. Version counter monotonicity
# --------------------------------------------------------------------------- #

def test_update_version_counter_is_monotonic_and_server_sees_it_in_order(fake):
    svc, scripted = fake
    _seed_open_file(svc, "a.ts", "v1")
    svc._open.add("a.ts")   # already open -- update() must not re-didOpen

    versions = [svc.update("a.ts", f"v{n}") for n in (2, 3, 4)]
    assert versions == [2, 3, 4]

    assert _wait_until(lambda: len(scripted.notifications_for("textDocument/didChange")) == 3)
    changes = scripted.notifications_for("textDocument/didChange")
    seen_versions = [c["params"]["textDocument"]["version"] for c in changes]
    assert seen_versions == [2, 3, 4]
    assert scripted.notifications_for("textDocument/didOpen") == []


# --------------------------------------------------------------------------- #
# 4. Lazy didOpen: exactly once per path; update() opens an untouched path
# --------------------------------------------------------------------------- #

def test_didopen_sent_exactly_once_across_repeated_ops(fake):
    svc, scripted = fake
    scripted.responses["textDocument/definition"] = []
    _seed_open_file(svc, "a.ts", "const x = 1;\n")

    svc.definition("a.ts", 0)
    svc.definition("a.ts", 5)
    assert _wait_until(lambda: len(scripted.requests_for("textDocument/definition")) == 2)
    assert len(scripted.notifications_for("textDocument/didOpen")) == 1


def test_update_on_untouched_path_opens_it_first(fake):
    svc, scripted = fake
    _seed_open_file(svc, "b.ts", "old text")
    assert "b.ts" not in svc._open

    version = svc.update("b.ts", "new text")
    assert version == 2
    assert _wait_until(lambda: len(scripted.notifications_for("textDocument/didChange")) == 1)
    opens = scripted.notifications_for("textDocument/didOpen")
    assert len(opens) == 1
    assert opens[0]["params"]["textDocument"]["text"] == "old text"   # opened BEFORE the edit landed
    assert opens[0]["params"]["textDocument"]["version"] == 1
    changes = scripted.notifications_for("textDocument/didChange")
    assert changes[0]["params"]["textDocument"]["version"] == 2
    assert changes[0]["params"]["contentChanges"] == [{"text": "new text"}]


# --------------------------------------------------------------------------- #
# 5. Timeout is counted, not raised
# --------------------------------------------------------------------------- #

def test_completions_timeout_is_counted_not_raised(fake):
    svc, scripted = fake
    scripted.pending.add("textDocument/completion")
    _seed_open_file(svc, "a.ts", "x.")

    result = svc.completions("a.ts", 2)
    assert result == []
    assert svc.n_timeouts == 1


def test_definition_timeout_returns_empty_list(fake):
    svc, scripted = fake
    scripted.pending.add("textDocument/definition")
    _seed_open_file(svc, "a.ts", "x")

    assert svc.definition("a.ts", 0) == []
    assert svc.n_timeouts == 1


def test_quickinfo_timeout_returns_none(fake):
    svc, scripted = fake
    scripted.pending.add("textDocument/hover")
    _seed_open_file(svc, "a.ts", "x")

    assert svc.quickinfo("a.ts", 0) is None
    assert svc.n_timeouts == 1


# --------------------------------------------------------------------------- #
# 6. A JSON-RPC error reply propagates -- the deliberate asymmetry with timeouts
# --------------------------------------------------------------------------- #

def test_jsonrpc_error_reply_raises_runtime_error(fake):
    svc, scripted = fake
    scripted.responses["textDocument/definition"] = _RpcError("bad params")
    _seed_open_file(svc, "a.ts", "x")

    with pytest.raises(RuntimeError):
        svc.definition("a.ts", 0)


# --------------------------------------------------------------------------- #
# 7. diagnostics() version gating
# --------------------------------------------------------------------------- #

def test_diagnostics_version_gating(fake):
    svc, scripted = fake
    uri = _seed_open_file(svc, "a.ts", "old text", version=1)
    svc._open.add("a.ts")

    # v1 push arrives -- the cache-hit fast path for "unchanged document".
    scripted.push_diagnostics(uri, [_raw_diag(2339, message="old error")], version=1)
    assert _wait_until(lambda: uri in svc._diag_state)
    diags = svc.diagnostics("a.ts")
    assert [d.code for d in diags] == ["TS2339"]
    assert svc.n_timeouts == 0

    # Edit -> version 2.
    svc.update("a.ts", "new text")
    assert svc._versions["a.ts"] == 2

    # A STALE v1 push arrives AFTER the edit -- must not be mistaken for current.
    scripted.push_diagnostics(uri, [_raw_diag(9999, message="stale")], version=1)
    assert _wait_until(lambda: svc._diag_state[uri]["version"] == 1)
    stale_result = svc.diagnostics("a.ts")   # times out -- no v2 push yet
    assert stale_result == []
    assert svc.n_timeouts == 1

    # Now the real post-edit v2 push arrives.
    scripted.push_diagnostics(uri, [_raw_diag(1234, message="new error")], version=2)
    assert _wait_until(lambda: svc._diag_state[uri]["version"] == 2)
    fresh = svc.diagnostics("a.ts")
    assert [d.code for d in fresh] == ["TS1234"]


def test_diagnostics_falls_back_to_seq_when_server_omits_version(fake):
    svc, scripted = fake
    uri = _seed_open_file(svc, "a.ts", "text", version=1)
    svc._open.add("a.ts")
    with svc._diag_lock:
        marker_before = svc._diag_seq_counter

    scripted.push_diagnostics(uri, [_raw_diag(2339)])   # no version field
    assert _wait_until(lambda: uri in svc._diag_state)
    with svc._diag_lock:
        assert svc._diag_state[uri]["seq"] > marker_before
    diags = svc.diagnostics("a.ts")
    assert [d.code for d in diags] == ["TS2339"]


# --------------------------------------------------------------------------- #
# 8. Trap A / Trap B pinning on this class
# --------------------------------------------------------------------------- #

def test_trap_a_integer_code_becomes_ts_prefixed(fake):
    svc, _ = fake
    text = "const x: number = 'hi';\n"
    payload = [_raw_diag(2339, message="bad")]
    out = svc._filter_diagnostics_payload(payload, text)
    assert [d.code for d in out] == ["TS2339"]


def test_trap_b_severity_4_hint_is_dropped(fake):
    svc, _ = fake
    text = "const unused = 1;\n"
    payload = [_raw_diag(6133, severity=4)]
    assert svc._filter_diagnostics_payload(payload, text) == []


# --------------------------------------------------------------------------- #
# 9. _generate_project determinism (scripts/bench_ts_lsp.py)
# --------------------------------------------------------------------------- #

def _bench_module():
    spec = importlib.util.spec_from_file_location(
        "bench_ts_lsp", REPO_ROOT / "scripts" / "bench_ts_lsp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def test_generate_project_is_deterministic():
    mod = _bench_module()
    a = mod._generate_project(12, 0)
    b = mod._generate_project(12, 0)
    assert a == b


def test_generate_project_different_seed_differs():
    mod = _bench_module()
    a = mod._generate_project(12, 0)
    b = mod._generate_project(12, 1)
    assert a != b


def test_generate_project_imports_resolve_and_are_acyclic():
    mod = _bench_module()
    files = mod._generate_project(20, 0)
    index_of = {path: i for i, path in enumerate(sorted(files, key=mod._module_index))}
    for path, text in files.items():
        my_idx = mod._module_index(path)
        for m in mod._IMPORT_RE.finditer(text):
            target_rel = m.group(1)
            target_path = mod._resolve_import(path, target_rel)
            assert target_path in files, f"{path} imports missing {target_path}"
            assert mod._module_index(target_path) < my_idx or target_path == "src/hub.ts", (
                f"{path} imports {target_path}, not lower-indexed -- would create a cycle")
