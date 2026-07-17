"""Fake-transport tests for `src/lsp/jsonrpc.py` — no subprocess, no live language
server. Two `os.pipe()` pairs stand in for "what the server writes to the client"
and "what the client writes to the server", so the async reader-thread demux (the
hard part of #199 Stage A) is fully exercised in isolation before `ts_lsp.py` or
`opengrep.py` (which depend on it) are written.
"""

from __future__ import annotations

import io
import os
import threading
import time
from typing import BinaryIO, List, Tuple

import pytest

from src.lsp.jsonrpc import JsonRpcEndpoint, encode_message, read_message

_TIMEOUT = 5.0  # generous ceiling for a background thread on a local pipe


def _pipe_files() -> Tuple[BinaryIO, BinaryIO]:
    """One `os.pipe()` as a pair of unbuffered binary file objects (read end,
    write end)."""
    r_fd, w_fd = os.pipe()
    return os.fdopen(r_fd, "rb", buffering=0), os.fdopen(w_fd, "wb", buffering=0)


class _FakeServer:
    """The "other side" of the wire: a pipe pair the test writes server->client
    traffic into, and a pipe pair it reads client->server traffic from — wired to
    an `JsonRpcEndpoint` exactly like a real subprocess's stdout/stdin would be.
    """

    def __init__(self, on_notification=None):
        self.to_client_r, self.to_client_w = _pipe_files()   # server writes here
        self.to_server_r, self.to_server_w = _pipe_files()   # endpoint writes here
        self.endpoint = JsonRpcEndpoint(self.to_client_r, self.to_server_w,
                                         on_notification=on_notification)
        self.endpoint.start()

    def send(self, msg: dict) -> None:
        """Simulate the server sending `msg` to the client."""
        self.to_client_w.write(encode_message(msg))

    def recv(self, timeout: float = _TIMEOUT):
        """Read one message the client (endpoint) sent to the server. Runs the
        blocking read in this thread — callers only use it when a message is
        known to be in flight."""
        return read_message(self.to_server_r)

    def close(self) -> None:
        self.endpoint.close()
        for f in (self.to_client_r, self.to_client_w, self.to_server_r, self.to_server_w):
            try:
                f.close()
            except (OSError, ValueError):
                pass


@pytest.fixture
def server():
    s = _FakeServer()
    yield s
    s.close()


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #

def test_encode_read_message_round_trips():
    obj = {"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {"a": [1, 2, "x"]}}
    buf = io.BytesIO(encode_message(obj))
    assert read_message(buf) == obj


def test_read_message_returns_none_at_eof():
    assert read_message(io.BytesIO(b"")) is None


def test_read_message_handles_back_to_back_messages():
    one = {"jsonrpc": "2.0", "method": "a", "params": {}}
    two = {"jsonrpc": "2.0", "method": "b", "params": {}}
    buf = io.BytesIO(encode_message(one) + encode_message(two))
    assert read_message(buf) == one
    assert read_message(buf) == two
    assert read_message(buf) is None


# --------------------------------------------------------------------------- #
# response <-> request-by-id
# --------------------------------------------------------------------------- #

def test_response_resolves_request_by_id(server: _FakeServer):
    result_box: List[object] = []

    def _do_request():
        result_box.append(server.endpoint.request("initialize", {"foo": 1}, timeout=_TIMEOUT))

    t = threading.Thread(target=_do_request)
    t.start()

    sent = server.recv()
    assert sent["method"] == "initialize"
    assert sent["params"] == {"foo": 1}
    server.send({"jsonrpc": "2.0", "id": sent["id"], "result": {"ok": True}})

    t.join(timeout=_TIMEOUT)
    assert not t.is_alive()
    assert result_box == [{"ok": True}]


def test_response_error_raises_runtime_error(server: _FakeServer):
    error_box: List[BaseException] = []

    def _do_request():
        try:
            server.endpoint.request("foo", timeout=_TIMEOUT)
        except BaseException as e:  # noqa: BLE001 - captured for assertion below
            error_box.append(e)

    t = threading.Thread(target=_do_request)
    t.start()
    sent = server.recv()
    server.send({"jsonrpc": "2.0", "id": sent["id"], "error": {"code": -32601, "message": "nope"}})
    t.join(timeout=_TIMEOUT)

    assert len(error_box) == 1
    assert isinstance(error_box[0], RuntimeError)
    assert "nope" in str(error_box[0])


# --------------------------------------------------------------------------- #
# notification routing (what ts_lsp.py's per-uri publishDiagnostics wait builds on)
# --------------------------------------------------------------------------- #

def test_notifications_route_to_on_notification_without_crossing():
    received: List[dict] = []
    events_by_uri = {"a": threading.Event(), "b": threading.Event()}
    payload_by_uri = {}

    def _on_notification(msg):
        received.append(msg)
        if msg.get("method") == "textDocument/publishDiagnostics":
            uri = msg["params"]["uri"]
            payload_by_uri[uri] = msg["params"]["diagnostics"]
            events_by_uri[uri].set()

    s = _FakeServer(on_notification=_on_notification)
    try:
        # Interleave: a request response, then two notifications for DIFFERENT
        # uris, arriving in an order that would reveal any cross-routing bug.
        result_box: List[object] = []

        def _do_request():
            result_box.append(s.endpoint.request("ping", timeout=_TIMEOUT))

        t = threading.Thread(target=_do_request)
        t.start()
        sent = s.recv()

        s.send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                "params": {"uri": "b", "diagnostics": ["B_FINDING"]}})
        s.send({"jsonrpc": "2.0", "id": sent["id"], "result": "pong"})
        s.send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                "params": {"uri": "a", "diagnostics": ["A_FINDING"]}})

        t.join(timeout=_TIMEOUT)
        assert result_box == ["pong"]

        assert events_by_uri["a"].wait(_TIMEOUT)
        assert events_by_uri["b"].wait(_TIMEOUT)
        assert payload_by_uri["a"] == ["A_FINDING"]
        assert payload_by_uri["b"] == ["B_FINDING"]
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# failure modes: must never hang
# --------------------------------------------------------------------------- #

def test_missing_reply_raises_timeout_error_not_hang(server: _FakeServer):
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        server.endpoint.request("never_answered", timeout=0.3)
    assert time.monotonic() - start < _TIMEOUT, "request() must not block past its own timeout"


def test_eof_fails_all_pending_waiters(server: _FakeServer):
    errors: List[BaseException] = []

    def _do_request():
        try:
            server.endpoint.request("will_never_get_a_reply", timeout=_TIMEOUT)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=_do_request)
    t.start()
    server.recv()  # drain the outgoing request so the write doesn't matter

    # Simulate the server process dying: close its outgoing pipe -> EOF on the
    # endpoint's reader.
    server.to_client_w.close()

    t.join(timeout=_TIMEOUT)
    assert not t.is_alive(), "EOF must fail the pending waiter, not leave it hanging"
    assert len(errors) == 1
    assert isinstance(errors[0], ConnectionError)


# --------------------------------------------------------------------------- #
# server -> client requests must always get a reply
# --------------------------------------------------------------------------- #

def test_server_to_client_request_gets_a_null_reply(server: _FakeServer):
    server.send({"jsonrpc": "2.0", "id": "srv-1", "method": "window/workDoneProgress/create",
                 "params": {"token": "tok"}})
    reply = server.recv()
    assert reply == {"jsonrpc": "2.0", "id": "srv-1", "result": None}


def test_server_to_client_request_also_reaches_on_notification():
    seen: List[dict] = []
    s = _FakeServer(on_notification=lambda msg: seen.append(msg))
    try:
        s.send({"jsonrpc": "2.0", "id": "srv-2", "method": "workspace/configuration",
                "params": {"items": []}})
        reply = s.recv()
        assert reply["id"] == "srv-2"
        assert reply["result"] is None
        assert any(m.get("method") == "workspace/configuration" for m in seen)
    finally:
        s.close()
