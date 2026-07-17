"""Pure LSP-framing JSON-RPC client — split from any live endpoint so the async
demux (the hard part of this refactor) is testable with a fake `os.pipe()`
transport and no subprocess.

Framing is the LSP wire format: `Content-Length: <n>\\r\\n\\r\\n<n bytes of UTF-8 JSON>`
(`encode_message`/`read_message`). `JsonRpcEndpoint` runs a daemon reader thread that
demuxes three message shapes on every read:

  - `id` + (`result` | `error`) -> a response to one of OUR requests; resolves that
    id's waiter.
  - `method`, no `id` -> a notification (e.g. `textDocument/publishDiagnostics`);
    dispatched to `on_notification` if given.
  - `method` + `id` -> a server-to-client REQUEST (e.g. tsserver-lsp's
    `window/workDoneProgress/create`, or opengrep's `workspace/configuration`
    lookups). These must be replied to or a spec-conformant server will block
    waiting for the answer — the endpoint always answers `{"result": null}`
    (dispatching to `on_notification` first, purely for observation; the reply is
    unconditional and does not wait on it).

EOF or a dead process fails every pending waiter rather than leaving it hanging
forever — `request()` itself also always has a `timeout` and raises `TimeoutError`
rather than blocking indefinitely. Every above-the-seam caller (`ts_lsp.py`,
`opengrep.py`) depends on this: "every wait needs a timeout; the loop must never
hang on a scan or dead server."

`spawn()` also drains the child's stderr in a background thread (unbounded stderr
output — and opengrep's scan-status banner is verbose — would otherwise fill the
pipe buffer and deadlock the child on a blocking stderr write) and keeps a bounded
tail of it for debugging.

ABOVE THE SEAM — stdlib only (subprocess/threading/json). No `mlx`/`torch` import
anywhere in this module (guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import deque
from typing import Any, BinaryIO, Callable, Deque, Dict, List, Optional, Tuple

OnNotification = Callable[[dict], None]

_STDERR_TAIL_LINES = 200


def encode_message(obj: dict) -> bytes:
    """Frame `obj` as one `Content-Length`-delimited LSP message."""
    body = json.dumps(obj).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def read_message(stream: BinaryIO) -> Optional[dict]:
    """Read one framed LSP message from `stream`. Returns `None` at EOF (a short
    read anywhere — header or body — is treated as EOF, never as a partial
    message to wait longer for; the reader thread's caller decides what EOF
    means)."""
    header = b""
    while not header.endswith(b"\r\n\r\n"):
        b = stream.read(1)
        if not b:
            return None
        header += b

    length: Optional[int] = None
    for line in header.split(b"\r\n"):
        if b":" not in line:
            continue
        key, _, value = line.partition(b":")
        if key.strip().lower() == b"content-length":
            length = int(value.strip())
    if length is None:
        return None

    body = b""
    while len(body) < length:
        chunk = stream.read(length - len(body))
        if not chunk:
            return None
        body += chunk
    return json.loads(body.decode("utf-8"))


class _Waiter:
    """One in-flight request's rendezvous point. `set()` is called at most once,
    from the reader thread (a response) or from `_fail_all_pending` (EOF/timeout
    cleanup); `wait()` is called from the requesting thread. `None` is never a
    legitimate value to `set()` (a JSON-RPC response is always a dict, and
    failures pass an `Exception`), so it safely doubles as the "still nothing"
    sentinel distinguishing a timeout from a real result.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._value: Any = None

    def set(self, value: Any) -> None:
        self._value = value
        self._event.set()

    def wait(self, timeout: float) -> Any:
        if not self._event.wait(timeout):
            return None
        return self._value


class JsonRpcEndpoint:
    """A live JSON-RPC/LSP endpoint over any `(reader, writer)` binary-stream pair.

    `reader`/`writer` need only `.read(n)` / `.write(bytes)` / `.flush()` — an
    `os.pipe()` pair (via `os.fdopen`) satisfies this with no subprocess at all,
    which is what `tests/test_jsonrpc.py` drives against. `spawn()` below wires
    this to a real child process's stdio.
    """

    def __init__(self, reader: BinaryIO, writer: BinaryIO,
                 on_notification: Optional[OnNotification] = None) -> None:
        self._reader = reader
        self._writer = writer
        self._on_notification = on_notification
        self._next_id = 1
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: Dict[int, _Waiter] = {}
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def start(self) -> None:
        """Start the daemon reader thread. Must be called once before `request`/
        `notify` can see any server-initiated traffic."""
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while True:
            try:
                msg = read_message(self._reader)
            except (ValueError, OSError):
                msg = None
            if msg is None:
                self._fail_all_pending(ConnectionError("jsonrpc endpoint: EOF or dead process"))
                return
            self._dispatch(msg)

    def _dispatch(self, msg: dict) -> None:
        has_id = "id" in msg
        has_result_or_error = "result" in msg or "error" in msg
        has_method = "method" in msg

        if has_id and has_result_or_error and not has_method:
            waiter = None
            with self._lock:
                waiter = self._pending.pop(msg["id"], None)
            if waiter is not None:
                waiter.set(msg)
            return

        if has_method and not has_id:
            # Notification.
            self._notify_caller(msg)
            return

        if has_method and has_id:
            # Server -> client request: observe, then always reply so the
            # server can never block waiting on us.
            self._notify_caller(msg)
            self._reply_null(msg["id"])
            return
        # Malformed/unrecognized shape — drop it rather than raise out of the
        # reader thread (that would silently kill the demux for everyone).

    def _notify_caller(self, msg: dict) -> None:
        if self._on_notification is None:
            return
        try:
            self._on_notification(msg)
        except Exception:
            pass

    def _reply_null(self, request_id: Any) -> None:
        self._write({"jsonrpc": "2.0", "id": request_id, "result": None})

    def request(self, method: str, params: Optional[dict] = None, *,
                timeout: float) -> Any:
        """Send a request, block up to `timeout` seconds for its response, and
        return `result`. Raises `TimeoutError` (never hangs) or `RuntimeError` if
        the server replied with a JSON-RPC `error`."""
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            waiter = _Waiter()
            self._pending[request_id] = waiter

        msg: dict = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)

        got = waiter.wait(timeout)
        if got is None:
            with self._lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"timed out after {timeout}s waiting for {method!r} (id={request_id})")
        if isinstance(got, BaseException):
            raise got
        if "error" in got:
            raise RuntimeError(f"{method} failed: {got['error']}")
        return got.get("result")

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Fire-and-forget notification — no response expected."""
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._write(msg)

    def _write(self, obj: dict) -> None:
        data = encode_message(obj)
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
        self._fail_all_pending(ConnectionError("jsonrpc endpoint closed"))


def spawn(argv: List[str], cwd: Optional[str] = None,
          env: Optional[Dict[str, str]] = None) -> Tuple[subprocess.Popen, JsonRpcEndpoint]:
    """Start `argv` as a child process and wrap its stdio in a started
    `JsonRpcEndpoint`. Drains stderr in the background (see module docstring) so
    verbose server logging can never deadlock the child."""
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None

    stderr_tail: Deque[bytes] = deque(maxlen=_STDERR_TAIL_LINES)

    def _drain_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_tail.append(line)

    threading.Thread(target=_drain_stderr, daemon=True).start()

    endpoint = JsonRpcEndpoint(proc.stdout, proc.stdin)
    endpoint.stderr_tail = stderr_tail  # type: ignore[attr-defined]
    endpoint.start()
    return proc, endpoint
