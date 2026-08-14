// JsonRpc.swift — LSP wire framing + the async demux, a direct port of
// `src/lsp/jsonrpc.py`. Framing is `Content-Length: <n>\r\n\r\n<n bytes of UTF-8 JSON>`;
// `JsonRpcEndpoint` runs a daemon reader thread that demuxes three message shapes on every
// read, exactly like the Python reference's module docstring describes:
//
//   - `id` + (`result` | `error`), no `method` -> a response to one of OUR requests;
//     resolves that id's waiter.
//   - `method`, no `id` -> a notification; dispatched to `onNotification` if given.
//   - `method` + `id` -> a server-to-client REQUEST. Observed via `onNotification`, then
//     ALWAYS answered `{"result": null}` — without this a spec-conformant server blocks
//     waiting for our answer.
//
// EOF fails every pending waiter rather than leaving it hanging forever; `request()` always
// has a timeout and throws rather than blocking indefinitely. Threading model mirrors the
// Python reference (a lock-guarded pending-request table + a condition-variable waiter per
// request) rather than async/await — `swift/` is tools-version 5.9 (language mode 5), and
// the reference itself is thread-based.

import Foundation

public enum JsonRpcError: Error, CustomStringConvertible {
    case timeout(method: String, id: Int)
    case closed
    case rpcError(method: String, payload: Any)
    case malformedResponse(method: String)

    public var description: String {
        switch self {
        case .timeout(let method, let id):
            return "timed out waiting for \(method) (id=\(id))"
        case .closed:
            return "jsonrpc endpoint closed (EOF or dead process)"
        case .rpcError(let method, let payload):
            return "\(method) failed: \(payload)"
        case .malformedResponse(let method):
            return "\(method): malformed JSON-RPC response"
        }
    }
}

/// One in-flight request's rendezvous point. `set()` is called at most once — from the
/// reader thread (a real response) or from `failAllPending` (EOF/close cleanup); `wait()` is
/// called from the requesting thread.
final class RpcWaiter {
    enum Result { case message([String: Any]); case failure(Error) }

    private let cond = NSCondition()
    private var result: Result?

    func set(_ r: Result) {
        cond.lock()
        defer { cond.unlock() }
        guard result == nil else { return }
        result = r
        cond.signal()
    }

    /// `nil` on timeout (never hangs past `timeout` seconds).
    func wait(timeout: TimeInterval) -> Result? {
        cond.lock()
        defer { cond.unlock() }
        let deadline = Date().addingTimeInterval(timeout)
        while result == nil {
            if !cond.wait(until: deadline) { return result }
        }
        return result
    }
}

public final class JsonRpcEndpoint {
    public typealias OnNotification = ([String: Any]) -> Void

    private let stream: ByteStream
    private let onNotification: OnNotification?
    private var nextId = 1
    private let stateLock = NSLock()
    private let writeLock = NSLock()
    private var pending: [Int: RpcWaiter] = [:]
    private var closed = false

    public init(stream: ByteStream, onNotification: OnNotification? = nil) {
        self.stream = stream
        self.onNotification = onNotification
    }

    /// Start the daemon reader thread. Must be called once before `request`/`notify` can
    /// see any server-initiated traffic.
    public func start() {
        let t = Thread { [weak self] in self?.readLoop() }
        t.name = "MonicaLSP.JsonRpcEndpoint.reader"
        t.stackSize = 1 << 20
        t.start()
    }

    private func readLoop() {
        while true {
            guard let msg = Self.readMessage(stream) else {
                failAllPending(JsonRpcError.closed)
                return
            }
            dispatch(msg)
        }
    }

    private func dispatch(_ msg: [String: Any]) {
        let hasId = msg["id"] != nil
        let hasResultOrError = msg["result"] != nil || msg["error"] != nil
        let hasMethod = (msg["method"] as? String) != nil

        if hasId, hasResultOrError, !hasMethod {
            guard let id = Self.asInt(msg["id"]) else { return }
            stateLock.lock()
            let waiter = pending.removeValue(forKey: id)
            stateLock.unlock()
            waiter?.set(.message(msg))
            return
        }
        if hasMethod, !hasId {
            notifyCaller(msg)
            return
        }
        if hasMethod, hasId {
            // Server -> client request: observe, then ALWAYS reply so the server can never
            // block waiting on us. The reply is unconditional and does not wait on the
            // observer.
            notifyCaller(msg)
            replyNull(msg["id"])
            return
        }
        // Malformed/unrecognized shape — drop it rather than crash the reader thread (that
        // would silently kill the demux for every other pending request too).
    }

    private func notifyCaller(_ msg: [String: Any]) {
        onNotification?(msg)
    }

    private func replyNull(_ id: Any?) {
        let msg: [String: Any] = ["jsonrpc": "2.0", "id": id ?? NSNull(), "result": NSNull()]
        _ = writeMessage(msg)
    }

    /// Send a request, block up to `timeout` seconds for its response, and return `result`.
    /// Throws `JsonRpcError.timeout` (never hangs) or `.rpcError` if the server replied with
    /// a JSON-RPC `error`.
    @discardableResult
    public func request(_ method: String, params: [String: Any]? = nil, timeout: TimeInterval) throws -> Any? {
        let waiter = RpcWaiter()
        let id: Int
        stateLock.lock()
        id = nextId
        nextId += 1
        pending[id] = waiter
        stateLock.unlock()

        var msg: [String: Any] = ["jsonrpc": "2.0", "id": id, "method": method]
        if let params { msg["params"] = params }
        guard writeMessage(msg) else {
            stateLock.lock(); pending.removeValue(forKey: id); stateLock.unlock()
            throw JsonRpcError.closed
        }

        guard let got = waiter.wait(timeout: timeout) else {
            stateLock.lock(); pending.removeValue(forKey: id); stateLock.unlock()
            throw JsonRpcError.timeout(method: method, id: id)
        }
        switch got {
        case .failure(let e):
            throw e
        case .message(let m):
            if let err = m["error"] {
                throw JsonRpcError.rpcError(method: method, payload: err)
            }
            return m["result"]
        }
    }

    /// Fire-and-forget notification — no response expected.
    public func notify(_ method: String, params: [String: Any]? = nil) {
        var msg: [String: Any] = ["jsonrpc": "2.0", "method": method]
        if let params { msg["params"] = params }
        _ = writeMessage(msg)
    }

    private func writeMessage(_ obj: [String: Any]) -> Bool {
        guard let data = try? Self.encodeMessage(obj) else { return false }
        writeLock.lock()
        defer { writeLock.unlock() }
        return stream.write(data)
    }

    private func failAllPending(_ error: Error) {
        stateLock.lock()
        let waiters = Array(pending.values)
        pending.removeAll()
        stateLock.unlock()
        for w in waiters { w.set(.failure(error)) }
    }

    /// Idempotent. Fails any still-pending waiter rather than leaving it to time out on its
    /// own. Closes the write side; the reader thread notices EOF on its own and exits.
    public func close() {
        stateLock.lock()
        if closed { stateLock.unlock(); return }
        closed = true
        stateLock.unlock()
        stream.closeWrite()
        failAllPending(JsonRpcError.closed)
    }

    // MARK: - framing (static — usable directly by a fake-endpoint self-test)

    public static func encodeMessage(_ obj: [String: Any]) throws -> Data {
        let body = try JSONSerialization.data(withJSONObject: obj, options: [])
        let header = "Content-Length: \(body.count)\r\n\r\n"
        var out = Data(header.utf8)
        out.append(body)
        return out
    }

    /// Read one framed LSP message from `stream`. Returns `nil` at EOF (a short read
    /// anywhere — header or body — is EOF, never a partial message to wait longer for).
    public static func readMessage(_ stream: ByteStream) -> [String: Any]? {
        var header = Data()
        let terminator: [UInt8] = [13, 10, 13, 10]  // \r\n\r\n
        while true {
            guard let b = stream.readByte() else { return nil }
            header.append(b)
            if header.count >= 4, Array(header.suffix(4)) == terminator { break }
        }

        var length: Int?
        let headerStr = String(decoding: header, as: UTF8.self)
        for line in headerStr.components(separatedBy: "\r\n") {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[line.startIndex..<colon].trimmingCharacters(in: .whitespaces)
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            if key.lowercased() == "content-length" {
                length = Int(value)
            }
        }
        guard let length else { return nil }
        guard let body = stream.readExact(length) else { return nil }
        guard let obj = try? JSONSerialization.jsonObject(with: body, options: [.fragmentsAllowed]),
              let dict = obj as? [String: Any] else { return nil }
        return dict
    }

    static func asInt(_ any: Any?) -> Int? {
        if let i = any as? Int { return i }
        if let n = any as? NSNumber { return n.intValue }
        if let s = any as? String, let i = Int(s) { return i }
        return nil
    }
}
