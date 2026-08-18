// TsLspClient.swift — port of `src/lsp/ts_service.py`'s `TsLspService`: one persistent
// `typescript-language-server --stdio` process carrying a warm, multi-file program (stable
// URIs, monotonically versioned `didChange` edits), answering `completions` (the fast loop's
// op — debounce-free, #278), `diagnostics`, `definition`, `references`, `quickinfo`,
// `documentSymbols`.
//
// Two hard-won invariants carried over verbatim from the Python reference:
//
//   - **First-push-wins diagnostics, no settle window** (#211 — 0/97 #194 candidates
//     disagreed between the first `publishDiagnostics` push and the union of all pushes on
//     the pinned typescript-language-server 5.3.0). Do NOT "fix" this to settle like
//     `opengrep.py` — that would put a floor of *seconds* on every call and defeat the
//     reason `completions` can sustain 2,500+ calls/s.
//   - **No pull diagnostics** — 5.3.0 advertises no `capabilities.diagnosticProvider`.
//
// Not safe for concurrent use — one client instance == one sequential open/edit/query
// stream, matching the Python reference's contract. `ProcessSupervisor` is the only thing
// that owns the child process; this class only ever touches it through `supervisor.endpoint`
// and `supervisor.shutdown()`.

import Foundation

public enum TsLspClientError: Error, CustomStringConvertible {
    case noToolchain
    case noOpenProject
    case emptyProject
    case alreadyOpened
    case invalidOffset(offset: Int, length: Int)

    public var description: String {
        switch self {
        case .noToolchain:
            return "no typescript-language-server toolchain resolvable"
        case .noOpenProject:
            return "openProject was never called"
        case .emptyProject:
            return "openProject requires at least one file"
        case .alreadyOpened:
            return "openProject called twice -- one TsLspClient instance == one project"
        case .invalidOffset(let offset, let length):
            return "offset \(offset) out of range for source of length \(length)"
        }
    }
}

/// Resolve the argv prefix to invoke `typescript-language-server`, or `nil` if no usable
/// toolchain exists. Mirrors `src/lsp/ts_lsp.py:55` exactly: requires both the local bin
/// (`npm i -D typescript-language-server` in `setDir`) and `node` on PATH (the bin's shebang
/// needs it).
public func resolveTsLsp(setDir: URL) -> [String]? {
    let bin = setDir.appendingPathComponent("node_modules/.bin/typescript-language-server")
    guard FileManager.default.isExecutableFile(atPath: bin.path) else { return nil }
    guard which("node") != nil else { return nil }
    return [bin.path]
}

func which(_ name: String) -> String? {
    guard let pathEnv = ProcessInfo.processInfo.environment["PATH"] else { return nil }
    for dir in pathEnv.split(separator: ":") {
        let candidate = "\(dir)/\(name)"
        if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
    }
    return nil
}

// MARK: - text <-> LSP position (UTF-16 columns, 0-indexed both axes) — port of
// `src/lsp/diagnostics.py`'s `offset_to_lsp_position`/`line_col_to_offset`.

enum LspText {
    static func lineStartOffsets(_ chars: [Character]) -> [Int] {
        var starts = [0]
        for (i, c) in chars.enumerated() where c == "\n" { starts.append(i + 1) }
        return starts
    }

    /// 0-indexed character offset into `chars` -> an LSP `Position` (0-indexed line +
    /// UTF-16 `character`). Throws on an out-of-range offset rather than silently clamping —
    /// a wrong-but-plausible position gets back confidently wrong completions, the exact
    /// BLIND failure mode this module exists to prevent.
    static func offsetToPosition(_ chars: [Character], offset: Int) throws -> [String: Any] {
        guard offset >= 0, offset <= chars.count else {
            throw TsLspClientError.invalidOffset(offset: offset, length: chars.count)
        }
        let starts = lineStartOffsets(chars)
        // bisect_right(starts, offset) - 1
        var lo = 0, hi = starts.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if starts[mid] <= offset { lo = mid + 1 } else { hi = mid }
        }
        let line0 = lo - 1
        let lineStart = starts[line0]
        let lineEnd = (line0 + 1 < starts.count) ? starts[line0 + 1] - 1 : chars.count
        var units = 0
        for i in lineStart..<offset where i < lineEnd {
            units += String(chars[i]).utf16.count
        }
        return ["line": line0, "character": units]
    }

    /// 1-indexed (line, col) UTF-16 position -> 0-indexed character offset. Inverse
    /// convention used by `tsc`/LSP diagnostic payloads.
    static func lineColToOffset(_ chars: [Character], line: Int, col: Int) -> Int {
        let starts = lineStartOffsets(chars)
        let line0 = max(0, min(line - 1, starts.count - 1))
        let lineStart = starts[line0]
        let lineEnd = (line0 + 1 < starts.count) ? starts[line0 + 1] - 1 : chars.count
        var units = 0
        var idx = lineStart
        while idx < lineEnd {
            if units >= col - 1 { break }
            units += String(chars[idx]).utf16.count
            idx += 1
        }
        return idx
    }
}

public final class TsLspClient {
    public struct Location: Equatable {
        public let uri: String
        public let path: String
        public let line, col, endLine, endCol: Int
    }
    public struct CompletionItem: Equatable {
        public let label: String
        public let kind: Int?
        public let detail: String?
        public let insertText: String?
        public let sortText: String?
    }
    public struct Symbol: Equatable {
        public let name: String
        public let kind: Int
        public let line, col: Int
        public let container: String?
    }
    public struct Diagnostic: Equatable {
        public let code: String
        public let line, col: Int
        public let message: String
        public let offset: Int
        public let severity: Int
    }

    public private(set) var nCalls = 0
    public private(set) var wallS: Double = 0
    public private(set) var nTimeouts = 0
    public private(set) var nNoPublish = 0
    public private(set) var nRestarts = 0
    public private(set) var opCounts: [String: Int] = [:]
    public private(set) var opWallS: [String: Double] = [:]
    public private(set) var coldLoadS: Double?
    public private(set) var lastCompletionIncomplete = false

    private let argv: [String]
    private let timeoutS: Double
    private let tsconfigText: String
    public let scratchDir: URL
    private let scratchOwner: Bool

    private var supervisor: ProcessSupervisor?
    private var projectOpened = false
    private var files: [String: String] = [:]
    private var versions: [String: Int] = [:]
    private var open: Set<String> = []

    private let diagLock = NSLock()
    private var diagEvents: [String: ManualResetEvent] = [:]
    private var diagState: [String: (seq: Int, version: Int?, payload: [[String: Any]])] = [:]
    private var diagSeqCounter = 0
    private var updateMarker: [String: Int] = [:]

    private static let teardownTimeoutS = 5.0

    private static let clientCapabilities: [String: Any] = [
        "textDocument": [
            "synchronization": ["dynamicRegistration": false, "willSave": false,
                                 "willSaveWaitUntil": false, "didSave": true],
            "publishDiagnostics": ["relatedInformation": false, "versionSupport": true],
            "definition": ["dynamicRegistration": false, "linkSupport": false],
            "references": ["dynamicRegistration": false],
            "completion": ["dynamicRegistration": false, "contextSupport": true,
                            "completionItem": ["snippetSupport": false,
                                                "documentationFormat": ["plaintext"]]],
            "hover": ["dynamicRegistration": false, "contentFormat": ["plaintext", "markdown"]],
            "documentSymbol": ["dynamicRegistration": false,
                                "hierarchicalDocumentSymbolSupport": true],
        ],
        "workspace": ["workspaceFolders": true],
    ]

    /// - Parameters:
    ///   - argv: `typescript-language-server` invocation prefix (from `resolveTsLsp`).
    ///   - scratchParent: nested scratch dir parent — load-bearing exactly as the Python
    ///     reference's is: TypeScript's default `typeRoots` walk needs to find
    ///     `scratchParent/node_modules/@types/node` from wherever the scratch dir sits.
    public init(argv: [String], timeoutS: Double = 10.0, tsconfigText: String,
                scratchParent: URL) throws {
        self.argv = argv
        self.timeoutS = timeoutS
        self.tsconfigText = tsconfigText
        let dir = scratchParent.appendingPathComponent("lsp_scratch_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try tsconfigText.write(to: dir.appendingPathComponent("tsconfig.json"),
                                atomically: true, encoding: .utf8)
        self.scratchDir = dir
        self.scratchOwner = true
        try startServer()
    }

    deinit {
        // NOT relied on for reaping (Swift gives no guarantee this runs at process exit) —
        // `ProcessSupervisor`'s own signal/atexit backstop covers that. This only cleans up
        // the ordinary in-process lifecycle.
        supervisor?.shutdown()
    }

    // MARK: - path/uri helpers

    private func absPath(_ path: String) -> URL { scratchDir.appendingPathComponent(path) }
    private func uri(for path: String) -> String { absPath(path).absoluteString }

    // MARK: - server lifecycle

    private func onNotification(_ msg: [String: Any]) {
        guard (msg["method"] as? String) == "textDocument/publishDiagnostics" else { return }
        guard let params = msg["params"] as? [String: Any], let uri = params["uri"] as? String else { return }
        let payload = params["diagnostics"] as? [[String: Any]] ?? []
        let version = (params["version"] as? NSNumber)?.intValue
        var ev: ManualResetEvent?
        diagLock.lock()
        diagSeqCounter += 1
        diagState[uri] = (seq: diagSeqCounter, version: version, payload: payload)
        ev = diagEvents[uri]
        diagLock.unlock()
        ev?.set()
    }

    private func startServer() throws {
        let sup = try ProcessSupervisor.spawn(
            argv: argv + ["--stdio"], cwd: scratchDir.path,
            onNotification: { [weak self] msg in self?.onNotification(msg) })
        supervisor = sup
        let params: [String: Any] = [
            "processId": Int(getpid()),
            "rootUri": scratchDir.absoluteString,
            "capabilities": Self.clientCapabilities,
            "initializationOptions": [String: Any](),
        ]
        _ = try sup.endpoint.request("initialize", params: params, timeout: timeoutS)
        sup.endpoint.notify("initialized", params: [String: Any]())
    }

    private func ensureAlive() {
        if let sup = supervisor, sup.isAlive { return }
        restart()
    }

    /// Restart loses the warm program; files are already on disk (both `openProject` and
    /// `update` write through), so re-`didOpen` every previously-open path at its CURRENT
    /// version — version counters stay monotonic across restarts (LSP requires
    /// per-document monotonicity).
    private func restart() {
        nRestarts += 1
        let openPaths = Array(open)
        supervisor?.shutdown()
        supervisor = nil
        diagLock.lock()
        diagState.removeAll(); diagEvents.removeAll(); updateMarker.removeAll()
        diagLock.unlock()
        open.removeAll()
        try? startServer()
        for p in openPaths { ensureOpen(p) }
    }

    // MARK: - project / document lifecycle

    /// Materialize `files` on disk, then `didOpen` exactly one (the first key, or
    /// `warmPath`) to force the server to load the configured project, blocking on its first
    /// `publishDiagnostics` (bounded by `timeoutS`, counted as a timeout, never a hang).
    public func openProject(_ files: [String: String], warmPath: String? = nil) throws {
        guard !projectOpened else { throw TsLspClientError.alreadyOpened }
        guard !files.isEmpty else { throw TsLspClientError.emptyProject }
        ensureAlive()

        for (rel, text) in files {
            let abs = absPath(rel)
            try FileManager.default.createDirectory(at: abs.deletingLastPathComponent(),
                                                      withIntermediateDirectories: true)
            try text.write(to: abs, atomically: true, encoding: .utf8)
            self.files[rel] = text
            versions[rel] = 1
        }
        projectOpened = true

        let warm = warmPath ?? files.keys.sorted().first!
        let u = uri(for: warm)
        let t0 = Date()
        let ev = ManualResetEvent()
        diagLock.lock(); diagEvents[u] = ev; diagLock.unlock()
        ensureOpen(warm)
        let got = ev.wait(timeout: timeoutS)
        diagLock.lock(); diagEvents.removeValue(forKey: u); diagLock.unlock()
        coldLoadS = Date().timeIntervalSince(t0)
        if !got { nTimeouts += 1 }
    }

    private func notifyRetrying(_ method: String, params: [String: Any]) {
        guard let sup = supervisor else { return }
        sup.endpoint.notify(method, params: params)
    }

    private func ensureOpen(_ path: String) {
        guard !open.contains(path) else { return }
        let u = uri(for: path)
        let text = files[path] ?? ""
        diagLock.lock()
        if updateMarker[u] == nil { updateMarker[u] = diagSeqCounter }
        diagLock.unlock()
        notifyRetrying("textDocument/didOpen", params: [
            "textDocument": ["uri": u, "languageId": "typescript",
                              "version": versions[path] ?? 1, "text": text],
        ])
        open.insert(path)
    }

    /// `ensureOpen`, write `text` through to disk, bump the version, send
    /// `textDocument/didChange` (full-text). Returns the new version.
    @discardableResult
    public func update(_ path: String, text: String) throws -> Int {
        ensureAlive()
        ensureOpen(path)
        let u = uri(for: path)
        try text.write(to: absPath(path), atomically: true, encoding: .utf8)
        files[path] = text
        versions[path, default: 1] += 1
        let version = versions[path]!
        diagLock.lock()
        updateMarker[u] = diagSeqCounter
        // A push already recorded before this edit is not an answer to it (#278) — see the
        // Python reference's comment on the exact same pop.
        diagState.removeValue(forKey: u)
        diagLock.unlock()
        notifyRetrying("textDocument/didChange", params: [
            "textDocument": ["uri": u, "version": version],
            "contentChanges": [["text": text]],
        ])
        return version
    }

    // MARK: - diagnostics — version-aware first-push-wins

    private func pushIsCurrent(_ state: (seq: Int, version: Int?, payload: [[String: Any]]),
                                wantVersion: Int, marker: Int) -> Bool {
        if let v = state.version { return v == wantVersion }
        return state.seq > marker
    }

    /// Every `severity == 1` diagnostic for `path`'s CURRENT version. First-push-wins, no
    /// settle window (see the file header). `[]` covers two distinguishable-only-by-counter
    /// situations: a genuinely clean re-check, and "nothing published since the edit"
    /// (`nNoPublish`) — never raises on timeout or a dead server.
    public func diagnostics(_ path: String) -> [Diagnostic] {
        ensureAlive()
        ensureOpen(path)
        let u = uri(for: path)
        let wantVersion = versions[path] ?? 1
        let t0 = Date()

        diagLock.lock()
        if let state = diagState[u], pushIsCurrent(state, wantVersion: wantVersion,
                                                     marker: updateMarker[u] ?? 0) {
            let payload = state.payload
            diagLock.unlock()
            recordOp("diagnostics", Date().timeIntervalSince(t0))
            return filterDiagnostics(payload, source: files[path] ?? "")
        }
        let ev = ManualResetEvent()
        diagEvents[u] = ev
        diagLock.unlock()

        _ = ev.wait(timeout: timeoutS)

        diagLock.lock()
        diagEvents.removeValue(forKey: u)
        let state = diagState[u]
        let marker = updateMarker[u] ?? 0
        diagLock.unlock()

        recordOp("diagnostics", Date().timeIntervalSince(t0))

        guard let state else {
            nNoPublish += 1
            return []
        }
        guard pushIsCurrent(state, wantVersion: wantVersion, marker: marker) else {
            nTimeouts += 1
            return []
        }
        return filterDiagnostics(state.payload, source: files[path] ?? "")
    }

    private func filterDiagnostics(_ payload: [[String: Any]], source: String) -> [Diagnostic] {
        let chars = Array(source)
        var out: [Diagnostic] = []
        for d in payload {
            let severity = (d["severity"] as? NSNumber)?.intValue ?? 1
            guard severity == 1 else { continue }
            guard let range = d["range"] as? [String: Any],
                  let start = range["start"] as? [String: Any],
                  let line0 = (start["line"] as? NSNumber)?.intValue,
                  let char0 = (start["character"] as? NSNumber)?.intValue,
                  let code = d["code"] else { continue }
            let line = line0 + 1, col = char0 + 1
            let codeStr = (code as? String) ?? "\((code as? NSNumber)?.intValue ?? 0)"
            let message = d["message"] as? String ?? ""
            out.append(Diagnostic(code: "TS\(codeStr)", line: line, col: col, message: message,
                                   offset: LspText.lineColToOffset(chars, line: line, col: col),
                                   severity: severity))
        }
        return out
    }

    // MARK: - the other four ops

    private func recordOp(_ opName: String, _ elapsed: Double) {
        wallS += elapsed
        nCalls += 1
        opCounts[opName, default: 0] += 1
        opWallS[opName, default: 0] += elapsed
    }

    /// `restart()` swallows a failed `startServer()` (`try?`) so `ensureAlive()` never
    /// throws; that leaves `supervisor` nil, and a request helper force-unwrapping it would
    /// crash instead of taking the `.closed` timeout path `timedRequest` already handles.
    /// Throwing `JsonRpcError.closed` here routes that same nil straight into that path.
    private func requireSupervisor() throws -> ProcessSupervisor {
        guard let sup = supervisor else { throw JsonRpcError.closed }
        return sup
    }

    /// `ensureAlive`, run `fn`, record cost. `.timeout`/`.closed` are counted (`nTimeouts`)
    /// and return `empty` — never propagate. `.rpcError` (the server rejected our params) IS
    /// re-thrown: hiding a real protocol bug as "no completions" would be exactly the BLIND
    /// failure mode this repo guards hardest.
    private func timedRequest<T>(_ opName: String, empty: T, _ fn: () throws -> T) throws -> T {
        ensureAlive()
        let t0 = Date()
        defer { recordOp(opName, Date().timeIntervalSince(t0)) }
        do {
            return try fn()
        } catch let e as JsonRpcError {
            switch e {
            case .timeout, .closed:
                nTimeouts += 1
                return empty
            case .rpcError, .malformedResponse:
                throw e
            }
        }
    }

    public func definition(_ path: String, offset: Int) throws -> [Location] {
        try timedRequest("definition", empty: []) {
            ensureOpen(path)
            let u = uri(for: path)
            let pos = try LspText.offsetToPosition(Array(files[path] ?? ""), offset: offset)
            let raw = try requireSupervisor().endpoint.request(
                "textDocument/definition",
                params: ["textDocument": ["uri": u], "position": pos], timeout: timeoutS)
            return Self.mapLocations(raw, root: scratchDir)
        }
    }

    public func references(_ path: String, offset: Int, includeDeclaration: Bool = false) throws -> [Location] {
        try timedRequest("references", empty: []) {
            ensureOpen(path)
            let u = uri(for: path)
            let pos = try LspText.offsetToPosition(Array(files[path] ?? ""), offset: offset)
            let raw = try requireSupervisor().endpoint.request(
                "textDocument/references",
                params: ["textDocument": ["uri": u], "position": pos,
                         "context": ["includeDeclaration": includeDeclaration]], timeout: timeoutS)
            return Self.mapLocations(raw, root: scratchDir)
        }
    }

    public func completions(_ path: String, offset: Int) throws -> [CompletionItem] {
        try timedRequest("completions", empty: []) {
            ensureOpen(path)
            let u = uri(for: path)
            let pos = try LspText.offsetToPosition(Array(files[path] ?? ""), offset: offset)
            let raw = try requireSupervisor().endpoint.request(
                "textDocument/completion",
                params: ["textDocument": ["uri": u], "position": pos], timeout: timeoutS)
            let dict = raw as? [String: Any]
            lastCompletionIncomplete = (dict?["isIncomplete"] as? Bool) ?? false
            let items = (dict?["items"] as? [[String: Any]]) ?? (raw as? [[String: Any]]) ?? []
            return items.map {
                CompletionItem(label: $0["label"] as? String ?? "",
                                kind: ($0["kind"] as? NSNumber)?.intValue,
                                detail: $0["detail"] as? String,
                                insertText: $0["insertText"] as? String,
                                sortText: $0["sortText"] as? String)
            }
        }
    }

    public func quickinfo(_ path: String, offset: Int) throws -> String? {
        try timedRequest("quickinfo", empty: nil) {
            ensureOpen(path)
            let u = uri(for: path)
            let pos = try LspText.offsetToPosition(Array(files[path] ?? ""), offset: offset)
            let raw = try requireSupervisor().endpoint.request(
                "textDocument/hover",
                params: ["textDocument": ["uri": u], "position": pos], timeout: timeoutS)
            guard let dict = raw as? [String: Any], let contents = dict["contents"] else { return nil }
            if let s = contents as? String { return s }
            if let d = contents as? [String: Any] { return d["value"] as? String }
            return nil
        }
    }

    public func documentSymbols(_ path: String) throws -> [Symbol] {
        try timedRequest("document_symbols", empty: []) {
            ensureOpen(path)
            let u = uri(for: path)
            let raw = try requireSupervisor().endpoint.request(
                "textDocument/documentSymbol", params: ["textDocument": ["uri": u]], timeout: timeoutS)
            return Self.mapSymbols(raw)
        }
    }

    // MARK: - result mapping

    private static func mapLocations(_ raw: Any?, root: URL) -> [Location] {
        var items: [[String: Any]] = []
        if let arr = raw as? [[String: Any]] { items = arr }
        else if let one = raw as? [String: Any] { items = [one] }
        var out: [Location] = []
        for item in items {
            let uriStr: String
            let range: [String: Any]
            if let target = item["targetUri"] as? String {
                uriStr = target
                range = (item["targetSelectionRange"] as? [String: Any])
                    ?? (item["targetRange"] as? [String: Any]) ?? [:]
            } else if let u = item["uri"] as? String {
                uriStr = u
                range = item["range"] as? [String: Any] ?? [:]
            } else { continue }
            guard let start = range["start"] as? [String: Any],
                  let end = range["end"] as? [String: Any] else { continue }
            let sLine = (start["line"] as? NSNumber)?.intValue ?? 0
            let sChar = (start["character"] as? NSNumber)?.intValue ?? 0
            let eLine = (end["line"] as? NSNumber)?.intValue ?? 0
            let eChar = (end["character"] as? NSNumber)?.intValue ?? 0
            let path = relativePath(uriStr, root: root)
            out.append(Location(uri: uriStr, path: path, line: sLine + 1, col: sChar + 1,
                                 endLine: eLine + 1, endCol: eChar + 1))
        }
        return out
    }

    private static func relativePath(_ uriStr: String, root: URL) -> String {
        guard let u = URL(string: uriStr) else { return uriStr }
        let path = u.path
        let rootPath = root.path
        if path.hasPrefix(rootPath) {
            var rel = String(path.dropFirst(rootPath.count))
            if rel.hasPrefix("/") { rel.removeFirst() }
            return rel
        }
        return path
    }

    private static func mapSymbols(_ raw: Any?) -> [Symbol] {
        guard let arr = raw as? [[String: Any]] else { return [] }
        var out: [Symbol] = []
        func flatten(_ items: [[String: Any]], container: String?) {
            for it in items {
                let name = it["name"] as? String ?? ""
                let kind = (it["kind"] as? NSNumber)?.intValue ?? 0
                if let sel = it["selectionRange"] as? [String: Any],
                   let start = sel["start"] as? [String: Any] {
                    let line = (start["line"] as? NSNumber)?.intValue ?? 0
                    let col = (start["character"] as? NSNumber)?.intValue ?? 0
                    out.append(Symbol(name: name, kind: kind, line: line + 1, col: col + 1,
                                       container: container))
                    if let children = it["children"] as? [[String: Any]], !children.isEmpty {
                        flatten(children, container: name)
                    }
                } else if let loc = it["location"] as? [String: Any],
                          let range = loc["range"] as? [String: Any],
                          let start = range["start"] as? [String: Any] {
                    let line = (start["line"] as? NSNumber)?.intValue ?? 0
                    let col = (start["character"] as? NSNumber)?.intValue ?? 0
                    out.append(Symbol(name: name, kind: kind, line: line + 1, col: col + 1,
                                       container: it["containerName"] as? String))
                }
            }
        }
        flatten(arr, container: nil)
        return out
    }

    // MARK: - misc

    public var callsPerS: Double { wallS > 0 ? Double(nCalls) / wallS : 0 }

    public func close() {
        supervisor?.shutdown()
        supervisor = nil
        if scratchOwner {
            try? FileManager.default.removeItem(at: scratchDir)
        }
    }
}

/// A simple manual-reset event over `NSCondition` — the Swift analogue of
/// `threading.Event`, used for the diagnostics wait exactly like the Python reference.
final class ManualResetEvent {
    private let cond = NSCondition()
    private var isSet = false

    func set() {
        cond.lock()
        isSet = true
        cond.signal()
        cond.unlock()
    }

    /// Returns whether it was set (mirrors `Event.wait`'s bool return — `true` even if it
    /// was already set before `wait` was called).
    @discardableResult
    func wait(timeout: TimeInterval) -> Bool {
        cond.lock()
        defer { cond.unlock() }
        let deadline = Date().addingTimeInterval(timeout)
        while !isSet {
            if !cond.wait(until: deadline) { return isSet }
        }
        return true
    }
}
