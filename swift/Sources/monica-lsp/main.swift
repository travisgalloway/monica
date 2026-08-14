// monica-lsp — dependency-free-runner CLI for MonicaLSP (#197, M13 #163). Mirrors
// monica-selfcheck/monica-parity/monica-bench's style: hand-rolled arg parsing, a
// `failures` array, `exit(1)` on failure. `swift test` is a no-op in this package (no
// .testTarget, matching MonicaTokenizer) — this binary IS the test runner.
//
//   --self-test            binary-free: framing/demux/trie/scanner. Runs on macOS AND Linux.
//   --emit-mask-parity OUT deterministic allowed-id sets for a fixed synthetic vocab/label
//                          pair, compared against a Python snippet run at CI time (V5).
//   --probe-reap           drives every ProcessSupervisor exit path against a real
//                          typescript-language-server and reports pid + liveness verdicts.
//   --bench                the SLA harness, comparable to scripts/bench_ts_lsp.py.
//
// --probe-reap/--bench need node + eval_sets/ts_error_injection/node_modules (macOS CI +
// local dev only); absent that toolchain they print SKIP and exit 0, mirroring
// scripts/bench_ts_lsp.py's own "no toolchain resolvable -> clean exit" contract.

import Foundation
import MonicaLSP
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    exit(1)
}

func parseFlags(_ args: [String]) -> [String: String] {
    var out: [String: String] = [:]
    var i = 0
    while i < args.count {
        let a = args[i]
        guard a.hasPrefix("--") else { i += 1; continue }
        let key = String(a.dropFirst(2))
        if i + 1 < args.count && !args[i + 1].hasPrefix("--") {
            out[key] = args[i + 1]; i += 2
        } else {
            out[key] = ""; i += 1
        }
    }
    return out
}

func intFlag(_ flags: [String: String], _ name: String, default def: Int) -> Int {
    guard let raw = flags[name], !raw.isEmpty, let v = Int(raw) else { return def }
    return v
}

// MARK: - repo-root / eval-set resolution
//
// Every invocation in CI runs with `working-directory: swift`, so `..` is the repo root.
// `--eval-set-dir` overrides for a different CWD (e.g. running the binary directly).

func evalSetDir(_ flags: [String: String]) -> URL {
    if let override = flags["eval-set-dir"] {
        return URL(fileURLWithPath: override)
    }
    return URL(fileURLWithPath: "..").appendingPathComponent("eval_sets/ts_error_injection")
}

func defaultTsconfigText(_ setDir: URL) -> String {
    (try? String(contentsOf: setDir.appendingPathComponent("tsconfig.json"), encoding: .utf8))
        ?? "{\"compilerOptions\":{\"strict\":true,\"target\":\"es2020\",\"module\":\"commonjs\"}}"
}

// =============================================================================================
// MARK: - --self-test (binary-free; no subprocess, no node)
// =============================================================================================

func runSelfTest() -> Never {
    var failures: [String] = []
    func check(_ cond: Bool, _ msg: String) { if !cond { failures.append(msg) } }
    func eq<T: Equatable>(_ a: T, _ b: T, _ msg: String) {
        if a != b { failures.append("\(msg): \(a) != \(b)") }
    }

    // --- framing round trip (no transport at all) ---
    do {
        let msg: [String: Any] = ["jsonrpc": "2.0", "id": 7, "method": "initialize",
                                   "params": ["rootUri": "file:///tmp/x"]]
        let data = try! JsonRpcEndpoint.encodeMessage(msg)
        let (a, b) = PipeTransport.pipePair()
        check(b.write(data), "framing: write succeeds")
        let decoded = JsonRpcEndpoint.readMessage(a)
        check(decoded != nil, "framing: message round-trips")
        eq((decoded?["id"] as? NSNumber)?.intValue, 7, "framing: id round-trips")
        eq(decoded?["method"] as? String, "initialize", "framing: method round-trips")
        a.closeWrite(); a.closeRead(); b.closeWrite()
    }

    // --- multi-message framing with split reads over a real pipe ---
    do {
        let (a, b) = PipeTransport.pipePair()
        let m1: [String: Any] = ["jsonrpc": "2.0", "id": 1, "method": "a"]
        let m2: [String: Any] = ["jsonrpc": "2.0", "id": 2, "method": "b"]
        _ = b.write(try! JsonRpcEndpoint.encodeMessage(m1))
        _ = b.write(try! JsonRpcEndpoint.encodeMessage(m2))
        let d1 = JsonRpcEndpoint.readMessage(a)
        let d2 = JsonRpcEndpoint.readMessage(a)
        eq((d1?["id"] as? NSNumber)?.intValue, 1, "multi-message: first message intact")
        eq((d2?["id"] as? NSNumber)?.intValue, 2, "multi-message: second message intact")
        a.closeWrite(); a.closeRead(); b.closeWrite()
    }

    // --- demux: response arriving out of order (id 2 replies before id 1) ---
    do {
        let (client, fakeServer) = PipeTransport.pipePair()
        let endpoint = JsonRpcEndpoint(stream: client)
        endpoint.start()

        let r1Done = DispatchSemaphore(value: 0)
        let r2Done = DispatchSemaphore(value: 0)
        var r1Result: Any?, r2Result: Any?
        DispatchQueue.global().async {
            r1Result = try? endpoint.request("methodOne", timeout: 5.0)
            r1Done.signal()
        }
        DispatchQueue.global().async {
            r2Result = try? endpoint.request("methodTwo", timeout: 5.0)
            r2Done.signal()
        }
        // Drain both requests off the fake-server side, then reply id=2 BEFORE id=1.
        let req1 = JsonRpcEndpoint.readMessage(fakeServer)
        let req2 = JsonRpcEndpoint.readMessage(fakeServer)
        let id1 = (req1?["id"] as? NSNumber)?.intValue ?? -1
        let id2 = (req2?["id"] as? NSNumber)?.intValue ?? -1
        _ = fakeServer.write(try! JsonRpcEndpoint.encodeMessage(
            ["jsonrpc": "2.0", "id": id2, "result": "second"]))
        _ = fakeServer.write(try! JsonRpcEndpoint.encodeMessage(
            ["jsonrpc": "2.0", "id": id1, "result": "first"]))
        _ = r1Done.wait(timeout: .now() + 5)
        _ = r2Done.wait(timeout: .now() + 5)
        eq(r1Result as? String, "first", "demux: out-of-order response resolves the right waiter (1)")
        eq(r2Result as? String, "second", "demux: out-of-order response resolves the right waiter (2)")
        endpoint.close(); fakeServer.closeWrite(); fakeServer.closeRead()
    }

    // --- demux: notification (no id) reaches onNotification, is not treated as a response ---
    do {
        let (client, fakeServer) = PipeTransport.pipePair()
        var seen: [String: Any]?
        let gotIt = DispatchSemaphore(value: 0)
        let endpoint = JsonRpcEndpoint(stream: client, onNotification: { msg in
            seen = msg
            gotIt.signal()
        })
        endpoint.start()
        _ = fakeServer.write(try! JsonRpcEndpoint.encodeMessage(
            ["jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
             "params": ["uri": "file:///x.ts", "diagnostics": []]]))
        _ = gotIt.wait(timeout: .now() + 5)
        eq(seen?["method"] as? String, "textDocument/publishDiagnostics",
           "demux: notification dispatched to onNotification")
        endpoint.close(); fakeServer.closeWrite(); fakeServer.closeRead()
    }

    // --- demux: server -> client REQUEST gets an unconditional {"result": null} reply ---
    do {
        let (client, fakeServer) = PipeTransport.pipePair()
        var observed = false
        let endpoint = JsonRpcEndpoint(stream: client, onNotification: { msg in
            if msg["method"] as? String == "window/workDoneProgress/create" { observed = true }
        })
        endpoint.start()
        _ = fakeServer.write(try! JsonRpcEndpoint.encodeMessage(
            ["jsonrpc": "2.0", "id": 99, "method": "window/workDoneProgress/create",
             "params": ["token": "t1"]]))
        let reply = JsonRpcEndpoint.readMessage(fakeServer)
        check(observed, "server->client request: observed via onNotification")
        eq((reply?["id"] as? NSNumber)?.intValue, 99, "server->client request: reply carries the same id")
        check(reply?["result"] is NSNull, "server->client request: auto-reply result is null")
        endpoint.close(); fakeServer.closeWrite(); fakeServer.closeRead()
    }

    // --- EOF fails every pending waiter rather than hanging ---
    do {
        let (client, fakeServer) = PipeTransport.pipePair()
        let endpoint = JsonRpcEndpoint(stream: client)
        endpoint.start()
        let threw = DispatchSemaphore(value: 0)
        var caught: Error?
        DispatchQueue.global().async {
            do { _ = try endpoint.request("neverAnswered", timeout: 10.0) }
            catch { caught = error }
            threw.signal()
        }
        // Give the request time to register, then close the SERVER's write side so the
        // client's reader thread sees EOF.
        Thread.sleep(forTimeInterval: 0.1)
        fakeServer.closeWrite()
        let waited = threw.wait(timeout: .now() + 5)
        check(waited == .success, "EOF: pending request fails promptly rather than hanging the full timeout")
        check(caught is JsonRpcError, "EOF: pending request fails with a JsonRpcError")
        fakeServer.closeRead()
    }

    // --- request timeout throws rather than hanging past the deadline ---
    do {
        let (client, fakeServer) = PipeTransport.pipePair()
        let endpoint = JsonRpcEndpoint(stream: client)
        endpoint.start()
        let t0 = Date()
        var caught: Error?
        do { _ = try endpoint.request("neverAnswered", timeout: 0.2) }
        catch { caught = error }
        let elapsed = Date().timeIntervalSince(t0)
        check(caught is JsonRpcError, "timeout: request throws JsonRpcError")
        check(elapsed < 2.0, "timeout: returns promptly (\(elapsed)s), does not hang")
        endpoint.close(); fakeServer.closeWrite(); fakeServer.closeRead()
    }

    // --- VocabTrie / allowedExtensions ---
    do {
        let vocab: VocabTable = ["x", "y", "length", "len", "to", "toString", nil, "."]
        let trie = VocabTrie(vocab: vocab)
        eq(Set(trie.prefixMatches("length")), Set([2, 3]), "trie: prefix matches both 'length' and 'len'")
        let allowed = allowedExtensions(trie: trie, labels: ["length", "toString"], prefix: "", exitIds: [7])
        check(allowed.contains(2), "allowedExtensions: 'length' reachable from empty prefix")
        check(allowed.contains(5), "allowedExtensions: 'toString' (id 5) reachable from empty prefix")
        let completeLabel = allowedExtensions(trie: trie, labels: ["x"], prefix: "x", exitIds: [7])
        eq(completeLabel, [7], "allowedExtensions: complete-label prefix unions in exitIds only")
    }

    // --- SourceScan: identifier predicates + string/comment masking ---
    do {
        check(SourceScan.isIdentChar("a"), "isIdentChar: letter")
        check(SourceScan.isIdentChar("_"), "isIdentChar: underscore")
        check(!SourceScan.isIdentChar("."), "isIdentChar: dot is not an identifier char")
        check(SourceScan.isIdentStart("_"), "isIdentStart: underscore")
        check(!SourceScan.isIdentStart("1"), "isIdentStart: digit cannot start an identifier")

        let masked = SourceScan.maskStringsAndComments("const s = \"a.b.c\"; x.y;")
        check(!masked.contains("a.b.c"), "maskStringsAndComments: string contents blanked")
        check(masked.contains("x.y"), "maskStringsAndComments: real code left intact")
        eq(masked.count, "const s = \"a.b.c\"; x.y;".count,
           "maskStringsAndComments: preserves length")

        let commentMasked = SourceScan.maskStringsAndComments("x.y; // a.b.c\nz.w;")
        check(!commentMasked.contains("a.b.c"), "maskStringsAndComments: line comment blanked")
        check(commentMasked.contains("z.w"), "maskStringsAndComments: code after comment intact")
    }

    // --- CompletionMasker: one query per span, not per token; never opens inside a string ---
    do {
        final class RecordingLabels: LabelSource {
            var calls: [(text: String, anchorOffset: Int)] = []
            func query(path: String, text: String, anchorOffset: Int) -> [String]? {
                calls.append((text, anchorOffset))
                return ["length", "toString"]
            }
        }
        let src = RecordingLabels()
        let decode: ([Int]) -> String = { ids in ids.map { String(UnicodeScalar($0)!) }.joined() }
        let masker = CompletionMasker(labelSource: src, path: "a.ts", decode: decode)

        _ = masker.maskFor("const x = obj")
        _ = masker.maskFor("const x = obj.")
        eq(src.calls.count, 1, "CompletionMasker: exactly one query fires when the span opens")
        let allowed = masker.maskFor("const x = obj.le", vocabSize: 128)
        check(allowed != nil, "CompletionMasker: mid-span prefix 'le' still constrains toward 'length'")
        _ = masker.maskFor("const x = obj.length ")
        eq(src.calls.count, 1, "CompletionMasker: closing the span issues no extra query")

        let strMasker = CompletionMasker(labelSource: src, path: "b.ts", decode: decode)
        _ = strMasker.maskFor("const s = \"a.")
        eq(src.calls.count, 1, "CompletionMasker: a '.' inside a string never opens a span")
    }

    if failures.isEmpty {
        print("monica-lsp --self-test: OK — all checks passed")
        exit(0)
    } else {
        for f in failures { FileHandle.standardError.write(Data("FAIL: \(f)\n".utf8)) }
        FileHandle.standardError.write(Data("monica-lsp --self-test: \(failures.count) failure(s)\n".utf8))
        exit(1)
    }
}

// =============================================================================================
// MARK: - --emit-mask-parity OUT
//
// A FIXED synthetic vocab/label pair, hardcoded IDENTICALLY here and in the Python one-liner
// `.github/workflows/ci.yml` runs against `src.serve.constrained` (V5). Deliberately no
// tokenizer/decode involved — this exercises VocabTrie + allowedExtensions directly, pure
// integer/string logic, so the two languages' outputs are bit-stable by construction. Output
// is a hand-assembled, whitespace-free JSON string (not JSONSerialization's default
// formatting) so a `cmp` against the Python side's `json.dumps(..., separators=(",", ":"))`
// output is comparing semantics, not incidental library pretty-printing.
// =============================================================================================

let maskParityVocab: [String] = [
    "x", "y", "length", "len", "toString", "to", "String", ".", " ",
    "toLowerCase", "toUpperCase", "foo", "fooBar", "_bar", "1", "2", "a", "ab", "abc", "!",
]
let maskParityLabels = ["length", "toString", "toLowerCase", "toUpperCase", "x", "y", "abc"]
let maskParityExitIds = [7, 8, 19]  // ".", " ", "!" — the non-identifier-leading pieces above
let maskParityPrefixes = ["", "to", "len", "x", "abc", "ab", "zzz"]

func runEmitMaskParity(out: String) {
    let vocab: VocabTable = maskParityVocab
    let trie = VocabTrie(vocab: vocab)
    var caseStrings: [String] = []
    for prefix in maskParityPrefixes {
        let allowed = allowedExtensions(trie: trie, labels: maskParityLabels, prefix: prefix,
                                         exitIds: maskParityExitIds).sorted()
        let idsStr = allowed.map(String.init).joined(separator: ",")
        caseStrings.append("{\"prefix\":\"\(prefix)\",\"allowed\":[\(idsStr)]}")
    }
    let json = "{\"cases\":[\(caseStrings.joined(separator: ","))]}\n"
    do {
        try json.write(toFile: out, atomically: true, encoding: .utf8)
        print("emit-mask-parity: wrote \(maskParityPrefixes.count) cases to \(out)")
    } catch {
        fail("emit-mask-parity: could not write \(out): \(error)")
    }
}

// =============================================================================================
// MARK: - --probe-reap
//
// Drives every ProcessSupervisor exit path against a REAL typescript-language-server and
// reports, per scenario, the spawned child's pid and a post-teardown liveness verdict. This
// is what makes the CI/local orphan check (V7) non-blind: it prints `spawned_pid=<n>` for
// every scenario, so a check that never sees that line reports BLIND rather than a vacuous
// PASS.
// =============================================================================================

func minimalWarmProject() -> [String: String] {
    ["src/a.ts": "export const x = 1;\nexport function f(v: number): number { return v + x; }\n"]
}

func runProbeReapScenarios(argv: [String], tsconfigText: String, scratchParent: URL) {
    var overallFail = false

    func scenario(_ name: String, _ body: () throws -> Int32) {
        print("--- probe-reap: \(name) ---")
        do {
            let pid = try body()
            print("spawned_pid=\(pid)")
            // Bounded settle window for the OS to actually reap the zombie/finish teardown.
            var alive = ProcessSupervisor.isPidAlive(pid)
            var waited = 0.0
            while alive && waited < 3.0 {
                Thread.sleep(forTimeInterval: 0.1)
                waited += 0.1
                alive = ProcessSupervisor.isPidAlive(pid)
            }
            if alive {
                print("FAIL: \(name): pid \(pid) still alive after teardown")
                overallFail = true
            } else {
                print("OK: \(name): pid \(pid) reaped")
            }
        } catch {
            print("FAIL: \(name): threw during setup: \(error)")
            overallFail = true
        }
    }

    // 1. Clean close: spawn, initialize, normal return through withServer's `defer`.
    scenario("clean-close") {
        try ProcessSupervisor.withServer(argv: argv + ["--stdio"], cwd: scratchParent.path) { sup in
            _ = try sup.endpoint.request("initialize", params: [
                "processId": Int(getpid()), "rootUri": scratchParent.absoluteString,
                "capabilities": [String: Any](), "initializationOptions": [String: Any](),
            ], timeout: 10.0)
            sup.endpoint.notify("initialized", params: [String: Any]())
            return sup.pid
        }
    }

    // 2. Thrown error mid-session: withServer's `defer` must still reap.
    scenario("thrown-error") {
        struct Probe: Error {}
        do {
            _ = try ProcessSupervisor.withServer(argv: argv + ["--stdio"], cwd: scratchParent.path) { sup -> Int32 in
                _ = try sup.endpoint.request("initialize", params: [
                    "processId": Int(getpid()), "rootUri": scratchParent.absoluteString,
                    "capabilities": [String: Any](), "initializationOptions": [String: Any](),
                ], timeout: 10.0)
                throw Probe()
            }
            fail("thrown-error scenario: body should have thrown")
        } catch is Probe {
            // Expected — but we need the pid, which withServer's throw path doesn't return.
            // Spawn a second time just to report the pid the FIRST spawn's supervisor used
            // is unavailable here, so re-derive by re-running with a captured pid instead.
        }
        // Re-run capturing pid via a local var, since `withServer`'s throw discards the
        // return value.
        var capturedPid: Int32 = -1
        do {
            _ = try ProcessSupervisor.withServer(argv: argv + ["--stdio"], cwd: scratchParent.path) { sup -> Int32 in
                capturedPid = sup.pid
                _ = try sup.endpoint.request("initialize", params: [
                    "processId": Int(getpid()), "rootUri": scratchParent.absoluteString,
                    "capabilities": [String: Any](), "initializationOptions": [String: Any](),
                ], timeout: 10.0)
                throw Probe()
            }
        } catch is Probe {
            return capturedPid
        }
        throw Probe()
    }

    // 3. Request timeout: the endpoint throws JsonRpcError.timeout; caller reaps explicitly.
    scenario("request-timeout") {
        let sup = try ProcessSupervisor.spawn(argv: argv + ["--stdio"], cwd: scratchParent.path)
        let pid = sup.pid
        _ = try? sup.endpoint.request("initialize", params: [
            "processId": Int(getpid()), "rootUri": scratchParent.absoluteString,
            "capabilities": [String: Any](), "initializationOptions": [String: Any](),
        ], timeout: 10.0)
        sup.endpoint.notify("initialized", params: [String: Any]())
        // A request the server will never answer (unknown method, no id collision) with a
        // near-zero timeout.
        do {
            _ = try sup.endpoint.request("workspace/thisMethodDoesNotExist", timeout: 0.05)
        } catch {
            // expected — JsonRpcError.timeout OR .rpcError, either way we now reap.
        }
        sup.shutdown()
        return pid
    }

    if overallFail {
        FileHandle.standardError.write(Data("probe-reap: one or more in-process scenarios FAILED\n".utf8))
        exit(1)
    }
}

/// `--probe-reap-signal-child <sigint|sigkill>` — an internal sub-mode `--probe-reap`
/// launches AS A SEPARATE PROCESS (via `Process`, re-execing this same binary) so a real
/// SIGINT/SIGKILL can be delivered to a live `monica-lsp` process from the outside, the only
/// way to genuinely exercise the async-signal-safe handler / the LSP `processId` parent-death
/// backstop. Spawns tsserver, prints `spawned_pid=<n>`, flushes, then blocks until signaled.
func runProbeReapSignalChild(argv: [String], scratchParent: URL) -> Never {
    guard let sup = try? ProcessSupervisor.spawn(argv: argv + ["--stdio"], cwd: scratchParent.path) else {
        FileHandle.standardError.write(Data("probe-reap-signal-child: spawn failed\n".utf8))
        exit(2)
    }
    _ = try? sup.endpoint.request("initialize", params: [
        "processId": Int(getpid()), "rootUri": scratchParent.absoluteString,
        "capabilities": [String: Any](), "initializationOptions": [String: Any](),
    ], timeout: 10.0)
    sup.endpoint.notify("initialized", params: [String: Any]())
    print("spawned_pid=\(sup.pid)")
    fflush(stdout)
    // Block until an external signal arrives (SIGINT via the installed handler, or SIGKILL
    // which no in-process code ever runs for). No sleep-loop upper bound here on purpose —
    // the PARENT enforces the timeout by killing this child if it doesn't exit promptly.
    while true { Thread.sleep(forTimeInterval: 1.0) }
}

func runProbeReapSignalScenarios(selfBinary: String, scratchParent: URL) {
    func runOne(_ mode: String, deliver: Int32, label: String) {
        print("--- probe-reap: \(label) (informational) ---")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: selfBinary)
        p.arguments = ["--probe-reap-signal-child"]
        let stdout = Pipe()
        p.standardOutput = stdout
        p.standardError = FileHandle.nullDevice
        do { try p.run() } catch {
            print("SKIP: \(label): could not spawn sub-child: \(error)")
            return
        }
        // Read spawned_pid= off the child's stdout with a bounded wait.
        var tsserverPid: Int32?
        let deadline = Date().addingTimeInterval(10.0)
        var buffer = Data()
        while tsserverPid == nil && Date() < deadline {
            let chunk = stdout.fileHandleForReading.availableData
            if chunk.isEmpty { Thread.sleep(forTimeInterval: 0.05); continue }
            buffer.append(chunk)
            let text = String(decoding: buffer, as: UTF8.self)
            if let range = text.range(of: "spawned_pid=") {
                let rest = text[range.upperBound...]
                let digits = rest.prefix { $0.isNumber }
                tsserverPid = Int32(digits)
            }
        }
        guard let tsPid = tsserverPid else {
            print("BLIND: \(label): child never printed spawned_pid= — nothing was verified")
            p.terminate()
            return
        }
        print("spawned_pid=\(tsPid)")
        kill(p.processIdentifier, deliver)
        let childDeadline = Date().addingTimeInterval(5.0)
        while p.isRunning && Date() < childDeadline { Thread.sleep(forTimeInterval: 0.05) }
        if p.isRunning { p.terminate() }
        Thread.sleep(forTimeInterval: 0.5)  // let the OS finish reaping / the server notice processId death
        if ProcessSupervisor.isPidAlive(tsPid) {
            print("\(mode == "sigint" ? "FAIL" : "INFO"): \(label): tsserver pid \(tsPid) still alive")
        } else {
            print("OK: \(label): tsserver pid \(tsPid) is gone")
        }
    }
    runOne("sigint", deliver: SIGINT, label: "SIGINT to monica-lsp (handler kills the child)")
    runOne("sigkill", deliver: SIGKILL, label: "SIGKILL to monica-lsp (processId parent-death backstop)")
}

// =============================================================================================
// MARK: - --bench
//
// Reuses the same deterministic synthetic project generator shape as scripts/bench_ts_lsp.py
// (seeded, same flag surface, same per-op median/mean/p95/calls-per-s JSON schema) so the two
// columns are directly comparable, and carries over its BLIND-guarded sanity probe.
// =============================================================================================

struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}

func modPath(_ i: Int) -> String { "src/mod_\(String(format: "%04d", i)).ts" }

func generateProject(nFiles: Int, seed: UInt64) -> [String: String] {
    var rng = SplitMix64(seed: seed)
    var files: [String: String] = [
        "src/hub.ts": """
        export interface Vec { x: number; y: number }

        export function scale(v: Vec, k: number): Vec {
          return { x: v.x * k, y: v.y * k };
        }

        export const ORIGIN: Vec = { x: 0, y: 0 };

        """,
    ]
    for i in 1..<max(nFiles, 2) {
        let lowerMods = Array(1..<i)
        let k = lowerMods.isEmpty ? 0 : Int.random(in: 1...min(3, lowerMods.count), using: &rng)
        let chosen = lowerMods.shuffled(using: &rng).prefix(k).sorted()

        var importLines = ["import { Vec, scale } from \"./hub\";"]
        var uses = ["scale(v, 2)"]
        for idx in chosen {
            let stem = URL(fileURLWithPath: modPath(idx)).deletingPathExtension().lastPathComponent
            importLines.append("import { f\(idx) } from \"./\(stem)\";")
            uses.append("f\(idx)(v)")
        }
        let bodyUses = uses.enumerated().map { "  const u\($0.offset) = \($0.element);" }.joined(separator: "\n")
        files[modPath(i)] = """
        \(importLines.joined(separator: "\n"))

        export function f\(i)(v: Vec): Vec {
        \(bodyUses)
          const vx = v.x;
          return scale(v, 1);
        }

        export interface T\(i) { v: Vec; tag: number }

        """
    }
    return files
}

func percentile(_ sorted: [Double], _ p: Double) -> Double {
    guard !sorted.isEmpty else { return 0 }
    let idx = min(sorted.count - 1, Int(Double(sorted.count) * p))
    return sorted[idx]
}

func summarize(_ samples: [Double]) -> [String: Any] {
    guard !samples.isEmpty else { return ["n": 0] }
    let sorted = samples.sorted()
    let sum = samples.reduce(0, +)
    let mean = sum / Double(samples.count)
    let median = percentile(sorted, 0.5)
    let p95 = percentile(sorted, 0.95)
    let callsPerS = sum > 0 ? Double(samples.count) / sum : 0
    return ["n": samples.count, "median_ms": median * 1000, "mean_ms": mean * 1000,
            "p95_ms": p95 * 1000, "min_ms": (sorted.first ?? 0) * 1000, "calls_per_s": callsPerS]
}

func runBench(flags: [String: String], argv: [String], tsconfigText: String, scratchParent: URL) {
    let nFiles = intFlag(flags, "n-files", default: 200)
    let iters = intFlag(flags, "iters", default: 50)
    let seed = UInt64(intFlag(flags, "seed", default: 1))
    let outPath = flags["out"]

    let files = generateProject(nFiles: nFiles, seed: seed)
    guard let client = try? TsLspClient(argv: argv, tsconfigText: tsconfigText, scratchParent: scratchParent) else {
        fail("bench: could not start typescript-language-server")
    }
    defer { client.close() }
    do {
        try client.openProject(files, warmPath: "src/hub.ts")
    } catch {
        fail("bench: openProject failed: \(error)")
    }

    // --- BLIND-guarded sanity probe: known-answer checks BEFORE any timing starts. ---
    let target = modPath(1)
    guard let modText = files[target] else { fail("bench: generated project missing \(target)") }
    var checks: [String: Bool] = [:]

    if let callRange = modText.range(of: "scale(") {
        let offset = modText.distance(from: modText.startIndex, to: callRange.lowerBound)
        let defs = (try? client.definition(target, offset: offset)) ?? []
        checks["definition_resolves_to_hub"] = defs.contains { $0.path == "src/hub.ts" }
    } else { checks["definition_resolves_to_hub"] = false }

    let dotOffset = (modText.range(of: "v.x").map {
        modText.distance(from: modText.startIndex, to: $0.lowerBound) + 2
    }) ?? -1
    let completions = dotOffset >= 0 ? ((try? client.completions(target, offset: dotOffset)) ?? []) : []
    checks["completions_contains_x"] = completions.contains { $0.label == "x" }

    let broken = modText.replacingOccurrences(of: "v.x", with: "v.gorblak", range: modText.range(of: "v.x"))
    _ = try? client.update(target, text: broken)
    let brokenDiags = client.diagnostics(target)
    checks["diagnostics_detects_break"] = brokenDiags.contains { $0.code == "TS2339" }
    _ = try? client.update(target, text: modText)
    let cleanDiags = client.diagnostics(target)
    checks["diagnostics_clean_after_revert"] = cleanDiags.isEmpty

    let failedChecks = checks.filter { !$0.value }.keys.sorted()
    if !failedChecks.isEmpty {
        print("verdict=BLIND failed_checks=\(failedChecks.joined(separator: ","))")
        if let outPath {
            let json = "{\"verdict\":\"BLIND\",\"failed_checks\":[\(failedChecks.map { "\"\($0)\"" }.joined(separator: ","))]}\n"
            try? json.write(toFile: outPath, atomically: true, encoding: .utf8)
        }
        exit(2)
    }

    // --- timed measurement ---
    var completionsSamples: [Double] = []
    var quickinfoSamples: [Double] = []
    for i in 0..<iters {
        let idx = 1 + (i % max(nFiles - 1, 1))
        let path = modPath(idx)
        guard let text = files[path] else { continue }
        let offset = min((text.range(of: "v.x").map { text.distance(from: text.startIndex, to: $0.lowerBound) + 2 }) ?? 0, text.count)
        let t0 = Date()
        _ = try? client.completions(path, offset: offset)
        completionsSamples.append(Date().timeIntervalSince(t0))
        let t1 = Date()
        _ = try? client.quickinfo(path, offset: offset)
        quickinfoSamples.append(Date().timeIntervalSince(t1))
    }

    let result: [String: Any] = [
        "verdict": "PASS",
        "n_files": nFiles, "iters": iters, "seed": seed,
        "completions": summarize(completionsSamples),
        "quickinfo": summarize(quickinfoSamples),
        "diagnostics_note": "not separately timed here — unchanged by construction, "
            + "client-side debounce, see #278/#279",
        "client_n_restarts": client.nRestarts,
        "client_n_timeouts": client.nTimeouts,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]) {
        print(String(decoding: data, as: UTF8.self))
        if let outPath { try? data.write(to: URL(fileURLWithPath: outPath)) }
    }
}

// =============================================================================================
// MARK: - dispatch
// =============================================================================================

let argvAll = Array(CommandLine.arguments.dropFirst())
let flags = parseFlags(argvAll)

if argvAll.contains("--self-test") {
    runSelfTest()
}

if argvAll.contains("--probe-reap-signal-child") {
    let setDir = evalSetDir(flags)
    guard let argv = resolveTsLsp(setDir: setDir) else {
        FileHandle.standardError.write(Data("probe-reap-signal-child: no toolchain\n".utf8))
        exit(2)
    }
    runProbeReapSignalChild(argv: argv, scratchParent: setDir)
}

if let outPath = flags["emit-mask-parity"], !outPath.isEmpty {
    runEmitMaskParity(out: outPath)
    exit(0)
}

if argvAll.contains("--probe-reap") {
    let setDir = evalSetDir(flags)
    guard let argv = resolveTsLsp(setDir: setDir) else {
        print("SKIP: --probe-reap needs a typescript-language-server toolchain in "
            + "\(setDir.path)/node_modules (run `npm ci` there) — nothing was measured")
        exit(0)
    }
    let tsconfigText = defaultTsconfigText(setDir)
    runProbeReapScenarios(argv: argv, tsconfigText: tsconfigText, scratchParent: setDir)
    runProbeReapSignalScenarios(selfBinary: CommandLine.arguments[0], scratchParent: setDir)
    exit(0)
}

if argvAll.contains("--bench") {
    let setDir = evalSetDir(flags)
    guard let argv = resolveTsLsp(setDir: setDir) else {
        print("SKIP: --bench needs a typescript-language-server toolchain in "
            + "\(setDir.path)/node_modules (run `npm ci` there) — nothing was measured")
        exit(0)
    }
    let tsconfigText = defaultTsconfigText(setDir)
    runBench(flags: flags, argv: argv, tsconfigText: tsconfigText, scratchParent: setDir)
    exit(0)
}

fail("provide --self-test, --emit-mask-parity <out.json>, --probe-reap, or --bench")
