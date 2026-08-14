// ProcessSupervisor.swift — the ONLY thing in this codebase allowed to own a child
// `Process`. `TsLspClient` cannot spawn or signal a server directly; it only ever holds a
// `ProcessSupervisor` and talks to its `endpoint`. This is the reaping contract
// (docs: .claude/plans/issue-197.md step 2), mirroring `src/lsp/ts_service.py:446`'s
// `_teardown_process`: `terminate` -> bounded wait -> `kill` -> bounded wait, "already dead"
// counted as success not error.
//
// `Process`/`Pipe` are in Foundation's portable subset (present under
// swift-corelibs-foundation on Linux), so this file builds and runs on both `swift-macos`
// and `swift-linux` — the CI job that actually exercises it end to end
// (`--probe-reap`/`--bench`) is macOS-only (needs the pinned node_modules toolchain), but
// nothing here is macOS-specific.

import Foundation
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

public enum ProcessSupervisorError: Error, CustomStringConvertible {
    case emptyArgv
    case spawnFailed(Underlying: Error)
    public var description: String {
        switch self {
        case .emptyArgv: return "ProcessSupervisor.spawn: argv must not be empty"
        case .spawnFailed(let e): return "ProcessSupervisor.spawn failed: \(e)"
        }
    }
}

public final class ProcessSupervisor {
    public struct Timeouts {
        public var lspShutdown: TimeInterval = 1.0
        public var terminateWait: TimeInterval = 5.0
        public var killWait: TimeInterval = 2.0
        public init() {}
    }

    public let endpoint: JsonRpcEndpoint
    public let pid: Int32

    private let process: Process
    private let stdinPipe: Pipe
    private let stderrPipe: Pipe
    private var timeouts = Timeouts()

    private let stateLock = NSLock()
    private var reaped = false

    private let stderrLock = NSLock()
    private var stderrTailLines: [String] = []
    private static let stderrTailCap = 200

    // MARK: - process-wide signal / atexit backstop (item 4 of the reaping contract)
    //
    // Async-signal-safe by construction: a `sig_atomic_t` global + a bare `kill(2)`, no
    // allocation, no lock acquisition inside the handler itself (only reading/writing a
    // `sig_atomic_t`, which is signal-safe by definition). This covers exactly ONE "current"
    // child at a time — the process-management model this package uses everywhere (one
    // `monica-lsp`/`monica-generate --lsp-mask` invocation talks to one tsserver at a time),
    // which is also why `--probe-reap`'s signal scenarios each run in their OWN child
    // `monica-lsp` invocation (see `main.swift`) rather than sharing one process.
    private static var currentPID: sig_atomic_t = 0
    private static let installLock = NSLock()
    private static var handlersInstalled = false

    private static func installHandlersOnce() {
        installLock.lock()
        defer { installLock.unlock() }
        guard !handlersInstalled else { return }
        handlersInstalled = true

        // `signal()`/`raise()` are NOT on POSIX's async-signal-safe list (only `kill(2)` and
        // `_exit(2)` are, of the operations relevant here) — re-arming the default disposition
        // and re-raising from inside the handler was itself a violation of the safety this
        // block's own doc comment claims. `_exit` after the `kill(2)` backstop keeps every
        // operation in the handler signal-safe while still terminating the process.
        signal(SIGINT) { sig in
            let p = ProcessSupervisor.currentPID
            if p > 0 { kill(pid_t(p), SIGKILL) }
            _exit(128 + sig)
        }
        signal(SIGTERM) { sig in
            let p = ProcessSupervisor.currentPID
            if p > 0 { kill(pid_t(p), SIGKILL) }
            _exit(128 + sig)
        }
        signal(SIGHUP) { sig in
            let p = ProcessSupervisor.currentPID
            if p > 0 { kill(pid_t(p), SIGKILL) }
            _exit(128 + sig)
        }
        atexit {
            let p = ProcessSupervisor.currentPID
            if p > 0 { kill(pid_t(p), SIGKILL) }
        }
    }

    private static func setCurrent(_ pid: Int32) {
        installHandlersOnce()
        currentPID = sig_atomic_t(pid)
    }

    private static func clearCurrent(_ pid: Int32) {
        if currentPID == sig_atomic_t(pid) { currentPID = 0 }
    }

    // MARK: - lifecycle

    private init(process: Process, stdinPipe: Pipe, stderrPipe: Pipe, endpoint: JsonRpcEndpoint, pid: Int32) {
        self.process = process
        self.stdinPipe = stdinPipe
        self.stderrPipe = stderrPipe
        self.endpoint = endpoint
        self.pid = pid
    }

    /// Spawn `argv` and wrap its stdio in a started `JsonRpcEndpoint`. Drains stderr in a
    /// background thread (`src/lsp/jsonrpc.py:261`'s trap: an undrained stderr pipe fills
    /// its OS buffer and deadlocks the child on a blocking write — a deadlocked child is
    /// also an un-reapable one).
    public static func spawn(argv: [String], cwd: String? = nil, env: [String: String]? = nil,
                              onNotification: JsonRpcEndpoint.OnNotification? = nil) throws -> ProcessSupervisor {
        guard !argv.isEmpty else { throw ProcessSupervisorError.emptyArgv }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: argv[0])
        process.arguments = Array(argv.dropFirst())
        if let cwd { process.currentDirectoryURL = URL(fileURLWithPath: cwd) }
        if let env { process.environment = env }

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        do {
            try process.run()
        } catch {
            throw ProcessSupervisorError.spawnFailed(Underlying: error)
        }

        let pid = process.processIdentifier
        let stream = PipeTransport(readHandle: stdoutPipe.fileHandleForReading,
                                    writeHandle: stdinPipe.fileHandleForWriting)
        let endpoint = JsonRpcEndpoint(stream: stream, onNotification: onNotification)
        let supervisor = ProcessSupervisor(process: process, stdinPipe: stdinPipe,
                                            stderrPipe: stderrPipe, endpoint: endpoint, pid: pid)
        setCurrent(pid)
        supervisor.drainStderr()
        endpoint.start()
        return supervisor
    }

    private func drainStderr() {
        let fd = stderrPipe.fileHandleForReading.fileDescriptor
        let t = Thread { [weak self] in
            var buffer = Data()
            var buf = [UInt8](repeating: 0, count: 4096)
            while true {
                let n = buf.withUnsafeMutableBytes { ptr -> Int in
                    #if canImport(Glibc)
                    return Glibc.read(fd, ptr.baseAddress, 4096)
                    #else
                    return Darwin.read(fd, ptr.baseAddress, 4096)
                    #endif
                }
                if n <= 0 { break }
                buffer.append(contentsOf: buf[0..<n])
                while let nlIdx = buffer.firstIndex(of: 0x0A) {
                    let lineData = buffer.subdata(in: buffer.startIndex..<nlIdx)
                    buffer.removeSubrange(buffer.startIndex...nlIdx)
                    self?.appendStderrLine(String(decoding: lineData, as: UTF8.self))
                }
            }
        }
        t.name = "MonicaLSP.ProcessSupervisor.stderr"
        t.start()
    }

    private func appendStderrLine(_ line: String) {
        stderrLock.lock()
        defer { stderrLock.unlock() }
        stderrTailLines.append(line)
        if stderrTailLines.count > Self.stderrTailCap {
            stderrTailLines.removeFirst(stderrTailLines.count - Self.stderrTailCap)
        }
    }

    public var stderrTail: [String] {
        stderrLock.lock()
        defer { stderrLock.unlock() }
        return stderrTailLines
    }

    public var isAlive: Bool {
        process.isRunning
    }

    /// Idempotent and bounded, mirroring `ts_service.py:446`: close stdin (server sees EOF)
    /// -> best-effort LSP `shutdown`/`exit` -> `SIGTERM` -> bounded wait -> `SIGKILL` ->
    /// bounded wait. "Already dead" is success, not an error. Safe to call from `defer`,
    /// from a `catch`, or twice.
    public func shutdown(sendLspShutdown: Bool = true) {
        stateLock.lock()
        if reaped { stateLock.unlock(); return }
        reaped = true
        stateLock.unlock()

        if sendLspShutdown && isAlive {
            _ = try? endpoint.request("shutdown", timeout: timeouts.lspShutdown)
            endpoint.notify("exit")
        }
        // `endpoint.close()` already closes the write side via `stream.closeWrite()`
        // (`PipeTransport.closeWrite()` -> raw `close(writeFD)`). Closing
        // `stdinPipe.fileHandleForWriting` again here would be a second `close(2)` on that
        // same fd number, which the OS may have already reassigned to something unrelated —
        // a real double-close, not a harmless idempotent no-op.
        endpoint.close()

        if waitUntilExit(timeout: 0.3) { Self.clearCurrent(pid); return }

        kill(pid_t(pid), SIGTERM)
        if waitUntilExit(timeout: timeouts.terminateWait) { Self.clearCurrent(pid); return }

        kill(pid_t(pid), SIGKILL)
        _ = waitUntilExit(timeout: timeouts.killWait)
        Self.clearCurrent(pid)
    }

    private func waitUntilExit(timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if !process.isRunning { return true }
            Thread.sleep(forTimeInterval: 0.02)
        }
        return !process.isRunning
    }

    /// `kill(pid, 0) == 0` liveness probe usable from anywhere (e.g. `--probe-reap`'s
    /// post-teardown assertion), independent of any particular `ProcessSupervisor` instance.
    public static func isPidAlive(_ pid: Int32) -> Bool {
        kill(pid_t(pid), 0) == 0
    }

    /// A scoped entry point: `spawn`, run `body`, and reap on every exit path (normal
    /// return, thrown error, or the caller's own early exit via `return`/`throw` inside
    /// `body`) via `defer`. `deinit` is deliberately NOT relied on for reaping — Swift does
    /// not guarantee it runs at process exit (e.g. on `exit(_:)` or a signal) — the process-
    /// wide signal/atexit backstop above covers those instead.
    public static func withServer<T>(argv: [String], cwd: String? = nil, env: [String: String]? = nil,
                                      onNotification: JsonRpcEndpoint.OnNotification? = nil,
                                      _ body: (ProcessSupervisor) throws -> T) throws -> T {
        let supervisor = try spawn(argv: argv, cwd: cwd, env: env, onNotification: onNotification)
        defer { supervisor.shutdown() }
        return try body(supervisor)
    }
}
