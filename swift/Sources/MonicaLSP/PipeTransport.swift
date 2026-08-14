// PipeTransport — buffered read(2)/write(2) over raw file descriptors, with the exact
// "short read == EOF, never a partial wait" semantics `JsonRpc.swift`'s framing needs
// (`src/lsp/jsonrpc.py:56`'s `read_message` contract). `FileHandle.availableData` does
// NOT give this guarantee (it can return fewer bytes than requested for reasons other
// than EOF), so this type talks to the POSIX layer directly instead.
//
// Works two ways:
//   - `PipeTransport.pipePair()` — two connected ends over `pipe(2)`, no subprocess at
//     all. This is what makes `monica-lsp --self-test`'s framing/demux tests binary-free
//     and runnable on Linux (mirrors `tests/test_jsonrpc.py`'s `os.pipe()` fixture).
//   - `init(readHandle:writeHandle:)` — wraps a real child process's stdio pipes
//     (`Foundation.Pipe`'s `fileHandleForReading`/`fileHandleForWriting`), extracting the
//     raw fd rather than going through `FileHandle`'s buffered read API.
//
// `Process`/`Pipe`/raw POSIX `read`/`write`/`pipe`/`close` are all in Foundation's
// portable subset — this file builds and runs identically on macOS and Linux (`swift-linux`
// CI, #246's existing guarantee, now covers this package too).

import Foundation
#if canImport(Glibc)
import Glibc
#else
import Darwin
#endif

/// The minimal contract `JsonRpc.swift`'s framing needs from a byte-oriented transport.
public protocol ByteStream: AnyObject {
    /// One byte, or `nil` at EOF.
    func readByte() -> UInt8?
    /// Up to `n` bytes in one `read(2)` call; `nil` if that single call returned 0 bytes
    /// (EOF) or failed. Never blocks waiting for more than one kernel read to arrive.
    func readChunk(upTo n: Int) -> Data?
    /// `true` if every byte was written, `false` on any write error (treated as EOF for
    /// framing purposes — no partial-write recovery is attempted, matching the reference's
    /// "the writer either works or the endpoint is dead" contract).
    func write(_ data: Data) -> Bool
    /// Close the write side (lets a downstream reader on the other end see EOF). Idempotent.
    func closeWrite()
}

public extension ByteStream {
    /// Read exactly `n` bytes, or `nil` if EOF is hit before all `n` are available —
    /// `src/lsp/jsonrpc.py:56`'s "a short read anywhere is EOF, never a partial message to
    /// wait longer for."
    func readExact(_ n: Int) -> Data? {
        if n == 0 { return Data() }
        var out = Data()
        out.reserveCapacity(n)
        while out.count < n {
            guard let chunk = readChunk(upTo: n - out.count) else { return nil }
            out.append(chunk)
        }
        return out
    }
}

public final class PipeTransport: ByteStream {
    private let readFD: Int32
    private let writeFD: Int32
    private let lock = NSLock()
    private var writeClosed = false
    private var readClosed = false

    public init(readFD: Int32, writeFD: Int32) {
        self.readFD = readFD
        self.writeFD = writeFD
    }

    /// Wrap a `Foundation.Pipe` pair's raw file descriptors (a real child process's stdio).
    public convenience init(readHandle: FileHandle, writeHandle: FileHandle) {
        self.init(readFD: readHandle.fileDescriptor, writeFD: writeHandle.fileDescriptor)
    }

    /// Two connected `PipeTransport`s over `pipe(2)`, no subprocess — the fixture
    /// `monica-lsp --self-test` drives the framing/demux tests against (mirrors
    /// `tests/test_jsonrpc.py`'s `os.pipe()` pattern).
    public static func pipePair() -> (a: PipeTransport, b: PipeTransport) {
        var fds1: [Int32] = [0, 0]
        var fds2: [Int32] = [0, 0]
        let r1 = fds1.withUnsafeMutableBufferPointer { pipe($0.baseAddress) }
        let r2 = fds2.withUnsafeMutableBufferPointer { pipe($0.baseAddress) }
        // A failed `pipe(2)` leaves the fds at their initialized 0/0, which would silently
        // alias stdin — assert instead of building a `PipeTransport` fixture that can
        // accidentally read/write stdio and make the self-test non-deterministic.
        precondition(r1 == 0 && r2 == 0, "PipeTransport.pipePair: pipe(2) failed")
        // a reads fds1's read end (fed by b's write end fds1[1]); a writes to fds2's write
        // end (read by b's read end fds2[0]).
        let a = PipeTransport(readFD: fds1[0], writeFD: fds2[1])
        let b = PipeTransport(readFD: fds2[0], writeFD: fds1[1])
        return (a, b)
    }

    public func readByte() -> UInt8? {
        guard let d = readChunk(upTo: 1), !d.isEmpty else { return nil }
        return d[d.startIndex]
    }

    public func readChunk(upTo n: Int) -> Data? {
        guard n > 0 else { return Data() }
        var buf = [UInt8](repeating: 0, count: n)
        let got = buf.withUnsafeMutableBytes { ptr -> Int in
            #if canImport(Glibc)
            return Glibc.read(readFD, ptr.baseAddress, n)
            #else
            return Darwin.read(readFD, ptr.baseAddress, n)
            #endif
        }
        if got <= 0 { return nil }
        return Data(buf[0..<got])
    }

    public func write(_ data: Data) -> Bool {
        // Held for the ENTIRE write, not just the `writeClosed` check — otherwise
        // `closeWrite()` can interleave between this check and the `write(2)` loop below (or
        // between individual `write(2)` calls) and close the fd mid-message, putting a
        // partial JSON-RPC frame on the wire. Making `write`/`closeWrite` share one
        // lock-for-the-duration critical section is what makes them mutually exclusive.
        lock.lock()
        defer { lock.unlock() }
        if writeClosed { return false }
        var written = 0
        let bytes = [UInt8](data)
        let total = bytes.count
        while written < total {
            let n = bytes[written...].withUnsafeBufferPointer { ptr -> Int in
                #if canImport(Glibc)
                return Glibc.write(writeFD, ptr.baseAddress, ptr.count)
                #else
                return Darwin.write(writeFD, ptr.baseAddress, ptr.count)
                #endif
            }
            if n <= 0 { return false }
            written += n
        }
        return true
    }

    public func closeWrite() {
        lock.lock()
        defer { lock.unlock() }
        guard !writeClosed else { return }
        writeClosed = true
        close(writeFD)
    }

    /// Only meaningful for the `pipePair()` fixture — a real subprocess's read fd is owned
    /// by the `Pipe`/`Process` machinery and closed there instead.
    public func closeRead() {
        lock.lock()
        defer { lock.unlock() }
        guard !readClosed else { return }
        readClosed = true
        close(readFD)
    }
}
