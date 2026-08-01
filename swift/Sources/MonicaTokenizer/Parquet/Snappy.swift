// Raw Snappy block decompression (NOT the framed/streaming format — Parquet's SNAPPY codec is
// the raw block format applied once per page). Spec: https://github.com/google/snappy/blob/main/format_description.txt

import Foundation

enum SnappyError: Error, CustomStringConvertible {
    case truncated(String)
    case malformed(String)

    var description: String {
        switch self {
        case .truncated(let m): return "truncated Snappy data: \(m)"
        case .malformed(let m): return "malformed Snappy data: \(m)"
        }
    }
}

enum Snappy {
    /// Decompress one raw Snappy block. Throws (never traps) on any malformed tag, an
    /// out-of-range backreference, or a final length that disagrees with the declared preamble.
    static func decompress(_ input: [UInt8]) throws -> [UInt8] {
        var pos = 0
        func readByte() throws -> UInt8 {
            guard pos < input.count else { throw SnappyError.truncated("byte at offset \(pos)") }
            defer { pos += 1 }
            return input[pos]
        }
        func readLE(_ n: Int) throws -> Int {
            guard pos + n <= input.count else {
                throw SnappyError.truncated("\(n)-byte little-endian field at offset \(pos)")
            }
            var v = 0
            for i in 0..<n { v |= Int(input[pos + i]) << (8 * i) }
            pos += n
            return v
        }
        func readVarint() throws -> Int {
            var result = 0, shift = 0, i = 0
            while true {
                guard i < 5 else { throw SnappyError.malformed("preamble varint exceeds 5 bytes") }
                let b = try readByte()
                result |= Int(b & 0x7f) << shift
                if b & 0x80 == 0 { break }
                shift += 7; i += 1
            }
            return result
        }

        let uncompressedLength = try readVarint()
        guard uncompressedLength >= 0 else {
            throw SnappyError.malformed("negative preamble length \(uncompressedLength)")
        }
        var out: [UInt8] = []
        out.reserveCapacity(uncompressedLength)

        while out.count < uncompressedLength {
            let tag = try readByte()
            switch tag & 0x03 {
            case 0:   // LITERAL
                let n = Int(tag >> 2)
                let litLen: Int
                if n < 60 {
                    litLen = n + 1
                } else {
                    let v = try readLE(n - 59)
                    litLen = v + 1
                }
                guard pos + litLen <= input.count else {
                    throw SnappyError.truncated("\(litLen)-byte literal at offset \(pos)")
                }
                out.append(contentsOf: input[pos..<(pos + litLen)])
                pos += litLen
            case 1:   // COPY_1B — 11-bit offset
                let length = 4 + Int((tag >> 2) & 0x07)
                let lo = try readByte()
                let offset = (Int(tag >> 5) << 8) | Int(lo)
                try copy(into: &out, offset: offset, length: length)
            case 2:   // COPY_2B
                let length = Int(tag >> 2) + 1
                let offset = try readLE(2)
                try copy(into: &out, offset: offset, length: length)
            case 3:   // COPY_4B
                let length = Int(tag >> 2) + 1
                let offset = try readLE(4)
                try copy(into: &out, offset: offset, length: length)
            default:
                fatalError("unreachable: tag & 0x03 is in [0, 3]")
            }
        }

        guard out.count == uncompressedLength else {
            throw SnappyError.malformed(
                "decompressed length \(out.count) != declared preamble length \(uncompressedLength)")
        }
        return out
    }

    /// A copy op with `1 <= offset <= out.count`, appended byte-by-byte since Snappy
    /// backreferences can overlap the just-written output (a bulk memcpy would be wrong).
    private static func copy(into out: inout [UInt8], offset: Int, length: Int) throws {
        guard offset >= 1 && offset <= out.count else {
            throw SnappyError.malformed("copy offset \(offset) out of range (have \(out.count) bytes)")
        }
        guard length >= 0 else { throw SnappyError.malformed("negative copy length \(length)") }
        var src = out.count - offset
        for _ in 0..<length {
            out.append(out[src])
            src += 1
        }
    }
}
