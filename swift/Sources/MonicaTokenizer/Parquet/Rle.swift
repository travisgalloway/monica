// Parquet's "RLE / bit-packed hybrid" — used for definition levels (bit width 1) and
// dictionary-encoded data-page indices (bit width = leading byte of the values section).
// Spec: https://parquet.apache.org/docs/file-format/data-pages/encodings/#run-length-encoding--bit-packing-hybrid-rle--3

import Foundation

enum RleError: Error, CustomStringConvertible {
    case truncated(String)
    case malformed(String)

    var description: String {
        switch self {
        case .truncated(let m): return "truncated RLE/bit-packed data: \(m)"
        case .malformed(let m): return "malformed RLE/bit-packed data: \(m)"
        }
    }
}

enum RleHybrid {
    /// Decode exactly `count` unsigned values of `bitWidth` bits from `bytes`. Throws if the
    /// input runs out before `count` values have been produced — never silently short-reads.
    static func decode(_ bytes: [UInt8], count: Int, bitWidth: Int) throws -> [Int] {
        guard count >= 0 else { throw RleError.malformed("negative count \(count)") }
        // Every value is 0 and there is nothing to consume — guards an infinite loop on a
        // zero-width column (e.g. a single-value dictionary) rather than reading garbage.
        if bitWidth == 0 { return Array(repeating: 0, count: count) }
        guard bitWidth > 0 && bitWidth <= 32 else {
            throw RleError.malformed("bit width \(bitWidth) out of range [1, 32]")
        }

        var out: [Int] = []
        out.reserveCapacity(count)
        var pos = 0
        let byteWidth = (bitWidth + 7) / 8

        func readByte() throws -> UInt8 {
            guard pos < bytes.count else { throw RleError.truncated("byte at offset \(pos)") }
            defer { pos += 1 }
            return bytes[pos]
        }
        func readVarint() throws -> UInt64 {
            var result: UInt64 = 0, shift: UInt64 = 0, i = 0
            while true {
                guard i < 10 else { throw RleError.malformed("varint exceeds 10 bytes") }
                let b = try readByte()
                result |= UInt64(b & 0x7f) << shift
                if b & 0x80 == 0 { break }
                shift += 7; i += 1
            }
            return result
        }

        while out.count < count {
            let header = try readVarint()
            if header & 1 == 1 {
                // Bit-packed run: `groups` groups of 8 values, packed LSB-first with bits
                // flowing across byte boundaries. Trailing values in the last group are padding
                // (the final truncate-to-`count` below drops them).
                let groups = Int(header >> 1)
                let values = groups * 8
                let nBytes = groups * bitWidth
                guard pos + nBytes <= bytes.count else {
                    throw RleError.truncated("\(nBytes) bit-packed bytes at offset \(pos)")
                }
                let base = pos
                var bitPos = 0
                for _ in 0..<values {
                    var v = 0
                    for b in 0..<bitWidth {
                        let absBit = bitPos + b
                        let byteIdx = base + absBit / 8
                        let bit = (bytes[byteIdx] >> (absBit % 8)) & 1
                        v |= Int(bit) << b
                    }
                    out.append(v)
                    bitPos += bitWidth
                }
                pos += nBytes
            } else {
                // RLE run: a little-endian integer over `ceil(bitWidth/8)` bytes, repeated.
                let runLen = Int(header >> 1)
                guard pos + byteWidth <= bytes.count else {
                    throw RleError.truncated("\(byteWidth)-byte RLE value at offset \(pos)")
                }
                var value = 0
                for i in 0..<byteWidth {
                    value |= Int(bytes[pos + i]) << (8 * i)
                }
                pos += byteWidth
                if runLen > 0 { out.append(contentsOf: repeatElement(value, count: runLen)) }
            }
        }
        if out.count > count { out.removeLast(out.count - count) }
        return out
    }
}
