// Minimal Thrift compact-protocol reader over an in-memory byte buffer — just enough to
// decode Parquet's FileMetaData/PageHeader structures (see ParquetMetadata.swift). Untrusted
// input (a Parquet footer handed to us on disk), so every read is bounds-checked and throws on
// malformed/truncated data rather than trapping.
//
// Reference: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md

import Foundation

enum ThriftError: Error, CustomStringConvertible {
    case truncated(String)
    case malformed(String)

    var description: String {
        switch self {
        case .truncated(let m): return "truncated Thrift compact data: \(m)"
        case .malformed(let m): return "malformed Thrift compact data: \(m)"
        }
    }
}

/// Compact-protocol field/element type codes (not a full Thrift `TType` — only what Parquet
/// metadata uses).
enum ThriftType {
    static let boolTrue: UInt8 = 1
    static let boolFalse: UInt8 = 2
    static let i8: UInt8 = 3
    static let i16: UInt8 = 4
    static let i32: UInt8 = 5
    static let i64: UInt8 = 6
    static let double: UInt8 = 7
    static let binary: UInt8 = 8
    static let list: UInt8 = 9
    static let set: UInt8 = 10
    static let map: UInt8 = 11
    static let structType: UInt8 = 12
}

struct ThriftFieldHeader {
    let id: Int16
    let type: UInt8
}

/// A cursor over `bytes`. `lastFieldId` is deliberately **not** tracked here across nested
/// structs — each struct-decoding function keeps its own local `var lastFieldId: Int16 = 0`
/// and threads it through `readFieldBegin(lastFieldId:)`, which is what makes "per-struct"
/// automatic: entering a nested struct starts a fresh local, and returning to the caller's
/// loop resumes the caller's own local, already where it left off.
struct ThriftCompactReader {
    private let bytes: [UInt8]
    private(set) var pos: Int

    init(_ bytes: [UInt8], at start: Int = 0) {
        self.bytes = bytes
        self.pos = start
    }

    var isAtEnd: Bool { pos >= bytes.count }

    mutating func readByte() throws -> UInt8 {
        guard pos < bytes.count else { throw ThriftError.truncated("byte at offset \(pos)") }
        defer { pos += 1 }
        return bytes[pos]
    }

    mutating func readBytes(_ n: Int) throws -> [UInt8] {
        guard n >= 0 else { throw ThriftError.malformed("negative length \(n)") }
        guard pos + n <= bytes.count else {
            throw ThriftError.truncated("\(n) bytes at offset \(pos) (have \(bytes.count - pos))")
        }
        let r = Array(bytes[pos..<(pos + n)])
        pos += n
        return r
    }

    /// Unsigned LEB128, 7 bits/byte LSB-first, continuation bit 0x80. Caps at 10 bytes (enough
    /// for any 64-bit value) so a corrupt buffer of all-0x80 bytes can't spin forever.
    mutating func readVarint() throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        var i = 0
        while true {
            guard i < 10 else { throw ThriftError.malformed("varint exceeds 10 bytes") }
            let b = try readByte()
            result |= UInt64(b & 0x7f) << shift
            if b & 0x80 == 0 { break }
            shift += 7
            i += 1
        }
        return result
    }

    mutating func readZigzag32() throws -> Int32 {
        let n = try readVarint()
        guard n <= UInt64(UInt32.max) else { throw ThriftError.malformed("zigzag32 overflow") }
        let u = UInt32(truncatingIfNeeded: n)
        return Int32(bitPattern: (u >> 1) ^ (0 &- (u & 1)))
    }

    mutating func readZigzag64() throws -> Int64 {
        let n = try readVarint()
        return Int64(bitPattern: (n >> 1) ^ (0 &- (n & 1)))
    }

    mutating func readBinary() throws -> [UInt8] {
        let n = try readVarint()
        guard n <= UInt64(Int32.max) else { throw ThriftError.malformed("binary length too large") }
        return try readBytes(Int(n))
    }

    mutating func readString() throws -> String {
        String(decoding: try readBinary(), as: UTF8.self)
    }

    /// One field header, or `nil` at STOP (`0x00`). `lastFieldId` is the caller's per-struct
    /// cursor: updated in place so the next call resumes correctly.
    mutating func readFieldBegin(lastFieldId: inout Int16) throws -> ThriftFieldHeader? {
        let header = try readByte()
        if header == 0x00 { return nil }
        let delta = (header & 0xf0) >> 4
        let type = header & 0x0f
        let id: Int16
        if delta == 0 {
            id = Int16(truncatingIfNeeded: try readZigzag32())
        } else {
            id = lastFieldId + Int16(delta)
        }
        lastFieldId = id
        return ThriftFieldHeader(id: id, type: type)
    }

    /// `(size << 4) | elemType`, with the size-15 escape to a following varint. Used for both
    /// LIST and SET (identical wire layout).
    mutating func readListHeader() throws -> (size: Int, elemType: UInt8) {
        let header = try readByte()
        var size = Int((header & 0xf0) >> 4)
        let elemType = header & 0x0f
        if size == 15 {
            let n = try readVarint()
            guard n <= UInt64(Int32.max) else { throw ThriftError.malformed("list size too large") }
            size = Int(n)
        }
        return (size, elemType)
    }

    /// A single integer-typed list/set element (I8 raw byte, I16/I32 zigzag varint, I64 zigzag
    /// varint). Used for `encodings: list<i32>` and similar.
    mutating func readIntElement(_ elemType: UInt8) throws -> Int64 {
        switch elemType {
        case ThriftType.i8: return Int64(Int8(bitPattern: try readByte()))
        case ThriftType.i16, ThriftType.i32: return Int64(try readZigzag32())
        case ThriftType.i64: return try readZigzag64()
        default: throw ThriftError.malformed("expected an integer list element, got type \(elemType)")
        }
    }

    /// Consume and discard one value of `type` (a field's body, or an unrecognized field we
    /// don't decode). Recurses into LIST/SET/MAP/STRUCT. Every case here must stay in sync with
    /// `ThriftType` — an unknown code is an error, not a silent no-op.
    mutating func skip(type: UInt8) throws {
        switch type {
        case ThriftType.boolTrue, ThriftType.boolFalse:
            return   // value is in the field-header nibble itself; no body
        case ThriftType.i8:
            _ = try readByte()
        case ThriftType.i16, ThriftType.i32, ThriftType.i64:
            _ = try readVarint()
        case ThriftType.double:
            _ = try readBytes(8)
        case ThriftType.binary:
            _ = try readBinary()
        case ThriftType.list, ThriftType.set:
            try skipList()
        case ThriftType.map:
            try skipMap()
        case ThriftType.structType:
            try skipStruct()
        default:
            throw ThriftError.malformed("unknown Thrift compact type code \(type)")
        }
    }

    mutating func skipStruct() throws {
        var lastFieldId: Int16 = 0
        while let fh = try readFieldBegin(lastFieldId: &lastFieldId) {
            try skip(type: fh.type)
        }
    }

    mutating func skipList() throws {
        let (size, elemType) = try readListHeader()
        for _ in 0..<size {
            // Edge case: unlike a bool *field* (value in the header nibble), a bool *list
            // element* does carry an explicit 0x00/0x01 body byte. We don't hit this decoding
            // real Parquet metadata, but a future field could, so handle it rather than trap.
            if elemType == ThriftType.boolTrue || elemType == ThriftType.boolFalse {
                _ = try readByte()
            } else {
                try skip(type: elemType)
            }
        }
    }

    /// Map wire format: varint `size` (a bare `0x00` byte *is* that varint when size is 0, so
    /// there is no separate "single byte" case to special-case), then — only if `size > 0` — one
    /// byte `(keyType << 4) | valType`, then `size` key/value pairs.
    mutating func skipMap() throws {
        let n = try readVarint()
        guard n <= UInt64(Int32.max) else { throw ThriftError.malformed("map size too large") }
        let size = Int(n)
        guard size > 0 else { return }
        let types = try readByte()
        let keyType = (types & 0xf0) >> 4
        let valType = types & 0x0f
        for _ in 0..<size {
            try skip(type: keyType)
            try skip(type: valType)
        }
    }
}
