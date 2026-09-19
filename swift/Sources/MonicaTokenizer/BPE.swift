// BPE core — the merge loop. stdlib-only, no Foundation, no allocations in the hot path
// beyond a per-call scratch array. Operates entirely on integer token ids (tiktoken-style):
// no GPT-2 bytes→printable-char remap, no `String` work.
//
// Id layout (deterministic, positional):
//   [0 ..< specialCount)                  special tokens (EOS/FIM/mask)
//   [specialCount ..< specialCount+256)   the 256 raw base bytes  (byte v → specialCount + v)
//   [specialCount+256 ...]                one id per merge, in merge order

public final class BPE: @unchecked Sendable {   // immutable after init → safe to share

    let specialCount: Int
    let baseOffset: Int
    let mergeRank: [UInt32: Int]      // packed pair (UInt32: a << 16 | b) → merge order index (lower = earlier)
    let idToBytes: [[UInt8]]          // token id → its raw bytes (for decode)
    public let vocabSize: Int

    /// Pack an ordered id pair into one `UInt64` key (for backward compatibility).
    @inline(__always)
    public static func key(_ a: Int, _ b: Int) -> UInt64 {
        (UInt64(UInt32(a)) << 32) | UInt64(UInt32(b))
    }

    /// Fast 32-bit key packing for tokens <= 65536.
    @inline(__always)
    public static func key32(_ a: Int, _ b: Int) -> UInt32 {
        (UInt32(a) << 16) | UInt32(b)
    }

    public init(format: TokenizerFormat) {
        let sc = format.specialTokens.count
        let base = sc + 256
        specialCount = sc
        baseOffset = base

        var idBytes: [[UInt8]] = []
        idBytes.reserveCapacity(base + format.merges.count)
        for s in format.specialTokens { idBytes.append(Array(s.utf8)) }   // specials
        for v in 0..<256 { idBytes.append([UInt8(v)]) }                   // base bytes

        var rank: [UInt32: Int] = [:]
        rank.reserveCapacity(format.merges.count)
        for (m, pair) in format.merges.enumerated() {
            let a = pair[0], b = pair[1]
            idBytes.append(idBytes[a] + idBytes[b])
            rank[BPE.key32(a, b)] = m
        }

        mergeRank = rank
        idToBytes = idBytes
        vocabSize = idBytes.count
    }

    /// Encode one pre-token's raw bytes, appending its ids to `out`.
    public func encodePretoken(_ bytes: [UInt8], into out: inout [Int]) {
        var syms: [Int] = []
        syms.reserveCapacity(bytes.count)
        encodePretoken(bytes, syms: &syms, into: &out)
    }

    /// Encode one pre-token's raw bytes using a caller-supplied scratch buffer to avoid heap allocations.
    public func encodePretoken(_ bytes: [UInt8], syms: inout [Int], into out: inout [Int]) {
        if bytes.isEmpty { return }
        syms.removeAll(keepingCapacity: true)
        let byteOffset = specialCount
        for b in bytes { syms.append(byteOffset + Int(b)) }

        while syms.count >= 2 {
            var bestRank = Int.max
            var bestPos = -1
            for p in 0..<(syms.count - 1) {
                if let r = mergeRank[BPE.key32(syms[p], syms[p + 1])], r < bestRank {
                    bestRank = r
                    bestPos = p
                }
            }
            if bestPos < 0 { break }
            syms[bestPos] = baseOffset + bestRank
            syms.remove(at: bestPos + 1)
        }
        out.append(contentsOf: syms)
    }

    /// Concatenate the raw bytes of each id and decode as UTF-8 (lossless: the full 256-byte
    /// alphabet is always in-vocab). Out-of-range ids are skipped.
    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids where id >= 0 && id < idToBytes.count {
            bytes.append(contentsOf: idToBytes[id])
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
