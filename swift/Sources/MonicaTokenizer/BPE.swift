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
    let mergeRank: [UInt64: Int]      // packed pair → merge order index (lower = earlier)
    let pairToMerged: [UInt64: Int]   // packed pair → resulting merged token id
    let idToBytes: [[UInt8]]          // token id → its raw bytes (for decode)
    public let vocabSize: Int

    /// Pack an ordered id pair into one `UInt64` key. Token ids must fit in `UInt32`; the code
    /// tokenizer's vocab is ≤ 16384 (uint16-packable, #191) — far within that bound.
    @inline(__always)
    public static func key(_ a: Int, _ b: Int) -> UInt64 {
        (UInt64(UInt32(a)) << 32) | UInt64(UInt32(b))
    }

    public init(format: TokenizerFormat) {
        let sc = format.specialTokens.count
        let baseOffset = sc + 256
        specialCount = sc

        var idBytes: [[UInt8]] = []
        idBytes.reserveCapacity(baseOffset + format.merges.count)
        for s in format.specialTokens { idBytes.append(Array(s.utf8)) }   // specials
        for v in 0..<256 { idBytes.append([UInt8(v)]) }                   // base bytes

        var rank: [UInt64: Int] = [:]
        var p2m: [UInt64: Int] = [:]
        rank.reserveCapacity(format.merges.count)
        p2m.reserveCapacity(format.merges.count)
        for (m, pair) in format.merges.enumerated() {
            let a = pair[0], b = pair[1]
            idBytes.append(idBytes[a] + idBytes[b])
            let k = BPE.key(a, b)
            rank[k] = m
            p2m[k] = baseOffset + m
        }

        mergeRank = rank
        pairToMerged = p2m
        idToBytes = idBytes
        vocabSize = idBytes.count
    }

    /// Encode one pre-token's raw bytes, appending its ids to `out`. Repeatedly applies the
    /// adjacent pair with the lowest merge rank (a linear min-scan — pre-tokens are short, so
    /// this beats a heap in practice) until no adjacent pair has a rank.
    public func encodePretoken(_ bytes: [UInt8], into out: inout [Int]) {
        if bytes.isEmpty { return }
        let byteOffset = specialCount
        var syms: [Int] = []
        syms.reserveCapacity(bytes.count)
        for b in bytes { syms.append(byteOffset + Int(b)) }

        while syms.count >= 2 {
            var bestRank = Int.max
            var bestPos = -1
            for p in 0..<(syms.count - 1) {
                if let r = mergeRank[BPE.key(syms[p], syms[p + 1])], r < bestRank {
                    bestRank = r
                    bestPos = p
                }
            }
            if bestPos < 0 { break }
            syms[bestPos] = pairToMerged[BPE.key(syms[bestPos], syms[bestPos + 1])]!
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
