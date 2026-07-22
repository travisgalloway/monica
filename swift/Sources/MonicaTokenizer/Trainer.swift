// Deterministic byte-level BPE trainer → native TokenizerFormat. stdlib-only.
//
// Determinism: the trained vocab depends only on the multiset of pre-tokens (order-independent
// counting) and a total deterministic merge-selection order — highest pair frequency, ties
// broken by the smaller packed-pair key. So re-training the same corpus yields identical merges
// on any platform (the regenerable-fixtures guarantee).
//
// Complexity: recomputes pair frequencies each round (O(rounds · unique-pre-token-symbols)).
// Fine for a one-time offline train over a representative sample; incremental pair-count
// maintenance is a documented follow-up if the sample grows large.

public enum Trainer {

    /// Train `vocabSize`-token BPE over `corpus`. The 6 (or N) `specialTokens` occupy ids
    /// 0 ..< N, the 256 base bytes the next block, and merges fill the remainder.
    public static func train(corpus: [String],
                             vocabSize: Int,
                             specialTokens: [String],
                             digitGroup: Int) -> TokenizerFormat {
        let specialCount = specialTokens.count
        let baseOffset = specialCount + 256
        // The vocab can never be smaller than the specials + 256 base bytes, so a smaller
        // `vocabSize` isn't an achievable cap — fail immediately rather than silently emitting
        // a vocab that exceeds the requested size.
        precondition(vocabSize >= baseOffset,
                     "vocabSize \(vocabSize) must be >= specialTokens.count + 256 (\(baseOffset))")

        // 1. Count unique pre-tokens (by raw bytes) across the corpus.
        var wordCounts: [[UInt8]: Int] = [:]
        for doc in corpus {
            for pt in Pretokenizer.pretokenize(doc, digitGroup: digitGroup) {
                wordCounts[pt, default: 0] += 1
            }
        }

        // 2. Each unique pre-token → its base-byte id sequence + count.
        var words: [(syms: [Int], count: Int)] = wordCounts.map { bytes, count in
            (bytes.map { specialCount + Int($0) }, count)
        }

        // 3. Greedily merge the most frequent adjacent pair until the vocab is full.
        var merges: [[Int]] = []
        let targetMerges = max(0, vocabSize - baseOffset)
        for _ in 0..<targetMerges {
            var pairCounts: [UInt64: Int] = [:]
            for w in words where w.syms.count >= 2 {
                for p in 0..<(w.syms.count - 1) {
                    pairCounts[BPE.key(w.syms[p], w.syms[p + 1]), default: 0] += w.count
                }
            }
            if pairCounts.isEmpty { break }

            // Deterministic pick: max count, tie → smaller packed key.
            var bestKey: UInt64 = .max
            var bestCount = 0
            for (k, c) in pairCounts where c > bestCount || (c == bestCount && k < bestKey) {
                bestCount = c
                bestKey = k
            }

            let a = Int(bestKey >> 32)
            let b = Int(bestKey & 0xffff_ffff)
            let newId = baseOffset + merges.count
            merges.append([a, b])

            for wi in 0..<words.count {
                words[wi].syms = applyMerge(words[wi].syms, a: a, b: b, newId: newId)
            }
        }

        return TokenizerFormat(version: 1, specialTokens: specialTokens,
                               digitGroup: digitGroup, merges: merges)
    }

    /// Replace every non-overlapping adjacent `(a, b)` in `syms` with `newId`, left to right.
    static func applyMerge(_ syms: [Int], a: Int, b: Int, newId: Int) -> [Int] {
        guard syms.count >= 2 else { return syms }
        var out: [Int] = []
        out.reserveCapacity(syms.count)
        var i = 0
        while i < syms.count {
            if i < syms.count - 1 && syms[i] == a && syms[i + 1] == b {
                out.append(newId)
                i += 2
            } else {
                out.append(syms[i])
                i += 1
            }
        }
        return out
    }
}
