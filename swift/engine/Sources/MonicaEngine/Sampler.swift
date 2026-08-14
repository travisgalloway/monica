// Sampler — the Swift port of `src/serve/sampling.py`'s `sample()`, order-for-order (#167).
//
// Works in `Double` over a `[Double]` copy of the logits row, matching Python's
// `np.asarray(logits, dtype=np.float64)`. Exactly the same pipeline, same order:
//
//   1. `allowedIds` mask (add -inf outside the set) — BEFORE everything else, so a
//      repetition penalty can never resurrect a masked token and top-p renormalizes over
//      survivors only. Empty (or entirely out-of-range) set THROWS, never a uniform draw.
//   2. Repetition penalty (CTRL-style) + `noRepeatNgramSize`.
//   3. All-non-finite fallback: uniform over `allowedIds` if set, else over the vocab.
//   4. `temperature == 0` -> argmax (greedy — the parity contract). `< 0` throws.
//   5. Divide by temperature; top-k (tie-inclusive); softmax; top-p (keep the crossing
//      token); renormalize; draw.
//
// PARITY NOTE: only greedy (`temperature == 0`) is a bit-for-bit cross-language contract
// (#167 AC1). Sampled draws (`temperature > 0`) are *semantically* identical to Python's —
// same math, same order of operations — but NOT id-identical, because numpy's `Generator`
// stream (PCG64) cannot be reproduced here. Draws below use a seeded SplitMix64 + inverse-CDF,
// which is reproducible run-to-run in Swift but not against the Python reference.
//
// The RNG is a self-contained ~10-line SplitMix64 duplicate of
// `MonicaTokenizer.SplitMix64` (see swift/Sources/MonicaTokenizer/FIM.swift) — deliberately
// NOT shared, so the `MonicaEngine` LIBRARY keeps its current dependency shape (no tokenizer
// edge). Only the `monica-generate` EXECUTABLE gains that edge (Package.swift).

import Foundation

public enum SamplerError: Error, CustomStringConvertible {
    case emptyAllowedIds
    case invalidRepetitionPenalty(Double)
    case invalidTemperature(Double)

    public var description: String {
        switch self {
        case .emptyAllowedIds:
            return "allowedIds is empty — the caller must never hand sample() an empty "
                + "constraint set (that is a masker bug, not 'nothing is valid'; the caller "
                + "should bypass the mask for this step instead)"
        case .invalidRepetitionPenalty(let p):
            return "repetitionPenalty \(p) must be >= 1.0 (1.0 = off; CTRL-style penalty "
                + "only suppresses, never boosts)"
        case .invalidTemperature(let t):
            return "temperature \(t) must be >= 0"
        }
    }
}

/// A tiny, self-contained SplitMix64 PRNG (Steele/Lea/Flood) — see the file header for why
/// this is a deliberate duplicate of `MonicaTokenizer.SplitMix64` rather than a shared type.
struct SamplerRNG {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// A uniform double in `[0, 1)`, from the top 53 bits of one draw (the standard
    /// double-from-uint64 construction).
    mutating func nextUnit() -> Double {
        Double(next() >> 11) * (1.0 / 9007199254740992.0)   // 2^53
    }

    /// Unbiased draw in `[0, bound)` by rejection sampling.
    mutating func next(upperBound bound: Int) -> Int {
        if bound <= 1 { return 0 }
        let b = UInt64(bound)
        let threshold = (0 &- b) % b
        while true {
            let r = next()
            if r >= threshold { return Int(r % b) }
        }
    }
}

/// The sampler's configuration — a struct so `--self-test` and the decode loop can share one
/// value. `temperature == 0` is greedy; everything else mirrors `src/serve/sampling.py`.
public struct Sampler {
    public var temperature: Double
    public var topK: Int?
    public var topP: Double?
    public var repetitionPenalty: Double
    public var noRepeatNgramSize: Int?
    public var seed: UInt64

    private var rng: SamplerRNG

    public init(temperature: Double = 1.0, topK: Int? = nil, topP: Double? = nil,
                repetitionPenalty: Double = 1.0, noRepeatNgramSize: Int? = nil,
                seed: UInt64 = 0) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.noRepeatNgramSize = noRepeatNgramSize
        self.seed = seed
        self.rng = SamplerRNG(seed: seed)
    }

    /// Sample one token id from a 1-D logits row. `previousTokens` drives repetition
    /// control; `allowedIds` restricts the draw to exactly this id set (#226's constrained-
    /// decode seam). See the file header for the exact contract.
    public mutating func sample(_ logitsIn: [Float], previousTokens: [Int] = [],
                                allowedIds: [Int]? = nil) throws -> Int {
        var logits = logitsIn.map { Double($0) }
        let vocab = logits.count

        // --- 1. allowedIds mask (before everything else) ---
        if let allowed = allowedIds {
            if allowed.isEmpty { throw SamplerError.emptyAllowedIds }
            let ids = allowed.filter { $0 >= 0 && $0 < vocab }
            if ids.isEmpty { throw SamplerError.emptyAllowedIds }
            var mask = [Double](repeating: -Double.infinity, count: vocab)
            for id in ids { mask[id] = 0.0 }
            for i in 0..<vocab { logits[i] += mask[i] }
        }

        // --- 2. repetition penalty + no-repeat-ngram ---
        if !previousTokens.isEmpty
            && (repetitionPenalty != 1.0 || (noRepeatNgramSize ?? 0) != 0) {
            if repetitionPenalty != 1.0 {
                if repetitionPenalty < 1.0 {
                    throw SamplerError.invalidRepetitionPenalty(repetitionPenalty)
                }
                var seen = Set<Int>()
                for t in previousTokens where t >= 0 && t < vocab { seen.insert(t) }
                for id in seen {
                    let v = logits[id]
                    logits[id] = v > 0 ? v / repetitionPenalty : v * repetitionPenalty
                }
            }
            if let n = noRepeatNgramSize, n > 0 {
                for id in bannedNgramTokens(previousTokens, n: n, vocabSize: vocab) {
                    logits[id] = -Double.infinity
                }
            }
        }

        // --- 3. all-non-finite fallback ---
        if !logits.contains(where: { $0.isFinite }) {
            if let allowed = allowedIds {
                let ids = allowed.filter { $0 >= 0 && $0 < vocab }
                return ids[rng.next(upperBound: ids.count)]
            }
            return rng.next(upperBound: vocab)
        }

        // --- 4. greedy ---
        if temperature == 0 { return argmaxFirstTie(logits) }
        if temperature < 0 { throw SamplerError.invalidTemperature(temperature) }

        // --- 5. temperature / top-k / softmax / top-p / draw ---
        for i in 0..<vocab { logits[i] /= temperature }

        if let k = topK, k > 0 && k < vocab {
            let kth = kthLargest(logits, k)
            for i in 0..<vocab where logits[i] < kth { logits[i] = -Double.infinity }
        }

        var probs = softmax(logits)

        if let p = topP, p > 0.0 && p < 1.0 {
            let order = (0..<vocab).sorted { probs[$0] > probs[$1] }
            var cumulative = 0.0
            var keep = [Bool](repeating: false, count: vocab)
            for (rank, idx) in order.enumerated() {
                // `(cumulative - sorted) < top_p`: the mass strictly BEFORE this token must
                // already be < top_p for the token to survive — the crossing token included.
                keep[idx] = rank == 0 || cumulative < p
                cumulative += probs[idx]
            }
            var sum = 0.0
            for i in 0..<vocab {
                probs[i] = keep[i] ? probs[i] : 0.0
                sum += probs[i]
            }
            for i in 0..<vocab { probs[i] /= sum }
        }

        return drawFrom(probs)
    }

    // --- internals ---

    private mutating func drawFrom(_ probs: [Double]) -> Int {
        let u = rng.nextUnit()
        var cumulative = 0.0
        for i in 0..<probs.count {
            cumulative += probs[i]
            if u < cumulative { return i }
        }
        return probs.count - 1   // floating-point residue guard, mirrors rng.choice's edge
    }
}

/// `np.argmax` keeps the FIRST maximum on ties — load-bearing for AC1 cross-language parity.
func argmaxFirstTie(_ v: [Double]) -> Int {
    var best = 0
    var bestVal = v[0]
    for i in 1..<v.count where v[i] > bestVal {
        best = i
        bestVal = v[i]
    }
    return best
}

/// The k-th largest value (1-indexed: `k=1` is the max), matching
/// `np.partition(logits, -top_k)[-top_k]`. Tie-inclusive top-k then keeps every logit >= this.
///
/// Quickselect (Lomuto partition, average O(V)) rather than a full O(V log V) sort — `sample`
/// calls this every decode step at vocab-sized inputs (50k-150k), so this matches the Python
/// reference's `np.partition` complexity class instead of paying for a full ordering nobody
/// needs. The median-of-three pivot only changes runtime, never the returned value, which is
/// exactly what a full sort would produce.
func kthLargest(_ v: [Double], _ k: Int) -> Double {
    var a = v
    let target = a.count - k   // k-th largest == (count - k)-th smallest, 0-indexed ascending
    var lo = 0
    var hi = a.count - 1
    while lo < hi {
        let p = quickselectPartition(&a, lo, hi)
        if p == target { return a[p] }
        if p < target { lo = p + 1 } else { hi = p - 1 }
    }
    return a[lo]
}

/// Lomuto partition of `a[lo...hi]` around a median-of-three pivot; returns the pivot's final
/// index once every element left of it is `< pivot` and every element right is `>= pivot`.
/// Median-of-three keeps quickselect off its O(V^2) worst case on already-sorted or
/// reverse-sorted logits (a plain last-element pivot degrades exactly there).
private func quickselectPartition(_ a: inout [Double], _ lo: Int, _ hi: Int) -> Int {
    let mid = lo + (hi - lo) / 2
    if a[mid] < a[lo] { a.swapAt(mid, lo) }
    if a[hi] < a[lo] { a.swapAt(hi, lo) }
    if a[hi] < a[mid] { a.swapAt(hi, mid) }
    a.swapAt(mid, hi)   // move the median (now at `mid`) into the pivot slot
    let pivot = a[hi]
    var i = lo
    for j in lo..<hi where a[j] < pivot {
        a.swapAt(i, j)
        i += 1
    }
    a.swapAt(i, hi)
    return i
}

/// Stable softmax (max-subtracted, `-inf`-safe).
func softmax(_ logits: [Double]) -> [Double] {
    let m = logits.max() ?? 0
    var exps = logits.map { exp($0 - m) }
    let sum = exps.reduce(0, +)
    for i in 0..<exps.count { exps[i] /= sum }
    return exps
}

/// Token ids that would complete a repeated `n`-gram in `prev` — the port of
/// `src/serve/sampling.py`'s `_banned_ngram_tokens`.
func bannedNgramTokens(_ prev: [Int], n: Int, vocabSize: Int) -> [Int] {
    if n < 1 { return [] }
    let n1 = n - 1
    if prev.count < n1 { return [] }
    let prefix = n1 > 0 ? Array(prev[(prev.count - n1)...]) : []
    var banned = Set<Int>()
    for i in 0..<(prev.count - n1) {
        let candidate = n1 > 0 ? Array(prev[i..<(i + n1)]) : []
        if candidate == prefix { banned.insert(prev[i + n1]) }
    }
    return banned.filter { $0 >= 0 && $0 < vocabSize }.sorted()
}
