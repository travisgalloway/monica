// Fill-in-the-middle (FIM) document transform (#215).
//
// WHERE THIS LIVES AND WHY (settled — see docs/design/13-code-model-moe.md, MHM-P2):
// FIM insertion happens at **pack** time, in Swift, not in a Python collator. FIM needs whole
// **documents**, and pack time is the only place they still exist as objects: `src/data/split.py`
// drops the `.bounds` sidecars, so the Python trainer reads a flat `train.bin` through
// `PackedLoader`, which cuts fixed `seq_len` windows and cannot see doc structure. Doing the
// transform here is far smaller than reconstructing boundaries in the loader.
//
// Accepted trade-offs (recorded, not open questions):
//   * Changing the FIM rate requires a **re-pack** of the corpus, not a config edit.
//   * This code sits outside the pytest gate — it is Swift. `monica-selfcheck` is the test runner
//     (the package declares no `.testTarget`), backed by `tests/test_swift_fim.py`, which shells
//     out to the built binary.
//
// DETERMINISM IS A HARD CI REQUIREMENT. `.github/workflows/ci.yml`'s `swift-parity` job `cmp`s
// macOS-packed shards against Linux-packed shards, so every decision here must be bit-identical
// across platforms and toolchain versions. That rules out, deliberately:
//   * `SystemRandomNumberGenerator`, `arc4random`, `Date`, `UUID`, `ProcessInfo` — non-reproducible;
//   * `Int.random(in:using:)` / `shuffled(using:)` / `randomElement(using:)` — the number of
//     `next()` calls and the bias-correction algorithm are stdlib *implementation details*, not
//     ABI, and may change between Swift versions. The bounded draw is implemented here instead,
//     and `SplitMix64` deliberately does NOT conform to `RandomNumberGenerator` so no stdlib
//     convenience can be reached through it by accident;
//   * `Double`/`Float` anywhere in the transform — the rate arrives as integer basis points, and
//     the one Double→Int conversion happens at the CLI boundary (`monica-tokenize`);
//   * iterating a `Set`/`Dictionary` — Swift's `Hasher` is per-process randomly seeded, so their
//     order is not stable even run-to-run on one machine. Arrays only;
//   * `Character`/grapheme-cluster indexing — grapheme breaking depends on the toolchain's Unicode
//     tables, which is exactly the kind of thing that can differ between the macOS and Linux
//     builds. Cut points are UTF-8 **byte** offsets snapped forward with table-free bit arithmetic.
//
// The RNG is seeded **per document** (from the global seed mixed with the document's index), not
// as one stream across the corpus. That makes each document's outcome independent of processing
// order, so the packed output is identical whether `cmdPack` stays serial or later moves to
// `batchEncode`-style bounded concurrency. This is the property that lets the design survive a
// future parallelization without breaking the cross-platform `cmp`.
//
// PSM only. SPM (suffix-prefix-middle) mode is deliberately out of scope for #215 — its absence
// is a scoping decision, not an oversight.

/// SplitMix64 (Steele/Lea/Flood) — a fixed, fully specified integer PRNG. Every operation is an
/// explicitly wrapping 64-bit integer op, so the stream is identical on every platform and Swift
/// version. Intentionally not `RandomNumberGenerator` (see the file header).
public struct SplitMix64 {
    private var state: UInt64

    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Unbiased draw in `[0, bound)` by rejection sampling. Written out here on purpose rather
    /// than calling a stdlib convenience: the rejection threshold and the redraw count are part
    /// of this file's reproducibility contract.
    public mutating func next(upperBound bound: UInt64) -> UInt64 {
        if bound <= 1 { return 0 }
        let threshold = (0 &- bound) % bound   // 2^64 mod bound — the biased tail to reject
        while true {
            let r = next()
            if r >= threshold { return r % bound }
        }
    }
}

/// How `FIM.transform` behaves. `rateBasisPoints` is integer parts-per-10000 (4500 = 0.45);
/// `0` disables the transform entirely, which is the default and keeps `pack` output
/// byte-identical to the pre-#215 pipeline.
public struct FIMOptions {
    public var rateBasisPoints: Int
    public var seed: UInt64
    public var prefixId: Int
    public var suffixId: Int
    public var middleId: Int

    public init(rateBasisPoints: Int = 0, seed: UInt64 = 0,
                prefixId: Int = 1, suffixId: Int = 3, middleId: Int = 2) {
        self.rateBasisPoints = rateBasisPoints
        self.seed = seed
        self.prefixId = prefixId
        self.suffixId = suffixId
        self.middleId = middleId
    }

    public var isEnabled: Bool { rateBasisPoints > 0 }
}

/// Counts for the one-line report `pack` prints. `eligible` is documents that passed the
/// short/sentinel filters and therefore got a rate roll; `transformed` is the subset the roll
/// selected. Skips are reported, never silent — a corpus that silently stopped being FIM'd would
/// be indistinguishable from one that never was.
public struct FIMStats {
    public var eligible: Int = 0
    public var transformed: Int = 0
    public var skippedShort: Int = 0
    public var skippedSentinel: Int = 0
    public init() {}
}

public enum FIM {

    /// The three reserved sentinel ids in *stream* order for PSM: prefix, suffix, middle.
    public static func sentinels(_ o: FIMOptions) -> [Int] { [o.prefixId, o.suffixId, o.middleId] }

    /// Derive this document's RNG seed from the global seed and the document's index. One
    /// SplitMix64 step over a Weyl-shifted index, so neighbouring indices land far apart in the
    /// stream and no document's outcome depends on any other's.
    public static func documentSeed(_ globalSeed: UInt64, _ index: Int) -> UInt64 {
        let idx = UInt64(bitPattern: Int64(index))
        var g = SplitMix64(seed: globalSeed &+ (idx &* 0x9E37_79B9_7F4A_7C15))
        return g.next()
    }

    /// Token ids for one document, FIM-transformed with probability `options.rateBasisPoints`.
    /// EOS is the **caller's** job (matching `cmdPack`), and `Packing.pack` is untouched — that
    /// matters because `tests/test_swift_parquet.py` asserts byte-identical `pack` output.
    ///
    /// PSM layout (the stream order the selfcheck pins as ids `1, 3, 2`):
    ///
    ///     [fim_prefix] encode(prefix) [fim_suffix] encode(suffix) [fim_middle] encode(middle)
    ///
    /// so `decode(prefix) + decode(middle) + decode(suffix)` reassembles the original document.
    public static func transform(document: String, index: Int, tokenizer: Tokenizer,
                                 options: FIMOptions, stats: inout FIMStats) -> [Int] {
        if !options.isEnabled { return tokenizer.encode(document) }

        let bytes = Array(document.utf8)
        if bytes.isEmpty { return [] }

        // Nothing meaningful to split below 3 bytes (an empty prefix AND an empty middle AND an
        // empty suffix is the only reachable shape), so skip rather than emit a degenerate frame.
        if bytes.count < 3 {
            stats.skippedShort += 1
            return tokenizer.encode(document)
        }

        // A document that literally contains one of the reserved special strings would have it
        // split out into a real sentinel id by `Tokenizer.encode` (Tokenizer.swift:70-84), which
        // would corrupt the PSM frame — two `<|fim_prefix|>` ids in one stream, or a sentinel
        // inside the middle span. Skip FIM for that document; never emit a malformed frame.
        // The scan is on raw UTF-8 bytes (not `String.contains`, which compares Characters and
        // therefore drags in grapheme/canonical-equivalence tables) to keep it platform-stable.
        for special in tokenizer.specials where containsSubsequence(bytes, Array(special.text.utf8)) {
            stats.skippedSentinel += 1
            return tokenizer.encode(document)
        }

        stats.eligible += 1
        var rng = SplitMix64(seed: documentSeed(options.seed, index))
        if rng.next(upperBound: 10000) >= UInt64(options.rateBasisPoints) {
            return tokenizer.encode(document)
        }
        stats.transformed += 1

        // Uniform two-point sampling over byte offsets (Bavarian et al.). An empty prefix, middle,
        // or suffix is a legitimate draw and is KEPT — the model must learn both "complete from
        // nothing" and "nothing left to complete". Resampling would distort the distribution.
        let n = UInt64(bytes.count + 1)
        var a = Int(rng.next(upperBound: n))
        var b = Int(rng.next(upperBound: n))
        if a > b { let t = a; a = b; b = t }
        a = snapToScalarBoundary(bytes, a)
        b = snapToScalarBoundary(bytes, b)   // snapping is monotonic, so a <= b still holds

        // Split the document TEXT, not the token ids. Slicing ids instead would make every middle
        // begin exactly on a token boundary, teaching the model it never has to complete a partial
        // token — the classic silent FIM distribution bug.
        let prefix = String(decoding: bytes[0..<a], as: UTF8.self)
        let middle = String(decoding: bytes[a..<b], as: UTF8.self)
        let suffix = String(decoding: bytes[b...], as: UTF8.self)

        var ids: [Int] = []
        ids.reserveCapacity(bytes.count / 2 + 3)
        ids.append(options.prefixId)
        tokenizer.encode(prefix, into: &ids)
        ids.append(options.suffixId)
        tokenizer.encode(suffix, into: &ids)
        ids.append(options.middleId)
        tokenizer.encode(middle, into: &ids)
        return ids
    }

    // MARK: - internals (stdlib-only, table-free)

    /// Smallest index `>= i` that starts a UTF-8 scalar. Continuation bytes match `0b10xxxxxx`;
    /// `count` (one past the end) is always a boundary. Pure bit arithmetic — deliberately not
    /// `String.Index`/`Character` arithmetic, which depends on the toolchain's Unicode tables.
    static func snapToScalarBoundary(_ bytes: [UInt8], _ i: Int) -> Int {
        var j = i
        while j < bytes.count && (bytes[j] & 0xC0) == 0x80 { j += 1 }
        return j
    }

    /// Naive byte-level substring search. The needles are the six ASCII special-token strings and
    /// are short, so this is cheap next to BPE; correctness and platform-stability beat cleverness.
    static func containsSubsequence(_ haystack: [UInt8], _ needle: [UInt8]) -> Bool {
        if needle.isEmpty || needle.count > haystack.count { return false }
        let last = haystack.count - needle.count
        var i = 0
        while i <= last {
            var k = 0
            while k < needle.count && haystack[i + k] == needle[k] { k += 1 }
            if k == needle.count { return true }
            i += 1
        }
        return false
    }
}
