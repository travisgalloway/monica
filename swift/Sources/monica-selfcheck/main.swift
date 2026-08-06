// monica-selfcheck — dependency-free test runner (no XCTest, so it runs on macOS Command Line
// Tools AND Linux identically). Exits non-zero on any failure. This is the cross-platform
// verification gate: run it on both platforms and confirm identical results.

import Foundation
import MonicaTokenizer

var failures: [String] = []
func check(_ cond: Bool, _ msg: String) { if !cond { failures.append(msg) } }
func eq<T: Equatable>(_ a: T, _ b: T, _ msg: String) {
    if a != b { failures.append("\(msg): \(a) != \(b)") }
}

// A small, code-flavored corpus, repeated so BPE has merges to learn.
let SAMPLE: [String] = Array(repeating: [
    "function add(a: number, b: number): number { return a + b; }",
    "const greet = (name: string): string => `hello ${name}`;",
    "interface Point { x: number; y: number; }",
    "export class Vec { constructor(public x: number, public y: number) {} }",
    "for (let i = 0; i < 1000; i++) { total += values[i]; }",
], count: 40).flatMap { $0 }

let SPECIALS = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>",
                "<|fim_suffix|>", "<|fim_pad|>", "<mask>"]

func trained(_ vocab: Int = 2000) -> TokenizerFormat {
    Trainer.train(corpus: SAMPLE, vocabSize: vocab, specialTokens: SPECIALS, digitGroup: 3)
}

func pretok(_ s: String) -> [String] {
    Pretokenizer.pretokenize(s, digitGroup: 3).map { String(decoding: $0, as: UTF8.self) }
}

// MARK: training

// Determinism → regenerable fixtures, identical on any platform.
eq(trained().merges, trained().merges, "training is deterministic")

do {
    let fmt = trained(2000)
    let vocab = SPECIALS.count + 256 + fmt.merges.count
    check(vocab <= 2000, "vocab respects cap (\(vocab) <= 2000)")
    check(vocab <= 65536, "vocab is uint16-packable (\(vocab) <= 65536)")
}

// MARK: vocab-size invariants (#251)
//
// The vocab sweep trains ONCE at the largest size and derives the smaller ones by truncating
// `merges`. That is only sound because `Trainer.train` is greedy with no lookahead, which makes
// a k-vocab merge list a strict prefix of any larger run's. Without a guard that is an
// undocumented assumption a future trainer optimization could silently break, invalidating the
// ratified vocab size.

do {
    // Calibrate off what this corpus can actually produce: `Trainer.train` stops early once no
    // pair repeats, so a hard-coded small size could silently become vacuous (both runs
    // exhausting at the same merge count would make the prefix check trivially true). Asking
    // for exactly half of the big run's merges keeps the check non-vacuous by construction.
    let big = trained(2000)
    let half = big.merges.count / 2
    check(half > 0, "corpus produces enough merges to test the prefix invariant")
    let small = trained(SPECIALS.count + 256 + half)
    eq(small.merges.count, half, "smaller run produces exactly the requested merge count")
    eq(small.merges, Array(big.merges.prefix(half)),
       "merges of a smaller vocab are a strict prefix of a larger one's")

    // A truncated artifact must still be a *valid* artifact — merge m's parents are all
    // < baseOffset + m, so prefixes are self-contained. The sweep writes these to disk.
    let truncated = TokenizerFormat(specialTokens: big.specialTokens, digitGroup: big.digitGroup,
                                    merges: Array(big.merges.prefix(half)))
    do { try truncated.validate() } catch { failures.append("truncated format rejected: \(error)") }

    // Special-token layout is vocab-independent: `baseOffset = specialCount + 256`, so the
    // FIM sentinels keep their ids at every vocab size. True by construction — verified, not
    // assumed, because the sweep compares token streams across sizes.
    eq(small.specialTokens, big.specialTokens, "special tokens identical across vocab sizes")
    let fimProbe = "<|fim_prefix|>x<|fim_suffix|>y<|fim_middle|>"
    let smallIds = Tokenizer(format: small).encode(fimProbe)
    let bigIds = Tokenizer(format: big).encode(fimProbe)
    eq(smallIds.filter { $0 < SPECIALS.count }, [1, 3, 2], "FIM sentinel ids are 1,3,2")
    eq(smallIds.filter { $0 < SPECIALS.count }, bigIds.filter { $0 < SPECIALS.count },
       "FIM sentinel ids identical across vocab sizes")

    // Pretokenization never sees the merges, so pre-token counts are a vocab-independent
    // control in the sweep's per-language table.
    for s in ["const x = 1234567;", "  indented\tline", "function f(a: number) {}"] {
        eq(Pretokenizer.pretokenize(s, digitGroup: small.digitGroup).count,
           Pretokenizer.pretokenize(s, digitGroup: big.digitGroup).count,
           "pre-token count is vocab-independent")
    }
}

// MARK: format validation (a corrupt artifact must fail with an actionable error, not crash)

do {
    do { try trained().validate() } catch { failures.append("valid format rejected: \(error)") }

    func rejects(_ fmt: TokenizerFormat) -> Bool {
        do { try fmt.validate(); return false } catch { return true }
    }
    check(rejects(TokenizerFormat(specialTokens: SPECIALS, digitGroup: 3, merges: [[99999, 0]])),
          "out-of-range merge id is rejected")
    check(rejects(TokenizerFormat(specialTokens: SPECIALS, digitGroup: 3, merges: [[0, 1, 2]])),
          "malformed merge (not a pair) is rejected")
    check(rejects(TokenizerFormat(specialTokens: SPECIALS, digitGroup: 0, merges: [])),
          "non-positive digit_group is rejected")
    check(rejects(TokenizerFormat(version: 2, specialTokens: SPECIALS, digitGroup: 3, merges: [])),
          "unsupported version is rejected")
    check(rejects(TokenizerFormat(specialTokens: ["<mask>", "<|endoftext|>"], digitGroup: 3, merges: [])),
          "EOS not at id 0 is rejected")
}

// MARK: specials

do {
    let tok = Tokenizer(format: trained())
    eq(tok.eosTokenId, 0, "EOS id is 0")
    eq(tok.encode("<|endoftext|>"), [0], "EOS string → [0]")
    eq(tok.encode("<mask>"), [5], "<mask> → [5]")
}

// MARK: round trip

do {
    let tok = Tokenizer(format: trained())
    for s in ["const x = 'π≈3.14'; // 数字",
              "function add(a: number, b: number) { return a + b; }",
              "\t\tif (x) {\n\t\t\treturn 0;\n\t\t}",
              "emoji 🚀 and tabs\t\tand  spaces",
              "prefix <|fim_prefix|> body <mask> end"] {
        eq(tok.decode(tok.encode(s)), s, "round-trip")
    }
    eq(tok.encode(""), [], "empty encodes to []")
    eq(tok.decode([]), "", "empty decodes to \"\"")
}

// MARK: pretokenizer scheme

eq(pretok("1234567"), ["123", "456", "7"], "digit runs split at 3")
eq(pretok("    end"), ["   ", " end"], "indentation run grouped")
eq(pretok("hello world"), ["hello", " world"], "leading space attaches to word")
eq(pretok("it's"), ["it", "'s"], "contraction split")

// MARK: batch encode

do {
    let tok = Tokenizer(format: trained())
    let batched = await tok.batchEncode(SAMPLE)
    eq(batched, SAMPLE.map { tok.encode($0) }, "batchEncode matches serial")
}

// MARK: pack (shard.py-compatible layout)

do {
    let tok = Tokenizer(format: trained())
    let docs = ["function f(x: number) { return x + 1; }", "const y = f(2);"]
    let tokenized = docs.map { d -> [Int] in var i = tok.encode(d); i.append(tok.eosTokenId); return i }
    let dir = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("monica-pack-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }
    let seqLen = 16
    do {
        let m = try Packing.pack(docs: tokenized, outDir: dir, seqLen: seqLen, shardSizeMB: 1)
        eq(m.dtype, "uint16", "pack dtype is uint16")
        eq(m.n_tokens % seqLen, 0, "packed tokens are whole sequences")
        eq(m.n_sequences, m.n_tokens / seqLen, "n_sequences consistent")
        var totalBytes = 0
        for s in m.shards {
            let bin = try Data(contentsOf: dir.appendingPathComponent("\(s.name).bin"))
            let bounds = try Data(contentsOf: dir.appendingPathComponent("\(s.name).bounds"))
            eq(bin.count, s.n_tokens * 2, "shard .bin is 2 bytes/token")
            eq(bounds.count, s.n_tokens, "shard .bounds is 1 byte/token")
            totalBytes += bin.count
        }
        eq(totalBytes, m.n_tokens * 2, "total .bin bytes match manifest")
        check(FileManager.default.fileExists(atPath: dir.appendingPathComponent("manifest.json").path),
              "manifest.json written")
    } catch {
        failures.append("pack threw: \(error)")
    }

    // pack throws (catchable), does not trap, on invalid args / out-of-range token ids.
    func packThrows(_ body: () throws -> Void) -> Bool { do { try body(); return false } catch { return true } }
    check(packThrows { _ = try Packing.pack(docs: [[1]], outDir: dir, seqLen: 0) },
          "pack throws on non-positive seqLen")
    check(packThrows { _ = try Packing.pack(docs: [[70000]], outDir: dir, seqLen: 8) },
          "pack throws on out-of-uint16 token id")
    check(packThrows { _ = try Packing.pack(docs: [[1]], outDir: dir, seqLen: 8, shardSizeMB: 0) },
          "pack throws on non-positive shardSizeMB")
    check(packThrows { _ = try Packing.pack(docs: [[1]], outDir: dir, seqLen: 8, shardSizeMB: Int.max) },
          "pack throws on overflowing shardSizeMB")
}

// MARK: FIM (#215)
//
// StarCoder2 shipped a silent FIM bug — a malformed frame trains fine and only shows up as a
// model that cannot infill. The reassembly assertion below is the guard: it must pass before any
// compute is spent. It extends the tokenizer-level sentinel check above (ids 1,3,2) to the actual
// pack path: transform → decode the three spans → get the original document back.

/// Split a PSM stream into its (prefix, suffix, middle) id runs. `nil` if the frame is malformed,
/// which the caller must report as a failure — never as "no FIM here".
func psmSpans(_ ids: [Int], _ o: FIMOptions) -> (p: [Int], s: [Int], m: [Int])? {
    guard let pi = ids.firstIndex(of: o.prefixId),
          let si = ids.firstIndex(of: o.suffixId),
          let mi = ids.firstIndex(of: o.middleId),
          pi == 0, pi < si, si < mi else { return nil }
    return (Array(ids[1..<si]), Array(ids[(si + 1)..<mi]), Array(ids[(mi + 1)...]))
}

do {
    // --- 1. RNG determinism and a known-answer vector -----------------------------------------
    // The known answers are the *published* SplitMix64 outputs for seed 0, not values harvested
    // from this implementation — an externally-anchored vector is what catches a toolchain or
    // refactor silently altering the stream (which would break the macOS-vs-Linux shard cmp).
    do {
        var r1 = SplitMix64(seed: 42)
        var r2 = SplitMix64(seed: 42)
        var a: [UInt64] = [], b: [UInt64] = []
        for _ in 0..<16 { a.append(r1.next()); b.append(r2.next()) }
        eq(a, b, "SplitMix64 is reproducible from a seed")
        check(Set(a).count > 8, "SplitMix64 stream is not a constant")

        var ref = SplitMix64(seed: 0)
        eq(ref.next(), 0xE220_A839_7B1D_CDAF, "SplitMix64 reference vector, seed 0, draw 1")
        eq(ref.next(), 0x6E78_9E6A_A1B9_65F4, "SplitMix64 reference vector, seed 0, draw 2")
        eq(ref.next(), 0x06C4_5D18_8009_454F, "SplitMix64 reference vector, seed 0, draw 3")

        var bounded = SplitMix64(seed: 7)
        var maxDraw: UInt64 = 0
        for _ in 0..<20000 { maxDraw = max(maxDraw, bounded.next(upperBound: 10000)) }
        check(maxDraw < 10000, "next(upperBound: 10000) stays in range (max \(maxDraw))")
        check(maxDraw > 9000, "next(upperBound: 10000) covers the range (max \(maxDraw))")
        var degenerate = SplitMix64(seed: 7)
        eq(degenerate.next(upperBound: 1), 0, "next(upperBound: 1) is always 0")
        eq(degenerate.next(upperBound: 0), 0, "next(upperBound: 0) is 0, not a division trap")
    }

    let tok = Tokenizer(format: trained())
    let forceAll = FIMOptions(rateBasisPoints: 10000, seed: 1234)

    let fimDocs = [
        "function add(a: number, b: number): number { return a + b; }",
        "const greet = (name: string): string => `hello ${name}`;",
        "export class Vec {\n  constructor(public x: number, public y: number) {}\n}\n",
        "const s = 'π≈3.14'; // 数字 🚀 mixed-width unicode",
        "\t\tif (x) {\n\t\t\treturn 0;\n\t\t}",
        "abc",                                   // exactly the 3-byte minimum
        String(repeating: "let z = 0;\n", count: 40),
    ]

    // --- 2. Round trip: the headline check ----------------------------------------------------
    do {
        var stats = FIMStats()
        for (i, doc) in fimDocs.enumerated() {
            let ids = FIM.transform(document: doc, index: i, tokenizer: tok,
                                    options: forceAll, stats: &stats)
            eq(ids.first, forceAll.prefixId, "FIM stream starts with <|fim_prefix|> (doc \(i))")
            for sentinel in FIM.sentinels(forceAll) {
                eq(ids.filter { $0 == sentinel }.count, 1,
                   "sentinel \(sentinel) appears exactly once (doc \(i))")
            }
            guard let spans = psmSpans(ids, forceAll) else {
                failures.append("malformed PSM frame for doc \(i)")
                continue
            }
            eq(tok.decode(spans.p) + tok.decode(spans.m) + tok.decode(spans.s), doc,
               "prefix+middle+suffix reassembles the original document (doc \(i))")
            let specialsInSpans = (spans.p + spans.m + spans.s).filter { $0 < SPECIALS.count }
            eq(specialsInSpans, [], "no sentinel id leaks inside a span (doc \(i))")
        }
        eq(stats.transformed, fimDocs.count, "rate 1.0 transforms every eligible doc")
        eq(stats.skippedShort, 0, "no fixture doc was skipped as short")
        eq(stats.skippedSentinel, 0, "no fixture doc was skipped as sentinel-bearing")
    }

    // --- 3. The rate is actually honoured -----------------------------------------------------
    // A deliberately loose band: this is a wiring check (is the roll connected to the flag?),
    // not a statistics test, and it must never flake.
    do {
        var stats = FIMStats()
        let synthetic = (0..<200).map { "const value\($0) = compute(\($0)) + offset;" }
        for (i, doc) in synthetic.enumerated() {
            _ = FIM.transform(document: doc, index: i, tokenizer: tok,
                              options: FIMOptions(rateBasisPoints: 5000, seed: 99), stats: &stats)
        }
        eq(stats.eligible, synthetic.count, "every synthetic doc is FIM-eligible")
        check(stats.transformed >= 70 && stats.transformed <= 130,
              "rate 0.5 transforms roughly half (\(stats.transformed)/200)")
    }

    // --- 4. Determinism at the pack level, plus the anti-vacuity guard ------------------------
    // Same seed -> byte-identical shards; different seed -> different tokens. Without the second
    // half, a transform that silently did nothing would pass the identity check vacuously.
    do {
        func packWith(_ options: FIMOptions, into dir: URL) -> Bool {
            var stats = FIMStats()
            var tokenized: [[Int]] = []
            for (i, doc) in SAMPLE.enumerated() {
                var ids = FIM.transform(document: doc, index: i, tokenizer: tok,
                                        options: options, stats: &stats)
                ids.append(tok.eosTokenId)
                tokenized.append(ids)
            }
            // `try?` rather than do/catch: in top-level code the compiler already treats thrown
            // errors as handled, so a `catch` here warns as unreachable. A nil result is the
            // failure signal, and it is reported — never swallowed.
            guard (try? Packing.pack(docs: tokenized, outDir: dir, seqLen: 16, shardSizeMB: 1)) != nil else {
                failures.append("FIM pack failed for seed \(options.seed)")
                return false
            }
            return true
        }

        func artifacts(_ dir: URL) -> [String: Data] {
            var out: [String: Data] = [:]
            let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
            for name in names.sorted() {
                out[name] = (try? Data(contentsOf: dir.appendingPathComponent(name))) ?? Data()
            }
            return out
        }

        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("monica-fim-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let dirA = root.appendingPathComponent("a")
        let dirB = root.appendingPathComponent("b")
        let dirC = root.appendingPathComponent("c")
        let seeded = FIMOptions(rateBasisPoints: 5000, seed: 1234)
        let reseeded = FIMOptions(rateBasisPoints: 5000, seed: 4321)

        if packWith(seeded, into: dirA), packWith(seeded, into: dirB), packWith(reseeded, into: dirC) {
            let a = artifacts(dirA), b = artifacts(dirB), c = artifacts(dirC)
            check(!a.isEmpty, "FIM pack produced artifacts (an empty compare would be vacuous)")
            eq(a.keys.sorted(), b.keys.sorted(), "same-seed packs write the same file set")
            for name in a.keys.sorted() {
                eq(a[name], b[name], "same-seed pack is byte-identical: \(name)")
            }
            check(a["part-00000.bin"] != c["part-00000.bin"],
                  "a different --fim-seed produces different tokens (anti-vacuity)")
        }
    }

    // --- 5. Rate 0 is a true no-op (this is what keeps the pre-#215 pipeline byte-identical) --
    do {
        var stats = FIMStats()
        let off = FIMOptions(rateBasisPoints: 0, seed: 1234)
        for (i, doc) in fimDocs.enumerated() {
            eq(FIM.transform(document: doc, index: i, tokenizer: tok, options: off, stats: &stats),
               tok.encode(doc), "rate 0 returns a plain encode (doc \(i))")
        }
        eq(stats.transformed, 0, "rate 0 transforms nothing")
    }

    // --- 6. Edge cases ------------------------------------------------------------------------
    do {
        var stats = FIMStats()
        eq(FIM.transform(document: "", index: 0, tokenizer: tok, options: forceAll, stats: &stats),
           [], "empty doc yields no tokens and no sentinels")

        for short in ["a", "ab"] {
            let ids = FIM.transform(document: short, index: 1, tokenizer: tok,
                                    options: forceAll, stats: &stats)
            eq(ids, tok.encode(short), "sub-3-byte doc is left unchanged")
        }
        eq(stats.skippedShort, 2, "short docs are counted, not silently dropped")

        // A doc that literally contains a sentinel string would otherwise get a real sentinel id
        // mid-span from `Tokenizer.encode`, corrupting the frame. It must be skipped and counted.
        let hostile = "before <|fim_prefix|> after the sentinel string"
        let ids = FIM.transform(document: hostile, index: 2, tokenizer: tok,
                                options: forceAll, stats: &stats)
        eq(ids, tok.encode(hostile), "sentinel-bearing doc is left unchanged")
        eq(stats.skippedSentinel, 1, "sentinel-bearing docs are counted")

        // Empty prefix / middle / suffix are legitimate draws and must round-trip too.
        for (a, b) in [(0, 0), (0, 3), (3, 3)] {
            let doc = "abc"
            let bytes = Array(doc.utf8)
            var frame: [Int] = [forceAll.prefixId]
            tok.encode(String(decoding: bytes[0..<a], as: UTF8.self), into: &frame)
            frame.append(forceAll.suffixId)
            tok.encode(String(decoding: bytes[b...], as: UTF8.self), into: &frame)
            frame.append(forceAll.middleId)
            tok.encode(String(decoding: bytes[a..<b], as: UTF8.self), into: &frame)
            guard let spans = psmSpans(frame, forceAll) else {
                failures.append("degenerate PSM frame (\(a),\(b)) did not parse")
                continue
            }
            eq(tok.decode(spans.p) + tok.decode(spans.m) + tok.decode(spans.s), doc,
               "degenerate split (\(a),\(b)) still reassembles")
        }

        // Cut points never land inside a UTF-8 scalar: snapping forward is idempotent and
        // monotonic, so every transformed multi-byte-scalar doc must decode losslessly.
        let unicodeDoc = "π≈3.14 数字 🚀 πππ 数数数 🚀🚀🚀 tail"
        for seed in UInt64(0)..<64 {
            var s = FIMStats()
            let out = FIM.transform(document: unicodeDoc, index: 0, tokenizer: tok,
                                    options: FIMOptions(rateBasisPoints: 10000, seed: seed),
                                    stats: &s)
            guard let spans = psmSpans(out, forceAll) else {
                failures.append("unicode doc produced a malformed frame at seed \(seed)")
                continue
            }
            eq(tok.decode(spans.p) + tok.decode(spans.m) + tok.decode(spans.s), unicodeDoc,
               "unicode doc reassembles at seed \(seed)")
        }
    }
}

// MARK: parquet
//
// Covers the pure-Swift Parquet reader (swift/Sources/MonicaTokenizer/Parquet/, #247) on both
// macOS and Linux — the fixture parity test (tests/test_swift_parquet.py) needs Python +
// pyarrow + a built binary in the same environment, which only the macOS CI job has, so this is
// what makes the reader a real gate on `swift-linux` too.

/// Resolve `swift/Fixtures/`: `$MONICA_FIXTURES` override, then `#filePath` (this file is
/// `.../swift/Sources/monica-selfcheck/main.swift` — three `deletingLastPathComponent()`s up is
/// `swift/`), then `./Fixtures` (running from the `swift/` directory). `nil` if none resolves —
/// callers must treat that as a failure, not a skip.
func resolveFixturesDir() -> URL? {
    let fm = FileManager.default
    if let env = ProcessInfo.processInfo.environment["MONICA_FIXTURES"], !env.isEmpty {
        let u = URL(fileURLWithPath: env)
        if fm.fileExists(atPath: u.path) { return u }
    }
    let fromSource = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Sources/monica-selfcheck/ -> drop main.swift
        .deletingLastPathComponent()   // Sources/monica-selfcheck -> Sources/
        .deletingLastPathComponent()   // Sources/ -> swift/
        .appendingPathComponent("Fixtures")
    if fm.fileExists(atPath: fromSource.path) { return fromSource }
    let cwd = URL(fileURLWithPath: "Fixtures")
    if fm.fileExists(atPath: cwd.path) { return cwd }
    return nil
}

if let fixturesDir = resolveFixturesDir() {
    func parquetThrows(_ body: () throws -> Void) -> Bool {
        do { try body(); return false } catch { return true }
    }

    // 7 rows, nulls, single row group, dictionary-encoded, snappy — the exact worked example
    // from the #247 plan (dictionary "a"=0, ""=1, "ccc"=2; def levels 1,0,1,1,1,0,1).
    do {
        let url = fixturesDir.appendingPathComponent("parquet-snappy-dict.parquet")
        do {
            let (values, nulls) = try Parquet.readStringColumn(contentsOf: url)
            eq(values, ["a", "", "ccc", "a", ""], "parquet snappy+dict values")
            eq(nulls, 2, "parquet snappy+dict nullsSkipped")
        } catch {
            failures.append("parquet snappy+dict fixture threw: \(error)")
        }
    }

    // PLAIN encoding, multiple row groups, multiple data pages per row group.
    do {
        let url = fixturesDir.appendingPathComponent("parquet-plain-multipage.parquet")
        do {
            let (values, nulls) = try Parquet.readStringColumn(contentsOf: url)
            eq(values.count, 24, "parquet PLAIN multipage row count")
            eq(nulls, 0, "parquet PLAIN multipage has no nulls")
            if values.count == 24 {
                eq(values[0], "doc 000 " + String(repeating: "x", count: 250),
                   "parquet PLAIN multipage first value")
                eq(values[23], "doc 023 " + String(repeating: "x", count: 250),
                   "parquet PLAIN multipage last value")
            }
        } catch {
            failures.append("parquet PLAIN multipage fixture threw: \(error)")
        }
    }

    // zstd is a named, actionable error — not a silent skip and not a trap.
    do {
        let url = fixturesDir.appendingPathComponent("parquet-zstd.parquet")
        do {
            _ = try Parquet.readStringColumn(contentsOf: url)
            failures.append("parquet zstd fixture: expected a thrown error, got none")
        } catch {
            let msg = "\(error)"
            check(msg.contains("ZSTD"), "parquet zstd error names the codec: \(msg)")
            check(msg.contains("snappy"), "parquet zstd error names the fix: \(msg)")
        }
    }

    // Corrupted input throws (catchable), never traps. Built in memory from a valid fixture's
    // bytes, written to scratch files (the public API reads from a URL, not raw bytes).
    do {
        let validURL = fixturesDir.appendingPathComponent("parquet-snappy-dict.parquet")
        if let validData = try? Data(contentsOf: validURL) {
            let scratch = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("monica-parquet-corrupt-\(UUID().uuidString)")
            try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: scratch) }

            func writeAndCheck(_ name: String, _ bytes: Data, _ desc: String) {
                let f = scratch.appendingPathComponent(name)
                do {
                    try bytes.write(to: f)
                    check(parquetThrows { _ = try Parquet.readStringColumn(contentsOf: f) },
                          "parquet \(desc) throws, not traps")
                } catch {
                    failures.append("could not write corrupted fixture \(name): \(error)")
                }
            }

            // Truncated: drop the last 20 bytes (loses part or all of the footer).
            writeAndCheck("truncated.parquet", validData.dropLast(20), "truncated file")

            // Corrupt the leading PAR1 magic.
            var badHead = validData
            badHead.replaceSubrange(0..<4, with: [0, 0, 0, 0])
            writeAndCheck("bad-head-magic.parquet", badHead, "corrupt head magic")

            // Absurd footer length (bigger than the file).
            var badFooterLen = validData
            let n = badFooterLen.count
            let hugeLen: [UInt8] = [0xff, 0xff, 0xff, 0x7f]   // ~2 GB, far larger than this file
            badFooterLen.replaceSubrange((n - 8)..<(n - 4), with: hugeLen)
            writeAndCheck("absurd-footer-len.parquet", badFooterLen, "absurd footer length")
        } else {
            failures.append("could not read parquet-snappy-dict.parquet to build corrupted variants")
        }
    }
} else {
    failures.append("could not resolve swift/Fixtures (checked $MONICA_FIXTURES, #filePath, ./Fixtures)")
}

// MARK: report

if failures.isEmpty {
    print("monica-selfcheck: OK — all checks passed")
} else {
    for f in failures { FileHandle.standardError.write(Data("FAIL: \(f)\n".utf8)) }
    FileHandle.standardError.write(Data("monica-selfcheck: \(failures.count) failure(s)\n".utf8))
    exit(1)
}
