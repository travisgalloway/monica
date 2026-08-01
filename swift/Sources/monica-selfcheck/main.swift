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
