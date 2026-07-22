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
}

// MARK: report

if failures.isEmpty {
    print("monica-selfcheck: OK — all checks passed")
} else {
    for f in failures { FileHandle.standardError.write(Data("FAIL: \(f)\n".utf8)) }
    FileHandle.standardError.write(Data("monica-selfcheck: \(failures.count) failure(s)\n".utf8))
    exit(1)
}
