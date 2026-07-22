// monica-tokenize — native CLI for the Monica code tokenizer.
//
//   monica-tokenize train  --in <corpus> --out <tokenizer.json> [--vocab-size 16384]
//   monica-tokenize encode --tokenizer <tokenizer.json> [--in <file>] [--json]
//   monica-tokenize decode --tokenizer <tokenizer.json> [--in <file>]
//   monica-tokenize pack   --tokenizer <tokenizer.json> --in <jsonl|txt> --out <dir>
//                          [--seq-len 8192] [--shard-size-mb 512] [--chunk-align N]
//
// `--in` reads stdin when omitted (encode/decode). `train` corpus = a directory of source
// files (one doc each), a `.jsonl` of {"text": ...} rows, or a single text file (one doc).

import Foundation
import MonicaTokenizer

// The reserved special tokens (ids 0..5). Kept in lockstep with the retired Python trainer's
// SPECIAL_TOKENS so the model's vocab layout is unchanged: EOS first, then FIM, then <mask>.
let SPECIAL_TOKENS = [
    "<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>",
    "<|fim_suffix|>", "<|fim_pad|>", "<mask>",
]
let DEFAULT_VOCAB_SIZE = 16384
let DEFAULT_DIGIT_GROUP = 3

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    exit(1)
}

func warn(_ msg: String) {
    FileHandle.standardError.write(Data("warning: \(msg)\n".utf8))
}

/// Minimal `--flag value` parser. Bare flags (no following value) map to "".
func parseFlags(_ args: [String]) -> [String: String] {
    var out: [String: String] = [:]
    var i = 0
    while i < args.count {
        let a = args[i]
        guard a.hasPrefix("--") else { i += 1; continue }
        let key = String(a.dropFirst(2))
        if i + 1 < args.count && !args[i + 1].hasPrefix("--") {
            out[key] = args[i + 1]; i += 2
        } else {
            out[key] = ""; i += 1
        }
    }
    return out
}

func readStdin() -> String {
    String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
}

/// An integer flag, or `def` when absent. Fails fast on a present-but-non-integer value
/// (e.g. `--seq-len foo`) rather than silently falling back to the default.
func intFlag(_ flags: [String: String], _ name: String, default def: Int) -> Int {
    guard let raw = flags[name], !raw.isEmpty else { return def }
    guard let v = Int(raw) else { fail("--\(name) must be an integer, got '\(raw)'") }
    return v
}

/// Text from `--in <file>` (failing fast if it can't be read — never a silent empty
/// string that would tokenize the wrong input), or stdin when `--in` is absent.
func readInput(_ flags: [String: String]) -> String {
    guard let path = flags["in"] else { return readStdin() }
    guard let text = try? String(contentsOfFile: path, encoding: .utf8) else {
        fail("cannot read --in file \(path)")
    }
    return text
}

/// Load documents for train/pack. jsonl → each row's "text"; dir → each source file; else one doc.
func readDocs(_ path: String) -> [String] {
    let url = URL(fileURLWithPath: path)
    var isDir: ObjCBool = false
    FileManager.default.fileExists(atPath: path, isDirectory: &isDir)
    if isDir.boolValue {
        let exts: Set<String> = ["ts", "tsx", "js", "jsx", "py", "txt", "md", "json", "swift"]
        var files: [URL] = []
        if let en = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil) {
            for case let f as URL in en where exts.contains(f.pathExtension) { files.append(f) }
        }
        // Sort for a stable, deterministic doc order — `enumerator` traversal order is not
        // guaranteed, and for `pack` that would make shard output nondeterministic.
        files.sort { $0.path < $1.path }
        var docs: [String] = []
        var skipped = 0
        for f in files {
            if let t = try? String(contentsOf: f, encoding: .utf8) { docs.append(t) } else { skipped += 1 }
        }
        if skipped > 0 { warn("skipped \(skipped) unreadable file(s) under \(path)") }
        return docs
    }
    guard let content = try? String(contentsOf: url, encoding: .utf8) else {
        fail("cannot read \(path)")
    }
    if path.hasSuffix(".jsonl") {
        // Skip is deterministic (same input -> same result), but not silent: report the count
        // so a malformed corpus doesn't quietly change the training set unnoticed.
        var docs: [String] = []
        var skipped = 0
        for line in content.split(separator: "\n", omittingEmptySubsequences: true) {
            if let d = line.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
               let text = obj["text"] as? String {
                docs.append(text)
            } else {
                skipped += 1
            }
        }
        if skipped > 0 { warn("skipped \(skipped) malformed/text-less JSONL line(s) in \(path)") }
        return docs
    }
    return [content]
}

func loadTokenizer(_ flags: [String: String]) -> Tokenizer {
    guard let path = flags["tokenizer"] else { fail("--tokenizer <tokenizer.json> is required") }
    do { return try Tokenizer(contentsOf: URL(fileURLWithPath: path)) }
    catch { fail("failed to load tokenizer \(path): \(error)") }
}

// MARK: - subcommands

func cmdTrain(_ flags: [String: String]) {
    guard let inPath = flags["in"] else { fail("train: --in <corpus> is required") }
    guard let outPath = flags["out"] else { fail("train: --out <tokenizer.json> is required") }
    let vocab = intFlag(flags, "vocab-size", default: DEFAULT_VOCAB_SIZE)
    let minVocab = SPECIAL_TOKENS.count + 256   // specials + the 256 base bytes
    guard vocab >= minVocab else {
        fail("--vocab-size must be >= \(minVocab) (specials + 256 base bytes), got \(vocab)")
    }
    guard vocab <= 65536 else {
        fail("--vocab-size must be <= 65536 for the uint16 packing path, got \(vocab)")
    }
    let docs = readDocs(inPath)
    if docs.isEmpty { fail("train: no documents read from \(inPath)") }
    let fmt = Trainer.train(corpus: docs, vocabSize: vocab,
                            specialTokens: SPECIAL_TOKENS, digitGroup: DEFAULT_DIGIT_GROUP)
    do { try fmt.save(to: URL(fileURLWithPath: outPath)) }
    catch { fail("train: cannot write \(outPath): \(error)") }
    let vocabSize = SPECIAL_TOKENS.count + 256 + fmt.merges.count
    print("trained \(vocabSize) tokens (\(fmt.merges.count) merges, \(SPECIAL_TOKENS.count) special) -> \(outPath)")
}

func cmdEncode(_ flags: [String: String]) {
    let tok = loadTokenizer(flags)
    let ids = tok.encode(readInput(flags))
    if flags["json"] != nil {
        // ids are non-negative Ints -> always valid JSON; build the array directly (no throwing).
        print("[" + ids.map(String.init).joined(separator: ",") + "]")
    } else {
        print(ids.map(String.init).joined(separator: " "))
    }
}

func cmdDecode(_ flags: [String: String]) {
    let tok = loadTokenizer(flags)
    let ids = readInput(flags).split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "," })
        .compactMap { Int($0) }
    print(tok.decode(ids), terminator: "")
}

func cmdPack(_ flags: [String: String]) {
    let tok = loadTokenizer(flags)
    guard let inPath = flags["in"] else { fail("pack: --in <jsonl|txt> is required") }
    guard let outPath = flags["out"] else { fail("pack: --out <dir> is required") }
    let seqLen = intFlag(flags, "seq-len", default: 8192)
    let shardMB = intFlag(flags, "shard-size-mb", default: 512)
    let chunkAlign: Int?
    if let raw = flags["chunk-align"], !raw.isEmpty {
        guard let v = Int(raw) else { fail("--chunk-align must be an integer, got '\(raw)'") }
        chunkAlign = v
    } else {
        chunkAlign = nil
    }

    let docs = readDocs(inPath)
    let eos = tok.eosTokenId
    let tokenized = docs.map { doc -> [Int] in
        var ids = tok.encode(doc); ids.append(eos); return ids
    }
    do {
        let m = try Packing.pack(docs: tokenized, outDir: URL(fileURLWithPath: outPath),
                                 seqLen: seqLen, shardSizeMB: shardMB,
                                 tokenizer: "code", chunkAlign: chunkAlign)
        print("packed \(m.n_sequences) seq x \(seqLen) (\(m.n_tokens) tokens, \(m.shards.count) shard(s)) -> \(outPath)")
    } catch { fail("pack: \(error)") }
}

// MARK: - dispatch

let argv = Array(CommandLine.arguments.dropFirst())
guard let cmd = argv.first else {
    fail("usage: monica-tokenize <train|encode|decode|pack> [flags]")
}
let flags = parseFlags(Array(argv.dropFirst()))
switch cmd {
case "train":  cmdTrain(flags)
case "encode": cmdEncode(flags)
case "decode": cmdDecode(flags)
case "pack":   cmdPack(flags)
default:       fail("unknown subcommand '\(cmd)' (train|encode|decode|pack)")
}
