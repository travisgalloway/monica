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
        return files.compactMap { try? String(contentsOf: $0, encoding: .utf8) }
    }
    guard let content = try? String(contentsOf: url, encoding: .utf8) else {
        fail("cannot read \(path)")
    }
    if path.hasSuffix(".jsonl") {
        return content.split(separator: "\n", omittingEmptySubsequences: true).compactMap { line in
            guard let d = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any]
            else { return nil }
            return obj["text"] as? String
        }
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
    let vocab = flags["vocab-size"].flatMap { Int($0) } ?? DEFAULT_VOCAB_SIZE
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
    let seqLen = flags["seq-len"].flatMap { Int($0) } ?? 8192
    let shardMB = flags["shard-size-mb"].flatMap { Int($0) } ?? 512
    let chunkAlign = flags["chunk-align"].flatMap { Int($0) }

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
