// monica-tokenize — native CLI for the Monica code tokenizer.
//
//   monica-tokenize train  --in <corpus> --out <tokenizer.json> [--vocab-size 49152]
//   monica-tokenize encode --tokenizer <tokenizer.json> [--in <file>] [--json]
//   monica-tokenize decode --tokenizer <tokenizer.json> [--in <file>]
//   monica-tokenize pack   --tokenizer <tokenizer.json> --in <parquet|jsonl|txt|dir> --out <dir>
//                          [--seq-len 8192] [--shard-size-mb 512] [--chunk-align N]
//   monica-tokenize stats  --tokenizer <tokenizer.json> --in <jsonl> [--json]
//
// `--in` reads stdin when omitted (encode/decode). `train`/`pack` corpus = a `.parquet` file or
// a directory of `.parquet` shards (reads the `text` column directly, #247 — only
// UNCOMPRESSED/SNAPPY-compressed shards are supported, never zstd), a directory of source files
// (one doc each), a `.jsonl` of {"text": ...} rows, or a single text file (one doc).

import Foundation
import MonicaTokenizer

// The reserved special tokens (ids 0..5). Kept in lockstep with the retired Python trainer's
// SPECIAL_TOKENS so the model's vocab layout is unchanged: EOS first, then FIM, then <mask>.
let SPECIAL_TOKENS = [
    "<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>",
    "<|fim_suffix|>", "<|fim_pad|>", "<mask>",
]
// Ratified 2026-08-04 by the #251 sweep (see docs/design/13-code-model-moe.md). 16384 was sized
// when #193 was scoped TypeScript-only; on the #198 multilingual corpus it costs 7.6% of overall
// compression and 11.4% on Markdown against 49152. Still under the 65536 uint16 packing cap.
let DEFAULT_VOCAB_SIZE = 49152
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

/// Load documents for train/pack. parquet (file or dir of shards) → the `text` column (#247);
/// jsonl → each row's "text"; dir → each source file; else one doc.
func readDocs(_ path: String) -> [String] {
    let url = URL(fileURLWithPath: path)
    var isDir: ObjCBool = false
    FileManager.default.fileExists(atPath: path, isDirectory: &isDir)
    if isDir.boolValue {
        let exts: Set<String> = ["ts", "tsx", "js", "jsx", "py", "txt", "md", "json", "swift", "parquet"]
        var files: [URL] = []
        if let en = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil) {
            for case let f as URL in en where exts.contains(f.pathExtension) { files.append(f) }
        }
        // Sort for a stable, deterministic doc order — `enumerator` traversal order is not
        // guaranteed, and for `pack` that would make shard output nondeterministic.
        files.sort { $0.path < $1.path }
        // Parquet-wins: a directory of corpus shards may also contain a stray `manifest.json`
        // or similar — if any `.parquet` file is present, read only those (mirrors Python's
        // `_shard_paths`/`has_jsonl_shards` precedence). Directories with no Parquet are
        // byte-for-byte unchanged from before #247.
        let parquetFiles = files.filter { $0.pathExtension == "parquet" }
        if !parquetFiles.isEmpty {
            var docs: [String] = []
            var totalNulls = 0
            for f in parquetFiles {
                do {
                    let (d, nulls) = try Parquet.readStringColumn(contentsOf: f)
                    docs.append(contentsOf: d)
                    totalNulls += nulls
                } catch { fail("cannot read Parquet \(f.path): \(error)") }
            }
            if totalNulls > 0 { warn("skipped \(totalNulls) null `text` row(s) under \(path)") }
            return docs
        }
        var docs: [String] = []
        var skipped = 0
        for f in files {
            if let t = try? String(contentsOf: f, encoding: .utf8) { docs.append(t) } else { skipped += 1 }
        }
        if skipped > 0 { warn("skipped \(skipped) unreadable file(s) under \(path)") }
        return docs
    }
    if path.hasSuffix(".parquet") {
        do {
            let (docs, nulls) = try Parquet.readStringColumn(contentsOf: url)
            if nulls > 0 { warn("skipped \(nulls) null `text` row(s) in \(path)") }
            return docs
        } catch { fail("cannot read Parquet \(path): \(error)") }
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
    // Fail fast on any non-integer token — silently dropping it would decode the wrong text
    // (a real hazard when debugging pack/unpack), so a stray character is an error, not a skip.
    let ids = readInput(flags).split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "," })
        .map { field -> Int in
            guard let id = Int(field) else { fail("decode: non-integer token id '\(field)'") }
            return id
        }
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

// MARK: - stats

/// One language's accumulated measurement. `bytes` is raw UTF-8 in, `tokens` is what `pack`
/// would actually write (encode + the appended EOS), so bytes/token is the real packed ratio
/// rather than an idealized encode.
struct LangStats {
    var docs = 0
    var bytes = 0
    var tokens = 0
    var pretokens = 0
    var maxTokenId = 0

    mutating func add(bytes b: Int, ids: [Int], pretokens p: Int) {
        docs += 1
        bytes += b
        tokens += ids.count + 1        // + EOS, matching cmdPack
        pretokens += p
        for id in ids where id > maxTokenId { maxTokenId = id }
    }

    var json: [String: Any] {
        ["docs": docs, "bytes": bytes, "tokens": tokens, "pretokens": pretokens,
         "max_token_id": maxTokenId,
         "bytes_per_token": tokens > 0 ? Double(bytes) / Double(tokens) : 0]
    }
}

/// Read `{"text": ..., "lang": ...}` JSONL for `stats`. Deliberately separate from `readDocs`:
/// `train`/`pack` depend on that function's exact behavior, and `stats` needs the language tag
/// it discards. Rows without a `lang` land in "untagged" rather than being dropped silently.
func readTaggedDocs(_ path: String) -> [(text: String, lang: String)] {
    guard let content = try? String(contentsOfFile: path, encoding: .utf8) else {
        fail("stats: cannot read --in file \(path)")
    }
    var docs: [(text: String, lang: String)] = []
    var skipped = 0
    for line in content.split(separator: "\n", omittingEmptySubsequences: true) {
        guard let d = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
              let text = obj["text"] as? String else { skipped += 1; continue }
        docs.append((text, (obj["lang"] as? String) ?? "untagged"))
    }
    if skipped > 0 { warn("skipped \(skipped) malformed/text-less JSONL line(s) in \(path)") }
    return docs
}

/// Pre-token counts, computed with the same bounded-concurrency shape as `batchEncode`.
/// `Pretokenizer` never sees the merges, so this number must be identical across vocab sizes —
/// which is exactly why the sweep measures it (a mismatch means something real broke).
func pretokenCounts(_ docs: [String], digitGroup: Int) async -> [Int] {
    var out = [Int](repeating: 0, count: docs.count)
    let limit = max(1, ProcessInfo.processInfo.activeProcessorCount)
    await withTaskGroup(of: (Int, Int).self) { group in
        var next = 0
        while next < docs.count && next < limit {
            let i = next
            group.addTask { (i, Pretokenizer.pretokenize(docs[i], digitGroup: digitGroup).count) }
            next += 1
        }
        for await (i, n) in group {
            out[i] = n
            if next < docs.count {
                let j = next
                group.addTask { (j, Pretokenizer.pretokenize(docs[j], digitGroup: digitGroup).count) }
                next += 1
            }
        }
    }
    return out
}

func cmdStats(_ flags: [String: String]) async {
    let tok = loadTokenizer(flags)
    guard let inPath = flags["in"] else { fail("stats: --in <jsonl> is required") }
    let tagged = readTaggedDocs(inPath)
    if tagged.isEmpty { fail("stats: no documents read from \(inPath)") }

    let texts = tagged.map { $0.text }
    let idsPerDoc = await tok.batchEncode(texts)
    let pretoks = await pretokenCounts(texts, digitGroup: tok.digitGroup)

    var byLang: [String: LangStats] = [:]
    var overall = LangStats()
    for (i, doc) in tagged.enumerated() {
        let b = doc.text.utf8.count
        byLang[doc.lang, default: LangStats()].add(bytes: b, ids: idsPerDoc[i], pretokens: pretoks[i])
        overall.add(bytes: b, ids: idsPerDoc[i], pretokens: pretoks[i])
    }

    if flags["json"] != nil {
        let payload: [String: Any] = [
            "vocab_size": tok.vocabSize,
            "overall": overall.json,
            "by_lang": byLang.mapValues { $0.json },
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload,
                                                     options: [.prettyPrinted, .sortedKeys]) else {
            fail("stats: could not serialize results")
        }
        print(String(decoding: data, as: UTF8.self))
    } else {
        func line(_ name: String, _ s: LangStats) -> String {
            let pad = name.padding(toLength: max(12, name.count), withPad: " ", startingAt: 0)
            let bpt = s.tokens > 0 ? Double(s.bytes) / Double(s.tokens) : 0
            return "  \(pad) \(s.docs) docs, \(s.bytes) bytes, \(s.tokens) tokens, "
                + String(format: "%.4f bytes/token", bpt)
        }
        print("vocab_size \(tok.vocabSize)")
        for lang in byLang.keys.sorted() { print(line(lang, byLang[lang]!)) }
        print(line("overall", overall))
        print("  max_token_id \(overall.maxTokenId), pretokens \(overall.pretokens)")
    }
}

// MARK: - dispatch

let argv = Array(CommandLine.arguments.dropFirst())
guard let cmd = argv.first else {
    fail("usage: monica-tokenize <train|encode|decode|pack|stats> [flags]")
}
let flags = parseFlags(Array(argv.dropFirst()))
switch cmd {
case "train":  cmdTrain(flags)
case "encode": cmdEncode(flags)
case "decode": cmdDecode(flags)
case "pack":   cmdPack(flags)
case "stats":  await cmdStats(flags)
default:       fail("unknown subcommand '\(cmd)' (train|encode|decode|pack|stats)")
}
