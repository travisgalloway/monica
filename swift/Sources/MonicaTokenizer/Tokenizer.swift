// Public tokenizer API: load a native artifact, encode text → ids, decode ids → text.
// Composes Pretokenizer (split) + BPE (merge). Special-token strings embedded in text are
// split out and mapped to their reserved ids before pre-tokenization.

import Foundation

public final class Tokenizer: @unchecked Sendable {   // immutable after init → safe to share across tasks

    public let bpe: BPE
    public let digitGroup: Int
    public let eosTokenId: Int
    /// (special string, id), longest-first for greedy longest-match splitting.
    let specials: [(text: String, id: Int)]

    public var vocabSize: Int { bpe.vocabSize }

    public init(format: TokenizerFormat) {
        // Fail deterministically on a hand-constructed invalid format (the throwing
        // `init(contentsOf:)` path reports the same message cleanly before reaching here);
        // without this, a bad format would crash with an index-out-of-range inside BPE.init.
        do { try format.validate() } catch { preconditionFailure("\(error)") }
        bpe = BPE(format: format)
        digitGroup = format.digitGroup
        eosTokenId = 0
        specials = format.specialTokens.enumerated()
            .map { (text: $0.element, id: $0.offset) }
            .sorted { $0.text.count > $1.text.count }
    }

    public convenience init(contentsOf url: URL) throws {
        let format = try TokenizerFormat.load(from: url)
        try format.validate()   // deterministic, actionable failure on a corrupt artifact
        self.init(format: format)
    }

    public func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        encode(text, into: &ids)
        return ids
    }

    public func decode(_ ids: [Int]) -> String { bpe.decode(ids) }

    /// Append `text`'s ids to `ids`. Public so `FIM.transform` can assemble a stream.
    public func encode(_ text: String, into ids: inout [Int]) {
        var scratch: [Int] = []
        scratch.reserveCapacity(64)
        encode(text, scratch: &scratch, into: &ids)
    }

    /// Append `text`'s ids to `ids`, using a caller-provided scratch buffer for BPE symbols.
    public func encode(_ text: String, scratch: inout [Int], into ids: inout [Int]) {
        if specials.isEmpty { encodeSegment(text, scratch: &scratch, into: &ids); return }

        // Fast path: if all specials start with '<' and text contains no '<', none can match.
        if !text.utf8.contains(UInt8(ascii: "<")) {
            encodeSegment(text, scratch: &scratch, into: &ids)
            return
        }

        var idx = text.startIndex
        var segStart = idx
        let end = text.endIndex
        while idx < end {
            // Only inspect prefixes if the leading character matches a special token's start ('<')
            if text[idx] == "<" {
                var hit: (text: String, id: Int)? = nil
                for sp in specials where text[idx...].hasPrefix(sp.text) { hit = sp; break }
                if let m = hit {
                    if segStart < idx { encodeSegment(String(text[segStart..<idx]), scratch: &scratch, into: &ids) }
                    ids.append(m.id)
                    idx = text.index(idx, offsetBy: m.text.count)
                    segStart = idx
                    continue
                }
            }
            idx = text.index(after: idx)
        }
        if segStart < end { encodeSegment(String(text[segStart..<end]), scratch: &scratch, into: &ids) }
    }

    /// Encode many documents concurrently (data-parallel across docs; identical on Mac/Linux).
    /// Concurrency is **bounded** to `maxConcurrency` in-flight tasks (default = core count):
    /// a large corpus would otherwise spawn one task per document and pile up memory. Output
    /// order matches input order regardless of completion order.
    public func batchEncode(_ texts: [String],
                            maxConcurrency: Int = ProcessInfo.processInfo.activeProcessorCount) async -> [[Int]] {
        var result = [[Int]](repeating: [], count: texts.count)
        let limit = max(1, maxConcurrency)
        await withTaskGroup(of: (Int, [Int]).self) { group in
            var next = 0
            while next < texts.count && next < limit {           // prime up to `limit` tasks
                let i = next
                group.addTask {
                    var scratch: [Int] = []
                    scratch.reserveCapacity(64)
                    var ids: [Int] = []
                    ids.reserveCapacity(texts[i].utf8.count / 3 + 4)
                    self.encode(texts[i], scratch: &scratch, into: &ids)
                    return (i, ids)
                }
                next += 1
            }
            for await (i, ids) in group {                        // drain, refilling one-for-one
                result[i] = ids
                if next < texts.count {
                    let j = next
                    group.addTask {
                        var scratch: [Int] = []
                        scratch.reserveCapacity(64)
                        var ids: [Int] = []
                        ids.reserveCapacity(texts[j].utf8.count / 3 + 4)
                        self.encode(texts[j], scratch: &scratch, into: &ids)
                        return (j, ids)
                    }
                    next += 1
                }
            }
        }
        return result
    }

    // MARK: - internals

    private func encodeSegment(_ segment: String, scratch: inout [Int], into ids: inout [Int]) {
        for pretoken in Pretokenizer.pretokenize(segment, digitGroup: digitGroup) {
            bpe.encodePretoken(pretoken, syms: &scratch, into: &ids)
        }
    }
}
