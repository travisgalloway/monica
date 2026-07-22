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
                let i = next; group.addTask { (i, self.encode(texts[i])) }; next += 1
            }
            for await (i, ids) in group {                        // drain, refilling one-for-one
                result[i] = ids
                if next < texts.count {
                    let j = next; group.addTask { (j, self.encode(texts[j])) }; next += 1
                }
            }
        }
        return result
    }

    // MARK: - internals

    private func encode(_ text: String, into ids: inout [Int]) {
        if specials.isEmpty { encodeSegment(text, into: &ids); return }
        var idx = text.startIndex
        var segStart = idx
        let end = text.endIndex
        while idx < end {
            var hit: (text: String, id: Int)? = nil
            for sp in specials where text[idx...].hasPrefix(sp.text) { hit = sp; break }
            if let m = hit {
                if segStart < idx { encodeSegment(String(text[segStart..<idx]), into: &ids) }
                ids.append(m.id)
                idx = text.index(idx, offsetBy: m.text.count)
                segStart = idx
            } else {
                idx = text.index(after: idx)
            }
        }
        if segStart < end { encodeSegment(String(text[segStart..<end]), into: &ids) }
    }

    private func encodeSegment(_ segment: String, into ids: inout [Int]) {
        for pretoken in Pretokenizer.pretokenize(segment, digitGroup: digitGroup) {
            bpe.encodePretoken(pretoken, into: &ids)
        }
    }
}
