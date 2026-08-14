// CompletionMasker.swift — port of `src/lsp/completion_mask.py`, the #226 glue between a
// completion-list source and `VocabTrie`'s pure trie mechanism.
//
//   - `LabelSource` — where the valid-identifier list for the current cursor comes from.
//     `LspLabels` (a live `TsLspClient`) and `NullLabels` (the M4 control: pays the query
//     cost, applies no mask) mirror the Python reference; `OracleLabels` (reference-derived
//     ceiling) is left to a caller that has reference text, since this package has no eval
//     harness to source one from.
//   - The identifier-span state machine (`CompletionMasker.advance`) — a cheap left-to-right
//     scanner, string/comment-aware via `SourceScan.maskStringsAndComments`, that issues
//     EXACTLY ONE completion-list query per span, not per token — the load-bearing
//     performance decision (masking per-token would put an `update()`/`didChange` on every
//     decode step).
//   - `CompletionMasker.maskFor(_:vocabSize:)` — returns the allowed id set for the NEXT
//     token, or `nil` when no mask applies this step.

import Foundation

public protocol LabelSource: AnyObject {
    /// Called exactly once per span, with the text UP TO AND INCLUDING the anchor (e.g. the
    /// member-access `.`) and the offset generation should resume from. Returns the label
    /// list valid at that cursor, or `nil` to mean "no mask for this span" (the M4 null
    /// contract — see `NullLabels`).
    func query(path: String, text: String, anchorOffset: Int) -> [String]?
}

/// Labels from a live `TsLspClient`: `client.update(path, text)` then
/// `client.completions(path, anchorOffset)`. `nIncompleteLists` counts a truncated list — a
/// silently-treated-as-complete one would be a BLIND failure.
public final class LspLabels: LabelSource {
    private let client: TsLspClient
    private let path: String
    public private(set) var nIncompleteLists = 0

    public init(client: TsLspClient, path: String) {
        self.client = client
        self.path = path
    }

    public func query(path: String, text: String, anchorOffset: Int) -> [String]? {
        do {
            try client.update(path, text: text)
            let items = try client.completions(path, offset: anchorOffset)
            if client.lastCompletionIncomplete { nIncompleteLists += 1 }
            return items.map { $0.label }
        } catch {
            return []
        }
    }
}

/// The M4 null: delegates to `inner` so its query, latency, and counters ALL still happen
/// (the span is real, the cost is real), then returns `nil` — no mask is ever applied. This
/// is what makes "the mask helped" separable from "the arm ran a different code path".
public final class NullLabels: LabelSource {
    private let inner: LabelSource
    public init(inner: LabelSource) { self.inner = inner }
    public func query(path: String, text: String, anchorOffset: Int) -> [String]? {
        _ = inner.query(path: path, text: text, anchorOffset: anchorOffset)
        return nil
    }
}

public final class CompletionMasker {
    public enum MaskScope {
        /// Spans open on `.`/`?.` — the shape of the #194 injection set. Default.
        case member
        /// Additionally opens on any bare identifier at a word boundary. Risks masking
        /// keywords/locals; available but not default.
        case identifier
    }

    private let labelSource: LabelSource
    private let path: String
    private let decode: ([Int]) -> String
    private let encode: ((String) -> [Int])?
    private let maskScope: MaskScope

    private var vocab: VocabTable?
    private var trie: VocabTrie?
    private var exitIds: [Int] = []

    private var processedLen = 0
    private var spanOpen = false
    private var anchorOffset: Int?
    private var labels: [String]?

    public private(set) var nMaskSteps = 0
    public private(set) var nMaskBypass = 0
    public private(set) var nCompletionCalls = 0
    public private(set) var maskWallS: Double = 0

    /// - Parameters:
    ///   - decode: the tokenizer's decode closure (ids -> text). No tokenizer type
    ///     dependency — `MonicaTokenizer` supplies it in `monica-generate`, a synthetic
    ///     table stands in for `--self-test`.
    ///   - encode: optional; without it, vocab probing anchors on the empty context (a
    ///     known-degraded probe, not a crash — mirrors the Python reference).
    public init(labelSource: LabelSource, path: String, decode: @escaping ([Int]) -> String,
                maskScope: MaskScope = .member, encode: ((String) -> [Int])? = nil) {
        self.labelSource = labelSource
        self.path = path
        self.decode = decode
        self.encode = encode
        self.maskScope = maskScope
    }

    private func ensureVocab(_ vocabSize: Int) {
        guard trie == nil else { return }
        // Anchor on a short realistic member-access snippet ("x.") rather than the empty
        // context — `decode([i])` alone is unsound for byte-level BPE (see
        // `src/lsp/lm.py::offset_map`); every probe should happen mid-text, like the spans
        // this masker actually opens on.
        let anchorIds = encode?("x.") ?? []
        let v = buildVocabTable(vocabSize: vocabSize, anchorIds: anchorIds, decode: decode)
        vocab = v
        trie = VocabTrie(vocab: v)
        exitIds = v.enumerated().compactMap { (i, piece) -> Int? in
            guard let piece, let first = piece.first, !SourceScan.isIdentChar(first) else { return nil }
            return i
        }
    }

    private func openSpan(_ chars: [Character], anchorOffset: Int) {
        spanOpen = true
        self.anchorOffset = anchorOffset
        let t0 = Date()
        let prefixText = String(chars[0..<anchorOffset])
        let queried = labelSource.query(path: path, text: prefixText, anchorOffset: anchorOffset)
        maskWallS += Date().timeIntervalSince(t0)
        nCompletionCalls += 1
        labels = queried
    }

    private func closeSpan() {
        spanOpen = false
        anchorOffset = nil
        labels = nil
    }

    private func advance(_ text: String) {
        let chars = Array(text)
        if chars.count < processedLen {
            // Text shrank — shouldn't happen for append-only generation, but never trust
            // stale state over a scan that can't have kept up.
            processedLen = 0
            closeSpan()
        }
        let masked = Array(SourceScan.maskStringsAndComments(text))
        var i = processedLen
        while i < masked.count {
            let ch = masked[i]
            if spanOpen {
                if !SourceScan.isIdentChar(ch) { closeSpan() }
                i += 1
                continue
            }
            if ch == "." {
                openSpan(chars, anchorOffset: i + 1)
            } else if maskScope == .identifier && SourceScan.isIdentStart(ch) {
                let prev: Character = i > 0 ? masked[i - 1] : " "
                if !SourceScan.isIdentChar(prev) { openSpan(chars, anchorOffset: i) }
            }
            i += 1
        }
        processedLen = masked.count
    }

    /// Advance the scanner over `generatedText` (the FULL text generated so far) and return
    /// the allowed id set for the NEXT token, or `nil` if no mask applies this step.
    /// `vocabSize`, when given, primes the (once-built, cached) `VocabTrie` — the caller
    /// passes it from its own already-live logits size.
    ///
    /// Callers wiring this into a sampler MUST union in the model's EOS id(s) before use —
    /// masking must never make generation unstoppable (`src/lsp/masked_decode.py`'s
    /// invariant). That union happens at the call site (`monica-generate`), not here, so
    /// this class stays tokenizer/model-agnostic.
    public func maskFor(_ generatedText: String, vocabSize: Int? = nil) -> [Int]? {
        advance(generatedText)
        guard spanOpen, let labels else { return nil }
        if let vocabSize { ensureVocab(vocabSize) }
        guard let trie, let anchorOffset else { return nil }
        let chars = Array(generatedText)
        let prefix = String(chars[min(anchorOffset, chars.count)...])
        let allowed = allowedExtensions(trie: trie, labels: labels, prefix: prefix, exitIds: exitIds)
        nMaskSteps += 1
        if allowed.isEmpty {
            nMaskBypass += 1
            return nil
        }
        return allowed
    }
}
