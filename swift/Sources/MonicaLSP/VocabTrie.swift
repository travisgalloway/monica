// VocabTrie.swift — port of `src/serve/constrained.py`, the portable prefix-constraint
// mechanism #226 built for completion-list logit masking. Pure string/integer work: no
// floating point, no tokenizer dependency (the `decode` closure is supplied by the caller —
// `MonicaTokenizer` in `monica-generate`, a synthetic table in `--self-test`/
// `--emit-mask-parity`), so this file needs no dependency on the tokenizer package.

import Foundation

/// Vocab id -> token string, or `nil` for an id that could not be probed cleanly (empty
/// decode, or decodes to the Unicode replacement character — a byte-level-BPE id that only
/// makes sense as part of a multi-byte sequence with ITS OWN continuation, not this one).
public typealias VocabTable = [String?]

private let replacementChar: Character = "\u{FFFD}"

/// Probe every id in `0..<vocabSize` IN CONTEXT: `decode(anchorIds + [i])` minus the
/// `decode(anchorIds)` prefix. `decode([i])` alone is unsound for byte-level BPE (a lone id
/// can be a partial UTF-8 sequence) — probing in context is the fix. An id whose probed text
/// is empty, non-prefix-consistent, or contains the replacement character is recorded as
/// `nil` and excluded from every allowed set — exclusion is the conservative direction.
public func buildVocabTable(vocabSize: Int, anchorIds: [Int], decode: ([Int]) -> String) -> VocabTable {
    let anchorText = decode(anchorIds)
    var table: VocabTable = Array(repeating: nil, count: vocabSize)
    for i in 0..<vocabSize {
        let probed = decode(anchorIds + [i])
        guard probed.hasPrefix(anchorText) else { continue }
        let piece = String(probed.dropFirst(anchorText.count))
        if piece.isEmpty || piece.contains(replacementChar) { continue }
        table[i] = piece
    }
    return table
}

/// Char-trie over a `VocabTable`'s probeable (non-`nil`) token strings. `prefixMatches(s)`
/// walks `s` once (O(len(s))) and returns every token id whose string is a prefix of `s` —
/// i.e. every token that could legally come next if the remaining text to produce is exactly
/// `s`.
public final class VocabTrie {
    private final class Node {
        var children: [Character: Node] = [:]
        var ids: [Int] = []
    }

    private let root = Node()

    public init(vocab: VocabTable) {
        for (tokenId, piece) in vocab.enumerated() {
            guard let piece, !piece.isEmpty else { continue }
            insert(piece, tokenId)
        }
    }

    private func insert(_ piece: String, _ tokenId: Int) {
        var node = root
        for ch in piece {
            if let next = node.children[ch] {
                node = next
            } else {
                let next = Node()
                node.children[ch] = next
                node = next
            }
        }
        node.ids.append(tokenId)
    }

    /// Every token id whose string is a prefix of `s`, including the zero-length walk (the
    /// root's own `ids`, always empty by construction — no token string is empty).
    public func prefixMatches(_ s: String) -> [Int] {
        var node = root
        var out = node.ids
        for ch in s {
            guard let next = node.children[ch] else { break }
            node = next
            out.append(contentsOf: node.ids)
        }
        return out
    }
}

/// Every vocab id that keeps SOME `label` in `labels` reachable, given the identifier text
/// generated so far (`prefix`). For each label still consistent with `prefix`
/// (`label.hasPrefix(prefix)`), the remaining suffix is unioned in via `trie.prefixMatches`.
/// If `prefix` is ITSELF a complete label, `exitIds` (tokens beginning with a character that
/// cannot continue an identifier) are unioned in too — without this the model can never
/// leave the span once it has spelled a valid name.
public func allowedExtensions(trie: VocabTrie, labels: [String], prefix: String,
                               exitIds: [Int] = []) -> [Int] {
    var out = Set<Int>()
    var prefixIsCompleteLabel = false
    for label in labels {
        guard label.hasPrefix(prefix) else { continue }
        let suffix = String(label.dropFirst(prefix.count))
        if suffix.isEmpty {
            prefixIsCompleteLabel = true
            continue
        }
        out.formUnion(trie.prefixMatches(suffix))
    }
    if prefixIsCompleteLabel {
        out.formUnion(exitIds)
    }
    return out.sorted()
}
