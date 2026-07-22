// The native tokenizer artifact — own (tiktoken-style) JSON schema, versioned. Swift is the
// sole producer/consumer; no HF `tokenizers` compatibility is retained. The on-disk file is
// minimal and fully determines encode: the base 256 bytes are implicit ids
// [specialTokens.count ..< +256], so only the special tokens, the ordered merges (as parent-id
// pairs), and the pre-tokenizer config need to be stored.
//
// Foundation is used here only for Codable/JSON + file I/O (portable subset; present on Linux).

import Foundation

/// Raised when a loaded `tokenizer.json` is structurally invalid, so failures are
/// deterministic and actionable instead of an index-out-of-range crash deep in `BPE.init`.
public enum TokenizerError: Error, CustomStringConvertible {
    case invalidFormat(String)
    public var description: String {
        switch self { case .invalidFormat(let m): return "invalid tokenizer format: \(m)" }
    }
}

public struct TokenizerFormat: Codable, Equatable {
    public var version: Int
    /// Reserved special tokens, ids 0 ..< count. Index 0 is EOS/document separator.
    public var specialTokens: [String]
    /// Digit pre-token cap (o200k-style ≤ 3).
    public var digitGroup: Int
    /// Ordered BPE merges as `[leftId, rightId]` parent pairs; merge m → token id
    /// `specialTokens.count + 256 + m`.
    public var merges: [[Int]]

    enum CodingKeys: String, CodingKey {
        case version
        case specialTokens = "special_tokens"
        case digitGroup = "digit_group"
        case merges
    }

    public init(version: Int = 1, specialTokens: [String], digitGroup: Int, merges: [[Int]]) {
        self.version = version
        self.specialTokens = specialTokens
        self.digitGroup = digitGroup
        self.merges = merges
    }

    public func save(to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(self).write(to: url)
    }

    public static func load(from url: URL) throws -> TokenizerFormat {
        try JSONDecoder().decode(TokenizerFormat.self, from: Data(contentsOf: url))
    }

    /// Structural invariants a `BPE` relies on. A corrupt/hand-edited artifact fails here with
    /// an actionable message rather than crashing later. Each merge `m` produces id
    /// `specialTokens.count + 256 + m`, so its two parent ids must reference only tokens
    /// defined before it (< that id).
    public func validate() throws {
        guard digitGroup > 0 else {
            throw TokenizerError.invalidFormat("digit_group must be positive, got \(digitGroup)")
        }
        guard !specialTokens.isEmpty else {
            throw TokenizerError.invalidFormat("special_tokens must be non-empty")
        }
        let baseOffset = specialTokens.count + 256
        for (m, pair) in merges.enumerated() {
            guard pair.count == 2 else {
                throw TokenizerError.invalidFormat("merge \(m) must have exactly 2 ids, got \(pair.count)")
            }
            let ceiling = baseOffset + m
            for id in pair where id < 0 || id >= ceiling {
                throw TokenizerError.invalidFormat(
                    "merge \(m) references out-of-range id \(id) (valid range 0..<\(ceiling))")
            }
        }
    }
}
