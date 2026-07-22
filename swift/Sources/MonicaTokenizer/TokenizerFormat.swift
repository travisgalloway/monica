// The native tokenizer artifact — own (tiktoken-style) JSON schema, versioned. Swift is the
// sole producer/consumer; no HF `tokenizers` compatibility is retained. The on-disk file is
// minimal and fully determines encode: the base 256 bytes are implicit ids
// [specialTokens.count ..< +256], so only the special tokens, the ordered merges (as parent-id
// pairs), and the pre-tokenizer config need to be stored.
//
// Foundation is used here only for Codable/JSON + file I/O (portable subset; present on Linux).

import Foundation

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
}
