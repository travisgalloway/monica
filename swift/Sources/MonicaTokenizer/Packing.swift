// Native shard packer — replaces the Python code-tokenize+pack step. Emits the exact
// `src/data/shard.py` layout so the Python training loop (`shard.open_shard` / `PackedLoader`)
// reads the output unchanged:
//   part-NNNNN.bin      uint16 little-endian tokens (numpy native-endian on x86/arm64)
//   part-NNNNN.bounds   uint8 doc-start flags (1 at each doc's first token, else 0)
//   manifest.json       {seq_len, dtype, tokenizer, n_documents, n_sequences, n_tokens, shards[]}
//
// Mirrors `pack_sequences` semantics (shard.py:34-113): concatenate per-doc token lists into
// fixed `seqLen` sequences across few large shards; drop the final partial sequence.

import Foundation

/// Raised by `Packing.pack` on bad arguments or data, so a mistyped CLI flag or a mismatched
/// tokenizer artifact surfaces as a catchable, clean error rather than a process trap.
public enum PackingError: Error, CustomStringConvertible {
    case invalidArgument(String)
    case tokenOutOfRange(Int)
    public var description: String {
        switch self {
        case .invalidArgument(let m): return "invalid pack argument: \(m)"
        case .tokenOutOfRange(let v): return "token id \(v) out of uint16 range [0, 65535] " +
            "(is the tokenizer artifact consistent with this data?)"
        }
    }
}

public enum Packing {

    public struct ShardInfo: Codable, Equatable, Sendable {
        public let name: String
        public let n_sequences: Int
        public let n_tokens: Int

        public init(name: String, n_sequences: Int, n_tokens: Int) {
            self.name = name
            self.n_sequences = n_sequences
            self.n_tokens = n_tokens
        }
    }

    public struct Manifest: Codable, Equatable, Sendable {
        public let seq_len: Int
        public let dtype: String
        public let tokenizer: String
        public let n_documents: Int
        public let n_sequences: Int
        public let n_tokens: Int
        public let shards: [ShardInfo]

        public init(seq_len: Int, dtype: String, tokenizer: String,
                    n_documents: Int, n_sequences: Int, n_tokens: Int,
                    shards: [ShardInfo]) {
            self.seq_len = seq_len
            self.dtype = dtype
            self.tokenizer = tokenizer
            self.n_documents = n_documents
            self.n_sequences = n_sequences
            self.n_tokens = n_tokens
            self.shards = shards
        }
    }

    /// Single file entry inside a repository manifest.
    public struct RepoFileEntry: Codable, Equatable, Sendable {
        public let path: String
        public let content: String?
        public let text: String?
        public let tokens: [Int]?

        public init(path: String, content: String? = nil, text: String? = nil, tokens: [Int]? = nil) {
            self.path = path
            self.content = content
            self.text = text
            self.tokens = tokens
        }

        public var fileText: String {
            content ?? text ?? ""
        }
    }

    /// Multi-file repository project manifest (#359).
    public struct RepoProject: Codable, Equatable, Sendable {
        public let repo: String
        public let files: [RepoFileEntry]

        enum CodingKeys: String, CodingKey {
            case repo
            case repoName = "repo_name"
            case files
        }

        public init(repo: String, files: [RepoFileEntry]) {
            self.repo = repo
            self.files = files
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            if let r = try container.decodeIfPresent(String.self, forKey: .repo) {
                self.repo = r
            } else if let r = try container.decodeIfPresent(String.self, forKey: .repoName) {
                self.repo = r
            } else {
                self.repo = "repo"
            }
            self.files = try container.decode([RepoFileEntry].self, forKey: .files)
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(repo, forKey: .repo)
            try container.encode(files, forKey: .files)
        }
    }

    /// Internal representation of a document with boundary state control.
    public struct PackedDoc: Sendable {
        public let tokens: [Int]
        public let isBoundary: Bool

        public init(tokens: [Int], isBoundary: Bool = true) {
            self.tokens = tokens
            self.isBoundary = isBoundary
        }
    }

    /// Pack per-document token id lists (EOS already appended by the caller) into shards.
    /// `chunkAlign` (set it to the model's `chunk_size`) pads each doc up to a multiple of that
    /// length with `padId` so every doc starts on a chunk boundary (SSM reset, #68).
    @discardableResult
    public static func pack(docs: [[Int]], outDir: URL,
                            seqLen: Int = 8192, shardSizeMB: Int = 512,
                            tokenizer: String = "code",
                            chunkAlign: Int? = nil, padId: Int = 0) throws -> Manifest {
        try pack(documents: docs.map { PackedDoc(tokens: $0, isBoundary: true) },
                 outDir: outDir, seqLen: seqLen, shardSizeMB: shardSizeMB,
                 tokenizer: tokenizer, chunkAlign: chunkAlign, padId: padId)
    }

    /// Core packing engine with per-document boundary reset control.
    @discardableResult
    public static func pack(documents: [PackedDoc], outDir: URL,
                            seqLen: Int = 8192, shardSizeMB: Int = 512,
                            tokenizer: String = "code",
                            chunkAlign: Int? = nil, padId: Int = 0) throws -> Manifest {
        guard seqLen > 0 else {
            throw PackingError.invalidArgument("seqLen must be positive, got \(seqLen)")
        }
        if let ca = chunkAlign {
            guard ca > 0 else {
                throw PackingError.invalidArgument("chunkAlign must be positive, got \(ca)")
            }
            guard seqLen % ca == 0 else {
                throw PackingError.invalidArgument(
                    "seqLen \(seqLen) must be a multiple of chunkAlign \(ca)")
            }
        }
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        guard shardSizeMB > 0 else {
            throw PackingError.invalidArgument("shardSizeMB must be positive, got \(shardSizeMB)")
        }
        let bytesPerToken = 2
        // Guard the MB→bytes multiply against Int overflow for an unreasonable --shard-size-mb.
        let (byteBudget, overflow) = shardSizeMB.multipliedReportingOverflow(by: 1 << 20)
        guard !overflow else {
            throw PackingError.invalidArgument("shardSizeMB \(shardSizeMB) is too large")
        }
        var budget = max(seqLen, byteBudget / bytesPerToken)
        budget -= budget % seqLen

        var tokBuf: [UInt16] = []
        var bndBuf: [UInt8] = []
        var shards: [ShardInfo] = []
        var idx = 0, nDocs = 0, nSeqs = 0, nTokens = 0

        func emit(_ count: Int) throws {
            let name = String(format: "part-%05d", idx)
            let binURL = outDir.appendingPathComponent("\(name).bin")
            let boundsURL = outDir.appendingPathComponent("\(name).bounds")

            // Direct buffer write (UInt16 is native little-endian on all target platforms: arm64 & x86_64)
            try tokBuf[0..<count].withUnsafeBufferPointer { ptr in
                let bytePtr = UnsafeRawBufferPointer(start: ptr.baseAddress, count: count * bytesPerToken)
                let data = Data(bytes: bytePtr.baseAddress!, count: count * bytesPerToken)
                try data.write(to: binURL)
            }
            try Data(bndBuf[0..<count]).write(to: boundsURL)
            let seq = count / seqLen
            shards.append(ShardInfo(name: name, n_sequences: seq, n_tokens: count))
            idx += 1; nSeqs += seq; nTokens += count
            nDocs += bndBuf[0..<count].reduce(0) { $0 + Int($1) }
            tokBuf.removeFirst(count); bndBuf.removeFirst(count)
        }

        for doc in documents {
            if doc.tokens.isEmpty { continue }
            var ids = doc.tokens
            if let ca = chunkAlign {
                let rem = ids.count % ca
                if rem != 0 { ids.append(contentsOf: repeatElement(padId, count: ca - rem)) }
            }
            for v in ids {
                guard v >= 0 && v <= 0xffff else { throw PackingError.tokenOutOfRange(v) }
                tokBuf.append(UInt16(v))
            }
            bndBuf.append(doc.isBoundary ? 1 : 0)
            if ids.count > 1 { bndBuf.append(contentsOf: repeatElement(0, count: ids.count - 1)) }
            while tokBuf.count >= budget { try emit(budget) }
        }
        let full = (tokBuf.count / seqLen) * seqLen   // flush remaining complete sequences
        if full > 0 { try emit(full) }

        let manifest = Manifest(seq_len: seqLen, dtype: "uint16", tokenizer: tokenizer,
                                n_documents: nDocs, n_sequences: nSeqs, n_tokens: nTokens,
                                shards: shards)
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(manifest).write(to: outDir.appendingPathComponent("manifest.json"))
        return manifest
    }

    /// Pack topologically sorted files from repository manifests contiguously into large token windows
    /// (32k to 64k tokens, #359).
    ///
    /// Repository metadata tokens `<|repo_name|>` and `<|file_sep|>` delimit the repo and files.
    /// Document boundary state resets (`.bounds` = 1) are placed ONLY at the start of each repository,
    /// suppressing boundary resets between dependent files in the same repository.
    @discardableResult
    public static func packRepos(repos: [RepoProject], tokenizer: Tokenizer,
                                 outDir: URL, seqLen: Int = 32768, shardSizeMB: Int = 512,
                                 chunkAlign: Int? = nil, padId: Int = 0,
                                 fimOptions: FIMOptions? = nil) throws -> Manifest {
        guard seqLen > 0 else {
            throw PackingError.invalidArgument("seqLen must be positive, got \(seqLen)")
        }
        var docs: [PackedDoc] = []
        let eos = tokenizer.eosTokenId
        var fimStats = FIMStats()
        var docIndex = 0

        for repo in repos {
            if repo.files.isEmpty { continue }
            var repoTokens: [Int] = []
            repoTokens.append(contentsOf: tokenizer.encodeRepoHeader(repo: repo.repo))

            for file in repo.files {
                repoTokens.append(contentsOf: tokenizer.encodeFileSeparator(path: file.path))
                if let toks = file.tokens {
                    repoTokens.append(contentsOf: toks)
                } else if let opts = fimOptions, opts.rateBasisPoints > 0 {
                    let fTokens = FIM.transform(document: file.fileText, index: docIndex,
                                                tokenizer: tokenizer, options: opts,
                                                stats: &fimStats)
                    repoTokens.append(contentsOf: fTokens)
                } else {
                    repoTokens.append(contentsOf: tokenizer.encode(file.fileText))
                }
                docIndex += 1
            }
            repoTokens.append(eos)
            // The entire repository is packed with isBoundary: true at its first token,
            // so all internal tokens across dependent files have .bounds = 0.
            docs.append(PackedDoc(tokens: repoTokens, isBoundary: true))
        }

        return try pack(documents: docs, outDir: outDir, seqLen: seqLen,
                        shardSizeMB: shardSizeMB, tokenizer: "code",
                        chunkAlign: chunkAlign, padId: padId)
    }
}
