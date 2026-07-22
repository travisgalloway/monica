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

public enum Packing {

    public struct ShardInfo: Codable, Equatable {
        public let name: String
        public let n_sequences: Int
        public let n_tokens: Int
    }

    public struct Manifest: Codable, Equatable {
        public let seq_len: Int
        public let dtype: String
        public let tokenizer: String
        public let n_documents: Int
        public let n_sequences: Int
        public let n_tokens: Int
        public let shards: [ShardInfo]
    }

    /// Pack per-document token id lists (EOS already appended by the caller) into shards.
    /// `chunkAlign` (set it to the model's `chunk_size`) pads each doc up to a multiple of that
    /// length with `padId` so every doc starts on a chunk boundary (SSM reset, #68).
    @discardableResult
    public static func pack(docs: [[Int]], outDir: URL,
                            seqLen: Int = 8192, shardSizeMB: Int = 512,
                            tokenizer: String = "code",
                            chunkAlign: Int? = nil, padId: Int = 0) throws -> Manifest {
        precondition(seqLen > 0, "seqLen must be positive")
        if let ca = chunkAlign {
            precondition(ca > 0 && seqLen % ca == 0, "seqLen must be a multiple of chunkAlign")
        }
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        let bytesPerToken = 2
        var budget = max(seqLen, (shardSizeMB * (1 << 20)) / bytesPerToken)
        budget -= budget % seqLen

        var tokBuf: [UInt16] = []
        var bndBuf: [UInt8] = []
        var shards: [ShardInfo] = []
        var idx = 0, nDocs = 0, nSeqs = 0, nTokens = 0

        func emit(_ count: Int) throws {
            let name = String(format: "part-%05d", idx)
            var data = Data(); data.reserveCapacity(count * 2)
            for t in 0..<count {
                let v = tokBuf[t]
                data.append(UInt8(v & 0xff)); data.append(UInt8(v >> 8))   // little-endian
            }
            try data.write(to: outDir.appendingPathComponent("\(name).bin"))
            try Data(bndBuf[0..<count]).write(to: outDir.appendingPathComponent("\(name).bounds"))
            let seq = count / seqLen
            shards.append(ShardInfo(name: name, n_sequences: seq, n_tokens: count))
            idx += 1; nSeqs += seq; nTokens += count
            nDocs += bndBuf[0..<count].reduce(0) { $0 + Int($1) }
            tokBuf.removeFirst(count); bndBuf.removeFirst(count)
        }

        for doc in docs {
            if doc.isEmpty { continue }
            var ids = doc
            if let ca = chunkAlign {
                let rem = ids.count % ca
                if rem != 0 { ids += Array(repeating: padId, count: ca - rem) }
            }
            for v in ids {
                precondition(v >= 0 && v <= 0xffff, "token id \(v) out of uint16 range")
                tokBuf.append(UInt16(v))
            }
            bndBuf.append(1)
            if ids.count > 1 { bndBuf.append(contentsOf: Array(repeating: 0, count: ids.count - 1)) }
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
}
