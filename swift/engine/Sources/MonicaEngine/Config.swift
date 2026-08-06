// MambaConfig — the Swift mirror of `src/model/blocks.py`'s dataclass, decoded from the
// `<weights>.safetensors.config.json` sidecar that `src/train/checkpoint.py:save_weights`
// writes (`MambaConfig.to_dict()`, i.e. every field in snake_case).
//
// Only the fields the inference path reads are decoded; `Codable` ignores the rest
// (Muon / torch_compile / fp8_experts / the data-side knobs), so a sidecar from any
// config in the tree decodes without a Swift change.

import Foundation
import MLX

/// `dt_rank` is `Union[int, str]` in Python (`blocks.py:53`) — an explicit int or `"auto"`.
public enum DtRank: Sendable, Equatable {
    case auto
    case value(Int)
}

extension DtRank: Decodable {
    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let i = try? c.decode(Int.self) { self = .value(i); return }
        if let s = try? c.decode(String.self), s == "auto" { self = .auto; return }
        throw DecodingError.dataCorruptedError(
            in: c, debugDescription: "dt_rank must be an Int or the string \"auto\"")
    }
}

public enum ConfigError: Error, CustomStringConvertible {
    case invalid(String)

    public var description: String {
        switch self {
        case .invalid(let m): return "invalid config: \(m)"
        }
    }
}

public struct MambaConfig: Decodable, Sendable {
    // --- core dimensions ---
    public var dModel: Int
    public var nLayers: Int
    public var dState: Int = 16
    public var expand: Int = 2
    public var dConv: Int = 4
    public var headDim: Int = 64
    public var dtRank: DtRank = .auto

    // --- vocab / sequence ---
    public var vocabSize: Int = 50280
    public var seqLen: Int = 1024
    public var tieEmbeddings: Bool = true

    // --- numerics ---
    public var precision: String = "fp32"
    public var chunkSize: Int? = nil
    public var longCtxFactor: Float = 1.0

    // --- hybrid attention (#67) ---
    public var attnEvery: Int? = nil
    public var nAttnHeads: Int? = nil

    // --- sparse MoE (#53) ---
    public var moeEvery: Int? = nil
    public var nExperts: Int = 0
    public var topK: Int = 2
    public var moeDFF: Int? = nil

    // --- dt-projection bias init (LOAD-BEARING; see blocks.py:156-161) ---
    public var dtMin: Float = 1e-3
    public var dtMax: Float = 1e-1
    public var dtInitFloor: Float = 1e-4

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case nLayers = "n_layers"
        case dState = "d_state"
        case expand
        case dConv = "d_conv"
        case headDim = "head_dim"
        case dtRank = "dt_rank"
        case vocabSize = "vocab_size"
        case seqLen = "seq_len"
        case tieEmbeddings = "tie_embeddings"
        case precision
        case chunkSize = "chunk_size"
        case longCtxFactor = "long_ctx_factor"
        case attnEvery = "attn_every"
        case nAttnHeads = "n_attn_heads"
        case moeEvery = "moe_every"
        case nExperts = "n_experts"
        case topK = "top_k"
        case moeDFF = "moe_d_ff"
        case dtMin = "dt_min"
        case dtMax = "dt_max"
        case dtInitFloor = "dt_init_floor"
    }

    // --- derived properties: transliterated from blocks.py:163-209 ---
    public var dInner: Int { expand * dModel }
    public var nHeads: Int { dInner / headDim }

    public var dtRankResolved: Int {
        switch dtRank {
        case .auto: return Int((Double(dModel) / 16.0).rounded(.up))
        case .value(let v): return v
        }
    }

    public var nAttnHeadsResolved: Int { nAttnHeads ?? max(1, dModel / 64) }
    public var attnHeadDim: Int { dModel / nAttnHeadsResolved }
    public var moeDFFResolved: Int { moeDFF ?? dInner }
    public var q: Int { chunkSize ?? 64 }

    public func isAttentionLayer(_ i: Int) -> Bool {
        guard let e = attnEvery, e != 0 else { return false }
        return (i + 1) % e == 0
    }

    /// Attention takes PRECEDENCE — a layer the hybrid claims is never also MoE
    /// (`blocks.py:205-209`).
    public func isMoELayer(_ i: Int) -> Bool {
        guard let e = moeEvery, e != 0 else { return false }
        return (i + 1) % e == 0 && !isAttentionLayer(i)
    }

    /// Count of layers `isMoELayer` selects (`blocks.py:212-213`'s `n_moe_layers`).
    public var nMoeLayers: Int {
        (0..<nLayers).filter(isMoELayer).count
    }

    /// Compute dtype for the heavy GEMMs (`mlx_backend.py:_DTYPES`).
    public var cd: DType {
        switch precision {
        case "fp16": return .float16
        case "bf16": return .bfloat16
        default: return .float32
        }
    }

    /// Only the invariants the Swift inference path can actually violate. Throws
    /// rather than trapping, so a bad sidecar names itself.
    public func validate() throws {
        if nLayers <= 0 { throw ConfigError.invalid("n_layers=\(nLayers) must be >= 1") }
        if dModel <= 0 { throw ConfigError.invalid("d_model=\(dModel) must be >= 1") }
        if !["fp32", "fp16", "bf16"].contains(precision) {
            throw ConfigError.invalid("unknown precision '\(precision)'")
        }
        if dConv < 1 { throw ConfigError.invalid("d_conv must be >= 1") }
        if headDim <= 0 || dInner % headDim != 0 {
            throw ConfigError.invalid(
                "head_dim=\(headDim) must divide d_inner=\(dInner)")
        }
        if let c = chunkSize, c <= 0 {
            throw ConfigError.invalid("chunk_size must be positive or null")
        }
        if longCtxFactor < 1.0 {
            throw ConfigError.invalid("long_ctx_factor must be >= 1.0 (1.0 = off)")
        }
        if let ae = attnEvery {
            if ae <= 0 { throw ConfigError.invalid("attn_every must be a positive int or null") }
            let nah = nAttnHeadsResolved
            if nah <= 0 || dModel % nah != 0 {
                throw ConfigError.invalid(
                    "n_attn_heads=\(nah) must divide d_model=\(dModel)")
            }
            if attnHeadDim % 2 != 0 {
                throw ConfigError.invalid(
                    "attn_head_dim=\(attnHeadDim) must be even (RoPE splits it in half)")
            }
        }
        if let me = moeEvery {
            if me <= 0 { throw ConfigError.invalid("moe_every must be a positive int or null") }
            if nExperts < 2 { throw ConfigError.invalid("n_experts must be >= 2 when moe_every is set") }
            if !(1 <= topK && topK <= nExperts) {
                throw ConfigError.invalid("top_k=\(topK) must be in [1, n_experts=\(nExperts)]")
            }
            if moeDFFResolved <= 0 { throw ConfigError.invalid("moe_d_ff must be positive or null") }
            if nMoeLayers == 0 {
                throw ConfigError.invalid(
                    "moe_every=\(me) selects no layer at n_layers=\(nLayers) (after attention precedence)")
            }
        }
    }

    /// Decode a sidecar written next to a weights file (`<weights>.config.json`).
    public static func load(sidecar url: URL) throws -> MambaConfig {
        let data = try Data(contentsOf: url)
        let cfg = try JSONDecoder().decode(MambaConfig.self, from: data)
        try cfg.validate()
        return cfg
    }
}
