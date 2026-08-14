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

/// Needed only so `MambaConfig` (which gains `Encodable` for the sidecar WRITER, #196)
/// can round-trip this field. Mirrors the decoder's two-shape contract in reverse.
extension DtRank: Encodable {
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .auto: try c.encode("auto")
        case .value(let v): try c.encode(v)
        }
    }
}

/// A minimal `Any`-Codable for JSON values Swift has no static type for — needed because
/// the sidecar (`<weights>.config.json`) is `dataclasses.asdict(MambaConfig)` on the
/// Python side (EVERY field: Muon / `torch_compile` / `fp8_experts` / data-side knobs /
/// `quant`), while Swift's `MambaConfig` decodes only the ~22 fields the inference path
/// reads. Re-encoding only the known subset on save would silently reset every other
/// field to a Python dataclass default — a quiet corruption of the cross-backend bridge
/// (#196). `JSONValue` is the passthrough vehicle: `MambaConfig.rawSidecar` below stores
/// the WHOLE decoded sidecar as `[String: JSONValue]`, and `save(sidecar:)` overlays the
/// known fields' current values on top of it rather than discarding the rest.
///
/// Order matters: try `Int` BEFORE `Double`, or an integer field (e.g. `"d_model": 64`)
/// decodes as `64.0` and changes the sidecar's JSON type across the round trip.
public enum JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])
}

extension JSONValue: Decodable {
    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let i = try? c.decode(Int.self) { self = .int(i); return }
        if let d = try? c.decode(Double.self) { self = .double(d); return }
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let a = try? c.decode([JSONValue].self) { self = .array(a); return }
        if let o = try? c.decode([String: JSONValue].self) { self = .object(o); return }
        throw DecodingError.dataCorruptedError(
            in: c, debugDescription: "unsupported JSON value")
    }
}

extension JSONValue: Encodable {
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
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

// `Equatable` is synthesized (every stored property — including `DtRank` and the
// `rawSidecar` passthrough — is itself Equatable): #196's round-trip gate uses
// `model.config == modelB.config` as a single check that BOTH the known fields AND the
// raw-JSON passthrough survived a save/load cycle unchanged.
public struct MambaConfig: Decodable, Encodable, Equatable, Sendable {
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

    // --- sidecar fidelity (#196) ---
    // The whole sidecar JSON as decoded by `load(sidecar:)`, or `nil` for a config built
    // in Swift rather than loaded (e.g. a test fixture constructed with the memberwise
    // init). Deliberately has NO CodingKeys case: the compiler-synthesized
    // `init(from:)`/`encode(to:)` for the KNOWN fields below then simply leaves this at
    // its default (never decodes it) and never encodes it — the ~22 modeled fields stay
    // exactly as narrow as before. `save(sidecar:)` is what actually reads/writes it.
    public var rawSidecar: [String: JSONValue]? = nil

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

    /// Decode a sidecar written next to a weights file (`<weights>.config.json`). Decodes
    /// the file TWICE — once into `MambaConfig` (the ~22 known fields), once into
    /// `[String: JSONValue]` (everything) — and keeps the latter in `rawSidecar` so
    /// `save(sidecar:)` can write the whole thing back rather than only what Swift
    /// models. Both decodes read the SAME bytes, so if the first succeeds the second is
    /// not expected to fail; letting it throw here (rather than swallowing with `try?`)
    /// means a case where it somehow could fail is loud, not a silently-lossy `rawSidecar
    /// == nil`.
    public static func load(sidecar url: URL) throws -> MambaConfig {
        let data = try Data(contentsOf: url)
        var cfg = try JSONDecoder().decode(MambaConfig.self, from: data)
        cfg.rawSidecar = try JSONDecoder().decode([String: JSONValue].self, from: data)
        try cfg.validate()
        return cfg
    }

    /// Write this config back out as a sidecar (`<url>`, matching the `<weights>
    /// .config.json` naming `Checkpoint.load`/`save` both use).
    ///
    /// If this config carries a `rawSidecar` (i.e. it came from `load(sidecar:)`), that
    /// full JSON object is written back VERBATIM, overlaid with this config's own
    /// current known-field values — so fields Swift does not model (Muon /
    /// `torch_compile` / `fp8_experts` / data-side knobs / `quant`) survive the round
    /// trip untouched, while a config that WAS mutated in Swift after loading (e.g. by
    /// `Quantization.apply`, which does not touch `MambaConfig` today but a future
    /// caller might) still writes its current values, not the stale originals. Overlay
    /// direction: known fields win over the raw dict for keys present in both.
    ///
    /// If there is no `rawSidecar` (a config built directly in Swift, e.g. in a test),
    /// only the known fields are written — LOSSY BY CONSTRUCTION, since there is no
    /// "everything else" to preserve. Any checkpoint destined to be read back by Python
    /// should carry a `load(sidecar:)`-sourced config.
    ///
    /// Deliberately NOT byte-identical to Python's `json.dumps(cfg, indent=2)`: Python
    /// writes fields in `dataclasses.fields()` order, which this function cannot
    /// reproduce without hardcoding that order (and having it rot the next time a field
    /// is added). The round-trip gate this feeds (`scripts/check_swift_checkpoint.py`)
    /// therefore compares SEMANTIC dict equality, not `cmp` — do not "fix" this into a
    /// byte-for-byte writer.
    public func save(sidecar url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let knownData = try encoder.encode(self)
        var merged: [String: JSONValue]
        if let raw = rawSidecar {
            merged = raw
            let known = try JSONDecoder().decode([String: JSONValue].self, from: knownData)
            for (k, v) in known { merged[k] = v }
        } else {
            merged = try JSONDecoder().decode([String: JSONValue].self, from: knownData)
        }
        let mergedData = try encoder.encode(merged)
        try mergedData.write(to: url, options: .atomic)
    }
}
