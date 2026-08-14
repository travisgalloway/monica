// Quantization (#168) — decode the `quant` sidecar block written by
// `src/eval/quantize.quantize_portable_state_dict` (via `save_weights(..., quant=...)`)
// and reshape a freshly-built `MonicaModel`'s `Linear`/`Embedding` leaves into their
// `QuantizedLinear`/`QuantizedEmbedding` mlx-swift counterparts so the checkpoint's
// packed `weight`/`scales`/`biases` load straight into `update(parameters:verify: .all)`
// — the same "quantize (from placeholder weights) then overwrite with the real packed
// arrays" flow mlx-swift-lm uses to load quantized checkpoints.

import Foundation
import MLX
import MLXNN

/// The `quant` sidecar block: which module paths were quantized, at what group size and
/// bit-width, and with which affine variant. Mirrors `quantize_portable_state_dict`'s
/// `quant_block` (`src/eval/quantize.py`) byte-for-byte — only `mode`/`group_size`/
/// `targets` are ever written there, so this struct decodes exactly those three keys
/// (no separate top-level `bits`; each target already carries its own).
public struct QuantSpec: Decodable, Sendable {
    public let mode: String
    public let groupSize: Int
    /// module path (NOT parameter name — no trailing `.weight`) -> bits.
    public let targets: [String: Int]

    enum CodingKeys: String, CodingKey {
        case mode
        case groupSize = "group_size"
        case targets
    }

    /// Decode the `quant` key of `<weights>.config.json` (the same sidecar
    /// `MambaConfig.load` reads), or `nil` if that key is absent — the fp-checkpoint
    /// case. A malformed `quant` object throws rather than silently loading as fp
    /// (mirrors the sidecar's other "known keys only, unknowns throw" contract).
    public static func load(sidecar url: URL) throws -> QuantSpec? {
        struct Wrapper: Decodable {
            let quant: QuantSpec?
        }
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Wrapper.self, from: data).quant
    }
}

public enum Quantization {
    /// Reshape `model`'s `Linear`/`Embedding` leaves named in `spec.targets` into their
    /// quantized counterparts, IN PLACE. Must run BEFORE `Checkpoint.loadInto` — the
    /// checkpoint's flat key set includes `{path}.scales`/`{path}.biases` only for
    /// quantized paths, and `update(parameters:verify: .all)` demands an EXACT match
    /// against the model's current parameter tree, so the module swap has to happen
    /// first or the key-set comparison is meaningless (a quantized checkpoint loaded
    /// into a still-all-fp model would report every `.scales`/`.biases` key as
    /// "unexpected" and every packed `.weight` as a shape mismatch).
    ///
    /// The quantize-then-overwrite two-step (this function only does step one; step two
    /// is `Checkpoint.loadInto`'s ordinary `update(parameters:)`) recomputes bogus
    /// scales/biases from the model's placeholder random-init weights — that's fine,
    /// because every quantized array gets fully overwritten immediately after by the
    /// checkpoint's real packed values. Throws if any named path does not resolve to a
    /// leaf module, or resolves but is not `Quantizable` (e.g. a name typo, or a config
    /// mismatch between the checkpoint and the model being built) — a spec that
    /// silently quantizes nothing must never read green.
    public static func apply(_ spec: QuantSpec, to model: MonicaModel) throws {
        guard spec.mode == "affine" else {
            throw EngineError.badCheckpoint(
                "unsupported quant mode '\(spec.mode)' — only 'affine' is implemented")
        }
        guard spec.groupSize > 0 else {
            throw EngineError.badCheckpoint("quant spec group_size must be positive")
        }

        var applied = Set<String>()
        MLXNN.quantize(
            model: model,
            filter: { path, module in
                guard let bits = spec.targets[path] else { return nil }
                guard module is Quantizable else { return nil }
                applied.insert(path)
                return (groupSize: spec.groupSize, bits: bits, mode: .affine)
            })

        let missing = Set(spec.targets.keys).subtracting(applied)
        if !missing.isEmpty {
            throw EngineError.badCheckpoint(
                "quant spec names path(s) that did not quantize (not found as a leaf "
                + "Quantizable module, or already quantized): \(missing.sorted())")
        }
    }
}
