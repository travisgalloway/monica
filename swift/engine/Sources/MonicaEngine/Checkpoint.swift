// Portable-checkpoint loading — the Swift mirror of `_load_portable`
// (`src/model/mlx_backend.py:984-1007`) plus the sidecar decode that
// `src/train/checkpoint.py:save_weights` writes.
//
// TWO THINGS A THIRD LOADER GETS WRONG, both handled here:
//
// 1. THE DEPTHWISE CONV WEIGHT LOADS AS-IS. NO TRANSPOSE. The portable format is the
//    MLX-canonical `(outChannels, kernelSize, inChannels/groups)`, and mlx-swift's
//    `MLXNN.Conv1d.weight` is that exact layout. `src/model/cuda_backend.py:908-926`
//    transposes on the way out and back on the way in ONLY because torch's Conv1d is
//    `(out, in/groups, k)`. Copying that transpose here would silently reverse the kernel.
//
// 2. TIED EMBEDDINGS => THERE IS NO `lm_head` TENSOR. Every config in scope ties, so
//    `_portable_state_dict` emits no head weight; the head is `h @ embedding.weight.T`.
//
// `moe_route_bias.{i}` is an extra NON-PARAMETER key (#213 D3), emitted only for an MoE
// layer whose bias was activated. It must be popped BEFORE `update(parameters:)`, which
// only knows the parameter tree.

import Foundation
import MLX
import MLXNN

public enum Checkpoint {
    static let moeBiasPrefix = "moe_route_bias."

    /// Load `<weightsURL>` (safetensors) plus `<weightsURL>.config.json` into a fresh model.
    /// Returns the model and the checkpoint's parameter key set (the caller asserts it
    /// against `model.parameters()` — belt and braces on the highest-risk step).
    public static func load(weights weightsURL: URL) throws -> (MonicaModel, Set<String>) {
        let sidecar = URL(fileURLWithPath: weightsURL.path + ".config.json")
        let config = try MambaConfig.load(sidecar: sidecar)
        let model = try MonicaModel(config)
        let keys = try loadInto(model, weights: weightsURL)
        return (model, keys)
    }

    /// Load portable weights into an already-built model. Returns the parameter key set.
    @discardableResult
    public static func loadInto(_ model: MonicaModel, weights weightsURL: URL) throws -> Set<String> {
        let arrays = try loadArrays(url: weightsURL)

        var biases: [Int: MLXArray] = [:]
        var params: [(String, MLXArray)] = []
        for (k, v) in arrays {
            if k.hasPrefix(moeBiasPrefix) {
                guard let i = Int(k.dropFirst(moeBiasPrefix.count)) else {
                    throw EngineError.badCheckpoint("unparseable MoE bias key '\(k)'")
                }
                biases[i] = v
            } else {
                params.append((k, v))
            }
        }

        // `verify: .all` is the point: it throws on an unknown key, an unused key, AND a
        // shape mismatch, so a silent partial load is impossible.
        let nested = ModuleParameters.unflattened(params)
        try model.update(parameters: nested, verify: .all)

        for (i, vec) in biases.sorted(by: { $0.key < $1.key }) {
            // Check the key names a real MoE layer BEFORE indexing: a balanced checkpoint
            // loaded into a dense or differently-interleaved config would otherwise die on
            // an index error with nothing pointing at the actual mismatch
            // (mirrors mlx_backend.py:1000-1005).
            guard i >= 0, i < model.layers.count,
                  let block = model.layers[i] as? MoEBlock else {
                throw EngineError.badCheckpoint(
                    "checkpoint has \(moeBiasPrefix)\(i), but layer \(i) of this config is "
                    + "not an MoE layer — the weights and the config disagree about the "
                    + "MoE interleave")
            }
            try block.setRouteBias(vec.asType(.float32).asArray(Float.self))
        }

        // `MLX.eval` — realize the lazy parameter graph now, so a load error surfaces here
        // rather than inside the first forward. (Nothing to do with code evaluation.)
        MLX.eval(model.parameters())
        return Set(params.map { $0.0 })
    }
}
