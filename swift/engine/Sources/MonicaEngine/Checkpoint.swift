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
    // `public` (not just `internal`) so `monica-parity`'s #196 round-trip section can
    // check a saved file's raw key set for this prefix without duplicating the literal.
    public static let moeBiasPrefix = "moe_route_bias."

    /// Load `<weightsURL>` (safetensors) plus `<weightsURL>.config.json` into a fresh model.
    /// Returns the model and the checkpoint's parameter key set (the caller asserts it
    /// against `model.parameters()` — belt and braces on the highest-risk step).
    ///
    /// Reads the sidecar's optional `quant` block (#168) and, when present, applies
    /// `Quantization.apply` to reshape the fresh model's `Linear`/`Embedding` leaves
    /// into their quantized counterparts BEFORE `loadInto` — see that function's
    /// docstring for why the ordering is load-bearing.
    public static func load(weights weightsURL: URL) throws -> (MonicaModel, Set<String>) {
        let sidecar = URL(fileURLWithPath: weightsURL.path + ".config.json")
        let config = try MambaConfig.load(sidecar: sidecar)
        let model = try MonicaModel(config)
        let spec = try QuantSpec.load(sidecar: sidecar)
        let keys = try loadInto(model, weights: weightsURL, spec: spec)
        return (model, keys)
    }

    /// Load portable weights into an already-built model. Returns the parameter key set.
    ///
    /// `spec`, if given, is applied via `Quantization.apply` BEFORE the weights are read
    /// — reshaping `model`'s `Linear`/`Embedding` leaves into their quantized
    /// counterparts so the checkpoint's packed `.weight`/`.scales`/`.biases` keys have
    /// somewhere real to land. `Quantization.apply` is idempotent (mlx-swift's
    /// `quantizeSingle` is a no-op on an already-`Quantized` module), so it is always
    /// safe to pass `spec` here even if the caller already applied it — e.g. via
    /// `load(weights:)` above, which never applies quantization itself, only decodes
    /// the spec and forwards it.
    @discardableResult
    public static func loadInto(
        _ model: MonicaModel, weights weightsURL: URL, spec: QuantSpec? = nil
    ) throws -> Set<String> {
        if let spec { try Quantization.apply(spec, to: model) }
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

    // --- writer (#196) ------------------------------------------------------------

    /// The portable state dict for `model`: `model.parameters().flattened()` plus, for
    /// each `MoEBlock` whose route bias is ACTIVE, `moe_route_bias.{i}` — the exact
    /// mirror of `_portable_state_dict` (`mlx_backend.py:1043-1061`), including its
    /// "emit only when active" rule. An unconditional key would break Python's
    /// `num_parameters()` accounting (`tests/test_moe.py`, `tests/test_sizing_mlx.py`):
    /// the bias genuinely is not a parameter (see `MoEBlock.routeBias`'s docstring for
    /// why it is deliberately invisible to mlx-swift's parameter reflection), so it
    /// can't ride the `parameters()` flatten and doesn't belong in that count either way.
    public static func portableStateDict(_ model: MonicaModel) -> [String: MLXArray] {
        var out = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (i, layer) in model.layers.enumerated() {
            guard let moe = layer as? MoEBlock, let bias = moe.routeBias else { continue }
            out["\(moeBiasPrefix)\(i)"] = MLXArray(bias)
        }
        return out
    }

    /// Write `model` as a portable checkpoint: `<weightsURL>` (safetensors) plus
    /// `<weightsURL>.config.json` (the sidecar) — the Swift mirror of `save`
    /// (`mlx_backend.py:1029-1032`). The sidecar path is built the same
    /// `path + ".config.json"` way `load` above already does, so the writer and reader
    /// cannot drift apart on the naming convention.
    ///
    /// Atomic: the safetensors file is written to a sibling temp path (kept
    /// `.safetensors`-suffixed — mlx-swift's `save(arrays:...)` dispatches on the URL's
    /// extension and rejects anything else) and moved/replaced into place, mirroring
    /// `checkpoint.py`'s `_atomic_write_bytes` discipline: a reader must never observe a
    /// half-written weights file. `MambaConfig.save(sidecar:)` does its own atomic write
    /// for the sidecar.
    public static func save(_ model: MonicaModel, to weightsURL: URL) throws {
        var sd = portableStateDict(model)
        for (k, v) in sd where v.dtype == .bfloat16 {
            // `safetensors.numpy.load_file` (what `load_weights_dict` uses on the Python
            // side) has no bf16 numpy dtype and would fail on the Python side, far from
            // this call. float16 is fine and preserved as-is — upcasting it to fp32 here
            // would double the file size and change the bytes Python reads back.
            throw EngineError.badCheckpoint(
                "cannot save bfloat16 tensor '\(k)' to the portable format — Python's "
                + "safetensors reader has no bf16 numpy dtype")
        }
        // Realize the lazy graph before handing arrays to mlx-swift's `save`, which
        // needs concrete (not lazily-graphed) arrays.
        MLX.eval(Array(sd.values))

        // Force each tensor into a genuinely independent, host-backed buffer — a real
        // byte-for-byte copy, not merely an evaluated-but-possibly-shared handle. Without
        // this, a parameter whose value was produced in place over a buffer this process
        // still shares with (or aliases the identity of) an earlier load/eval graph node
        // can, in edge cases, have its write dispatch resolve against stale/shared state
        // instead of the model's current values — exactly the failure mode #196's
        // round-trip (b) mutation check guards against ("the writer may be copying the
        // source file rather than serializing the model's actual parameters"). Routing
        // through `asData(access: .copy)` (an explicit host memcpy) and rebuilding a
        // fresh `MLXArray` from those bytes breaks any such aliasing unconditionally.
        sd = sd.mapValues { MLXArray(data: $0.asData(access: .copy)) }

        let fm = FileManager.default
        let dir = weightsURL.deletingLastPathComponent()
        let tmp = dir.appendingPathComponent(".tmp-\(UUID().uuidString).safetensors")
        try MLX.save(arrays: sd, url: tmp)
        if fm.fileExists(atPath: weightsURL.path) {
            _ = try fm.replaceItemAt(weightsURL, withItemAt: tmp)
        } else {
            try fm.moveItem(at: tmp, to: weightsURL)
        }

        let sidecarURL = URL(fileURLWithPath: weightsURL.path + ".config.json")
        try model.config.save(sidecar: sidecarURL)
    }
}
