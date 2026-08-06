// MonicaModel — the model shell, `src/model/mlx_backend.py:746-826` and `:924-939`.
//
// embedding -> N residual blocks -> final RMSNorm -> tied LM head (fp32).
//
// `grad_checkpoint` has no mlx-swift equivalent and is a TRAINING concern; `forward` here
// calls the raw layer forwards, which is exactly what Python's `prefill`/`step` already do
// (`mlx_backend.py:812`). See docs/design/14-inference-engine.md.

import MLX
import MLXNN

public final class MonicaModel: Module {
    public let config: MambaConfig

    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "layers") var layers: [Block]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm
    /// Only present when `tie_embeddings` is false. Every config in scope ties, so the
    /// portable checkpoint carries no head tensor (see Checkpoint.swift).
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: MambaConfig) throws {
        try config.validate()
        self.config = config
        self._embedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel)
        // Hybrid (#67): attention blocks replace Mamba blocks at the gated positions;
        // MoE (#53): sparse-FFN blocks likewise. Attention takes precedence on a collision,
        // matching `MambaConfig.isMoELayer`.
        self._layers.wrappedValue = (0..<config.nLayers).map { i -> Block in
            if config.isAttentionLayer(i) { return AttentionBlock(config) }
            if config.isMoELayer(i) { return MoEBlock(config) }
            return MambaBlock(config)
        }
        self._normF.wrappedValue = RMSNorm(config.dModel)
        self._lmHead.wrappedValue = config.tieEmbeddings
            ? nil : Linear(config.dModel, config.vocabSize, bias: false)
        super.init()
    }

    /// `_head` (`:778-784`) — logits run in fp32 (wide-vocab softmax stability) regardless
    /// of the compute dtype.
    func head(_ hIn: MLXArray) -> MLXArray {
        let h = f32(hIn)
        if let lm = lmHead { return lm(h) }
        return matmul(h, embedding.weight.transposed(1, 0))
    }

    /// `forward` (`:787-796`), the `seg_ids == nil` arm. `tokens` is `(B, L)` int32.
    public func forward(_ tokens: MLXArray) -> MLXArray {
        var h = cast(embedding(tokens), config.cd)
        for layer in layers { h = layer.forwardSeq(h) }
        return head(normF(h))
    }

    /// Per-layer hidden states — the embedding output followed by each layer's output
    /// (length `nLayers + 1`), the HF convention. Mirrors `hidden_states` (`:855-867`);
    /// `monica-parity` uses it to localize a divergence to a single block.
    public func hiddenStates(_ tokens: MLXArray) -> [MLXArray] {
        var h = cast(embedding(tokens), config.cd)
        var hs = [h]
        for layer in layers {
            h = layer.forwardSeq(h)
            hs.append(h)
        }
        return hs
    }

    /// `step` (`:820-826`). `token` is `(B,)` int32; `state` has one entry per layer.
    public func step(_ token: MLXArray, _ state: [LayerState]) throws -> (MLXArray, [LayerState]) {
        guard state.count == layers.count else {
            throw EngineError.stateMismatch(
                "got \(state.count) layer states for \(layers.count) layers")
        }
        var h = cast(embedding(token), config.cd)
        var newState: [LayerState] = []
        newState.reserveCapacity(layers.count)
        for (layer, st) in zip(layers, state) {
            let (h2, st2) = try layer.step(h, st)
            h = h2
            newState.append(st2)
        }
        return (head(normF(h)), newState)
    }

    /// `init_state` (`:924-939`).
    public func initState(batch: Int) -> [LayerState] {
        layers.map { $0.initState(batch: batch) }
    }

    /// This model's MoE layers, in layer order (`moe_blocks`, `:884-887`).
    public func moeBlocks() -> [MoEBlock] {
        layers.compactMap { $0 as? MoEBlock }
    }
}
