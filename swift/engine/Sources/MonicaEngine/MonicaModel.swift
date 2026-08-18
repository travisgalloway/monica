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
    // `layers`/`normF` are `public` (unlike `embedding`, which no external caller needs)
    // so #196's round-trip section in `monica-parity` — a separate module — can walk
    // `MoEBlock`s for the route-bias check and mutate `normF.weight` for the
    // anti-file-copy check, without adding new API surface to `MonicaModel` itself.
    @ModuleInfo(key: "layers") public var layers: [Block]
    @ModuleInfo(key: "norm_f") public var normF: RMSNorm
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
    ///
    /// Uses `embedding.asLinear(h)` rather than `matmul(h, embedding.weight.transposed(1,
    /// 0))` (#168). For the plain `Embedding` base class `asLinear` IS exactly that
    /// matmul, so this is a no-op for the fp32 path — but under `MLXNN.quantize`,
    /// `embedding` becomes a `QuantizedEmbedding` whose `weight` is the PACKED uint32
    /// codes, not float weights; the old matmul would silently compute nonsense on that
    /// tensor, while `asLinear` dispatches to `QuantizedEmbedding`'s override
    /// (`quantizedMM`) automatically. Since the tied head shares `embedding`'s weight,
    /// quantizing the embedding quantizes the head too — this is what makes that work.
    func head(_ hIn: MLXArray) -> MLXArray {
        let h = f32(hIn)
        if let lm = lmHead { return lm(h) }
        return embedding.asLinear(h)
    }

    /// `forward` (`:787-796`). `segIds` (B, L) document ids, `nil` by default, packs
    /// multiple documents into one sequence and masks each block's recurrent/attention
    /// state so it can't bleed across a boundary (#68/#263). `tokens` is `(B, L)` int32.
    /// These are plain `final`-in-effect concrete methods (not the overridable `Block`
    /// seam), so a default-nil parameter is safe here — see `LayerState.swift`.
    public func forward(_ tokens: MLXArray, segIds: MLXArray? = nil) -> MLXArray {
        var h = cast(embedding(tokens), config.cd)
        for layer in layers { h = layer.forwardSeq(h, segIds) }
        return head(normF(h))
    }

    /// Per-layer hidden states — the embedding output followed by each layer's output
    /// (length `nLayers + 1`), the HF convention. Mirrors `hidden_states` (`:855-867`);
    /// `monica-parity` uses it to localize a divergence to a single block.
    public func hiddenStates(_ tokens: MLXArray, segIds: MLXArray? = nil) -> [MLXArray] {
        var h = cast(embedding(tokens), config.cd)
        var hs = [h]
        for layer in layers {
            h = layer.forwardSeq(h, segIds)
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

    /// `prefill` (#169, `:867-887`) — one parallel scan over the whole prompt, returning
    /// `(logits, state)` where `state` is exactly what `nLayers` walks of `step` over
    /// `tokens` would have left from a fresh `initState`. `lastOnly` slices the head input
    /// to the final position BEFORE the vocab projection, skipping the (B, L, V) logits
    /// materialization for positions 1..<L-1 — the point at poc scale (V >= 50k).
    ///
    /// Fresh-session-only and no `seg_ids`, mirroring #165's seam contract
    /// (`src/model/interface.py:81-95`): the attention RoPE positions are seeded from 0, so
    /// a prompt may not be appended to a session that has already consumed tokens, and there
    /// is no packing-aware carry-masking here. Walks `layers` directly (not through any
    /// grad-checkpoint closure — that is a training concern; prefill is inference, matching
    /// Python's `prefill` which also bypasses `_layer_fns`).
    public func prefill(_ tokens: MLXArray, lastOnly: Bool = false) -> (MLXArray, [LayerState]) {
        precondition(
            tokens.ndim == 2 && tokens.dim(1) > 0,
            "prefill: tokens must be rank-2 (B, L) with L > 0, got shape \(tokens.shape)")
        var h = cast(embedding(tokens), config.cd)
        var state: [LayerState] = []
        state.reserveCapacity(layers.count)
        for layer in layers {
            let (h2, st) = layer.forwardPrefill(h)
            h = h2
            state.append(st)
        }
        h = normF(h)
        if lastOnly {
            let l = h.dim(1)
            h = h[0..., (l - 1)..<l, 0...].squeezed(axis: 1)   // (B, dModel)
        }
        return (head(h), state)
    }

    /// This model's MoE layers, in layer order (`moe_blocks`, `:884-887`).
    public func moeBlocks() -> [MoEBlock] {
        layers.compactMap { $0 as? MoEBlock }
    }

    // MARK: - Loss-Free-Balancing accessors (#213 / #265, `mlx_backend.py:963-1002`)

    /// `set_moe_biases` (`:969-983`) — push per-layer route-bias vectors into each MoE
    /// block, in layer order. A layer-count mismatch throws rather than `zip`-truncating:
    /// leaving some routers on the old bias while others take the new one is a
    /// partially-activated routing state that would be very hard to notice mid-run.
    /// Per-vector length is checked by `MoEBlock.setRouteBias`.
    public func setMoeBiases(_ biases: [[Float]]) throws {
        let blocks = moeBlocks()
        guard biases.count == blocks.count else {
            throw EngineError.badCheckpoint(
                "got \(biases.count) bias vectors for \(blocks.count) MoE layers")
        }
        for (block, vec) in zip(blocks, biases) {
            try block.setRouteBias(vec)
        }
    }

    /// `moe_biases` (`:985-990`) — per-MoE-layer route biases as host floats, in layer
    /// order (an inactive block reports `[]`, `routeBias ?? []`).
    public func moeBiases() -> [[Float]] {
        moeBlocks().map { $0.routeBias ?? [] }
    }

    /// `set_moe_load_counting` (`:992-996`) — enable/disable per-expert load counting on
    /// every MoE block. Off by default; the harness/training wiring turns it on explicitly.
    public func setMoeLoadCounting(_ flag: Bool) {
        for block in moeBlocks() { block.setLoadCounting(flag) }
    }

    /// `pop_moe_load` (`:998-1002`) — per-MoE-layer accumulated expert-load counts since
    /// the last pop, in layer order, resetting each block's accumulator. `[]` when the
    /// model has no MoE layers (dense/hybrid-only).
    public func popMoeLoad() -> [[Float]] {
        moeBlocks().map { $0.popLoad() }
    }

    // MARK: - interpretability accessors (`mixing_matrices`/`verify_block`, #264/#100/#52)

    /// `mixing_matrices` (`mlx_backend.py:938-950`, corrected guard): each Mamba layer's
    /// head-averaged mixing matrix `(B, L, L)` paired with its ABSOLUTE layer index.
    /// Attention AND MoE layers are skipped — `layer as? MambaBlock` is the natural Swift
    /// expression of that skip (an `AttentionBlock`/`MoEBlock` simply fails the cast).
    /// Computed on the layer's INPUT before advancing `h`: computing after the advance
    /// would pair layer `i`'s index with layer `i+1`'s input — same shape, wrong values,
    /// the off-by-one hazard #264 flags.
    public func mixingMatrices(_ tokens: MLXArray) -> [(Int, MLXArray)] {
        var h = cast(embedding(tokens), config.cd)
        var out: [(Int, MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            if let mamba = layer as? MambaBlock {
                out.append((i, mamba.mixingMatrix(h).mean(axis: 1)))   // head-average -> (B,L,L)
            }
            h = layer.forwardSeq(h, nil)
        }
        return out
    }

    /// `verify_block` (`mlx_backend.py:897-920`, #52's speculative-decoding prerequisite):
    /// consumes `tokens` (HOST-side ids) through the `step` recurrence from `state`,
    /// returning the per-token next-token logits and per-token states in a SINGLE
    /// `MLX.eval`. `states[k]` is the state after consuming `tokens[0...k]`, so a caller
    /// can roll back to an accepted prefix without recomputing — the wall-clock win comes
    /// from one graph eval, not per-token launch/sync. Empty `tokens` returns `([], [])`.
    ///
    /// Throws `EngineError.stateMismatch` if any incoming state leaf's batch is not 1 —
    /// `step`'s own `MLXArray([Int32(tok)])` token is shape `(1,)`, so this rejects
    /// nothing a well-formed single-sequence `state` would pass; it just fails legibly
    /// instead of via a broadcast error deep inside `step`. Checks EVERY leaf of every
    /// `LayerState` (`.arrays`, not just `.arrays.first`) — a mamba/attention state
    /// carries two leaves (conv+ssm, k+v), and a mismatch confined to the second leaf
    /// must not slip past this guard and surface later as a less legible broadcast/shape
    /// error inside `step`.
    public func verifyBlock(_ tokens: [Int], _ state: [LayerState])
        throws -> (logits: [MLXArray], states: [[LayerState]])
    {
        for st in state {
            for leaf in st.arrays where leaf.dim(0) != 1 {
                throw EngineError.stateMismatch(
                    "verifyBlock: state batch must be 1, got \(leaf.dim(0))")
            }
        }
        guard !tokens.isEmpty else { return ([], []) }
        var logitsList: [MLXArray] = []
        var stateList: [[LayerState]] = []
        var hState = state
        for tok in tokens {
            let tokArr = MLXArray([Int32(tok)])
            let (logit, newState) = try step(tokArr, hState)
            logitsList.append(logit)
            stateList.append(newState)
            hState = newState
        }
        // One eval realizes the whole block (logits + every intermediate state) together.
        var leaves = logitsList
        leaves += stateList.flatMap { $0.flatMap(\.arrays) }
        MLX.eval(leaves)
        return (logitsList, stateList)
    }
}
