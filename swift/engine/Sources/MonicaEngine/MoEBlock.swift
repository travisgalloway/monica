// MoEBlock + Expert — `src/model/mlx_backend.py:578-740`.
//
// pre-norm -> router (softmax over nExperts) -> keep the top_k experts per token,
// renormalize their gates -> weighted sum of those experts' SwiGLU outputs, with a
// residual. POINTWISE over the sequence, so forward and step compute the same function by
// construction and no recurrent state is carried.
//
// #265 ports the two training-side surfaces #166 deliberately left out and #195 deferred
// again: load counting (`_count_loads`/`pop_load`) and the `set_route_bias`/`set_moe_biases`
// WRITE path (model-level plumbing in `MonicaModel.swift`). Load counting does not depend on
// the balancer being active — `set_moe_load_counting` is an independent switch from
// `moe_balance_rate` (`mlx_backend.py:992-996`) — so it is exercised directly by the parity
// harness, not gated behind any config. #217's entropy diagnostic (`pop_routing_stats`/
// `_entropy_sum`) is gated by the SAME Python flag but remains out of scope here.

import MLX
import MLXNN

/// A SwiGLU FFN expert: `down(silu(gate(x)) * up(x))`. Bias-free, matmuls in `cd`.
public final class Expert: Module {
    @ModuleInfo(key: "gate") var gateProj: Linear
    @ModuleInfo(key: "up") var upProj: Linear
    @ModuleInfo(key: "down") var downProj: Linear

    public init(dModel: Int, dFF: Int) {
        self._gateProj.wrappedValue = Linear(dModel, dFF, bias: false)
        self._upProj.wrappedValue = Linear(dModel, dFF, bias: false)
        self._downProj.wrappedValue = Linear(dFF, dModel, bias: false)
        super.init()
    }

    public func callAsFunction(_ xn: MLXArray, _ cd: DType) -> MLXArray {
        linear(downProj, silu(linear(gateProj, xn, cd)) * linear(upProj, xn, cd), cd)
    }
}

/// Per-expert load-count accumulator (#213 D4 / #265), boxed in a plain (non-`Module`,
/// non-`MLXArray`) reference type — deliberately NOT a stored `MLXArray` property on
/// `MoEBlock` itself. mlx-swift discovers parameters by reflecting over stored properties:
/// `ModuleItem.build` maps ANY bare `MLXArray` — including `Optional<MLXArray>` and
/// `[MLXArray]` — to `.value(.parameters)`, which `parameters()`/`update(verify: .all)` then
/// walks; anything else falls through to `.value(.other)` and is invisible to both. That is
/// the same reflection rule that already forces `routeBias` below to be `[Float]?` rather
/// than an `MLXArray?`. A plain class holding the counts sidesteps it without giving up the
/// lazy-graph accumulation Python's `mx.array` gets for free.
final class LoadCounter {
    var counts: MLXArray
    var enabled = false
    init(_ nExperts: Int) { counts = MLXArray.zeros([nExperts]) }
}

public final class MoEBlock: Block {
    let config: MambaConfig

    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "router") var router: Linear
    @ModuleInfo(key: "experts") var experts: [Expert]

    /// Loss-Free-Balancing route bias (#213), held as HOST floats — deliberately NOT an
    /// `MLXArray` property. mlx-swift discovers parameters by reflecting over `MLXArray`
    /// (and `Module`) stored properties, so a `[Float]` is invisible to `parameters()` by
    /// construction. That is the Swift analogue of Python's leading-underscore trick
    /// (`mlx_backend.py:615-636`): the bias must live outside the parameter tree, or
    /// `update(parameters:verify: .all)` would demand a checkpoint key for it and a future
    /// training path would silently differentiate it. `nil` == never activated, which takes
    /// the ORIGINAL pre-#213 ranking path verbatim.
    public private(set) var routeBias: [Float]?

    /// See `LoadCounter` above for why this is boxed rather than a stored `MLXArray`.
    let loadCounter: LoadCounter

    public init(_ config: MambaConfig) {
        self.config = config
        self._norm.wrappedValue = RMSNorm(config.dModel)
        self._router.wrappedValue = Linear(config.dModel, config.nExperts, bias: false)
        let dFF = config.moeDFFResolved
        self._experts.wrappedValue = (0..<config.nExperts).map { _ in
            Expert(dModel: config.dModel, dFF: dFF)
        }
        self.loadCounter = LoadCounter(config.nExperts)
        super.init()
    }

    /// `set_route_bias` (`:638-651`) — read path only (the checkpoint's `moe_route_bias.{i}`).
    /// A wrong-length vector raises here rather than broadcasting into a silently wrong
    /// ranking downstream.
    public func setRouteBias(_ vec: [Float]) throws {
        guard vec.count == config.nExperts else {
            throw EngineError.badCheckpoint(
                "route bias has \(vec.count) entries, expected n_experts=\(config.nExperts)")
        }
        routeBias = vec
    }

    /// `set_load_counting`/`_count_loads` (`mlx_backend.py:689-691`, `:651-655`). Loads
    /// ONLY — #217's entropy diagnostic (`pop_routing_stats`/`_entropy_sum`) is gated by the
    /// same Python flag but is not ported here (out of scope for #265).
    public func setLoadCounting(_ flag: Bool) {
        loadCounter.enabled = flag
    }

    /// `pop_load` (`:693-700`). Unconditional, exactly like Python: returns all-zeros when
    /// counting is off, never `nil`/throws. `MLX.eval` forces the accumulator to a concrete
    /// value first, so the lazy graph never outlives one step — the point of the gate below —
    /// then it is reset to zeros.
    public func popLoad() -> [Float] {
        MLX.eval(loadCounter.counts)
        let out = loadCounter.counts.asType(.float32).asArray(Float.self)
        loadCounter.counts = MLXArray.zeros([config.nExperts])
        return out
    }

    /// `_moe` (`:676-726`).
    func moe(_ xn: MLXArray) -> MLXArray {
        let cd = config.cd
        let e = config.nExperts
        let k = config.topK
        let logits = f32(linear(router, xn, cd))            // (..., E) — route in fp32
        let probs = softmax(logits, axis: -1)               // UNBIASED — always the gate weight
        var gate: MLXArray
        if k < e {
            // Loss-Free-Balancing (#213 D2): when active, rank by the BIASED selection score
            // `logits + routeBias`; the gate weight stays `probs` (unbiased). The bias steers
            // ROUTING only, never the combination weight.
            //
            // The DOUBLE ARGSORT is load-bearing: a plain `probs >= kth` threshold keeps MORE
            // than k experts on ties (uniform routing early in training), breaking the top_k
            // contract and the active-FLOP count. Ranking breaks ties by index, so exactly k
            // survive.
            let ranks: MLXArray
            if let bias = routeBias {
                let sel = logits + MLXArray(bias)
                ranks = argSort(argSort(-sel, axis: -1), axis: -1)
            } else {
                ranks = argSort(argSort(-probs, axis: -1), axis: -1)
            }
            let mask = ranks .< MLXArray(Int32(k))
            gate = which(mask, probs, MLXArray(Float(0)).asType(probs.dtype))
            gate = gate / MLX.sum(gate, axis: -1, keepDims: true)   // renormalize the kept gates
            if loadCounter.enabled {
                // Per-expert load bookkeeping for the balancer (`:765-784`): how many tokens
                // (summed over every non-expert axis) this forward routed to each expert,
                // detached from the graph (`stopGradient` — a count must never become a
                // training signal) and accumulated; `popLoad()` reads + resets it. Counted
                // only here, in the `k < E` branch: at `k == E` every expert takes every
                // token and balancing is vacuous (no mask exists).
                //
                // Two known, benign inexactnesses carried over from Python unchanged (there
                // is no `grad_checkpoint`/training step in this engine to exercise them, but
                // the reasoning is recorded so a future port does not "fix" a divergence that
                // isn't one):
                //  * With `grad_checkpoint: true` each layer's forward is RECOMPUTED in
                //    backward, so every count doubles (#213 D5). Both consumers are invariant
                //    to a uniform positive scale, so this is harmless.
                //  * On an fp16 overflow-skipped training step the counts are not popped, so
                //    they carry into the next pop. Harmless: the routing really did happen.
                let axes = Array(0..<(mask.ndim - 1))
                loadCounter.counts = loadCounter.counts + stopGradient(
                    MLX.sum(mask.asType(.float32), axes: axes))
            }
        } else {
            gate = probs                                    // softmax already sums to 1
        }
        // Every expert is evaluated and the top_k gates select the combination — a DENSE
        // compute with a SPARSE combination. Exact and FLOP-accountable, but not a
        // production MoE kernel; this mirrors the Python semantics and must, for parity.
        let outs = stacked(experts.map { $0(xn, cd) }, axis: -2)        // (..., E, dModel)
        let y = MLX.sum(gate.expandedDimensions(axis: -1) * f32(outs), axis: -2)
        return cast(y, cd)
    }

    /// `forward_seq` (`:728-729`) — pointwise, so `segIds` is accepted (the `Block` seam
    /// requires it, #263) and ignored: nothing here is recurrent or attends across
    /// positions, so packing can't perturb it.
    public override func forwardSeq(_ x: MLXArray, _ segIds: MLXArray?) -> MLXArray {
        x + moe(norm(x))
    }

    /// `step` (`:739-740`) — stateless: the state passes through unchanged.
    public override func step(_ x: MLXArray, _ state: LayerState) throws -> (MLXArray, LayerState) {
        (x + moe(norm(x)), state)
    }

    public override func initState(batch: Int) -> LayerState { .moe }

    /// `forward_prefill` (`:800-806`) — stateless, so this is `forwardSeq` plus the same
    /// `.moe` placeholder `initState`/`step` use. Python must emit a zero-sized tensor pair
    /// to keep its tuple unpack valid; Swift's enum needs no such placeholder.
    public override func forwardPrefill(_ x: MLXArray) -> (MLXArray, LayerState) {
        (x + moe(norm(x)), .moe)
    }
}
