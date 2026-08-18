// TrainStep — the Swift mirror of `src/model/mlx_train_step.py`'s `make_train_step` +
// `_accumulate_and_step` (#195). Forward+backward+optimizer via mlx-swift's own autodiff
// (`MLXNN.valueAndGrad(model:_:)`), NOT a hand-rolled backward pass — the same relationship
// `MonicaModel.forward` already has to `mlx_backend.py`'s `forward`.
//
// SCOPE (see `.claude/plans/issue-195.md`'s REPRODUCE/DEFER table): pretraining CE only.
// `accumulateAndStep` is factored as the SHARED TAIL exactly as Python's
// `_accumulate_and_step` is, so a future SFT masked-CE step (`make_sft_train_step`,
// DEFERRED to a follow-up) is a small delta — a different `lossAndGrad`, not a rewrite of
// this file. DPO/GRPO and MoE Loss-Free-Balancing are likewise deferred; see the plan.

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// One micro-batch: `inputs`/`targets` are `(B, L)` int32, `targets` shifted by one
/// position relative to `inputs` (the caller's job — mirrors the `(inputs, targets)`
/// tuple `mlx_train_step.py`'s `train_step` receives).
public struct MicroBatch {
    public let inputs: MLXArray
    public let targets: MLXArray

    public init(inputs: MLXArray, targets: MLXArray) {
        self.inputs = inputs
        self.targets = targets
    }
}

/// Mirrors the dict `_accumulate_and_step` returns. `lossScale`/`skipped` are nil/false
/// respectively when `scaler` is nil (fp32/bf16 path) — same shape as Python's dict only
/// gaining those keys under the `if scaler:` branch.
public struct TrainStepResult {
    public let loss: Float
    public let gradNorm: Float
    public let lossScale: Float?
    public let skipped: Bool
}

public typealias LossAndGrad = (MonicaModel, MicroBatch) -> (MLXArray, ModuleParameters)
public typealias TrainStepFn = (MonicaModel, [MicroBatch], Float) -> TrainStepResult

/// `backend.py:96-113`'s MLX `_make_optimizer`, transliterated: MLX `AdamW` holds no
/// parameter refs at construction. Throws loudly on a config asking for Muon or
/// `optimizer_8bit` rather than silently substituting AdamW — both are CUDA-only (#237,
/// #214) and neither has an MLX/Metal port.
public func makeOptimizer(_ model: MonicaModel, baseLR: Float) throws -> AdamW {
    if model.config.optimizer == "muon" {
        throw EngineError.unsupportedOptimizer(
            "Muon is CUDA-only (#237); the MLX Newton-Schulz port is a scoped follow-up, "
            + "not this train step.")
    }
    if model.config.optimizer != "adamw" {
        throw EngineError.unsupportedOptimizer("unknown optimizer '\(model.config.optimizer)'")
    }
    if model.config.optimizer8bit {
        throw EngineError.unsupportedOptimizer(
            "optimizer_8bit is CUDA-only (bitsandbytes has no MLX/Metal port).")
    }
    // mlx-swift's AdamW defaults (betas (0.9, 0.999), eps 1e-8, weightDecay 0.01,
    // biasCorrection false) are verified identical to Python MLX's `optim.AdamW` defaults
    // (`.claude/plans/issue-195.md` P1) — no need to pass them explicitly.
    return AdamW(learningRate: baseLR)
}

/// `_global_grad_norm` (`mlx_train_step.py:21-24`): `sqrt(sum over leaves of sum(g*g))`.
/// Summed in SORTED-KEY order (not `grads.flattened()`'s native order) so the reduction
/// is at least deterministic run-to-run; this does not reproduce Python's own
/// `tree_flatten`-order summation exactly, which is why `grad_norm` sits in the looser
/// `2e-4/1e-6` band rather than the `1e-4/1e-5` fp32 gate (R3 in the plan).
public func globalGradNorm(_ grads: ModuleParameters) -> MLXArray {
    let leaves = grads.flattened().sorted { $0.0 < $1.0 }
    var sq = MLXArray(Float(0))
    for (_, g) in leaves {
        sq = sq + MLX.sum(g * g)
    }
    return MLX.sqrt(sq)
}

/// Add two gradient trees leaf-wise, keyed by flattened name. `ModuleParameters` has no
/// built-in two-tree zip, so this goes through `flattened()`/`unflattened(_:)` (the same
/// round-trip `Checkpoint.loadInto` already uses for a single tree).
func addTrees(_ a: ModuleParameters, _ b: ModuleParameters) -> ModuleParameters {
    let aFlat = a.flattened()
    let bFlat = Dictionary(uniqueKeysWithValues: b.flattened())
    // Python's `tree_map`-based `_add` requires the two trees to match structurally, so
    // mirror that here: a `b` with extra keys beyond `a` must fail loudly rather than
    // silently drop those leaves from the sum (the missing-key case below already fails
    // loudly the other direction).
    guard bFlat.count == aFlat.count else {
        fatalError(
            "accumulateAndStep: gradient tree mismatch — a has \(aFlat.count) keys, "
            + "b has \(bFlat.count)")
    }
    let summed: [(String, MLXArray)] = aFlat.map { key, value in
        guard let bValue = bFlat[key] else {
            fatalError("accumulateAndStep: gradient tree mismatch — missing key '\(key)'")
        }
        return (key, value + bValue)
    }
    return ModuleParameters.unflattened(summed)
}

/// Scale every leaf of a tree by a shared factor — the Swift `_unscale` (`:255-257`).
/// Two overloads (`Float` for the plain `1/n` and `1/scale` cases, `MLXArray` for the
/// clip `factor`, which is itself a graph value) rather than boxing everything into an
/// `MLXArray`, so the common case reads exactly like `tree_map(lambda g: g * factor, ...)`.
func scaleTree(_ a: ModuleParameters, _ factor: Float) -> ModuleParameters {
    a.mapValues { $0 * factor }
}

func scaleTree(_ a: ModuleParameters, _ factor: MLXArray) -> ModuleParameters {
    a.mapValues { $0 * factor }
}

/// The SHARED accumulate -> (unscale) -> clip -> optimizer-step tail —
/// `_accumulate_and_step` (`mlx_train_step.py:27-97`), pretraining/SFT/DPO-agnostic in
/// Python via the `loss_and_grad` parameter; here via `lossAndGrad`. Only the
/// objective-specific piece (`lossAndGrad`) varies between train-step flavors; everything
/// below is identical, so a future SFT masked-CE step reuses this function unchanged.
///
/// `gradClip == 0` disables clipping — mirrors Python's `if grad_clip:` truthiness check
/// exactly (default `1.0`, `0` is the "off" sentinel; MLX's `grad_clip: float = 1.0`
/// signature). A negative `gradClip` is truthy in Python too (clipping stays enabled,
/// producing a negative — nonsensical but reproduced — scale factor), so this checks
/// `!= 0` rather than `> 0` to match that behavior bit-for-bit instead of silently
/// diverging on out-of-range inputs (#293 review).
/// MoE Loss-Free-Balancing (`balancer` in Python) is DEFERRED (see the plan) — this
/// function has no `balancer` parameter at all, matching the deferral rather than
/// threading an always-nil one through.
public func accumulateAndStep(
    model: MonicaModel,
    optimizer: AdamW,
    lossAndGrad: LossAndGrad,
    microBatches: [MicroBatch],
    lr: Float,
    gradClip: Float,
    scaler: DynamicLossScaler?,
    captureGrads: ((ModuleParameters) -> Void)? = nil
) -> TrainStepResult {
    precondition(!microBatches.isEmpty, "accumulateAndStep needs at least one micro-batch")
    let n = microBatches.count

    var accGrads: ModuleParameters? = nil
    var accLoss = MLXArray(Float(0))
    for mb in microBatches {
        let (loss, grads) = lossAndGrad(model, mb)
        accGrads = accGrads == nil ? grads : addTrees(accGrads!, grads)
        accLoss = accLoss + loss
        // Eval each micro-batch, as Python does at `:53` — without it the lazy graph
        // grows across the whole accumulation loop (the same failure `Generator.generate`
        // and `monica-parity`'s decode loops already guard against).
        MLX.eval(accGrads!, accLoss)
    }
    var grads = scaleTree(accGrads!, Float(1.0) / Float(n))
    var loss = accLoss / Float(n)

    let norm: MLXArray
    if let scaler {
        let inv = Float(1.0 / scaler.scale)
        grads = scaleTree(grads, inv)
        loss = loss * inv
        let computedNorm = globalGradNorm(grads)
        MLX.eval(computedNorm)
        let overflow = !MLX.isFinite(computedNorm).item(Bool.self)
        scaler.update(overflow: overflow)
        if overflow {
            // Drop the step — model untouched. `grad_norm` is reported as NaN, not the
            // (also non-finite) computed norm, matching `:66` literally.
            MLX.eval(model.parameters())
            return TrainStepResult(
                loss: loss.item(Float.self), gradNorm: Float.nan,
                lossScale: Float(scaler.scale), skipped: true)
        }
        norm = computedNorm
    } else {
        norm = globalGradNorm(grads)
    }

    // `monica-train`'s fixture gate hooks in here: the PRE-CLIP, post-(un)scale gradient
    // tree is exactly what `train.safetensors`'s `grad.{k}.<param>` was exported to
    // reproduce (`.claude/plans/issue-195.md` P2's "recompute on pristine state" trick, on
    // the Python side). Not exposed via `TrainStepResult` itself — only the fixture gate
    // needs the full tree; every other caller (the free-running `--train` mode, and
    // `makeTrainStep`'s production signature) only needs the summary.
    captureGrads?(grads)

    // R2: hand-rolled to match `min(1.0, grad_clip / (norm + 1e-6))` UNCONDITIONALLY —
    // deliberately NOT `MLXOptimizers.clipGradNorm`, whose strict `totalNorm .< maxNorm`
    // branch differs from Python's form at and just above the clip boundary.
    if gradClip != 0 {
        let factor = MLX.minimum(MLXArray(Float(1.0)), MLXArray(gradClip) / (norm + MLXArray(Float(1e-6))))
        grads = scaleTree(grads, factor)
    }
    optimizer.learningRate = lr
    optimizer.update(model: model, gradients: grads)
    MLX.eval(model, optimizer, loss)

    return TrainStepResult(
        loss: loss.item(Float.self), gradNorm: norm.item(Float.self),
        lossScale: scaler.map { Float($0.scale) }, skipped: false)
}

/// The pretraining objective's `lossAndGrad` — the ONLY piece `make_train_step`
/// (`mlx_train_step.py:111-126`) has that a masked-CE/DPO/GRPO variant would not share.
/// Exposed publicly (not inlined into `makeTrainStep` below) so `monica-train`'s fixture
/// gate can build the exact same gradient-producing closure the production train step
/// uses and drive it through `accumulateAndStep` itself with a `captureGrads` hook —
/// rather than duplicating this closure a second time the way the Python exporter's
/// oracle-recompute trick has to (it cannot reach into `mlx_train_step.py`'s closures).
public func makePretrainLossAndGrad(model: MonicaModel, scaler: DynamicLossScaler?) -> LossAndGrad {
    func lossFn(_ m: MonicaModel, _ inputs: MLXArray, _ targets: MLXArray) -> MLXArray {
        let logits = m.forward(inputs)                          // (B, L, V)
        let v = logits.dim(-1)
        let t = targets.reshaped([-1]).asType(.int32)
        // Cross-entropy in fp32 (wide-vocab softmax stability), reduction "mean" —
        // `nn.losses.cross_entropy(reshape(-1, V).astype(fp32), t, reduction="mean")`.
        // `MLXNN.crossEntropy` is `logSumExp(logits) - takeAlong(logits, targets)`, the
        // same expression Python's cross_entropy uses (Losses.swift:40-70) — unlike the
        // `softmaxLastDim` precedent in Ops.swift, no hand-rolling is needed here.
        let ce = MLXNN.crossEntropy(
            logits: f32(logits).reshaped([-1, v]), targets: t, reduction: .mean)
        return scaler != nil ? ce * Float(scaler!.scale) : ce
    }

    let vg = MLXNN.valueAndGrad(model: model) { m, arrays in
        [lossFn(m, arrays[0], arrays[1])]
    }

    return { m, mb in
        let (losses, grads) = vg(m, [mb.inputs, mb.targets])
        return (losses[0], grads)
    }
}

/// `make_train_step` (`mlx_train_step.py:100-132`) — pretraining CE. `scaler` is nil for
/// fp32/bf16 (toy/smoke); non-nil activates the loss-scale policy + skip control flow.
public func makeTrainStep(
    model: MonicaModel, optimizer: AdamW, gradClip: Float = 1.0,
    scaler: DynamicLossScaler? = nil
) -> TrainStepFn {
    let lossAndGrad = makePretrainLossAndGrad(model: model, scaler: scaler)
    return { m, microBatches, lr in
        accumulateAndStep(
            model: m, optimizer: optimizer, lossAndGrad: lossAndGrad,
            microBatches: microBatches, lr: lr, gradClip: gradClip, scaler: scaler)
    }
}
