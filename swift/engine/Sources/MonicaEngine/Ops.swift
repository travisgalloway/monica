// Elementwise / helper ops — transliterations of `src/model/mlx_backend.py:36-107` and
// `:458-484`. Each function names the Python line range it mirrors; keep them in step.

import Foundation
import MLX
import MLXNN

// MARK: - activations and precision plumbing (mlx_backend.py:36-76)

/// `_silu` (`:36-37`) — written out rather than using `MLXNN.silu` so it is the same
/// expression as Python's `x * sigmoid(x)`.
@inlinable
public func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// `_softplus` (`:40-42`) — `log(1 + exp(x))` via `logaddexp(x, 0)`. NOT a naive
/// `log(1 + exp(x))`: at poc scale the naive form overflows.
@inlinable
public func softplus(_ x: MLXArray) -> MLXArray {
    logAddExp(x, MLXArray.zeros(like: x))
}

/// `_f32` (`:57-59`) — upcast to fp32, returning `t` unchanged when already fp32.
@inlinable
public func f32(_ t: MLXArray) -> MLXArray {
    t.dtype == .float32 ? t : t.asType(.float32)
}

/// `_cast` (`:62-65`) — cast to compute dtype `cd`, unchanged when already `cd`, so the
/// fp32 path stays the original op verbatim.
@inlinable
public func cast(_ t: MLXArray, _ cd: DType) -> MLXArray {
    t.dtype == cd ? t : t.asType(cd)
}

/// `_linear` (`:68-76`) — `Linear` with BOTH operands cast to `cd`. fp32 routes to the
/// layer call verbatim (an fp16 @ fp32 matmul would promote back to fp32, so casting one
/// side is not enough).
@inlinable
public func linear(_ layer: Linear, _ x: MLXArray, _ cd: DType) -> MLXArray {
    if cd == .float32 { return layer(x) }
    var y = matmul(x.asType(cd), layer.weight.asType(cd).transposed(1, 0))
    if let b = layer.bias { y = y + b.asType(cd) }
    return y
}

// MARK: - segment sum (mlx_backend.py:98-107)

/// Lower-triangular segment-sum. `out[..., i, j] = sum_{j < k <= i} x_k` for `i >= j`, and
/// `-inf` above the diagonal. `exp(segsum(g))` is the within-window decay matrix of a
/// scalar SSM (a 1-semiseparable matrix); the `-inf` upper triangle exps to 0, enforcing
/// causality.
public func segsum(_ x: MLXArray) -> MLXArray {
    let t = x.dim(-1)
    let xc = cumsum(x, axis: -1)
    let seg = xc.expandedDimensions(axis: -1) - xc.expandedDimensions(axis: -2)
    let mask = tril(MLXArray.ones([t, t])) .> MLXArray(Float(0))
    return which(mask, seg, MLXArray(-Float.infinity).asType(seg.dtype))
}

// MARK: - RoPE (mlx_backend.py:458-478)

/// `_rope_cos_sin` (`:458-466`). Returns `(cos, sin)`, each `(T, headDim)`.
///
/// Hand-rolled ON PURPOSE — do NOT substitute `MLXNN.RoPE`, which implements the
/// traditional/interleaved convention. This is the GPT-NeoX **duplicated-half** layout
/// (`concat([cos(ang), cos(ang)])`), paired with `rotateHalf` below.
public func ropeCosSin(positions: MLXArray, headDim: Int) -> (MLXArray, MLXArray) {
    let half = headDim / 2
    let ar = MLXArray(Array(0..<half).map { Int32($0) }).asType(.float32)
    let invFreq = exp(MLXArray(-Float(log(10000.0))) * ar / MLXArray(Float(half)))
    let ang = positions.asType(.float32).expandedDimensions(axis: 1)
        * invFreq.expandedDimensions(axis: 0)                       // (T, half)
    let c = cos(ang)
    let s = sin(ang)
    return (concatenated([c, c], axis: -1), concatenated([s, s], axis: -1))
}

/// `_rotate_half` (`:469-471`).
public func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let parts = split(x, indices: [half], axis: -1)
    return concatenated([-parts[1], parts[0]], axis: -1)
}

/// `_apply_rope` (`:474-478`) — x `(B, H, T, Dh)`; cos/sin `(T, Dh)` broadcast over `(B, H)`.
public func applyRope(_ x: MLXArray, _ cosv: MLXArray, _ sinv: MLXArray) -> MLXArray {
    let c = cosv.expandedDimensions(axes: [0, 1])
    let s = sinv.expandedDimensions(axes: [0, 1])
    return x * c + rotateHalf(x) * s
}

/// `_softmax_lastdim` (`:481-484`) — the hand-rolled max-subtract softmax, NOT `MLX.softmax`.
/// Using the fused op here risks a different internal formulation and a needless parity delta
/// against the Python oracle, which uses exactly this expression.
public func softmaxLastDim(_ scores: MLXArray) -> MLXArray {
    let s = scores - MLX.max(scores, axis: -1, keepDims: true)
    let w = exp(s)
    return w / MLX.sum(w, axis: -1, keepDims: true)
}

/// Boolean lower-triangular causal mask `(L, L)`.
public func causalMask(_ l: Int) -> MLXArray {
    tril(MLXArray.ones([l, l])) .> MLXArray(Float(0))
}
