// AttentionBlock — `src/model/mlx_backend.py:487-572`.
//
// Pre-norm causal multi-head attention with RoPE (the hybrid's attention layer, #67).
// Scores and softmax run in fp32 regardless of `cd` (qkv/o_proj GEMMs in `cd`) so forward
// and step agree at the fp32 ~1e-4 parity tolerance.
//
// Not ported (out of scope for #166): the `seg_ids` block-diagonal mask.
// `forward_prefill` landed in #169 as `forwardPrefill`.

import Foundation
import MLX
import MLXNN

public final class AttentionBlock: Block {
    let config: MambaConfig
    let h: Int
    let dh: Int

    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public init(_ config: MambaConfig) {
        self.config = config
        self.h = config.nAttnHeadsResolved
        self.dh = config.attnHeadDim
        let dAttn = h * dh
        self._norm.wrappedValue = RMSNorm(config.dModel)
        self._qkvProj.wrappedValue = Linear(config.dModel, 3 * dAttn, bias: false)
        self._oProj.wrappedValue = Linear(dAttn, config.dModel, bias: false)
        super.init()
    }

    /// `_qkv` (`:507-514`) — one fused projection, split three ways, each reshaped to
    /// `(B, T, H, Dh)` and transposed to `(B, H, T, Dh)` in fp32.
    func qkv(_ xn: MLXArray, _ cd: DType) -> (MLXArray, MLXArray, MLXArray) {
        let b = xn.dim(0)
        let t = xn.ndim == 3 ? xn.dim(1) : 1
        let parts = split(linear(qkvProj, xn, cd), parts: 3, axis: -1)
        func heads(_ v: MLXArray) -> MLXArray {
            f32(v).reshaped([b, t, h, dh]).transposed(0, 2, 1, 3)
        }
        return (heads(parts[0]), heads(parts[1]), heads(parts[2]))
    }

    /// `forward_seq` (`:516-537`), the `seg_ids == nil` arm.
    public override func forwardSeq(_ x: MLXArray) -> MLXArray {
        let cd = config.cd
        let l = x.dim(1)
        let xn = norm(x)
        var (q, k, v) = qkv(xn, cd)                                     // (B,H,L,Dh) fp32
        let (cosv, sinv) = ropeCosSin(
            positions: MLXArray(Array(0..<l).map { Int32($0) }), headDim: dh)
        q = applyRope(q, cosv, sinv)
        k = applyRope(k, cosv, sinv)
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) / MLXArray(Float(sqrt(Double(dh))))
        scores = which(causalMask(l), scores,
                       MLXArray(-Float.infinity).asType(scores.dtype))
        var out = matmul(softmaxLastDim(scores), v)                     // (B,H,L,Dh)
        out = out.transposed(0, 2, 1, 3).reshaped([x.dim(0), l, h * dh])
        return x + linear(oProj, cast(out, cd), cd)
    }

    /// `step` (`:559-572`) — `t = k_cache.shape[2]` is the absolute position.
    public override func step(_ x: MLXArray, _ state: LayerState) throws -> (MLXArray, LayerState) {
        guard case .attention(let kCache0, let vCache0) = state else {
            throw EngineError.stateMismatch("AttentionBlock.step got a non-attention LayerState")
        }
        let cd = config.cd
        let t = kCache0.dim(2)
        let xn = norm(x)                                                // x: (B, dModel)
        var (q, k, v) = qkv(xn, cd)                                     // (B,H,1,Dh) fp32
        let (cosv, sinv) = ropeCosSin(positions: MLXArray([Int32(t)]), headDim: dh)
        q = applyRope(q, cosv, sinv)
        k = applyRope(k, cosv, sinv)
        let kCache = concatenated([kCache0, k], axis: 2)                 // (B,H,t+1,Dh)
        let vCache = concatenated([vCache0, v], axis: 2)
        let scores = matmul(q, kCache.transposed(0, 1, 3, 2))
            / MLXArray(Float(sqrt(Double(dh))))                          // (B,H,1,t+1)
        var out = matmul(softmaxLastDim(scores), vCache)                 // (B,H,1,Dh)
        out = out.transposed(0, 2, 1, 3).reshaped([x.dim(0), h * dh])
        return (x + linear(oProj, cast(out, cd), cd), .attention(k: kCache, v: vCache))
    }

    /// A zero-length KV cache, grown one token per `step` (`:933`).
    public override func initState(batch: Int) -> LayerState {
        let z = MLXArray.zeros([batch, h, 0, dh])
        return .attention(k: z, v: z)
    }

    /// `forward_prefill` (`:539-557`) — `forwardSeq`'s body verbatim, additionally emitting
    /// the (k, v) cache `step` would have built over the same tokens. `k` is POST-RoPE (the
    /// same tensor that built `scores` here) and `v` is unchanged, both fp32 `(B, Ha, L,
    /// Dh)`. RoPE positions run `0..<L`, matching `step`'s `t = kCache.dim(2)` seeding from
    /// a zero-length cache — hence fresh-session-only, same as Python's `prefill`.
    public override func forwardPrefill(_ x: MLXArray) -> (MLXArray, LayerState) {
        let cd = config.cd
        let l = x.dim(1)
        let xn = norm(x)
        var (q, k, v) = qkv(xn, cd)                                     // (B,H,L,Dh) fp32
        let (cosv, sinv) = ropeCosSin(
            positions: MLXArray(Array(0..<l).map { Int32($0) }), headDim: dh)
        q = applyRope(q, cosv, sinv)
        k = applyRope(k, cosv, sinv)
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) / MLXArray(Float(sqrt(Double(dh))))
        scores = which(causalMask(l), scores,
                       MLXArray(-Float.infinity).asType(scores.dtype))
        var out = matmul(softmaxLastDim(scores), v)                     // (B,H,L,Dh)
        out = out.transposed(0, 2, 1, 3).reshaped([x.dim(0), l, h * dh])
        let result = x + linear(oProj, cast(out, cd), cd)
        return (result, .attention(k: k, v: v))
    }
}
