// SelectiveSSM — scalar-A Mamba-2 / SSD selective state space.
// Port of `src/model/mlx_backend.py:134-311`.
//
// `d_inner` splits into `nHeads` heads of width `headDim` (P); each head has a SCALAR decay
// A (the SSD restriction that turns the scan into matmuls). B and C are a single group of
// width `dState` (N), shared across heads. Two code paths must compute the same function:
//
//   * `parallel(x)`     — the SSD chunked-matmul scan (training path)
//   * `recurrence(x, h)` — the matching one-step recurrence (inference path)
//
// Not ported (out of scope for #166, see the plan): the `seg_ids` packing-aware arms
// (`_chunk_seg_mask`) and `mixing_matrix`. `return_state` (prefill) landed in #169 as
// `parallelWithState`.

import Foundation
import MLX
import MLXNN

public final class SelectiveSSM: Module {
    let config: MambaConfig

    @ModuleInfo(key: "x_proj") var xProj: Linear
    @ModuleInfo(key: "dt_proj") var dtProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var d: MLXArray

    public init(_ config: MambaConfig) {
        self.config = config
        let dInner = config.dInner
        let dState = config.dState
        let dtRank = config.dtRankResolved
        let h = config.nHeads

        // x_proj produces (dt_pre, B, C); B and C are one group, shared across heads.
        self._xProj.wrappedValue = Linear(dInner, dtRank + 2 * dState, bias: false)
        // dt is PER-HEAD in Mamba-2: dt_proj maps dt_rank -> n_heads.
        self._dtProj.wrappedValue = Linear(dtRank, h, bias: true)
        // Scalar decay A per head, stored as log: A = -exp(A_log). S4D-real init.
        self._aLog.wrappedValue = MLX.log(
            MLXArray((1...h).map { Float($0) }))
        self._d.wrappedValue = MLXArray.ones([h])
        super.init()
        initDtBias()
    }

    /// `_init_dt_bias` (`:161-174`). LOAD-BEARING for TRAINING (inverse-softplus into a small
    /// positive range; without it the model fails to learn recall) — ported for shape and
    /// structure so #195/#196 inherit it, but **not on the parity path**: every parameter
    /// here is overwritten by `Checkpoint.load`'s `update(parameters:)`.
    ///
    /// The uniform draw uses Swift's own RNG rather than `MLXRandom` ON PURPOSE. Matching
    /// `mx.random.uniform`'s stream bit-for-bit is neither possible nor needed; nobody
    /// should later "fix" this by chasing MLX's PRNG.
    func initDtBias() {
        let c = config
        let lo = Foundation.log(Double(c.dtMin))
        let hi = Foundation.log(Double(c.dtMax))
        var rng = SystemRandomNumberGenerator()
        let dt: [Float] = (0..<c.nHeads).map { _ in
            let u = Double.random(in: lo..<hi, using: &rng)
            return Swift.max(Float(Foundation.exp(u)), c.dtInitFloor)
        }
        // bias = dt + log(-expm1(-dt))   (inverse softplus)
        let bias = dt.map { v -> Float in
            v + Float(Foundation.log(-Foundation.expm1(-Double(v))))
        }
        // `Linear.bias` is a `let` in mlx-swift, so the parameter tree is the way in.
        dtProj.update(parameters: ModuleParameters.unflattened([("bias", MLXArray(bias))]))
    }

    // MARK: - shared projections (`_project`, :177-196)

    /// `x: (..., dInner)` -> `delta (..., H)`, `a (H,)`, `B (..., N)`, `C (..., N)`.
    ///
    /// `x_proj` is the heavy SSM GEMM and runs in `cd`; its outputs upcast to fp32 so the
    /// rest of the scan (dt_proj, softplus, the decays, cumsum/segsum, every state einsum)
    /// runs in fp32.
    func project(_ x: MLXArray) -> (delta: MLXArray, a: MLXArray, b: MLXArray, c: MLXArray) {
        let cd = config.cd
        let dtRank = config.dtRankResolved
        let dState = config.dState
        let proj = linear(xProj, x, cd)
        let parts = split(proj, indices: [dtRank, dtRank + dState], axis: -1)
        let dtPre = f32(parts[0])
        let bm = f32(parts[1])
        let cm = f32(parts[2])
        var delta = softplus(dtProj(dtPre))            // (..., H) per head, fp32
        // Long-context extension (#54). Guarded so factor 1.0 (the default, and every
        // config in the parity gate) leaves `delta` byte-identical.
        if config.longCtxFactor != 1.0 {
            delta = delta / MLXArray(config.longCtxFactor)
        }
        let a = -exp(aLog)                             // (H,) scalar decay, fp32
        return (delta, a, bm, cm)
    }

    // MARK: - SSD chunked-matmul scan (`parallel`, :198-278)

    /// `x: (B, L, dInner) -> (B, L, dInner)`. Pads L up to a multiple of the chunk length Q
    /// (padded steps carry zero input and are trimmed). Every decay is `exp` of a
    /// non-positive sum, so the scan is overflow-safe by construction.
    public func parallel(_ x: MLXArray) -> MLXArray { scan(x).y }

    /// `parallel(..., return_state=True)` (`:198-278`, `:267-268`). Same scan, additionally
    /// returning the END-OF-SEQUENCE SSM state `(B, H, P, N)` fp32 — the entering-state row
    /// `scan` normally discards when it slices `sEnter = newStates[:, :-1]`. Exact even when
    /// `L` is padded to a chunk multiple: a padded step has `delta = 0`, so its log-decay
    /// `g = delta * a` is 0 (decay `exp(0) = 1`, identity) and its input `Xin = delta * X` is
    /// 0 — padded tail steps neither contribute nor decay, so the carry-out is the state at
    /// position `L-1` regardless of `L % Q`. Every decay is `exp` of a non-positive cumulative
    /// sum, so this stays overflow-safe by construction, same as `parallel`.
    public func parallelWithState(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let (y, carry) = scan(x)
        return (y, carry)
    }

    /// The shared scan body. Costs nothing extra when the caller only wants `y`: MLX is
    /// lazy, so an un-eval'd `carry` is never materialized.
    private func scan(_ x: MLXArray) -> (y: MLXArray, carry: MLXArray) {
        let bSize = x.dim(0)
        let l = x.dim(1)
        let dInner = x.dim(2)
        let h = config.nHeads
        let p = config.headDim
        let n = config.dState
        let q = config.q
        let cd = config.cd

        var (delta, a, bm, cm) = project(x)            // delta (B,L,H); bm,cm (B,L,N) fp32
        // Upcast the head inputs to fp32 so the whole scan (and the pad zeros, which inherit
        // the operand dtype) runs in fp32; the output casts back to cd for out_proj.
        var xh = f32(x).reshaped([bSize, l, h, p])

        let pad = (q - l % q) % q
        if pad > 0 {
            xh = concatenated([xh, MLXArray.zeros([bSize, pad, h, p], dtype: xh.dtype)], axis: 1)
            delta = concatenated([delta, MLXArray.zeros([bSize, pad, h], dtype: delta.dtype)], axis: 1)
            bm = concatenated([bm, MLXArray.zeros([bSize, pad, n], dtype: bm.dtype)], axis: 1)
            cm = concatenated([cm, MLXArray.zeros([bSize, pad, n], dtype: cm.dtype)], axis: 1)
        }
        let lp = l + pad
        let nc = lp / q

        let g = delta * a                              // (B,Lp,H) log-decay (<= 0)
        let xin = delta.expandedDimensions(axis: -1) * xh   // (B,Lp,H,P) input = dt * X

        let gc = g.reshaped([bSize, nc, q, h]).transposed(0, 3, 1, 2)   // (B,H,nc,Q)
        let xc = xin.reshaped([bSize, nc, q, h, p])
        let bc = bm.reshaped([bSize, nc, q, n])
        let cc = cm.reshaped([bSize, nc, q, n])
        let acum = cumsum(gc, axis: -1)                                  // (B,H,nc,Q)

        // 1) intra-chunk diagonal block (attention-like, within each chunk)
        let lmask = exp(segsum(gc))                                      // (B,H,nc,Q,Q)
        let cb = einsum("bcin,bcjn->bcij", cc, bc)                       // (B,nc,Q,Q)
        let ydiag = einsum("bhcij,bcij,bcjhp->bcihp", lmask, cb, xc)

        // 2) each chunk's final state, from that chunk's inputs only
        let acumLastKeep = split(acum, indices: [q - 1], axis: -1)[1]    // (B,H,nc,1)
        let decayEnd = exp(acumLastKeep - acum)                          // (B,H,nc,Q)
        var states = einsum("bhcj,bcjhp,bcjn->bchpn", decayEnd, xc, bc)

        // 3) inter-chunk recurrence over the nc chunk-states, as a matmul against an
        // (nc+1, nc+1) decay matrix (the canonical SSD form) rather than a sequential scan.
        states = concatenated(
            [MLXArray.zeros([bSize, 1, h, p, n], dtype: states.dtype), states], axis: 1)
        let chunkTot = concatenated(
            [MLXArray.zeros([bSize, h, 1], dtype: acum.dtype),
             acumLastKeep.squeezed(axis: -1)], axis: -1)                 // (B,H,nc+1)
        let decayChunk = exp(segsum(chunkTot))                           // (B,H,nc+1,nc+1)
        let newStates = einsum("bhzc,bchpn->bzhpn", decayChunk, states)
        // Split once and reuse both halves: [0] is the entering state per chunk (`sEnter`),
        // [1] is the state one chunk past the end — the carry-out returned below.
        let newStatesSplit = split(newStates, indices: [nc], axis: 1)
        let sEnter = newStatesSplit[0]                                   // (B,nc,H,P,N)

        // 4) off-diagonal output: entering state decayed to each position
        let outDecay = exp(acum)                                         // (B,H,nc,Q)
        let yoff = einsum("bcin,bchpn,bhci->bcihp", cc, sEnter, outDecay)

        var y = split((ydiag + yoff).reshaped([bSize, lp, h, p]), indices: [l], axis: 1)[0]
        y = y + split(xh, indices: [l], axis: 1)[0] * d.reshaped([1, 1, h, 1])   // skip
        // The carry-out is the OTHER half of the split that produced sEnter above: the
        // entering state one chunk past the end, i.e. newStates[:, -1] — fp32, never cast
        // to cd (state stays fp32, matching `recurrence`).
        let carry = newStatesSplit[1].squeezed(axis: 1)   // (B,H,P,N)
        return (cast(y.reshaped([bSize, l, dInner]), cd), carry)
    }

    // MARK: - one-step recurrence (`recurrence`, :300-311)

    /// `x: (B, dInner)`, state `h: (B, H, P, N)` -> `y: (B, dInner)`. State stays fp32.
    public func recurrence(_ x: MLXArray, _ state: MLXArray) -> (MLXArray, MLXArray) {
        let bSize = x.dim(0)
        let h = config.nHeads
        let p = config.headDim
        let cd = config.cd
        let (delta, a, bm, cm) = project(x)            // delta (B,H); bm,cm (B,N) fp32
        let xh = f32(x).reshaped([bSize, h, p])        // scan + state stay fp32
        let dA = exp(delta * a)                        // (B,H)
        let dBx = (delta.expandedDimensions(axis: -1) * xh).expandedDimensions(axis: -1)
            * bm.expandedDimensions(axes: [1, 2])      // (B,H,P,N)
        let newState = dA.expandedDimensions(axes: [2, 3]) * state + dBx
        let y = MLX.sum(newState * cm.expandedDimensions(axes: [1, 2]), axis: -1)
            + xh * d.reshaped([1, h, 1])
        return (cast(y.reshaped([bSize, -1]), cd), newState)
    }
}
