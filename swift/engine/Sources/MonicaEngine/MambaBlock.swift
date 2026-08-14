// MambaBlock — `src/model/mlx_backend.py:339-452`.
//
// pre-norm -> input proj -> split main+gate -> causal depthwise conv -> SiLU
// -> selective SSM -> * SiLU(gate) -> output proj, with a residual.
//
// `_conv_seq_seg` (the packing-aware conv) landed in #263 as `convSeqSeg`.
// `_conv_window`/`forward_prefill` landed in #169 as `convWindow`/`forwardPrefill`.
// `mixing_matrix` (#100's interpretability auxiliary) lands in #264 as `mixingMatrix`,
// backed by a new `ssmInput` (the block's front-end, `forwardSeq`'s `segIds == nil` arm up
// to and including the post-conv SiLU) that exposes the SSM's scan input so
// `monica-parity` can apply `ssm.mixingMatrix` to it directly without re-deriving the
// front-end.

import MLX
import MLXNN

/// `_conv_window` (`mlx_backend.py:314-336`) — the conv state left after consuming `xMain`
/// (B, L, dInner), prefill's mirror of what `MambaBlock.step` leaves behind.
///
/// `step` keeps `window[:, 1:]` where `window = concat([convState, xMain[:, None]])`, so
/// after `L` tokens the state is the LAST `k-1` rows of `xMain` — post-`in_proj`, PRE-conv,
/// pre-SiLU. Two edge cases, both live footguns (mirrored from Python, not reinvented):
///   * `L < k-1`: LEFT-zero-pad to `(B, k-1, dInner)` — exactly what an initially-zeroed
///     window plus `L` real steps leaves behind. A right-pad would compile, produce the
///     right shape, and misalign every conv tap.
///   * `k == 1`: width 0. Guarded explicitly rather than relying on `split`'s empty-slice
///     behavior, matching Python's own explicit guard (its comment warns `x[:, -0:]` is the
///     WHOLE array, not empty).
/// Emitted in fp32 so it is dtype-identical to a stepped window (`initState` allocates fp32
/// and `concat([fp32, cd])` promotes).
public func convWindow(_ xMain: MLXArray, _ k: Int) -> MLXArray {
    let bSize = xMain.dim(0)
    let l = xMain.dim(1)
    let di = xMain.dim(2)
    let w = k - 1
    if w <= 0 {
        return MLXArray.zeros([bSize, 0, di], dtype: .float32)
    }
    let tail = f32(split(xMain, indices: [max(0, l - w)], axis: 1)[1])   // (B, min(L,w), di)
    let missing = w - tail.dim(1)
    if missing > 0 {
        return concatenated(
            [MLXArray.zeros([bSize, missing, di], dtype: tail.dtype), tail], axis: 1)
    }
    return tail
}

public final class MambaBlock: Block {
    let config: MambaConfig

    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv") var conv: Conv1d
    // Public since #264, same precedent as `MonicaModel.layers`/`normF` (#196): lets
    // `monica-parity` apply `ssm.mixingMatrix` to this block's own `ssmInput` output.
    @ModuleInfo(key: "ssm") public var ssm: SelectiveSSM
    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(_ config: MambaConfig) {
        self.config = config
        let dInner = config.dInner
        self._norm.wrappedValue = RMSNorm(config.dModel)
        self._inProj.wrappedValue = Linear(config.dModel, 2 * dInner, bias: false)
        // Depthwise causal conv. mlx-swift's Conv1d weight layout is
        // [outputChannels, kernelSize, inputChannels/groups] — the SAME layout the portable
        // checkpoint stores, so the loader does no transpose. (See Checkpoint.swift.)
        self._conv.wrappedValue = Conv1d(
            inputChannels: dInner, outputChannels: dInner,
            kernelSize: config.dConv, padding: config.dConv - 1,
            groups: dInner, bias: true)
        self._ssm.wrappedValue = SelectiveSSM(config)
        self._outProj.wrappedValue = Linear(dInner, config.dModel, bias: false)
        super.init()
    }

    /// `_conv_seq` (`:354-363`) — causal depthwise conv in `cd`. fp32 routes to the layer
    /// verbatim.
    func convSeq(_ xMain: MLXArray, _ cd: DType) -> MLXArray {
        if cd == .float32 { return conv(xMain) }
        let y = conv(xMain.asType(cd))
        return y
    }

    /// `_conv_seq_seg` (`mlx_backend.py:365-392`, #68/#263). Boundary-aware causal
    /// depthwise conv: taps reaching into a previous document are zeroed, so the conv
    /// window can't bleed across a packed boundary (the conv is part of the per-doc
    /// recurrent state). `seg` is (B, L). Returns length L DIRECTLY — unlike `convSeq`
    /// (which pads both sides and needs a `[:L]` trim), this path never over-pads.
    func convSeqSeg(_ xMain: MLXArray, _ cd: DType, _ seg: MLXArray) -> MLXArray {
        let k = config.dConv
        let bSize = xMain.dim(0)
        let l = xMain.dim(1)
        let di = xMain.dim(2)
        let x = xMain.asType(cd)
        let w = conv.weight.asType(cd)     // (dInner, K, 1) — mlx-swift Conv1d layout
        var acc: MLXArray? = nil
        for kk in 0..<k {
            let shift = k - 1 - kk         // how far this tap reaches into the past
            let wk = w[0..., kk..<(kk + 1), 0..<1].reshaped([di])       // (di,)
            var xs: MLXArray
            if shift == 0 {
                xs = x
            } else if shift >= l {
                continue                   // tap reaches entirely before the sequence start
            } else {
                let xPrefix = x[0..., 0..<(l - shift), 0...]            // x[t-shift]
                let xZeroPad = MLXArray.zeros([bSize, shift, di], dtype: cd)
                xs = concatenated([xZeroPad, xPrefix], axis: 1)         // (B, L, di), left-padded
                let segShift = seg[0..., shift..<l]
                let segPrefix = seg[0..., 0..<(l - shift)]
                let same = (segShift .== segPrefix).asType(cd)          // (B, L-shift)
                let validZeroPad = MLXArray.zeros([bSize, shift], dtype: cd)
                let valid = concatenated([validZeroPad, same], axis: 1) // (B, L): 0 across a boundary
                xs = xs * valid.expandedDimensions(axis: -1)
            }
            let term = xs * wk
            acc = acc == nil ? term : acc! + term
        }
        if acc == nil {                    // K > L: every tap reaches before the start
            acc = MLXArray.zeros([bSize, l, di], dtype: cd)
        }
        return acc! + conv.bias!.asType(cd)
    }

    /// `forward_seq` (`:394-408`). `segIds == nil` is the original single-segment path;
    /// otherwise the conv is boundary-aware (`convSeqSeg`) and the SSM scan's inter-chunk
    /// carry is masked (`SelectiveSSM.parallel(_:segIds:)`) so recurrent state can't cross
    /// a packed document boundary (#68/#263).
    public override func forwardSeq(_ x: MLXArray, _ segIds: MLXArray?) -> MLXArray {
        let l = x.dim(1)
        let cd = config.cd
        let xn = norm(x)
        let proj = split(linear(inProj, xn, cd), parts: 2, axis: -1)   // (B,L,di) each
        let xMain = proj[0]
        let z = proj[1]
        // Causal depthwise conv: pad both sides (d_conv-1), keep the FIRST L outputs. With
        // segIds the conv is boundary-aware so its window can't cross a packed boundary.
        let xc: MLXArray
        if let seg = segIds {
            xc = silu(convSeqSeg(xMain, cd, seg))
        } else {
            xc = silu(split(convSeq(xMain, cd), indices: [l], axis: 1)[0])
        }
        var y = ssm.parallel(xc, segIds: segIds)
        y = y * silu(z)
        return x + linear(outProj, y, cd)
    }

    /// `step` (`:437-452`).
    public override func step(_ x: MLXArray, _ state: LayerState) throws -> (MLXArray, LayerState) {
        guard case .mamba(let convState, let ssmState) = state else {
            throw EngineError.stateMismatch("MambaBlock.step got a non-Mamba LayerState")
        }
        let cd = config.cd
        let k = config.dConv
        let xn = norm(x)
        let proj = split(linear(inProj, xn, cd), parts: 2, axis: -1)    // (B,di) each
        let xMain = proj[0]
        let z = proj[1]
        let window = concatenated(
            [convState, xMain.expandedDimensions(axis: 1)], axis: 1)    // (B,k,di)
        // Depthwise conv at this timestep: sum over kernel positions (in cd, matching the
        // conv in forwardSeq; the casts no-op for fp32).
        let wk = conv.weight.squeezed(axis: 2).transposed(1, 0)         // (k, di)
        let convOut = MLX.sum(window.asType(cd) * wk.asType(cd).expandedDimensions(axis: 0),
                              axis: 1) + conv.bias!.asType(cd)          // (B, di)
        let xc = silu(convOut)
        let (yRaw, newSSM) = ssm.recurrence(xc, ssmState)
        let y = yRaw * silu(z)
        let out = x + linear(outProj, y, cd)
        // The new conv window drops the oldest row. `k >= 1` is enforced by
        // MambaConfig.validate(); `k == 1` degenerates to a zero-width window, which the
        // split below produces naturally.
        return (out, .mamba(conv: split(window, indices: [1], axis: 1)[1], ssm: newSSM))
    }

    public override func initState(batch: Int) -> LayerState {
        .mamba(conv: MLXArray.zeros([batch, config.dConv - 1, config.dInner]),
               ssm: MLXArray.zeros([batch, config.nHeads, config.headDim, config.dState]))
    }

    /// `mixing_matrix`'s front-end (`mlx_backend.py:426-435`): `norm` -> `in_proj` main
    /// half (index 0 of the 2-way split — swapping main/gate here is a silent
    /// plausible-but-wrong failure) -> causal conv -> SiLU. This is `forwardSeq`'s
    /// `segIds == nil` arm up to (not including) the SSM scan, factored out so
    /// `mixingMatrix` and `monica-parity`'s in-process identity check can both reach the
    /// SSM's raw scan input without re-deriving it.
    public func ssmInput(_ x: MLXArray) -> MLXArray {
        let l = x.dim(1)
        let cd = config.cd
        let xn = norm(x)
        let xMain = split(linear(inProj, xn, cd), parts: 2, axis: -1)[0]
        return silu(split(convSeq(xMain, cd), indices: [l], axis: 1)[0])
    }

    /// `mixing_matrix` (`mlx_backend.py:426-435`, #264/#100): this block's head-split SSM
    /// mixing matrix `(B, H, L, L)`. Runs the block front-end (`ssmInput`) then
    /// materializes the SSM matrix on that input.
    public func mixingMatrix(_ x: MLXArray) -> MLXArray {
        ssm.mixingMatrix(ssmInput(x))
    }

    /// `forward_prefill` (`:410-424`) — `forwardSeq`'s body verbatim (deliberately
    /// duplicated, not shared, so the hot training/scan path `forwardSeq` stays free of a
    /// state-collection branch), additionally emitting the (conv_window, ssm_carry_out)
    /// pair `step` would have left after L tokens from a fresh state.
    public override func forwardPrefill(_ x: MLXArray) -> (MLXArray, LayerState) {
        let l = x.dim(1)
        let cd = config.cd
        let xn = norm(x)
        let proj = split(linear(inProj, xn, cd), parts: 2, axis: -1)   // (B,L,di) each
        let xMain = proj[0]
        let z = proj[1]
        let xc = silu(split(convSeq(xMain, cd), indices: [l], axis: 1)[0])
        let (yRaw, ssmState) = ssm.parallelWithState(xc)
        let y = yRaw * silu(z)
        let out = x + linear(outProj, y, cd)
        // convWindow reads xMain (pre-conv, post-in_proj), NOT xc.
        return (out, .mamba(conv: convWindow(xMain, config.dConv), ssm: ssmState))
    }
}
