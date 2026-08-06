// MambaBlock — `src/model/mlx_backend.py:339-452`.
//
// pre-norm -> input proj -> split main+gate -> causal depthwise conv -> SiLU
// -> selective SSM -> * SiLU(gate) -> output proj, with a residual.
//
// Not ported (out of scope for #166): `_conv_seq_seg` (the packing-aware conv),
// `_conv_window`/`forward_prefill` (#169), and `mixing_matrix`.

import MLX
import MLXNN

public final class MambaBlock: Block {
    let config: MambaConfig

    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "ssm") var ssm: SelectiveSSM
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

    /// `forward_seq` (`:394-408`), the `seg_ids == nil` arm.
    public override func forwardSeq(_ x: MLXArray) -> MLXArray {
        let l = x.dim(1)
        let cd = config.cd
        let xn = norm(x)
        let proj = split(linear(inProj, xn, cd), parts: 2, axis: -1)   // (B,L,di) each
        let xMain = proj[0]
        let z = proj[1]
        // Causal depthwise conv: pad both sides (d_conv-1), keep the FIRST L outputs.
        let xc = silu(split(convSeq(xMain, cd), indices: [l], axis: 1)[0])
        var y = ssm.parallel(xc)
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
}
