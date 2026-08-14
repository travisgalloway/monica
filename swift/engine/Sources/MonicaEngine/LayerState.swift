// Per-layer inference state and the common block interface.
//
// Python keeps state as a 2-tuple of arrays for EVERY layer type, including MoE, which
// carries none — the placeholder pair exists only to keep `clone_state`'s tuple unpack
// valid (`mlx_backend.py:936-937`). Swift can say what it means, so this is an enum.

import MLX
import MLXNN

public enum LayerState {
    /// Mamba: conv window `(B, k-1, dInner)` and SSM state `(B, H, P, N)`, both fp32.
    case mamba(conv: MLXArray, ssm: MLXArray)
    /// Attention: a `(k_cache, v_cache)` pair, each `(B, Ha, T, Dh)` fp32, grown per step.
    case attention(k: MLXArray, v: MLXArray)
    /// MoE FFN blocks are pointwise and stateless.
    case moe
}

public enum EngineError: Error, CustomStringConvertible {
    case stateMismatch(String)
    case badCheckpoint(String)
    case unsupportedOptimizer(String)

    public var description: String {
        switch self {
        case .stateMismatch(let m): return "state mismatch: \(m)"
        case .badCheckpoint(let m): return "bad checkpoint: \(m)"
        case .unsupportedOptimizer(let m): return "unsupported optimizer: \(m)"
        }
    }
}

/// The contract every layer type satisfies — the Swift analogue of the duck-typed
/// `forward_seq(x) -> x` / `step(x, state) -> (x, state)` the Python blocks share.
///
/// A class (not a protocol) because mlx-swift's parameter reflection walks arrays of
/// `Module`, and `[Block]` where `Block: Module` is exactly such an array. The Python
/// model's `self.layers` list flattens to `layers.{i}.…` the same way.
open class Block: Module {
    open func forwardSeq(_ x: MLXArray) -> MLXArray {
        fatalError("Block subclasses must override forwardSeq")
    }

    open func step(_ x: MLXArray, _ state: LayerState) throws -> (MLXArray, LayerState) {
        fatalError("Block subclasses must override step")
    }

    /// `forward_prefill` (#169, `mlx_backend.py:410-424`/`:539-557`/`:800-806`) — the same
    /// function as `forwardSeq(x) -> x`, additionally returning the `LayerState` a `step`
    /// walk over `x`'s tokens (from a fresh, zero-length state) would have left. Fresh-
    /// session-only: no `seg_ids` counterpart, matching #165's seam contract.
    open func forwardPrefill(_ x: MLXArray) -> (MLXArray, LayerState) {
        fatalError("Block subclasses must override forwardPrefill")
    }

    open func initState(batch: Int) -> LayerState {
        fatalError("Block subclasses must override initState")
    }
}
