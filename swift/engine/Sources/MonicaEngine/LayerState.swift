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

    public var description: String {
        switch self {
        case .stateMismatch(let m): return "state mismatch: \(m)"
        case .badCheckpoint(let m): return "bad checkpoint: \(m)"
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

    open func initState(batch: Int) -> LayerState {
        fatalError("Block subclasses must override initState")
    }
}
