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

    /// The `MLXArray` leaves of this state case (#264, `verifyBlock`'s prerequisite): lets
    /// a caller collect every intermediate state across a whole token batch into ONE
    /// `MLX.eval` call without a per-case switch of its own. `.moe` carries none.
    public var arrays: [MLXArray] {
        switch self {
        case .mamba(let conv, let ssm): return [conv, ssm]
        case .attention(let k, let v): return [k, v]
        case .moe: return []
        }
    }
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
    /// `forward_seq` (#68/#263: the `segIds` arm packs multiple documents into one
    /// sequence and masks recurrent state so it can't bleed across a boundary; `nil` is
    /// the original single-segment path).
    ///
    /// Deliberately NOT `segIds: MLXArray? = nil` on this `open` method: Swift resolves
    /// default argument values against the STATIC type, not the dynamic one, so a
    /// subclass that re-declares the default would silently diverge from a caller that
    /// holds a `Block`-typed reference — exactly the class of bug #263 exists to catch.
    /// The explicit two-arg override plus the `final` one-arg forwarder below has no
    /// such hole.
    open func forwardSeq(_ x: MLXArray, _ segIds: MLXArray?) -> MLXArray {
        fatalError("Block subclasses must override forwardSeq")
    }

    /// Convenience forwarder for the (still very common) single-segment call sites.
    /// `final` — not part of the overridable seam, so it carries no divergence risk.
    public final func forwardSeq(_ x: MLXArray) -> MLXArray { forwardSeq(x, nil) }

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
