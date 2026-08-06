// RMSNorm — `src/model/mlx_backend.py:82-95`.
//
// Hand-rolled rather than `MLXNN.RMSNorm`: the upstream module routes through the fused
// `fast.rmsNorm` kernel, and the parity oracle is the Python expression below. One
// parameter, keyed `weight`, matching the portable checkpoint.

import MLX
import MLXNN

public final class RMSNorm: Module, UnaryLayer {
    public var weight: MLXArray
    let eps: Float

    public init(_ dModel: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([dModel])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.dtype == .float32 {
            // fp32 path: the original op, bit-identical to Python's.
            let norm = rsqrt(mean(x * x, axis: -1, keepDims: true) + MLXArray(eps))
            return weight * (x * norm)
        }
        // Mixed precision: reduce in fp32 (the weight is fp32), return in x's dtype.
        let xf = x.asType(.float32)
        let norm = rsqrt(mean(xf * xf, axis: -1, keepDims: true) + MLXArray(eps))
        return (weight * (xf * norm)).asType(x.dtype)
    }
}
