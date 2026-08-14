// DynamicLossScaler — the Swift mirror of `src/train/loss_scale.py` (#195).
//
// PURE SWIFT ARITHMETIC. Deliberately NO `import MLX` — the actual inf/nan check on the
// gradient tensors lives in `TrainStep.swift` (it needs the hardware array type); this
// file holds only the number policy, exactly the way `loss_scale.py` sits above the
// Python seam so it is unit-testable without a backend. `monica-train --self-test` runs
// this file's logic even on a host where mlx-swift cannot execute (no compiled
// `default.metallib`), mirroring `tests/test_loss_scale.py`.
//
// fp16 gradients underflow to zero for small values and overflow to inf/nan for large
// ones. Scale the loss up by `scale` before backprop (shifting gradients into fp16's
// representable range), unscale the grads by `1/scale` before the optimizer step:
//   * on a non-finite gradient (overflow) — drop the step, multiply `scale` by `backoff`;
//   * after `growthInterval` consecutive clean steps — multiply `scale` by `growthFactor`.

import Foundation

public final class DynamicLossScaler {
    public private(set) var scale: Double
    public let growthFactor: Double
    public let backoff: Double
    public let growthInterval: Int
    public let minScale: Double
    public let maxScale: Double
    private var goodSteps: Int = 0

    public init(
        initScale: Double = pow(2.0, 13),
        growthFactor: Double = 2.0,
        backoff: Double = 0.5,
        growthInterval: Int = 2000,
        minScale: Double = 1.0,
        maxScale: Double = pow(2.0, 24)
    ) {
        self.scale = initScale
        self.growthFactor = growthFactor
        self.backoff = backoff
        self.growthInterval = growthInterval
        self.minScale = minScale
        self.maxScale = maxScale
    }

    /// Advance the scale given whether the last step's gradients overflowed.
    public func update(overflow: Bool) {
        if overflow {
            scale = Swift.max(minScale, scale * backoff)
            goodSteps = 0
            return
        }
        goodSteps += 1
        if goodSteps >= growthInterval {
            // Cap growth: without a ceiling the scale doubles unboundedly until the scaled
            // loss itself overflows, wasting a step every cycle. 2**24 matches PyTorch
            // AMP / Apex defaults (loss_scale.py:46-48).
            scale = Swift.min(maxScale, scale * growthFactor)
            goodSteps = 0
        }
    }

    public struct State: Codable {
        public let scale: Double
        public let goodSteps: Int
    }

    public func stateDict() -> State { State(scale: scale, goodSteps: goodSteps) }

    public func loadStateDict(_ state: State?) {
        guard let state else { return }
        scale = state.scale
        goodSteps = state.goodSteps
    }
}

/// Map a config `precision` to the loss scaler a train step needs — the single source of
/// truth for the precision->scaler wiring, mirroring `scaler_for_precision`
/// (`src/train/loss_scale.py:62-70`). Only fp16 needs dynamic loss scaling; fp32 and bf16
/// train unscaled.
public func scalerFor(precision: String, initScale: Double = pow(2.0, 13)) -> DynamicLossScaler? {
    precision == "fp16" ? DynamicLossScaler(initScale: initScale) : nil
}
