// Generator — the Swift port of `src/serve/generate.py`'s `generate()` (#167).
//
// Prefill (#169): the whole prompt is consumed in ONE parallel scan via `model.prefill`,
// which hands back the exact `State` an `L`-step walk of `model.step` would have left —
// `monica-parity`'s P3 check (and `src/conformance/prefill_decode_parity.py` on the Python
// side this mirrors) gates that equivalence element-wise. `usePrefill: Bool = true` keeps
// the OLD sequential per-token prefill reachable (`usePrefill: false`): it is the AC3 speed
// baseline, the AC1 A/B reference, and the one-flag rollback if the parallel path ever
// regresses generation output.
//
// Per decode step: sample, stop BEFORE appending on eos, append, `onToken`, feed back through
// `model.step`, then `stopFn` — feeding back before the stop check matches
// `src/serve/generate.py`'s comment that session state must always reflect every emitted id.
//
// `MLX.eval` is called on each step's logits+state so the lazy graph does not accumulate
// across the loop — the single most likely performance/memory trap in an mlx-swift decode
// loop. The prefill call gets the same treatment, for the same reason.

import MLX

public enum GeneratorError: Error, CustomStringConvertible {
    case emptyPrompt

    public var description: String {
        switch self {
        case .emptyPrompt: return "promptIds must be non-empty"
        }
    }
}

public enum Generator {
    /// Generate up to `maxNewTokens` continuation ids for `promptIds` against `model`.
    /// Returns only the generated ids (not the prompt), matching `src/serve/generate.py`.
    ///
    /// - `onToken`: called with each emitted id, in order (streaming hook).
    /// - `stopFn`: called with the generated-ids-so-far AFTER each token is fed back; `true`
    ///   ends generation.
    /// - `allowedIdsFor`: per-step constrained-decode hook (#226's surface) — called with the
    ///   generated-ids-so-far, returning the allowed id set for the NEXT token (`nil` = no
    ///   constraint). No LSP client is wired here; this only exposes the seam.
    /// - `usePrefill`: `true` (default) consumes the prompt in one `model.prefill` scan;
    ///   `false` falls back to the old per-token `model.step` loop. Both must produce the
    ///   same generated ids (`monica-parity`'s P5) — the flag exists for the AC3 timing
    ///   comparison and as a rollback if the parallel path ever misbehaves.
    public static func generate(
        model: MonicaModel, promptIds: [Int], sampler: Sampler, maxNewTokens: Int = 128,
        eosId: Int? = nil, onToken: ((Int) -> Void)? = nil,
        stopFn: (([Int]) -> Bool)? = nil, allowedIdsFor: (([Int]) -> [Int]?)? = nil,
        usePrefill: Bool = true
    ) throws -> [Int] {
        if promptIds.isEmpty { throw GeneratorError.emptyPrompt }

        var sampler = sampler
        var state: [LayerState]
        var logits: MLXArray

        if usePrefill {
            let promptArr = MLXArray(promptIds.map { Int32($0) }).reshaped([1, promptIds.count])
            let (lg, st) = model.prefill(promptArr, lastOnly: true)
            MLX.eval([lg] + st.flatMap(evalTargets))
            logits = lg
            state = st
        } else {
            state = model.initState(batch: 1)
            logits = MLXArray.zeros([1])   // overwritten before first use
            // Sequential prefill: step every prompt token, keeping only the last logits.
            for tok in promptIds {
                let tokArr = MLXArray([Int32(tok)])
                let (lg, st) = try model.step(tokArr, state)
                MLX.eval([lg] + st.flatMap(evalTargets))
                logits = lg
                state = st
            }
        }

        var generated: [Int] = []
        for _ in 0..<maxNewTokens {
            let row = logits.asType(.float32).reshaped([-1]).asArray(Float.self)
            let allowed = allowedIdsFor?(generated)
            let nxt = try sampler.sample(row, previousTokens: promptIds + generated,
                                         allowedIds: allowed)
            if let eos = eosId, nxt == eos { break }
            generated.append(nxt)
            onToken?(nxt)

            let tokArr = MLXArray([Int32(nxt)])
            let (lg, st) = try model.step(tokArr, state)
            MLX.eval([lg] + st.flatMap(evalTargets))
            logits = lg
            state = st

            if let stop = stopFn, stop(generated) { break }
        }

        return generated
    }

    /// The arrays inside one layer's state, so the caller can `MLX.eval` them and keep the
    /// lazy graph from accumulating across the decode loop.
    private static func evalTargets(_ s: LayerState) -> [MLXArray] {
        switch s {
        case .mamba(let conv, let ssm): return [conv, ssm]
        case .attention(let k, let v): return [k, v]
        case .moe: return []
        }
    }
}
