// Generator — the Swift port of `src/serve/generate.py`'s `generate()` (#167).
//
// Prefill = stepping every prompt token through `model.step`, batch 1, keeping the last
// logits. This is the PRE-#165 Python shape (before `SessionStore.prefill`'s one-shot
// parallel scan) and is numerically the same answer — `src/conformance/
// prefill_decode_parity.py` gates prefill == stepping in Python. The parallel-scan prefill
// is Swift #169's job; this file's sequential prefill is a deliberate, temporary shape, not
// an oversight — do not "optimize" it here.
//
// Per decode step: sample, stop BEFORE appending on eos, append, `onToken`, feed back through
// `model.step`, then `stopFn` — feeding back before the stop check matches
// `src/serve/generate.py`'s comment that session state must always reflect every emitted id.
//
// `MLX.eval` is called on each step's logits+state so the lazy graph does not accumulate
// across the loop — the single most likely performance/memory trap in an mlx-swift decode
// loop.

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
    public static func generate(
        model: MonicaModel, promptIds: [Int], sampler: Sampler, maxNewTokens: Int = 128,
        eosId: Int? = nil, onToken: ((Int) -> Void)? = nil,
        stopFn: (([Int]) -> Bool)? = nil, allowedIdsFor: (([Int]) -> [Int]?)? = nil
    ) throws -> [Int] {
        if promptIds.isEmpty { throw GeneratorError.emptyPrompt }

        var sampler = sampler
        var state = model.initState(batch: 1)
        var logits: MLXArray = MLXArray.zeros([1])   // overwritten before first use

        // Sequential prefill: step every prompt token, keeping only the last logits.
        for tok in promptIds {
            let tokArr = MLXArray([Int32(tok)])
            let (lg, st) = try model.step(tokArr, state)
            MLX.eval([lg] + st.flatMap(evalTargets))
            logits = lg
            state = st
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
