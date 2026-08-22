// SpecDecodeLoop — the MLX half of speculative decoding (#172), ported from
// `scripts/spec_decode.py::spec_decode`.
//
// Split out of `SpecDecode.swift` because the file boundary IS the provability boundary:
// everything in `SpecDecode.swift` is pure Swift and runs under `monica-bench --self-test`
// on any host, while everything here evaluates `MLXArray`s and therefore cannot execute on
// a Command-Line-Tools-only Mac (mlx-swift's `default.metallib` is an Xcode-only build
// product — see `.github/workflows/ci.yml`'s `swift-engine` job). The correctness claim
// this file carries — speculative output byte-identical to greedy output — is gated in CI
// by `monica-bench --mode spec`, and nowhere else.
//
// The verifier pass is `MonicaModel.verifyBlock`, landed by #264 as this issue's
// prerequisite and reused unchanged: it consumes a whole draft block through the `step`
// recurrence in ONE `MLX.eval` and hands back every intermediate state, so rolling back to
// an accepted prefix costs no recompute.
//
// GREEDY ONLY (see `SpecDecode.swift`): temperature>0 rejection sampling is a follow-up.

import Foundation
import MLX

extension SpecDecode {
    /// Greedy self-speculative decoding. The returned ids are byte-identical to what plain
    /// greedy decoding (`Generator.generate` with a temperature-0 sampler, or `Bench`'s
    /// argmax loop) would have produced from the same prompt and model.
    ///
    /// Loop structure is preserved from `scripts/spec_decode.py::spec_decode`:
    ///   1. Prefill the prompt in one parallel scan, keeping the last logits + state.
    ///   2. Each round, ask `drafter` for up to `min(gamma, remaining)` tokens. An empty
    ///      draft falls back to one ordinary `model.step`.
    ///   3. Otherwise verify the whole draft in ONE `verifyBlock` eval, take every greedy
    ///      prediction in ONE host transfer, accept the agreeing prefix, roll back, and
    ///      emit the verifier's own token there (a correction, or the free bonus token
    ///      when everything was accepted).
    ///
    /// - Parameters:
    ///   - gamma: draft length per round (>= 1; smaller values are clamped to 1).
    ///   - drafter: the `Drafter` seam — this function never names a concrete drafter.
    public static func generate(
        model: MonicaModel, promptIds: [Int], maxNewTokens: Int, gamma: Int, drafter: Drafter
    ) throws -> (ids: [Int], stats: SpecStats) {
        if promptIds.isEmpty { throw GeneratorError.emptyPrompt }
        let budget = max(0, maxNewTokens)
        let gammaClamped = max(1, gamma)

        // Prefill via the parallel scan — the modern path `Generator.generate` uses.
        let promptArr = MLXArray(promptIds.map { Int32($0) }).reshaped([1, promptIds.count])
        var (logits, state) = model.prefill(promptArr, lastOnly: true)
        MLX.eval([logits] + state.flatMap(\.arrays))

        var context = promptIds
        var generated: [Int] = []
        var drafted = 0, accepted = 0, rounds = 0

        while generated.count < budget {
            let remaining = budget - generated.count
            let draft = drafter.propose(context: context, gamma: min(gammaClamped, remaining))

            if draft.isEmpty {
                // No tail recurs — take one ordinary verifier step.
                let x = greedyArgmax(logits)
                let (lg, st) = try model.step(MLXArray([Int32(x)]), state)
                MLX.eval([lg] + st.flatMap(\.arrays))
                logits = lg
                state = st
                generated.append(x)
                context.append(x)
                continue
            }

            let (blockLogits, blockStates) = try model.verifyBlock(draft, state)   // one eval

            // Every greedy verifier prediction for draft positions 0...draft.count in ONE
            // host transfer (position i is the token after consuming draft[..<i]): pred[0]
            // from the pre-existing logits, pred[1...] from the block. Doing a per-position
            // `.item()` here would undo `verifyBlock`'s single eval — the exact trap
            // `scripts/spec_decode.py` calls out.
            // Each logits leaf is `(1, V)` (the same shape `monica-parity` stacks in its
            // #264 verifyBlock check); reshaping to `[1, -1]` first keeps this independent
            // of whether `prefill(lastOnly:)` hands back `(1, V)` or `(1, 1, V)`.
            let rows = ([logits] + blockLogits).map { $0.asType(.float32).reshaped([1, -1]) }
            let stackedLogits = MLX.stacked(rows, axis: 0).reshaped([rows.count, -1])
            let preds = MLX.argMax(stackedLogits, axis: -1)
                .asArray(Int32.self)
                .map(Int.init)

            let m = firstMismatch(draft, Array(preds.prefix(draft.count)))
            let accept = Array(draft.prefix(m))
            drafted += draft.count
            accepted += m
            rounds += 1

            if generated.count + m >= budget {
                // Accepted drafts already fill the budget — stop without the extra
                // correction/bonus token (and its extra step), so we never compute past
                // `maxNewTokens`.
                generated.append(contentsOf: accept)
                context.append(contentsOf: accept)
                break
            }

            // Roll back to the accepted prefix, then emit the verifier's own token there
            // (a correction if m < draft.count, the free bonus token if all accepted).
            // `blockStates[k]` is the state after consuming `draft[0...k]`, so a prefix of
            // length m rolls back to index m-1; m == 0 means the pre-block state.
            let x = preds[m]
            let baseState = (m == 0) ? state : blockStates[m - 1]
            let (lg, st) = try model.step(MLXArray([Int32(x)]), baseState)
            MLX.eval([lg] + st.flatMap(\.arrays))
            logits = lg
            state = st
            let emit = accept + [x]
            generated.append(contentsOf: emit)
            context.append(contentsOf: emit)
        }

        let ids = Array(generated.prefix(budget))
        return (ids, SpecStats(rounds: rounds, drafted: drafted, accepted: accepted,
                               generated: ids.count))
    }

    /// Greedy next token from a `(1, V)` (or `(V,)`) logits row, checked finite — a
    /// degenerate run must not silently produce an "identical" comparison.
    static func greedyArgmax(_ logits: MLXArray) -> Int {
        let row = logits.asType(.float32).reshaped([-1]).asArray(Float.self)
        precondition(row.allSatisfy { $0.isFinite },
                     "SpecDecode: non-finite logits — run is degenerate")
        var bestI = 0
        var bestV = -Float.infinity
        for (i, v) in row.enumerated() where v > bestV { bestV = v; bestI = i }
        return bestI
    }

    /// Plain greedy decode over the same prefill+`step` path, used as the byte-identity
    /// reference in `Bench.spec`. Kept here (rather than reusing `Generator.generate` with
    /// a temperature-0 sampler) so the two loops differ ONLY in the draft-and-verify block:
    /// a mismatch then means the speculative logic diverged, not that two samplers broke
    /// ties differently.
    public static func greedyReference(
        model: MonicaModel, promptIds: [Int], maxNewTokens: Int
    ) throws -> [Int] {
        if promptIds.isEmpty { throw GeneratorError.emptyPrompt }
        let promptArr = MLXArray(promptIds.map { Int32($0) }).reshaped([1, promptIds.count])
        var (logits, state) = model.prefill(promptArr, lastOnly: true)
        MLX.eval([logits] + state.flatMap(\.arrays))

        var generated: [Int] = []
        for _ in 0..<max(0, maxNewTokens) {
            let x = greedyArgmax(logits)
            generated.append(x)
            let (lg, st) = try model.step(MLXArray([Int32(x)]), state)
            MLX.eval([lg] + st.flatMap(\.arrays))
            logits = lg
            state = st
        }
        return generated
    }
}
