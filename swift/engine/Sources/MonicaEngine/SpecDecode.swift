// SpecDecode — the BACKEND-FREE half of speculative decoding (#172), the Swift port of
// `src/serve/spec_decode.py`.
//
// Deliberately mirrors its Python source in being backend-free, and mirrors
// `LossScaler.swift` in being pure-Swift policy: it is importable, compilable and
// testable even where mlx-swift cannot execute. That matters concretely here — the
// CLT-only dev host cannot load mlx-swift's `default.metallib` (an Xcode-only build
// product, see `.github/workflows/ci.yml`'s `swift-engine` job), so anything that
// evaluates an `MLXArray` is CI-only. Everything in THIS file runs locally under
// `monica-bench --self-test`; the MLX draft-and-verify loop lives in
// `SpecDecodeLoop.swift`, and the file boundary is exactly the provability boundary.
//
// GREEDY ONLY: `firstMismatch` compares draft tokens against the verifier's *argmax*, so
// the preserved output is the GREEDY decode. It is NOT distribution-preserving for
// temperature>0 / top-p sampling — that needs the Leviathan et al. rejection-sampling
// rule, which this does not implement and which #172 explicitly leaves as a follow-up.
//
// NO `import MLX` HERE. That is a contract, not a coincidence.

import Foundation

// MARK: - the drafter seam

/// A source of speculative continuations (#172's `Drafter` seam).
///
/// A drafter proposes up to `gamma` tokens continuing `context`; a wrong guess costs
/// nothing but time, because the verifier's greedy accept rule rejects it. That is what
/// lets the seam stay this small: an implementation can be arbitrarily wrong without
/// being able to change the output. Returning `[]` is always legal and means "no guess" —
/// the loop then takes one ordinary `step`.
public protocol Drafter {
    func propose(context: [Int], gamma: Int) -> [Int]
}

/// Prompt-lookup (a.k.a. n-gram / self-speculative) drafting — the Swift port of
/// `src/serve/spec_decode.py:propose`. Needs no second trained model: it proposes
/// continuations by finding where the current context's tail recurred earlier and copying
/// what followed.
public struct PromptLookupDrafter: Drafter {
    /// Longest tail pattern to try, mirroring the Python `max_n` (default 8).
    public let maxN: Int

    public init(maxN: Int = 8) {
        self.maxN = maxN
    }

    /// Up to `gamma` tokens continuing `context`, or `[]` when no tail recurs.
    ///
    /// A literal transliteration of `src/serve/spec_decode.py:propose`. Tries the longest
    /// tail first: for `n` from `min(maxN, L-1)` down to 1, take the last `n` tokens as a
    /// pattern and search for its most recent EARLIER occurrence (start positions scanned
    /// right-to-left over `ctx[0..<(L-n)]`, so the tail occurrence itself is excluded); on
    /// a hit, copy the up-to-`gamma` tokens that followed it. Longer matched patterns are
    /// preferred because they predict the continuation more reliably.
    ///
    /// The `break` in the empty-slice branch is carried over verbatim, and is the one easy
    /// porting bug: a pattern that recurred ONLY at the very end would have nothing
    /// following it, and the right move is to fall through to a SHORTER `n` (`break` out of
    /// the start loop), not to keep scanning earlier starts for the same `n` (`continue`).
    /// It is in fact unreachable in both languages — the start bound `L - n - 1` already
    /// excludes the tail occurrence, so a hit always leaves at least one following token —
    /// but it is kept so the two implementations stay line-for-line comparable. The
    /// reachable fall-through (no earlier occurrence at all for this `n`) is exercised by
    /// `monica-bench --self-test`.
    public func propose(context: [Int], gamma: Int) -> [Int] {
        let ctx = context
        let L = ctx.count
        if L < 2 || gamma <= 0 { return [] }
        var n = min(maxN, L - 1)
        while n > 0 {
            let pattern = Array(ctx[(L - n)...])
            var start = L - n - 1
            while start >= 0 {
                if Array(ctx[start..<(start + n)]) == pattern {
                    let lo = start + n
                    let hi = min(lo + gamma, L)
                    let draft = lo < hi ? Array(ctx[lo..<hi]) : []
                    if !draft.isEmpty { return draft }
                    break   // this pattern recurs only at the very end — try a shorter one
                }
                start -= 1
            }
            n -= 1
        }
        return []
    }
}

// MARK: - the greedy accept rule

public enum SpecDecode {
    /// Number of leading draft tokens the verifier agrees with (greedy acceptance) — the
    /// Swift port of `src/serve/spec_decode.py:first_mismatch`.
    ///
    /// `preds[i]` is the verifier's greedy next token at position `i` given the accepted
    /// prefix `draft[..<i]`. The accepted count is the first `i` where they differ (or
    /// `draft.count` if all agree) — exactly the prefix plain greedy decoding would also
    /// have produced, which is what makes speculative decoding output-preserving.
    ///
    /// Unequal-length inputs truncate to the shorter, exactly as Python's `zip` does.
    public static func firstMismatch(_ draft: [Int], _ preds: [Int]) -> Int {
        var m = 0
        let n = min(draft.count, preds.count)
        while m < n && draft[m] == preds[m] { m += 1 }
        return m
    }
}

// MARK: - stats

/// Draft-and-verify accounting for one `SpecDecode.generate` run — the Swift shape of the
/// `stats` dict `scripts/spec_decode.py:spec_decode` returns. Reported by
/// `monica-bench --mode spec`; informational, never threshold-gated.
public struct SpecStats: Codable, Sendable {
    /// Rounds in which a non-empty draft was proposed and verified.
    public let rounds: Int
    /// Total draft tokens proposed across those rounds.
    public let drafted: Int
    /// Total draft tokens the verifier accepted.
    public let accepted: Int
    /// `accepted / drafted`, or 0 when nothing was drafted.
    public let acceptRate: Double
    /// Emitted tokens per verification round, or 0 when there were no rounds.
    public let tokensPerRound: Double

    public init(rounds: Int, drafted: Int, accepted: Int, acceptRate: Double,
                tokensPerRound: Double) {
        self.rounds = rounds
        self.drafted = drafted
        self.accepted = accepted
        self.acceptRate = acceptRate
        self.tokensPerRound = tokensPerRound
    }

    /// Derive the two ratios rather than making callers compute them (and risk dividing
    /// by zero differently in each call site).
    public init(rounds: Int, drafted: Int, accepted: Int, generated: Int) {
        self.rounds = rounds
        self.drafted = drafted
        self.accepted = accepted
        self.acceptRate = drafted > 0 ? Double(accepted) / Double(drafted) : 0.0
        self.tokensPerRound = rounds > 0 ? Double(generated) / Double(rounds) : 0.0
    }
}
