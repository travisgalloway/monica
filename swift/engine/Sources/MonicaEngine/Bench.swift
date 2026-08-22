// Bench — the measurement core for `monica-bench` (#170), shared with
// `monica-generate --bench-prefill` (#169's AC3, now a thin delegator over
// `Bench.prefill` — see that file's `runBenchPrefill`).
//
// Every result type is a plain `Codable` struct so `--json` can serialize it directly.
// Every timed mode warms up (once, untimed) before resetting memory counters and
// starting the clock — first-call MLX graph compile/dispatch must never leak into a
// measured window. `Bench.prefill`'s body is `monica-generate`'s original
// `runBenchPrefill`, moved verbatim, so the recorded 55.30x figure
// (docs/design/14-inference-engine.md) stays comparable.
//
// Analytic byte arithmetic is a deliberate TRANSLITERATION (not a shared library) of two
// Python formulas, cited by source so the two-maintenance-surfaces cost is visible where
// the risk actually lives:
//   - src/serve/sessions.py:per_session_state_floats  (Mamba recurrent state)
//   - scripts/bench_context.py:analytic_state_bytes's attn branch  (attention KV cache)
//
// A monitor that cannot observe its target must not read healthy: GPU memory counters
// that read 0 (CPU device, or Metal not engaged) are reported as `nil`, never as a
// measured 0 — see `Bench.memory`.

import Foundation
import MLX
import MLXNN

// MARK: - result records

public struct PrefillResult: Codable, Sendable {
    public let promptLen: Int
    public let iterations: Int
    public let sequentialMs: Double
    public let parallelMs: Double
    public let speedup: Double
    public let sequentialArgmax: Int
    public let parallelArgmax: Int
    public let sequentialTokPerSec: Double
    public let parallelTokPerSec: Double

    // Explicit public init: a struct's synthesized memberwise init is module-internal,
    // and `monica-bench` (a separate target) constructs these directly.
    public init(promptLen: Int, iterations: Int, sequentialMs: Double, parallelMs: Double,
               speedup: Double, sequentialArgmax: Int, parallelArgmax: Int,
               sequentialTokPerSec: Double, parallelTokPerSec: Double) {
        self.promptLen = promptLen
        self.iterations = iterations
        self.sequentialMs = sequentialMs
        self.parallelMs = parallelMs
        self.speedup = speedup
        self.sequentialArgmax = sequentialArgmax
        self.parallelArgmax = parallelArgmax
        self.sequentialTokPerSec = sequentialTokPerSec
        self.parallelTokPerSec = parallelTokPerSec
    }
}

public struct DecodeResult: Codable, Sendable {
    /// The exact M7 protocol (`scripts/bench_train_step.py --mode decode`): 32 warmup,
    /// 256 measured tokens, batch 1. `tokPerSec` is directly comparable to the 94.7
    /// tok/s record.
    public let warmupTokens: Int
    public let measuredTokens: Int
    public let tokPerSec: Double
    /// `Generator.generate`'s end-to-end figure (prefill + sampled decode), when a
    /// sampler was supplied — `nil` otherwise.
    public let generateTokPerSec: Double?
    public let generateTokens: Int?

    public init(warmupTokens: Int, measuredTokens: Int, tokPerSec: Double,
               generateTokPerSec: Double?, generateTokens: Int?) {
        self.warmupTokens = warmupTokens
        self.measuredTokens = measuredTokens
        self.tokPerSec = tokPerSec
        self.generateTokPerSec = generateTokPerSec
        self.generateTokens = generateTokens
    }
}

public struct MemoryResult: Codable, Sendable {
    /// `nil` = unavailable (e.g. the CPU device, or Metal not engaged) — never reported
    /// as a measured 0.
    public let peakBytes: Int?
    public let activeBytes: Int?
    public let cacheBytes: Int?
    /// Hardware-independent: per-session Mamba recurrent-state bytes (config-only).
    public let analyticStateBytes: Int
    /// Hardware-independent: attention KV-cache bytes at `contextLength` (config-only).
    /// `nil` when the config has no attention layers.
    public let analyticKvBytes: Int?
    /// Sum of every parameter's on-device byte footprint, at the model's CURRENT dtype
    /// (fp or quantized packed codes).
    public let totalWeightBytes: Int
    public let contextLength: Int

    public init(peakBytes: Int?, activeBytes: Int?, cacheBytes: Int?, analyticStateBytes: Int,
               analyticKvBytes: Int?, totalWeightBytes: Int, contextLength: Int) {
        self.peakBytes = peakBytes
        self.activeBytes = activeBytes
        self.cacheBytes = cacheBytes
        self.analyticStateBytes = analyticStateBytes
        self.analyticKvBytes = analyticKvBytes
        self.totalWeightBytes = totalWeightBytes
        self.contextLength = contextLength
    }
}

public struct SpecResult: Codable, Sendable {
    public let promptLen: Int
    public let maxNewTokens: Int
    public let gamma: Int
    public let maxN: Int
    public let plainTokPerSec: Double
    public let specTokPerSec: Double
    /// Wall-clock ratio `plainSeconds / specSeconds`. **Informational only** — no CI step
    /// asserts a threshold on it (hosted-runner timing is noisy; see `docs/benchmarks.md`).
    public let speedup: Double
    public let drafted: Int
    public let accepted: Int
    public let acceptRate: Double
    public let tokensPerRound: Double
    public let rounds: Int
    /// #172's headline criterion: speculative ids == plain greedy ids, element for
    /// element. `Bench.spec` fails loud before this can ever be recorded `false`; the field
    /// exists so the JSON record carries the claim explicitly rather than by omission.
    public let identical: Bool

    public init(promptLen: Int, maxNewTokens: Int, gamma: Int, maxN: Int,
                plainTokPerSec: Double, specTokPerSec: Double, speedup: Double,
                drafted: Int, accepted: Int, acceptRate: Double, tokensPerRound: Double,
                rounds: Int, identical: Bool) {
        self.promptLen = promptLen
        self.maxNewTokens = maxNewTokens
        self.gamma = gamma
        self.maxN = maxN
        self.plainTokPerSec = plainTokPerSec
        self.specTokPerSec = specTokPerSec
        self.speedup = speedup
        self.drafted = drafted
        self.accepted = accepted
        self.acceptRate = acceptRate
        self.tokensPerRound = tokensPerRound
        self.rounds = rounds
        self.identical = identical
    }
}

/// Machine identity — stamped onto every JSON record so `--baseline` can refuse to
/// compare across machines rather than silently reading a hosted-runner number against
/// a developer laptop's baseline (or vice versa).
public struct MachineID: Codable, Sendable, Equatable {
    public let architecture: String   // GPU.deviceInfo().architecture, or "cpu" off-Metal
    public let memorySizeBytes: Int
    public let hwModel: String        // sysctl hw.model
    public let cpuCores: Int
    public let osVersion: String
    public let swiftVersion: String
    public let device: String         // "metal" | "cpu"

    public init(architecture: String, memorySizeBytes: Int, hwModel: String, cpuCores: Int,
               osVersion: String, swiftVersion: String, device: String) {
        self.architecture = architecture
        self.memorySizeBytes = memorySizeBytes
        self.hwModel = hwModel
        self.cpuCores = cpuCores
        self.osVersion = osVersion
        self.swiftVersion = swiftVersion
        self.device = device
    }
}

public struct BenchRecord: Codable, Sendable {
    public let mode: String           // "prefill" | "decode" | "memory" | "spec"
    public let machine: MachineID
    public let ci: Bool
    public let gitSha: String?
    public let source: String?        // fixture path or --config sidecar name
    public let randomInit: Bool
    public let quantized: Bool
    public let quantBits: Int?
    public let timestamp: String
    public let prefill: PrefillResult?
    public let decode: DecodeResult?
    public let memory: MemoryResult?
    /// #172's `spec` mode. Optional so previously-written baseline/record JSON (which has
    /// no `spec` key) still decodes — `--baseline` must not start failing on old files.
    public let spec: SpecResult?

    public init(mode: String, machine: MachineID, ci: Bool, gitSha: String?, source: String?,
               randomInit: Bool, quantized: Bool, quantBits: Int?, timestamp: String,
               prefill: PrefillResult?, decode: DecodeResult?, memory: MemoryResult?,
               spec: SpecResult? = nil) {
        self.mode = mode
        self.machine = machine
        self.ci = ci
        self.gitSha = gitSha
        self.source = source
        self.randomInit = randomInit
        self.quantized = quantized
        self.quantBits = quantBits
        self.timestamp = timestamp
        self.prefill = prefill
        self.decode = decode
        self.memory = memory
        self.spec = spec
    }
}

public enum RegressionStatus: String, Codable, Sendable {
    case ok = "OK"
    case regression = "REGRESSION"
    case skipped = "SKIPPED"
}

public struct BaselineComparison: Codable, Sendable {
    public let metric: String
    public let status: RegressionStatus
    public let baseline: Double?
    public let observed: Double?
    public let message: String

    public init(metric: String, status: RegressionStatus, baseline: Double?, observed: Double?,
               message: String) {
        self.metric = metric
        self.status = status
        self.baseline = baseline
        self.observed = observed
        self.message = message
    }
}

// MARK: - Bench

public enum Bench {
    /// The arrays inside one layer's state, so `MLX.eval` can force the recurrent state
    /// alongside logits each step — duplicated (not exported before now) from
    /// `monica-generate`'s original `benchEvalTargets`/`monica-parity`'s `evalTargets`.
    public static func evalTargets(_ s: LayerState) -> [MLXArray] {
        switch s {
        case .mamba(let conv, let ssm): return [conv, ssm]
        case .attention(let k, let v): return [k, v]
        case .moe: return []
        }
    }

    // MARK: prefill (#169 AC3, moved verbatim from monica-generate's runBenchPrefill)

    /// Sequential per-token `step` prefill vs the one-shot parallel-scan `model.prefill`,
    /// timed over `iterations` after a warm-up (once each). Every logits row is checked
    /// finite before being trusted for the argmax — a degenerate run must not print a
    /// throughput number.
    public static func prefill(model: MonicaModel, promptLen: Int, iterations: Int) -> PrefillResult {
        let vocab = model.config.vocabSize
        // Prompt length is independent of vocab size — do not clamp it against `vocab`.
        let clampedLen = max(1, promptLen)
        let clampedIterations = max(1, iterations)
        var rng = SystemRandomNumberGenerator()
        let promptIds = (0..<clampedLen).map { _ in Int.random(in: 0..<vocab, using: &rng) }
        let promptArr = MLXArray(promptIds.map { Int32($0) }).reshaped([1, promptIds.count])

        func sequentialPrefill() -> (MLXArray, [LayerState]) {
            var state = model.initState(batch: 1)
            var logits = MLXArray.zeros([1])
            for tok in promptIds {
                let (lg, st) = try! model.step(MLXArray([Int32(tok)]), state)
                MLX.eval([lg] + st.flatMap(evalTargets))
                logits = lg
                state = st
            }
            return (logits, state)
        }
        func parallelPrefill() -> (MLXArray, [LayerState]) {
            let (lg, st) = model.prefill(promptArr, lastOnly: true)
            MLX.eval([lg] + st.flatMap(evalTargets))
            return (lg, st)
        }

        // Warm-up (once each) so the first call's graph compile/dispatch is not timed.
        _ = sequentialPrefill()
        _ = parallelPrefill()

        func timeMs(_ body: () -> (MLXArray, [LayerState])) -> (ms: Double, argmax: Int) {
            var lastLogits = MLXArray.zeros([1])
            let start = Date()
            for _ in 0..<clampedIterations {
                let (lg, _) = body()
                lastLogits = lg
            }
            let elapsedMs = Date().timeIntervalSince(start) * 1000.0 / Double(clampedIterations)
            let row = lastLogits.asType(.float32).reshaped([-1]).asArray(Float.self)
            precondition(row.allSatisfy { $0.isFinite },
                        "Bench.prefill: non-finite logits — run is degenerate")
            var bestI = 0
            var bestV = -Float.infinity
            for (i, v) in row.enumerated() where v > bestV { bestV = v; bestI = i }
            return (elapsedMs, bestI)
        }

        let seq = timeMs(sequentialPrefill)
        let par = timeMs(parallelPrefill)
        let speedup = par.ms > 0 ? seq.ms / par.ms : Double.infinity
        let seqTokPerSec = seq.ms > 0 ? Double(clampedLen) / (seq.ms / 1000.0) : 0
        let parTokPerSec = par.ms > 0 ? Double(clampedLen) / (par.ms / 1000.0) : 0
        return PrefillResult(
            promptLen: clampedLen, iterations: clampedIterations,
            sequentialMs: seq.ms, parallelMs: par.ms, speedup: speedup,
            sequentialArgmax: seq.argmax, parallelArgmax: par.argmax,
            sequentialTokPerSec: seqTokPerSec, parallelTokPerSec: parTokPerSec)
    }

    // MARK: decode

    /// Batch-1 `initState`+`step` loop at the exact M7 protocol
    /// (`scripts/bench_train_step.py --mode decode`): 32 warmup, 256 measured tokens,
    /// `MLX.eval` per step. When `sampler` is given, also runs `Generator.generate`
    /// once (prefill + sampled decode) for an end-to-end CLI figure.
    public static func decode(
        model: MonicaModel, sampler: Sampler? = nil, promptLen: Int = 16, maxNewTokens: Int = 64
    ) -> DecodeResult {
        let vocab = model.config.vocabSize
        let warmupTokens = 32, measuredTokens = 256
        var rng = SystemRandomNumberGenerator()
        let tokens = (0..<(warmupTokens + measuredTokens)).map { _ in
            Int32(Int.random(in: 0..<vocab, using: &rng))
        }

        var state = model.initState(batch: 1)
        var last = MLXArray.zeros([1])
        for i in 0..<warmupTokens {
            let (lg, st) = try! model.step(MLXArray([tokens[i]]), state)
            MLX.eval([lg] + st.flatMap(evalTargets))
            last = lg; state = st
        }

        let start = Date()
        for i in warmupTokens..<(warmupTokens + measuredTokens) {
            let (lg, st) = try! model.step(MLXArray([tokens[i]]), state)
            MLX.eval([lg] + st.flatMap(evalTargets))
            last = lg; state = st
        }
        let elapsed = Date().timeIntervalSince(start)
        let row = last.asType(.float32).reshaped([-1]).asArray(Float.self)
        precondition(row.allSatisfy { $0.isFinite },
                    "Bench.decode: non-finite logits — run is degenerate")
        let tokPerSec = Double(measuredTokens) / elapsed

        var generateTokPerSec: Double? = nil
        var generateTokens: Int? = nil
        if let sampler {
            let promptIds = (0..<max(1, promptLen)).map { _ in Int.random(in: 0..<vocab, using: &rng) }
            let genStart = Date()
            let outIds = (try? Generator.generate(
                model: model, promptIds: promptIds, sampler: sampler, maxNewTokens: maxNewTokens)) ?? []
            let genElapsed = Date().timeIntervalSince(genStart)
            if !outIds.isEmpty && genElapsed > 0 {
                generateTokPerSec = Double(outIds.count) / genElapsed
                generateTokens = outIds.count
            }
        }

        return DecodeResult(warmupTokens: warmupTokens, measuredTokens: measuredTokens,
                            tokPerSec: tokPerSec, generateTokPerSec: generateTokPerSec,
                            generateTokens: generateTokens)
    }

    // MARK: spec (#172)

    /// Speculative decode vs plain greedy decode on the SAME prompt and the SAME model
    /// instance, then a **byte-identity assertion** on the two id arrays.
    ///
    /// The identity check — not the timing — is what this mode exists for, and it is the
    /// one part of #172 that cannot be observed on a Command-Line-Tools-only host: it needs
    /// mlx-swift's Metal `default.metallib`, an Xcode-only build product. Hence the
    /// assertion lives HERE (in the code CI runs) and fails loud, in the same shape as
    /// `Bench.prefill`'s argmax-agreement assertion, rather than being reported as a field
    /// somebody has to remember to read.
    ///
    /// Timing follows the file's warm-up-then-time convention: one untimed spec run and one
    /// untimed greedy prefill precede the measured windows. The speedup is informational —
    /// no caller asserts a threshold on it.
    public static func spec(
        model: MonicaModel, promptLen: Int = 32, maxNewTokens: Int = 64,
        gamma: Int = 4, maxN: Int = 8
    ) throws -> SpecResult {
        let vocab = model.config.vocabSize
        let clampedLen = max(1, promptLen)
        let clampedNew = max(1, maxNewTokens)
        var rng = SystemRandomNumberGenerator()
        // A random prompt gives the prompt-lookup drafter little to match, but the
        // model's own greedy continuation is highly repetitive at toy scale, so tails do
        // recur once decoding starts — and a low accept rate weakens only the (already
        // informational) speedup, never the identity claim.
        let promptIds = (0..<clampedLen).map { _ in Int.random(in: 0..<vocab, using: &rng) }
        let drafter = PromptLookupDrafter(maxN: max(1, maxN))

        // Warm-up (untimed): first-call graph compile/dispatch must not land in a window.
        _ = try SpecDecode.generate(model: model, promptIds: promptIds, maxNewTokens: 4,
                                    gamma: gamma, drafter: drafter)
        _ = try SpecDecode.greedyReference(model: model, promptIds: promptIds,
                                           maxNewTokens: 4)

        let plainStart = Date()
        let plain = try SpecDecode.greedyReference(model: model, promptIds: promptIds,
                                                   maxNewTokens: clampedNew)
        let plainElapsed = Date().timeIntervalSince(plainStart)

        let specStart = Date()
        let (specIds, stats) = try SpecDecode.generate(
            model: model, promptIds: promptIds, maxNewTokens: clampedNew,
            gamma: gamma, drafter: drafter)
        let specElapsed = Date().timeIntervalSince(specStart)

        // #172's headline criterion. Fail loud, naming the first differing index — a
        // silent `identical: false` in a JSON record would be exactly the "a check that
        // cannot observe its target must not read healthy" failure inverted.
        precondition(specIds.count == plain.count,
                    "Bench.spec: speculative produced \(specIds.count) tokens, plain greedy "
                    + "produced \(plain.count) — speculative decoding is NOT output-preserving")
        for (i, (a, b)) in zip(plain, specIds).enumerated() where a != b {
            preconditionFailure(
                "Bench.spec: SPEC DECODE MISMATCH at token \(i): plain=\(a) spec=\(b) — "
                + "speculative decoding is NOT output-preserving")
        }

        let plainTps = plainElapsed > 0 ? Double(plain.count) / plainElapsed : 0
        let specTps = specElapsed > 0 ? Double(specIds.count) / specElapsed : 0
        let speedup = specElapsed > 0 ? plainElapsed / specElapsed : Double.infinity

        return SpecResult(
            promptLen: clampedLen, maxNewTokens: clampedNew, gamma: max(1, gamma),
            maxN: max(1, maxN), plainTokPerSec: plainTps, specTokPerSec: specTps,
            speedup: speedup, drafted: stats.drafted, accepted: stats.accepted,
            acceptRate: stats.acceptRate, tokensPerRound: stats.tokensPerRound,
            rounds: stats.rounds, identical: true)
    }

    // MARK: memory

    /// `Memory.snapshot()` (after `GPU.resetPeakMemory()`) around a prefill+decode run,
    /// plus hardware-independent analytic numbers. GPU counters reading 0 are reported
    /// as `nil` — see the module doc.
    public static func memory(
        model: MonicaModel, contextLength: Int = 512, decodeTokens: Int = 32
    ) -> MemoryResult {
        let cfg = model.config
        GPU.resetPeakMemory()

        let vocab = cfg.vocabSize
        var rng = SystemRandomNumberGenerator()
        let clampedLen = max(1, contextLength)
        let promptIds = (0..<clampedLen).map { _ in Int32(Int.random(in: 0..<vocab, using: &rng)) }
        let promptArr = MLXArray(promptIds).reshaped([1, promptIds.count])
        let (lg0, state0) = model.prefill(promptArr, lastOnly: true)
        MLX.eval([lg0] + state0.flatMap(evalTargets))

        var state = state0
        for _ in 0..<max(0, decodeTokens) {
            let tok = Int32(Int.random(in: 0..<vocab, using: &rng))
            let (lg, st) = try! model.step(MLXArray([tok]), state)
            MLX.eval([lg] + st.flatMap(evalTargets))
            state = st
        }

        let snap = Memory.snapshot()
        let peak = snap.peakMemory > 0 ? snap.peakMemory : nil
        let active = snap.activeMemory > 0 ? snap.activeMemory : nil
        let cache = snap.cacheMemory > 0 ? snap.cacheMemory : nil

        return MemoryResult(
            peakBytes: peak, activeBytes: active, cacheBytes: cache,
            analyticStateBytes: analyticStateBytes(cfg),
            analyticKvBytes: cfg.attnEvery != nil ? analyticKvBytes(cfg, length: clampedLen) : nil,
            totalWeightBytes: totalWeightBytes(model), contextLength: clampedLen)
    }

    // MARK: analytic arithmetic (mirrored, not shared, from the Python formulas cited above)

    /// `src/serve/sessions.py:per_session_state_floats`, transliterated, then charged at
    /// 4 bytes/float — the recurrent conv/SSM state is always kept fp32 regardless of
    /// the model's compute precision (`LayerState.mamba`'s doc comment).
    public static func analyticStateBytes(_ cfg: MambaConfig) -> Int {
        let perLayer = (cfg.dConv - 1) * cfg.dInner + cfg.nHeads * cfg.headDim * cfg.dState
        return cfg.nLayers * perLayer * 4
    }

    /// `scripts/bench_context.py:analytic_state_bytes`'s attn branch, transliterated:
    /// `nAttnLayers * 2 (k and v) * nAttnHeadsResolved * attnHeadDim * length * 4` bytes
    /// (the KV cache is fp32 — `LayerState.attention`'s doc comment). 0 when the config
    /// has no attention layers (callers gate the `nil` case on `cfg.attnEvery`).
    public static func analyticKvBytes(_ cfg: MambaConfig, length: Int) -> Int {
        guard let every = cfg.attnEvery, every > 0 else { return 0 }
        let nAttnLayers = cfg.nLayers / every
        return nAttnLayers * 2 * cfg.nAttnHeadsResolved * cfg.attnHeadDim * length * 4
    }

    /// Sum of every parameter's byte footprint at the model's CURRENT dtype (fp, or
    /// quantized packed codes if `model` was quantized) — `MLXArray.nbytes` already
    /// accounts for packed-uint32 quantized storage, so this needs no quantization
    /// awareness of its own.
    public static func totalWeightBytes(_ model: MonicaModel) -> Int {
        model.parameters().flattened().reduce(0) { acc, kv in acc + kv.1.nbytes }
    }

    // MARK: quant target-selection filter (throughput/footprint approximation only)

    /// In-engine mirror of `src/eval/quantize.py:quant_targets`'s FILTER LOGIC (name
    /// exclusion + group-size divisibility + `head_bits` override) — used by
    /// `monica-bench --quantize` on a `--config` random-init model, where there is no
    /// checkpoint state_dict to walk. `leaves` is `(modulePath, lastAxisSize)` for every
    /// 2-D `.weight` in the model's flattened parameter tree (module path, matching what
    /// `MLXNN.quantize(filter:)` receives — NOT the parameter name).
    ///
    /// This is a THROUGHPUT/FOOTPRINT approximation of the Python target selection, not
    /// an accuracy claim — accuracy is gated by #168's fixtures, not here. The
    /// authoritative path for a real checkpoint is always `--weights` with its `quant`
    /// sidecar (`Checkpoint.load`), never this filter.
    public static func quantTargets(
        leaves: [(path: String, lastAxis: Int)], groupSize: Int, bits: Int,
        headBits: Int? = nil, exclude: [String] = ["dt_proj", "router"]
    ) -> [String: Int] {
        var out: [String: Int] = [:]
        for (path, lastAxis) in leaves {
            if exclude.contains(where: { path.contains($0) }) { continue }
            guard groupSize > 0, lastAxis % groupSize == 0 else { continue }
            out[path] = (headBits != nil && path == "embedding") ? headBits! : bits
        }
        return out
    }

    // MARK: baseline comparison / regression flagging

    /// Compare `record` against `baselines`, matching on machine id (architecture +
    /// hw model + memory size + device) AND mode. `hwModel` alone can under-identify
    /// hardware — Apple Silicon `hw.model` values do not encode RAM configuration, so
    /// two machines of the same model with different memory would otherwise compare —
    /// hence `memorySizeBytes` joins the predicate (#170 review). OS/Swift versions are
    /// deliberately excluded: routine toolchain/OS bumps on the same CI runner or
    /// laptop should not perpetually SKIP every comparison. No match => every metric is
    /// `.skipped`, never a green `.ok` — "a comparison that cannot observe its target
    /// must not read healthy." tok/s-style metrics regress when
    /// `observed < baseline*(1-tolerance)`; ms/byte-style metrics regress when
    /// `observed > baseline*(1+tolerance)`.
    public static func compareToBaseline(
        _ record: BenchRecord, baselines: [BenchRecord], tolerance: Double = 0.15
    ) -> [BaselineComparison] {
        guard let match = baselines.first(where: {
            $0.mode == record.mode
                && $0.machine.architecture == record.machine.architecture
                && $0.machine.hwModel == record.machine.hwModel
                && $0.machine.memorySizeBytes == record.machine.memorySizeBytes
                && $0.machine.device == record.machine.device
        }) else {
            return [BaselineComparison(
                metric: record.mode, status: .skipped, baseline: nil, observed: nil,
                message: "no baseline for this machine (\(record.machine.hwModel)/"
                    + "\(record.machine.device), mode=\(record.mode))")]
        }

        func cmp(_ name: String, baseline: Double?, observed: Double?, higherIsBetter: Bool)
            -> BaselineComparison?
        {
            guard let baseline, let observed else { return nil }
            let regressed = higherIsBetter
                ? observed < baseline * (1 - tolerance)
                : observed > baseline * (1 + tolerance)
            return BaselineComparison(
                metric: name, status: regressed ? .regression : .ok,
                baseline: baseline, observed: observed,
                message: regressed
                    ? "\(name): observed \(observed) vs baseline \(baseline) "
                        + "(tolerance \(tolerance))"
                    : "\(name): within tolerance")
        }

        var out: [BaselineComparison] = []
        if let p = record.prefill, let bp = match.prefill {
            if let c = cmp("prefill.sequentialMs", baseline: bp.sequentialMs,
                          observed: p.sequentialMs, higherIsBetter: false) { out.append(c) }
            if let c = cmp("prefill.parallelMs", baseline: bp.parallelMs,
                          observed: p.parallelMs, higherIsBetter: false) { out.append(c) }
        }
        if let d = record.decode, let bd = match.decode {
            if let c = cmp("decode.tokPerSec", baseline: bd.tokPerSec,
                          observed: d.tokPerSec, higherIsBetter: true) { out.append(c) }
        }
        if let m = record.memory, let bm = match.memory {
            if let c = cmp("memory.peakBytes",
                          baseline: bm.peakBytes.map(Double.init),
                          observed: m.peakBytes.map(Double.init), higherIsBetter: false) {
                out.append(c)
            }
        }
        if out.isEmpty {
            out.append(BaselineComparison(
                metric: record.mode, status: .skipped, baseline: nil, observed: nil,
                message: "baseline matched machine but had no comparable metrics "
                    + "for mode \(record.mode)"))
        }
        return out
    }
}
