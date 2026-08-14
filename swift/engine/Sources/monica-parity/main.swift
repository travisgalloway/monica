// monica-parity — the logit-parity gate for the Swift/MLX model port (#166).
//
// Mirrors `src/conformance/forward_step_parity.py`'s contract, one level up: instead of
// comparing Python's `forward` against Python's `step`, it compares BOTH Swift paths
// against a checked-in Python/MLX reference (`scripts/export_parity_fixture.py`). fp32,
// `rtol = 1e-4`, `atol = 1e-5`, with the PYTHON array as the reference operand — the
// numpy `allclose` formula is asymmetric, so operand order matters.
//
// Style follows `swift/Sources/monica-selfcheck/main.swift` exactly: a `failures` array, a
// final report, `exit(1)` on any failure. No XCTest (neither Swift package has a
// `.testTarget`; see .github/workflows/ci.yml).
//
// Usage:
//   swift run monica-parity                        # every checked-in fixture
//   swift run monica-parity --fixtures <dir> ...   # explicit fixture dirs (e.g. a poc export)

import Foundation
import MLX
import MLXNN
import MonicaEngine

// The fp32 gate's constants — UNCHANGED by #168/#266. A fixture's `meta.json` MAY carry
// its own `rtol`/`atol` (the #168/#266 hook); absent, every comparison below defaults to
// these exact values, so the pre-#168 fixtures behave byte-for-byte as before.
let rtol: Double = 1e-4
let atol: Double = 1e-5

// #168: the quantized-vs-fp statistical gate, mirroring
// `src/conformance/quant_parity.QUANT_THRESHOLDS` exactly (int8 nearly lossless, int4
// tolerates real but bounded drift).
let quantThresholds: [Int: (top1: Double, kl: Double)] = [8: (0.99, 1e-3), 4: (0.95, 5e-2)]

// #266: the precision -> elementwise-band table, hand-mirroring
// `src/conformance.tolerances.PARITY_TOLERANCES` (Swift cannot import Python, so
// `tests/test_lowp_parity_band.py`'s T5 is the structural guard against the two tables
// drifting apart — it walks every checked-in fixture and re-derives the RESOLVED band
// for the 8 pre-#266 fixtures, asserting it is unchanged). fp32's entry is
// BYTE-IDENTICAL to the bare `rtol`/`atol` constants above.
let precisionBands: [String: (rtol: Double, atol: Double)] = [
    "fp32": (1e-4, 1e-5),
    "fp16": (4e-3, 3e-2),
    "bf16": (3e-2, 2.5e-1),
]

// #266: the load-bearing low-precision distributional tier — mirrors
// `src/conformance.tolerances.LOWP_MEAN_KL_MAX`/`LOWP_TOP1_MIN`. fp32 has no entry: its
// elementwise band above IS the load-bearing gate (see tolerances.py's derivation,
// including why this is calibrated on the WORST of three toy configs — toy-moe.yaml —
// rather than toy.yaml alone).
let lowpKLMax: [String: Double] = ["fp16": 9e-7, "bf16": 6e-5]
let lowpTop1Min: Double = 0.99

var failures: [String] = []
// #266: per-precision fixture counts, for the final summary line (which can no longer
// hardcode "at rtol=1e-4 / atol=1e-5" now that fixtures run at different bands).
var precisionCounts: [String: Int] = [:]

/// A fixture's optional `meta.json` overrides — just the #168/#266 fields this runner
/// reads. Every field is optional so a pre-#168 `meta.json` (no `rtol`/`atol`/
/// `quant_bits`) decodes fine and changes nothing.
struct FixtureMeta: Decodable {
    /// #266: the dtype this fixture's reference logits were computed at. Required —
    /// missing or unrecognised is a FAILURE, never a default (the anti-BLIND rule this
    /// file already applies elsewhere): a fixture the runner cannot classify must never
    /// read green.
    let precision: String?
    let rtol: Double?
    let atol: Double?
    let quant_bits: Int?
    // #264: the ABSOLUTE layer indices `mixing.{i}` was exported for (attention AND MoE
    // layers skipped). Absent only for the two quantized fixtures — the exporter omits
    // the mixing oracle there (a lossy path must not drag the loose #168 tolerance hook
    // into this strict gate).
    let mixing_layers: [Int]?
    // #265: the ABSOLUTE layer indices `load.{i}` was exported for. Present only when the
    // fixture has unquantized MoE layers with `top_k < n_experts` — absent for a dense/
    // hybrid fixture (no MoE layers to count) or a quantized one (same carve-out as
    // `mixing_layers`). Its absence is legitimate ONLY in those two cases; see the #265
    // gate section below for the anti-BLIND skip-discipline check.
    let moe_load_layers: [Int]?
    /// #265: the smallest k-th/(k+1)-th selection-score gap over every MoE layer/token —
    /// the `greedy_margin_min` analogue that makes `load.{i}`'s EXACT comparison safe.
    /// Printed alongside a count mismatch so a near-tie flake is diagnosed in one read.
    let moe_route_margin_min: Double?
    /// #266 fallback ladder: DECLARES that the load-count oracle was deliberately
    /// omitted (route-margin hazard too close for this precision even after a seed
    /// search), so the anti-BLIND skip-discipline check below can accept it as a
    /// legitimate skip rather than a stale-fixture failure. Never present alongside
    /// `moe_load_layers`.
    let moe_load_omitted_reason: String?
}

func loadFixtureMeta(_ dir: URL) -> FixtureMeta {
    let empty = FixtureMeta(precision: nil, rtol: nil, atol: nil, quant_bits: nil,
                            mixing_layers: nil, moe_load_layers: nil,
                            moe_route_margin_min: nil, moe_load_omitted_reason: nil)
    guard let data = try? Data(contentsOf: dir.appendingPathComponent("meta.json")) else {
        return empty
    }
    return (try? JSONDecoder().decode(FixtureMeta.self, from: data)) ?? empty
}

/// Top-1 agreement + mean KL(fp || q) over every flattened `(B, T)` position — the
/// Swift mirror of `src/conformance/quant_parity.check_quant_parity`'s numerics (log-
/// softmax in float64 for KL stability, same as the Python reference).
func quantParity(fp: [Float], q: [Float], vocab: Int) -> (top1: Double, kl: Double) {
    let n = fp.count / vocab
    guard n > 0 else { return (0, .infinity) }
    var agree = 0
    var klSum = 0.0
    for i in 0..<n {
        let base = i * vocab
        var fpMax = -Double.infinity
        var fpArg = 0
        var qMax = -Double.infinity
        var qArg = 0
        for j in 0..<vocab {
            let fv = Double(fp[base + j])
            let qv = Double(q[base + j])
            if fv > fpMax { fpMax = fv; fpArg = j }
            if qv > qMax { qMax = qv; qArg = j }
        }
        if fpArg == qArg { agree += 1 }
        var fpSum = 0.0
        var qSum = 0.0
        for j in 0..<vocab {
            fpSum += exp(Double(fp[base + j]) - fpMax)
            qSum += exp(Double(q[base + j]) - qMax)
        }
        let fpLogZ = fpMax + log(fpSum)
        let qLogZ = qMax + log(qSum)
        var kl = 0.0
        for j in 0..<vocab {
            let fpLogp = Double(fp[base + j]) - fpLogZ
            let qLogp = Double(q[base + j]) - qLogZ
            kl += exp(fpLogp) * (fpLogp - qLogp)
        }
        klSum += kl
    }
    return (Double(agree) / Double(n), klSum / Double(n))
}

/// The arrays inside one layer's state, mirroring `Generator`'s private helper of the same
/// name — lets the AC1 greedy-id loop below `MLX.eval` the recurrent state alongside logits
/// each step, the same way `Generator.generate` keeps its lazy graph from growing unbounded.
func evalTargets(_ s: LayerState) -> [MLXArray] {
    switch s {
    case .mamba(let conv, let ssm): return [conv, ssm]
    case .attention(let k, let v): return [k, v]
    case .moe: return []
    }
}

/// #169 P3: one `LayerState`'s stateful leaves, named by SLOT (not layer index — the
/// caller prefixes that), matching `state.{i}.{slot}` in `prefill.safetensors`.
func layerStateSlots(_ s: LayerState) -> [(String, MLXArray)] {
    switch s {
    case .mamba(let conv, let ssm): return [("conv", conv), ("ssm", ssm)]
    case .attention(let k, let v): return [("k", k), ("v", v)]
    case .moe: return []
    }
}

/// numpy's `allclose` predicate, elementwise, with `ref` as the reference operand:
/// `|a - ref| <= atol + rtol * |ref|`. Returns the max absolute difference and whether
/// every element passed. `rtol`/`atol` default to the fp32 gate's constants; a
/// quantized fixture's loose comparison passes its own (#168/#266).
func compare(_ a: [Float], _ ref: [Float], rtol: Double = rtol, atol: Double = atol)
    -> (maxAbs: Double, ok: Bool)
{
    guard a.count == ref.count else { return (.infinity, false) }
    var maxAbs = 0.0
    var ok = true
    for i in 0..<a.count {
        let x = Double(a[i])
        let r = Double(ref[i])
        let d = abs(x - r)
        if d > maxAbs { maxAbs = d }
        if !(d <= atol + rtol * abs(r)) { ok = false }
    }
    return (maxAbs, ok)
}

/// Resolve `swift/engine/Fixtures/`: `$MONICA_ENGINE_FIXTURES`, then `#filePath` (this file
/// is `.../swift/engine/Sources/monica-parity/main.swift`, so three
/// `deletingLastPathComponent()`s up is `swift/engine/`), then `./Fixtures`. `nil` is a
/// FAILURE, not a skip — a parity gate that cannot see its fixtures must never read green.
func resolveFixturesDir() -> URL? {
    let fm = FileManager.default
    if let env = ProcessInfo.processInfo.environment["MONICA_ENGINE_FIXTURES"], !env.isEmpty {
        let u = URL(fileURLWithPath: env)
        if fm.fileExists(atPath: u.path) { return u }
    }
    let fromSource = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Sources/monica-parity/ -> drop main.swift
        .deletingLastPathComponent()   // -> Sources/
        .deletingLastPathComponent()   // -> swift/engine/
        .appendingPathComponent("Fixtures")
    if fm.fileExists(atPath: fromSource.path) { return fromSource }
    let cwd = URL(fileURLWithPath: "Fixtures")
    if fm.fileExists(atPath: cwd.path) { return cwd }
    return nil
}

let defaultFixtureNames = [
    "toy", "toy-hybrid", "toy-moe", "toy-moe-biased",
    "toy-moe-int8", "toy-moe-int4",  // #168
    "toy-short",  // #169: L=2, gates the conv-window LEFT-PAD branch
    "toy-fp16", "toy-bf16", "toy-moe-fp16", "toy-hybrid-fp16",  // #266
]

// --- argument parsing (hand-rolled; the runner takes repeatable/optional flags) ---
var explicitFixtures: [URL] = []
// #196: where to write Swift-authored round-trip checkpoints for the Python-side
// `scripts/check_swift_checkpoint.py` gate to read. `nil` (the default, and every
// pre-#196 invocation) means "use a scratch temp dir and don't keep it around" — CI is
// the only caller that passes this, so a plain local `swift run monica-parity` never
// litters the working tree.
var roundtripOut: URL? = nil
var args = Array(CommandLine.arguments.dropFirst())
while let first = args.first {
    args.removeFirst()
    switch first {
    case "--fixtures":
        guard let dir = args.first else {
            FileHandle.standardError.write(Data("--fixtures needs a directory\n".utf8))
            exit(2)
        }
        args.removeFirst()
        explicitFixtures.append(URL(fileURLWithPath: dir))
    case "--roundtrip-out":
        guard let dir = args.first else {
            FileHandle.standardError.write(Data("--roundtrip-out needs a directory\n".utf8))
            exit(2)
        }
        args.removeFirst()
        roundtripOut = URL(fileURLWithPath: dir)
    default:
        FileHandle.standardError.write(Data("unknown argument '\(first)'\n".utf8))
        exit(2)
    }
}

var fixtureDirs: [URL] = explicitFixtures
if fixtureDirs.isEmpty {
    guard let base = resolveFixturesDir() else {
        print("FAIL: could not resolve the fixtures directory "
              + "(set MONICA_ENGINE_FIXTURES, or run from swift/engine/)")
        exit(1)
    }
    fixtureDirs = defaultFixtureNames.map { base.appendingPathComponent($0) }
}

// --- the gate -----------------------------------------------------------------
func runGate() {
for dir in fixtureDirs {
    let name = dir.lastPathComponent
    let meta = loadFixtureMeta(dir)
    // #266: the precision -> band lookup. Missing/unrecognised `precision` is a FAILURE,
    // never a default — the same anti-BLIND rule this file already applies to a missing
    // fixtures directory and a missing generation.safetensors/prefill.safetensors.
    guard let precision = meta.precision, let band = precisionBands[precision] else {
        failures.append("\(name): meta.json has no recognised 'precision' (got "
                        + "\(meta.precision.map { "'\($0)'" } ?? "nil")) — a fixture the "
                        + "runner cannot classify must never read green")
        continue
    }
    // Mechanism 2 of the fp32/strict-gate guarantee: a per-fixture `rtol`/`atol`
    // override is accepted ONLY for a quantized fixture. A non-quantized fixture
    // carrying one is a hard failure with an explanatory message — never a silently
    // honoured loosening (this is what keeps an fp32 fixture from EVER reaching a
    // looser row of `precisionBands` through the back door).
    if (meta.rtol != nil || meta.atol != nil) && meta.quant_bits == nil {
        failures.append("\(name): meta.json carries rtol/atol but quant_bits is nil — "
                        + "only a quantized fixture may override the precision-derived "
                        + "band (precision=\(precision))")
        continue
    }
    // Mechanism 3: the override can only LOOSEN, never tighten below the physical
    // floor — `max(band, override)`. For the two quantized fixtures (fp32 + quant_bits)
    // this reproduces today's `max(1e-4, 2e-2) = 2e-2` exactly.
    let fxRtol = max(band.rtol, meta.rtol ?? 0)
    let fxAtol = max(band.atol, meta.atol ?? 0)
    precisionCounts[precision, default: 0] += 1
    do {
        let weights = dir.appendingPathComponent("weights.safetensors")
        guard FileManager.default.fileExists(atPath: weights.path) else {
            failures.append("\(name): no weights.safetensors at \(dir.path)")
            continue
        }
        let (model, ckptKeys) = try Checkpoint.load(weights: weights)

        // Belt and braces on the highest-risk step (flat-key <-> module-tree reflection):
        // `update(verify: .all)` already throws on unknown/unused/mis-shaped keys, but an
        // explicit set equality fails LOUDLY here rather than producing plausible-but-wrong
        // logits if that contract ever changes.
        let modelKeys = Set(model.parameters().flattened().map { $0.0 })
        if modelKeys != ckptKeys {
            let missing = ckptKeys.subtracting(modelKeys).sorted()
            let extra = modelKeys.subtracting(ckptKeys).sorted()
            failures.append("\(name): parameter key mismatch — "
                            + "in checkpoint only \(missing), in model only \(extra)")
            continue
        }

        let inputs = try loadArrays(url: dir.appendingPathComponent("inputs.safetensors"))
        let reference = try loadArrays(url: dir.appendingPathComponent("reference.safetensors"))
        guard let tokens = inputs["tokens"] else {
            failures.append("\(name): inputs.safetensors has no 'tokens'")
            continue
        }
        guard let refForward = reference["forward_logits"],
              let refStep = reference["step_logits"] else {
            failures.append("\(name): reference.safetensors is missing logits")
            continue
        }
        let batch = tokens.dim(0)
        let seq = tokens.dim(1)

        // --- AC1: greedy-decode id parity (#167) ---
        // A gate that cannot see its fixtures must never read green: a missing
        // generation.safetensors is a FAILURE, not a skip.
        let genPath = dir.appendingPathComponent("generation.safetensors")
        guard FileManager.default.fileExists(atPath: genPath.path) else {
            failures.append("\(name): no generation.safetensors at \(genPath.path) — "
                            + "regenerate with scripts/export_parity_fixture.py")
            continue
        }
        let genRef = try loadArrays(url: genPath)
        guard let refPromptIds = genRef["prompt_ids"], let refGreedyIds = genRef["greedy_ids"]
        else {
            failures.append("\(name): generation.safetensors is missing prompt_ids/greedy_ids")
            continue
        }
        let promptIds = refPromptIds.asType(.int32).asArray(Int32.self).map { Int($0) }
        let expectedGreedy = refGreedyIds.asType(.int32).asArray(Int32.self).map { Int($0) }
        do {
            var genState = model.initState(batch: 1)
            var genLogits: MLXArray? = nil
            for t in promptIds {
                let (lg, st) = try model.step(MLXArray([Int32(t)]), genState)
                // Eval logits AND state together, same as `Generator.generate` — leaving the
                // recurrent state arrays lazy lets the mlx graph grow unbounded across this
                // prompt+decode loop.
                MLX.eval([lg] + st.flatMap(evalTargets))
                genState = st
                genLogits = lg
            }
            var sampler = Sampler(temperature: 0)
            var actualGreedy: [Int] = []
            actualGreedy.reserveCapacity(expectedGreedy.count)
            for _ in 0..<expectedGreedy.count {
                guard let lg = genLogits else { break }
                let row = lg.asType(.float32).reshaped([-1]).asArray(Float.self)
                let nxt = try sampler.sample(row)
                actualGreedy.append(nxt)
                let (lg2, st2) = try model.step(MLXArray([Int32(nxt)]), genState)
                MLX.eval([lg2] + st2.flatMap(evalTargets))
                genState = st2
                genLogits = lg2
            }
            if actualGreedy != expectedGreedy {
                var firstDiff = -1
                for i in 0..<min(actualGreedy.count, expectedGreedy.count)
                where actualGreedy[i] != expectedGreedy[i] { firstDiff = i; break }
                if firstDiff == -1 { firstDiff = min(actualGreedy.count, expectedGreedy.count) }
                failures.append(String(
                    format: "%@: greedy id mismatch at index %d — swift=%d python=%d "
                    + "(swift=%@ python=%@)", name, firstDiff,
                    firstDiff < actualGreedy.count ? actualGreedy[firstDiff] : -1,
                    firstDiff < expectedGreedy.count ? expectedGreedy[firstDiff] : -1,
                    "\(actualGreedy)", "\(expectedGreedy)"))
            } else {
                print("\(name): greedy ids OK (\(expectedGreedy.count) steps): \(actualGreedy)")
            }
        } catch {
            failures.append("\(name): greedy-id check threw — \(error)")
        }

        // --- forward (the SSD chunked-matmul scan) ---
        let fwd = model.forward(tokens)
        MLX.eval(fwd)
        let fwdCmp = compare(fwd.asType(.float32).asArray(Float.self),
                             refForward.asType(.float32).asArray(Float.self),
                             rtol: fxRtol, atol: fxAtol)

        // --- stacked per-token step (the one-step recurrence) ---
        var state = model.initState(batch: batch)
        var stepLogits: [MLXArray] = []
        stepLogits.reserveCapacity(seq)
        for t in 0..<seq {
            let tok = tokens[0..<batch, t..<(t + 1)].reshaped([batch])
            let (lg, st) = try model.step(tok, state)
            state = st
            stepLogits.append(lg)
        }
        let stepped = stacked(stepLogits, axis: 1)
        MLX.eval(stepped)
        let stepCmp = compare(stepped.asType(.float32).asArray(Float.self),
                              refStep.asType(.float32).asArray(Float.self),
                              rtol: fxRtol, atol: fxAtol)

        // Always print the numbers, on success too (the smoke gate's habit): a max-abs-diff
        // drifting from 1e-6 to 9e-5 is the early warning that something moved.
        print(String(format: "%@: forward max|d| = %.3e  step max|d| = %.3e  (B=%d, L=%d, "
                     + "rtol=%.1e atol=%.1e) %@",
                     name, fwdCmp.maxAbs, stepCmp.maxAbs, batch, seq, fxRtol, fxAtol,
                     fwdCmp.ok && stepCmp.ok ? "OK" : "FAIL"))

        // --- #266: the low-precision LOAD-BEARING distributional tier -----------------
        // The elementwise band above is deliberately coarse at fp16/bf16 (see
        // src/conformance/tolerances.py's derivation Step 3/4) — mean KL + top-1
        // agreement against Python's OWN reference logits is what actually catches a
        // real defect at those precisions. An fp32 fixture's control flow is UNTOUCHED:
        // this tier only runs when `precision != "fp32"`.
        if precision != "fp32" {
            let klMax = lowpKLMax[precision] ?? .infinity
            let vocabLP = fwd.dim(-1)
            let (fwdTop1, fwdKL) = quantParity(
                fp: refForward.asType(.float32).asArray(Float.self),
                q: fwd.asType(.float32).asArray(Float.self), vocab: vocabLP)
            let (stepTop1, stepKL) = quantParity(
                fp: refStep.asType(.float32).asArray(Float.self),
                q: stepped.asType(.float32).asArray(Float.self), vocab: vocabLP)
            let lpOk = fwdTop1 >= lowpTop1Min && fwdKL <= klMax
                     && stepTop1 >= lowpTop1Min && stepKL <= klMax
            print(String(
                format: "%@: low-precision tier (%@) — forward top1=%.4f kl=%.3e, step "
                + "top1=%.4f kl=%.3e (want top1>=%.2f, kl<=%.1e)  %@",
                name, precision, fwdTop1, fwdKL, stepTop1, stepKL, lowpTop1Min, klMax,
                lpOk ? "OK" : "FAIL"))
            if !lpOk {
                failures.append(String(
                    format: "%@: low-precision distributional gate failed (%@) — forward "
                    + "top1=%.4f kl=%.3e, step top1=%.4f kl=%.3e (want top1>=%.2f, "
                    + "kl<=%.1e)", name, precision, fwdTop1, fwdKL, stepTop1, stepKL,
                    lowpTop1Min, klMax))
            }
        }

        // --- #264: hidden states, UNCONDITIONAL (previously only checked on a forward/step
        // failure, to localize it). Every hidden state has the SAME shape (B, L, dModel), so
        // a shape check cannot catch an off-by-one accessor (e.g. appending before applying
        // the layer, which puts the embedding at index 1 and mismatches from there on while
        // `forward`'s SEPARATE loop stays perfect) — only an unconditional per-index numeric
        // compare does. `hs.count` wrong, or a missing `hidden.{i}` key, is a FAILURE, not a
        // skip — the same "absent => failure" rule this file already applies to
        // generation.safetensors/prefill.safetensors.
        let hs = model.hiddenStates(tokens)
        MLX.eval(hs)
        if hs.count != model.layers.count + 1 {
            failures.append("\(name): hiddenStates returned \(hs.count) entries, want "
                            + "\(model.layers.count + 1) (nLayers + 1)")
        }
        var hsWorst = 0.0
        var hsFirstBad: Int? = nil
        for i in 0..<hs.count {
            guard let refH = reference["hidden.\(i)"] else {
                failures.append("\(name): reference.safetensors is missing hidden.\(i)")
                continue
            }
            let c = compare(hs[i].asType(.float32).asArray(Float.self),
                            refH.asType(.float32).asArray(Float.self),
                            rtol: fxRtol, atol: fxAtol)
            if c.maxAbs > hsWorst { hsWorst = c.maxAbs }
            if !c.ok && hsFirstBad == nil { hsFirstBad = i }
        }
        // Print the max|d| on success too, matching this file's habit elsewhere.
        print(String(format: "%@: hidden states max|d| = %.3e  %@", name, hsWorst,
                     hsFirstBad == nil ? "OK" : "FAIL"))

        if !fwdCmp.ok || !stepCmp.ok {
            if let bad = hsFirstBad {
                failures.append(String(
                    format: "%@: FIRST DIVERGENCE at hidden state %d (max|d| = %.3e) — "
                    + "that is layer %d's output (state 0 is the embedding)",
                    name, bad, hsWorst, bad - 1))
            } else {
                failures.append("\(name): logits differ but every checked-in hidden state "
                                + "matches — suspect norm_f or the tied head")
            }
            if !fwdCmp.ok {
                failures.append(String(format: "%@: forward logits differ (max|d| = %.3e)", name,
                                       fwdCmp.maxAbs))
            }
            if !stepCmp.ok {
                failures.append(String(format: "%@: step logits differ (max|d| = %.3e)", name,
                                       stepCmp.maxAbs))
            }
        } else if let bad = hsFirstBad {
            // forward/step both matched but a hidden state didn't -- can only happen if a
            // layer's PARTIAL computation is wrong in a way its own OUTPUT still cancels
            // out of by the time the next layer/forward's separate loop finishes. Still a
            // real divergence worth surfacing on its own.
            failures.append(String(
                format: "%@: hidden state %d diverges (max|d| = %.3e) even though forward/"
                + "step logits matched", name, bad, hsWorst))
        }

        // --- #264: mixing-matrix interpretability accessor -----------------------------
        // `mixing_layers` in meta.json is present for every non-quantized fixture and
        // absent only for the two quantized ones (a lossy path that must not drag the
        // loose #168 tolerance hook into this strict gate) — print an explicit SKIP line
        // for those rather than silently doing nothing.
        if let mixingLayers = meta.mixing_layers {
            let mixing = model.mixingMatrices(tokens)
            let swiftIndices = mixing.map { $0.0 }
            if swiftIndices != mixingLayers {
                failures.append("\(name): mixing-matrix layer index list \(swiftIndices) "
                                + "!= checked-in mixing_layers \(mixingLayers) — layer "
                                + "classification (attention/MoE skip) has diverged from "
                                + "Python")
            }
            var mixWorst = 0.0
            var mixFirstBad: Int? = nil
            var mixMaxAbs = 0.0   // anti-zero-matrix: the raw magnitude, not a diff
            for (i, m) in mixing {
                MLX.eval(m)
                let flat = m.asType(.float32).asArray(Float.self)
                for v in flat { mixMaxAbs = max(mixMaxAbs, Double(abs(v))) }
                guard let ref = reference["mixing.\(i)"] else {
                    failures.append("\(name): reference.safetensors is missing mixing.\(i)")
                    continue
                }
                let c = compare(flat, ref.asType(.float32).asArray(Float.self),
                                rtol: fxRtol, atol: fxAtol)
                if c.maxAbs > mixWorst { mixWorst = c.maxAbs }
                if !c.ok && mixFirstBad == nil { mixFirstBad = i }
            }
            print(String(format: "%@: mixing matrices max|d| = %.3e  (max|M| = %.3e)  %@",
                         name, mixWorst, mixMaxAbs, mixFirstBad == nil ? "OK" : "FAIL"))
            if let bad = mixFirstBad {
                failures.append(String(
                    format: "%@: mixing.%d drifted from the checked-in Swift-parity oracle "
                    + "(max|d| = %.3e)", name, bad, mixWorst))
            }
            if mixMaxAbs == 0 {
                failures.append("\(name): every mixing matrix is exactly all-zero — a "
                                + "degenerate/no-op accessor would look like this")
            }

            // (iii)/(iv)/(v): the FULL per-head matrix of the FIRST Mamba layer, computed
            // in-process — head-averaging (above) hides a transposed matrix's antisymmetric
            // structure, so causality and the defining identity must run on the un-averaged
            // (B, H, L, L) tensor.
            if let firstMamba = model.layers.enumerated().first(where: { $0.element is MambaBlock }) {
                let idx = firstMamba.offset
                // swiftlint-safe forced cast: the `first(where:)` predicate above already
                // proved this element is a MambaBlock.
                let block = firstMamba.element as! MambaBlock
                let xc = block.ssmInput(hs[idx])          // hs[idx] is layer idx's INPUT
                let mFull = block.ssm.mixingMatrix(xc)    // (B, H, L, L)
                MLX.eval(mFull)

                let bFull = mFull.dim(0), hFull = mFull.dim(1), lFull = mFull.dim(2)
                let mHost = mFull.asType(.float32).asArray(Float.self)
                var causalMax = 0.0
                var fullMaxAbs = 0.0
                for b in 0..<bFull {
                    for h in 0..<hFull {
                        let base = (b * hFull + h) * lFull * lFull
                        for i in 0..<lFull {
                            for j in 0..<lFull {
                                let v = Double(mHost[base + i * lFull + j])
                                fullMaxAbs = max(fullMaxAbs, abs(v))
                                if j > i { causalMax = max(causalMax, abs(v)) }
                            }
                        }
                    }
                }
                if causalMax != 0 {
                    failures.append(String(
                        format: "%@: mixing matrix at layer %d is NOT strictly causal — "
                        + "max|upper triangle| = %.3e (want exactly 0) — a transposed or "
                        + "wrongly-oriented matrix would mix future positions into the past",
                        name, idx, causalMax))
                }
                if fullMaxAbs == 0 {
                    failures.append("\(name): the full per-head mixing matrix at layer "
                                    + "\(idx) is exactly all-zero")
                }

                // (v) the defining identity: einsum("bhij,bjhp->bihp", M, X) == parallel(x).
                let bSize = xc.dim(0)
                let l = xc.dim(1)
                let nHeads = model.config.nHeads
                let p = model.config.headDim
                let xh = xc.reshaped([bSize, l, nHeads, p])
                let yFromMatrix = einsum("bhij,bjhp->bihp", mFull, xh)
                    .reshaped([bSize, l, nHeads * p])
                let yFromParallel = block.ssm.parallel(xc)
                // Evaluate separately, not as one combined multi-array `MLX.eval` call: see
                // tests/test_mlx_mixing_matrix.py's matching comment on the Python side —
                // this MLX/mlx-swift generation has been observed to corrupt an unrelated
                // later computation when two independent graphs are realized together here.
                MLX.eval(yFromMatrix)
                MLX.eval(yFromParallel)
                let identityCmp = compare(
                    yFromMatrix.asType(.float32).asArray(Float.self),
                    yFromParallel.asType(.float32).asArray(Float.self),
                    rtol: fxRtol, atol: fxAtol)
                print(String(
                    format: "%@: mixing-matrix identity (layer %d) max|d| = %.3e  "
                    + "causal max|upper| = %.3e  %@", name, idx, identityCmp.maxAbs,
                    causalMax, identityCmp.ok && causalMax == 0 ? "OK" : "FAIL"))
                if !identityCmp.ok {
                    failures.append(String(
                        format: "%@: mixing-matrix identity FAILED at layer %d — "
                        + "einsum(\"bhij,bjhp->bihp\", M, X) != ssm.parallel(x) "
                        + "(max|d| = %.3e)", name, idx, identityCmp.maxAbs))
                }
            }
        } else {
            print("\(name): mixing matrices — SKIP (no mixing oracle: quantized fixture)")
        }

        // --- #265: MoE load counting + route-bias write-back --------------------------
        // Placed right after mixing matrices, so a count failure is reported with the
        // logit/hidden numbers already printed above it.
        //
        // Skip discipline (the anti-BLIND rule): absence of `moe_load_layers` is only
        // legitimate for a fixture with no MoE layers or a quantized one. Anything else
        // missing it is a stale-fixture FAILURE, not a silent skip.
        let hasMoeBlocks = !model.moeBlocks().isEmpty
        if let loadLayers = meta.moe_load_layers {
            // (1) Drain, forward, pop — the pinned pass this fixture's counts describe.
            model.setMoeLoadCounting(true)
            _ = model.popMoeLoad()
            let loadFwd = model.forward(tokens)
            MLX.eval(loadFwd)
            let loads = model.popMoeLoad()
            model.setMoeLoadCounting(false)

            // (2) Layer-index list — a diverged MoE interleave fails here first.
            let swiftLoadIndices = model.layers.enumerated().compactMap { i, l in
                l is MoEBlock ? i : nil
            }
            if swiftLoadIndices != loadLayers {
                failures.append("\(name): MoE load-count layer index list "
                                + "\(swiftLoadIndices) != checked-in moe_load_layers "
                                + "\(loadLayers) — layer classification has diverged "
                                + "from Python")
            }

            let margin = meta.moe_route_margin_min.map { String(format: "%.3e", $0) } ?? "nil"
            var countsBad = false
            var totalsBad = false
            var anyNonZero = false
            var anyNonUniform = false
            let expectedTotal = Float(batch * seq * model.config.topK)
            for (i, counts) in zip(swiftLoadIndices, loads) {
                guard let ref = reference["load.\(i)"] else {
                    failures.append("\(name): reference.safetensors is missing load.\(i)")
                    continue
                }
                let refCounts = ref.asType(.float32).asArray(Float.self)
                // (3) EXACT equality — counts are exact integers, never rtol/atol.
                if counts != refCounts {
                    countsBad = true
                    failures.append("\(name): load.\(i) drifted from the checked-in "
                                    + "Swift-parity oracle (exact match required) — got "
                                    + "\(counts), want \(refCounts), "
                                    + "moe_route_margin_min=\(margin)")
                }
                // (4) Independent anti-degenerate invariants, asserted even when (3) passes.
                let total = counts.reduce(0, +)
                if abs(total - expectedTotal) > 1e-3 {
                    totalsBad = true
                    failures.append(String(
                        format: "%@: load.%d counts sum to %.1f, expected B*L*top_k = %.1f",
                        name, i, total, expectedTotal))
                }
                if counts.contains(where: { $0 != 0 }) { anyNonZero = true }
                if Set(counts).count > 1 { anyNonUniform = true }
            }
            if !anyNonZero {
                failures.append("\(name): every MoE load count is exactly zero across all "
                                + "layers — the counter looks like a no-op")
            }
            if !anyNonUniform {
                failures.append("\(name): every MoE load count is identical across every "
                                + "layer and expert — the counter looks like a no-op")
            }
            // (7) Print on success too — a distribution drifting toward uniform is the
            // early warning that routing changed, the smoke gate's habit.
            print("\(name): MoE load counts \(loads) "
                 + "\(countsBad || totalsBad ? "FAIL" : "OK") (moe_route_margin_min=\(margin))")

            // (5) Counting OFF -> forward -> pop returns all-zero. Catches an accumulator
            // that ignores its own gate — a leak that would grow the lazy graph for a
            // whole run.
            let offFwd = model.forward(tokens)
            MLX.eval(offFwd)
            let offLoads = model.popMoeLoad()
            if offLoads.contains(where: { $0.contains(where: { $0 != 0 }) }) {
                failures.append("\(name): MoE load counts are non-zero after "
                                + "setMoeLoadCounting(false) — the accumulator ignored "
                                + "its own gate (got \(offLoads))")
            }

            // (6) Route-bias WRITE-BACK, on a FRESHLY Checkpoint.load-ed second model from
            // the same weights (never the pristine `model` used above/below) — same
            // pattern the #196 round-trip section uses. No checked-in tensor of its own:
            // the saturating bias makes the outcome analytically known.
            do {
                let (modelW, _) = try Checkpoint.load(weights: weights)
                let nExperts = modelW.config.nExperts
                var saturating: [Float] = [1e4]
                saturating.append(contentsOf: Array(repeating: Float(0), count: nExperts - 1))
                let biases = loadLayers.map { _ in saturating }
                try modelW.setMoeBiases(biases)
                modelW.setMoeLoadCounting(true)
                _ = modelW.popMoeLoad()
                let biasedFwd = modelW.forward(tokens)
                MLX.eval(biasedFwd)
                let biasedLoads = modelW.popMoeLoad()
                modelW.setMoeLoadCounting(false)
                let expectedExpert0 = Float(batch * seq)
                for (i, counts) in zip(loadLayers, biasedLoads) {
                    if counts.first != expectedExpert0 {
                        failures.append(String(
                            format: "%@: round trip (e) saturating bias at layer %d did not "
                            + "drive expert 0's count to B*L=%.1f (got %@) — the "
                            + "route-bias WRITE path did not land",
                            name, i, expectedExpert0, "\(counts)"))
                    }
                    let total = counts.reduce(0, +)
                    if abs(total - expectedTotal) > 1e-3 {
                        failures.append(String(
                            format: "%@: round trip (e) layer %d counts sum to %.1f under a "
                            + "saturating bias, expected B*L*top_k=%.1f", name, i, total,
                            expectedTotal))
                    }
                }

                // moeBiases() reads back exactly what was written.
                let readBack = modelW.moeBiases()
                if readBack != biases {
                    failures.append("\(name): round trip (e) moeBiases() read-back "
                                    + "\(readBack) != what was written \(biases)")
                }

                // A wrong layer count, and a wrong-length vector, both throw.
                var threwOnLayerCount = false
                do { try modelW.setMoeBiases([saturating]) } catch { threwOnLayerCount = true }
                if !threwOnLayerCount && loadLayers.count != 1 {
                    failures.append("\(name): round trip (e) setMoeBiases with the wrong "
                                    + "layer count did not throw")
                }
                var threwOnVectorLength = false
                do {
                    try modelW.setMoeBiases(loadLayers.map { _ in [Float(0), Float(1)] })
                } catch { threwOnVectorLength = true }
                if !threwOnVectorLength {
                    failures.append("\(name): round trip (e) setMoeBiases with a "
                                    + "wrong-length vector did not throw")
                }

                // Restore and recompare — the write path must be reversible and must not
                // have corrupted anything else in the model. The ORIGINAL bias state for
                // this fixture is either "inactive" (toy-moe: `routeBias == nil`) or the
                // fixture's own checked-in bias (toy-moe-biased), read off the pristine
                // `model` used throughout this fixture's other checks. A zero vector is
                // the WRITE-PATH-representable equivalent of "inactive" — Python's own
                // comment on this: "ranking by logits is order-identical to ranking by
                // probs... a zero bias does not perturb routing" — so this restores
                // `forward_logits`-equivalence even though `setMoeBiases` has no
                // "deactivate" call of its own.
                let restoreBiases = loadLayers.map { i -> [Float] in
                    (model.layers[i] as? MoEBlock)?.routeBias
                        ?? [Float](repeating: 0, count: nExperts)
                }
                try modelW.setMoeBiases(restoreBiases)
                let restoredFwd = modelW.forward(tokens)
                MLX.eval(restoredFwd)
                let restoredCmp = compare(
                    restoredFwd.asType(.float32).asArray(Float.self),
                    refForward.asType(.float32).asArray(Float.self), rtol: fxRtol, atol: fxAtol)
                if !restoredCmp.ok {
                    failures.append(String(
                        format: "%@: round trip (e) forward logits after restoring the "
                        + "original bias do not match forward_logits (max|d| = %.3e) — the "
                        + "write path left the model corrupted", name, restoredCmp.maxAbs))
                }
            } catch {
                failures.append("\(name): round trip (e) route-bias write-back check "
                                + "threw: \(error)")
            }
        } else if let omitted = meta.moe_load_omitted_reason {
            // #266 fallback ladder: a DECLARED omission (the route-margin hazard was too
            // close for this precision even after a seed search) is a legitimate skip,
            // not a stale fixture — see scripts/export_parity_fixture.py's
            // --allow-moe-load-omit.
            print("\(name): MoE load counts — SKIP (declared omission: \(omitted))")
        } else if hasMoeBlocks && meta.quant_bits == nil {
            failures.append("\(name): stale fixture — model has MoE layers and is not "
                            + "quantized, but meta.json has no moe_load_layers and no "
                            + "moe_load_omitted_reason — regenerate with "
                            + "scripts/export_parity_fixture.py")
        } else {
            print("\(name): MoE load counts — SKIP (\(hasMoeBlocks ? "quantized fixture" : "no MoE layers"))")
        }

        // --- #169: prefill (parallel-scan) checks -------------------------------------
        // A missing prefill.safetensors is a FAILURE, not a skip — same rule the file
        // already applies to generation.safetensors.
        let preURL = dir.appendingPathComponent("prefill.safetensors")
        guard FileManager.default.fileExists(atPath: preURL.path) else {
            failures.append("\(name): no prefill.safetensors at \(preURL.path) — "
                            + "regenerate with scripts/export_parity_fixture.py")
            continue
        }
        let preRef = try loadArrays(url: preURL)
        guard let refPreLogits = preRef["prefill_logits"],
              let refPreLast = preRef["prefill_last_logits"] else {
            failures.append("\(name): prefill.safetensors is missing prefill_logits/"
                            + "prefill_last_logits")
            continue
        }

        // P1: prefill's full logits vs the Python oracle AND vs this fixture's own
        // forward_logits (prefill must not have drifted from the training path).
        let (preLogits, preState) = model.prefill(tokens)
        MLX.eval([preLogits] + preState.flatMap(evalTargets))
        let p1VsPython = compare(preLogits.asType(.float32).asArray(Float.self),
                                 refPreLogits.asType(.float32).asArray(Float.self),
                                 rtol: fxRtol, atol: fxAtol)
        let p1VsForward = compare(preLogits.asType(.float32).asArray(Float.self),
                                  refForward.asType(.float32).asArray(Float.self),
                                  rtol: fxRtol, atol: fxAtol)
        if !p1VsPython.ok {
            failures.append(String(
                format: "%@: P1 prefill logits vs Python oracle differ (max|d| = %.3e)",
                name, p1VsPython.maxAbs))
        }
        if !p1VsForward.ok {
            failures.append(String(
                format: "%@: P1 prefill logits vs this fixture's own forward logits differ "
                + "(max|d| = %.3e) — prefill has drifted from the training path",
                name, p1VsForward.maxAbs))
        }

        // P2: lastOnly honesty — vs the Python oracle AND vs the last row of P1's own logits.
        let (preLast, _) = model.prefill(tokens, lastOnly: true)
        MLX.eval(preLast)
        let lastOfFull = preLogits[0..., (seq - 1)..<seq, 0...].squeezed(axis: 1)
        MLX.eval(lastOfFull)
        let p2VsPython = compare(preLast.asType(.float32).asArray(Float.self),
                                 refPreLast.asType(.float32).asArray(Float.self),
                                 rtol: fxRtol, atol: fxAtol)
        let p2VsSelf = compare(preLast.asType(.float32).asArray(Float.self),
                               lastOfFull.asType(.float32).asArray(Float.self),
                               rtol: fxRtol, atol: fxAtol)
        if !p2VsPython.ok {
            failures.append(String(
                format: "%@: P2 lastOnly logits vs Python oracle differ (max|d| = %.3e)",
                name, p2VsPython.maxAbs))
        }
        if !p2VsSelf.ok {
            failures.append(String(
                format: "%@: P2 lastOnly logits vs the last row of prefill's own full logits "
                + "differ (max|d| = %.3e)", name, p2VsSelf.maxAbs))
        }

        // P3 — STATE HANDOFF, the load-bearing check. Two independent comparisons of the
        // same Swift-prefilled state:
        //   (a) cross-language: each leaf vs Python's exported state.{i}.{slot}.
        //   (b) intra-Swift: `state` (this loop already stepped all `seq` tokens from
        //       initState above, to build `stepped`) vs the prefilled state, element-wise.
        var p3WorstCross = 0.0
        var p3WorstIntra = 0.0
        var p3FirstBadCross: String? = nil
        var p3FirstBadIntra: String? = nil
        for i in 0..<preState.count {
            for (slot, arr) in layerStateSlots(preState[i]) {
                let key = "state.\(i).\(slot)"
                guard let ref = preRef[key] else {
                    failures.append("\(name): P3(a) prefill.safetensors is missing '\(key)' "
                                    + "— structural disagreement between Swift and Python "
                                    + "layer \(i)")
                    continue
                }
                MLX.eval(arr)
                let c = compare(arr.asType(.float32).asArray(Float.self),
                                ref.asType(.float32).asArray(Float.self),
                                rtol: fxRtol, atol: fxAtol)
                if c.maxAbs > p3WorstCross { p3WorstCross = c.maxAbs }
                if !c.ok && p3FirstBadCross == nil { p3FirstBadCross = "layer \(i) \(slot)" }
            }
            switch (preState[i], state[i]) {
            case (.mamba(let c1, let s1), .mamba(let c2, let s2)):
                for (label, a, b) in [("conv", c1, c2), ("ssm", s1, s2)] {
                    MLX.eval([a, b])
                    let c = compare(a.asType(.float32).asArray(Float.self),
                                    b.asType(.float32).asArray(Float.self),
                                    rtol: fxRtol, atol: fxAtol)
                    if c.maxAbs > p3WorstIntra { p3WorstIntra = c.maxAbs }
                    if !c.ok && p3FirstBadIntra == nil { p3FirstBadIntra = "layer \(i) \(label)" }
                }
            case (.attention(let k1, let v1), .attention(let k2, let v2)):
                for (label, a, b) in [("k", k1, k2), ("v", v1, v2)] {
                    MLX.eval([a, b])
                    let c = compare(a.asType(.float32).asArray(Float.self),
                                    b.asType(.float32).asArray(Float.self),
                                    rtol: fxRtol, atol: fxAtol)
                    if c.maxAbs > p3WorstIntra { p3WorstIntra = c.maxAbs }
                    if !c.ok && p3FirstBadIntra == nil { p3FirstBadIntra = "layer \(i) \(label)" }
                }
            case (.moe, .moe):
                break
            default:
                failures.append("\(name): P3(b) layer \(i) prefilled and stepped state are "
                                + "different LayerState cases — structural disagreement")
            }
        }
        print(String(format: "%@: P3 state handoff — cross-lang max|d| = %.3e, "
                     + "intra-Swift max|d| = %.3e  %@", name, p3WorstCross, p3WorstIntra,
                     p3FirstBadCross == nil && p3FirstBadIntra == nil ? "OK" : "FAIL"))
        if let bad = p3FirstBadCross {
            failures.append("\(name): P3(a) state handoff FIRST DIVERGENCE (vs Python) at "
                            + "\(bad) (max|d| = \(String(format: "%.3e", p3WorstCross)))")
        }
        if let bad = p3FirstBadIntra {
            failures.append("\(name): P3(b) state handoff FIRST DIVERGENCE (vs Swift's own "
                            + "step) at \(bad) (max|d| = \(String(format: "%.3e", p3WorstIntra)))")
        }

        // P4: prefill-then-decode vs pure step-by-step, anchored to the Python step_logits
        // oracle (the end-to-end property serving depends on).
        let nDecode = min(4, seq - 1)
        let promptLen = seq - nDecode
        let promptTok = tokens[0..<batch, 0..<promptLen]
        let (_, stateA0) = model.prefill(promptTok)
        var stateA = stateA0
        var mixed: [MLXArray] = []
        mixed.reserveCapacity(nDecode)
        for t in 0..<nDecode {
            let tok = tokens[0..<batch, (promptLen + t)..<(promptLen + t + 1)].reshaped([batch])
            let (lg, st) = try model.step(tok, stateA)
            MLX.eval([lg] + st.flatMap(evalTargets))
            stateA = st
            mixed.append(lg)
        }
        let mixedArr = stacked(mixed, axis: 1)                        // (B, nDecode, V)
        let refTail = refStep[0..., promptLen..<seq, 0...]
        MLX.eval([mixedArr, refTail])
        let p4Cmp = compare(mixedArr.asType(.float32).asArray(Float.self),
                            refTail.asType(.float32).asArray(Float.self),
                            rtol: fxRtol, atol: fxAtol)
        print(String(format: "%@: P4 prefill-then-decode vs pure step max|d| = %.3e  %@",
                     name, p4Cmp.maxAbs, p4Cmp.ok ? "OK" : "FAIL"))
        if !p4Cmp.ok {
            failures.append(String(
                format: "%@: P4 prefill-then-decode logits vs pure step-by-step differ "
                + "(max|d| = %.3e)", name, p4Cmp.maxAbs))
        }

        // P5: greedy ids through the prefill path — exact id equality against the SAME
        // generation.safetensors oracle the sequential AC1 check above used.
        do {
            let p5Sampler = Sampler(temperature: 0)
            let genIds = try Generator.generate(
                model: model, promptIds: promptIds, sampler: p5Sampler,
                maxNewTokens: expectedGreedy.count, usePrefill: true)
            if genIds != expectedGreedy {
                failures.append("\(name): P5 greedy ids via the prefill path mismatch — "
                                + "swift=\(genIds) python=\(expectedGreedy)")
            } else {
                print("\(name): P5 prefill-path greedy ids OK (\(genIds.count) steps)")
            }
        } catch {
            failures.append("\(name): P5 threw — \(error)")
        }

        // --- #264: verifyBlock, the #172 speculative-decoding prerequisite -------------
        // Gated against oracles that already exist (no new checked-in tensor, per
        // swift/engine/Fixtures/README.md's #195 note): row 0's logits vs `step_logits`,
        // the FINAL state vs `prefill.safetensors` (state after L tokens is, by contract,
        // the stepped state after L tokens — already gated Python-side by
        // `check_prefill_decode_parity`), and an in-process rollback identity that proves
        // `states[k]` is not simply the final state repeated.
        do {
            let row0Ids = tokens[0..<1, 0..<seq].reshaped([seq])
                .asType(.int32).asArray(Int32.self).map { Int($0) }
            let (vbLogits, vbStates) = try model.verifyBlock(row0Ids, model.initState(batch: 1))
            if vbLogits.count != row0Ids.count || vbStates.count != row0Ids.count {
                failures.append("\(name): verifyBlock returned \(vbLogits.count) logits / "
                                + "\(vbStates.count) states for \(row0Ids.count) tokens")
            } else {
                // (ii) row 0's per-token logits vs the checked-in step_logits[0:1].
                let stackedLogits = stacked(vbLogits, axis: 1)     // (1, T, V)
                MLX.eval(stackedLogits)
                let refRow0 = refStep[0..<1, 0..., 0...]
                let vbCmp = compare(stackedLogits.asType(.float32).asArray(Float.self),
                                    refRow0.asType(.float32).asArray(Float.self),
                                    rtol: fxRtol, atol: fxAtol)
                if !vbCmp.ok {
                    failures.append(String(
                        format: "%@: verifyBlock logits vs step_logits[0:1] differ "
                        + "(max|d| = %.3e)", name, vbCmp.maxAbs))
                }

                // (iii) the FINAL state vs prefill.safetensors, sliced to row 0.
                let finalState = vbStates[vbStates.count - 1]
                var vbFinalWorst = 0.0
                var vbFinalBad: String? = nil
                for i in 0..<finalState.count {
                    for (slot, arr) in layerStateSlots(finalState[i]) {
                        let key = "state.\(i).\(slot)"
                        guard let ref = preRef[key] else {
                            failures.append("\(name): verifyBlock final-state check: "
                                            + "prefill.safetensors is missing '\(key)'")
                            continue
                        }
                        let refRow0Slot = ref[0..<1]
                        MLX.eval(arr)
                        let c = compare(arr.asType(.float32).asArray(Float.self),
                                        refRow0Slot.asType(.float32).asArray(Float.self),
                                        rtol: fxRtol, atol: fxAtol)
                        if c.maxAbs > vbFinalWorst { vbFinalWorst = c.maxAbs }
                        if !c.ok && vbFinalBad == nil { vbFinalBad = "layer \(i) \(slot)" }
                    }
                }
                if let bad = vbFinalBad {
                    failures.append(String(
                        format: "%@: verifyBlock final state vs prefill.safetensors row 0 "
                        + "FIRST DIVERGENCE at %@ (max|d| = %.3e)", name, bad, vbFinalWorst))
                }

                // (iv) in-process rollback identity: states[k] must equal a SEQUENTIAL
                // `step` walk over tokens[0...k], at k=0 and mid-sequence — the property
                // #172 depends on, and what distinguishes real per-token states from the
                // final state simply repeated.
                var rollbackBad: [Int] = []
                for k in [0, row0Ids.count / 2] where k < vbStates.count {
                    var walkState = model.initState(batch: 1)
                    for t in 0...k {
                        let (_, st) = try model.step(MLXArray([Int32(row0Ids[t])]), walkState)
                        walkState = st
                    }
                    let vbArrays = vbStates[k].flatMap(\.arrays)
                    let walkArrays = walkState.flatMap(\.arrays)
                    MLX.eval(vbArrays)
                    MLX.eval(walkArrays)
                    var kBad = false
                    for (a, b) in zip(vbArrays, walkArrays) {
                        let c = compare(a.asType(.float32).asArray(Float.self),
                                        b.asType(.float32).asArray(Float.self),
                                        rtol: fxRtol, atol: fxAtol)
                        if !c.ok { kBad = true }
                    }
                    if kBad { rollbackBad.append(k) }
                }
                if !rollbackBad.isEmpty {
                    failures.append("\(name): verifyBlock intermediate-state rollback "
                                    + "FAILED at k=\(rollbackBad) — states[k] does not equal "
                                    + "a sequential step walk over tokens[0...k]")
                }
                print(String(
                    format: "%@: verifyBlock (%d tokens) logits max|d| = %.3e, final-state "
                    + "max|d| = %.3e  %@", name, row0Ids.count, vbCmp.maxAbs, vbFinalWorst,
                    vbCmp.ok && vbFinalBad == nil && rollbackBad.isEmpty ? "OK" : "FAIL"))
            }
        } catch {
            failures.append("\(name): verifyBlock threw — \(error)")
        }

        // --- #263 P6: packing-aware seg_ids forward path (chunk mask, boundary conv,
        // block-diagonal attention) --------------------------------------------------
        // `packed.safetensors` is OPTIONAL — absent means this fixture doesn't exercise
        // packing (skip), present-but-unreadable is a FAILURE (the `try loadArrays` below
        // bubbles into this section's `do`'s outer `catch`, same rule as every other
        // required-once-present artifact in this file).
        let packedURL = dir.appendingPathComponent("packed.safetensors")
        if FileManager.default.fileExists(atPath: packedURL.path) {
            let packedData = try loadArrays(url: packedURL)
            guard let packedTokens = packedData["packed_tokens"],
                  let packedSegIds = packedData["packed_seg_ids"],
                  let docLengths = packedData["doc_lengths"],
                  let packedLogitsRef = packedData["packed_logits"] else {
                failures.append("\(name): packed.safetensors is missing one of "
                                + "packed_tokens/packed_seg_ids/doc_lengths/packed_logits")
                continue
            }
            // Rebuild each document's REAL-token span within the packed sequence, using
            // the SAME chunk-aligned packing rule the exporter used
            // (`check_doc_boundary_parity`'s contract): doc `n` pads to a multiple of Q,
            // and consecutive docs are laid out back to back.
            let q = model.config.q
            let docLens = docLengths.asType(.int32).asArray(Int32.self).map { Int($0) }
            var spans: [(Int, Int)] = []
            var off = 0
            for n in docLens {
                let plen = ((n + q - 1) / q) * q
                spans.append((off, off + n))
                off += plen
            }
            if spans.count < 2 {
                failures.append("\(name): P6 packed.safetensors has fewer than 2 "
                                + "documents — a single-doc fixture cannot prove "
                                + "packing-awareness")
                continue
            }

            // Assertion (1) cross-language, and the anti-no-op baseline (3): the SAME
            // blind forward (no segIds) both feeds assertion (3) below and would be
            // exactly what a no-op seg_ids implementation returns for the packed-aware
            // call — so if this whole section silently degenerated to a no-op port,
            // assertion (1) would already fail against Python's packing-aware oracle.
            let blindLogits = model.forward(packedTokens)
            let packedAware = model.forward(packedTokens, segIds: packedSegIds)
            MLX.eval([blindLogits, packedAware])
            // #266: the DTYPE band (`fxRtol`/`fxAtol`, resolved above from `precision`),
            // not the file-level fp32 constants — byte-identical for every fp32 fixture
            // (the only kind that carries packed.safetensors today), and correct if a
            // low-precision packed fixture is ever added. Still NEVER the per-fixture
            // `rtol`/`atol` override hook (that stays reserved for quantized fixtures).
            let p6CrossLang = compare(packedAware.asType(.float32).asArray(Float.self),
                                      packedLogitsRef.asType(.float32).asArray(Float.self),
                                      rtol: fxRtol, atol: fxAtol)

            // Assertions (2) self-consistency and (3) anti-no-op, per document.
            var p6WorstSelf = 0.0
            var p6FirstBadSelf: Int? = nil
            var p6MinBlindGap = Double.infinity
            var p6BlindGapFails: [Int] = []
            // #266: scale the anti-no-op threshold from the bare `1e-2` to
            // `max(1e-2, 4 * fxAtol)` — unchanged at fp32 (`4 * 1e-5 = 4e-5 < 1e-2`), but
            // correct if a low-precision packed fixture is ever added (its own noise
            // floor could otherwise exceed a hardcoded 1e-2).
            let p6BlindGapThreshold = max(1e-2, 4 * fxAtol)
            for (d, span) in spans.enumerated() {
                let (s, e) = span
                let docTokens = packedTokens[0..<1, s..<e]
                let solo = model.forward(docTokens)                    // standalone, in-process
                let sub = packedAware[0..<1, s..<e, 0...]
                MLX.eval([solo, sub])
                let cSelf = compare(sub.asType(.float32).asArray(Float.self),
                                    solo.asType(.float32).asArray(Float.self),
                                    rtol: fxRtol, atol: fxAtol)
                if cSelf.maxAbs > p6WorstSelf { p6WorstSelf = cSelf.maxAbs }
                if !cSelf.ok && p6FirstBadSelf == nil { p6FirstBadSelf = d }

                if d > 0 {   // doc 0 never leaks a PRIOR document's state, so it proves
                             // nothing about packing-awareness — matches the plan's
                             // "for every doc after the first" wording.
                    let blindSub = blindLogits[0..<1, s..<e, 0...]
                    MLX.eval(blindSub)
                    let cBlind = compare(blindSub.asType(.float32).asArray(Float.self),
                                         solo.asType(.float32).asArray(Float.self),
                                         rtol: 0, atol: 0)   // only .maxAbs is read below
                    if cBlind.maxAbs < p6MinBlindGap { p6MinBlindGap = cBlind.maxAbs }
                    if cBlind.maxAbs <= p6BlindGapThreshold { p6BlindGapFails.append(d) }
                }
            }

            print(String(
                format: "%@: P6 packing-aware forward — cross-lang max|d| = %.3e, "
                + "self-consistency max|d| = %.3e, anti-no-op min gap = %.3e  %@",
                name, p6CrossLang.maxAbs, p6WorstSelf,
                p6MinBlindGap.isFinite ? p6MinBlindGap : 0,
                p6CrossLang.ok && p6FirstBadSelf == nil && p6BlindGapFails.isEmpty
                    ? "OK" : "FAIL"))
            if !p6CrossLang.ok {
                failures.append(String(
                    format: "%@: P6(1) packed-aware forward vs the Python packed_logits "
                    + "oracle differ (max|d| = %.3e)", name, p6CrossLang.maxAbs))
            }
            if let bad = p6FirstBadSelf {
                failures.append(String(
                    format: "%@: P6(2) FIRST DIVERGENCE — packed-aware doc %d's logit "
                    + "slice vs its own standalone forward (worst max|d| across docs = "
                    + "%.3e) — state leaked across a packed document boundary",
                    name, bad, p6WorstSelf))
            }
            if !p6BlindGapFails.isEmpty {
                failures.append(String(
                    format: "%@: P6(3) anti-no-op check failed for doc(s) %@ — the "
                    + "packing-BLIND forward is not provably different from the "
                    + "standalone forward at the %.1e threshold (min observed gap = "
                    + "%.3e); this makes P6(2) unable to distinguish a real port from a "
                    + "no-op one on this fixture — see swift/engine/Fixtures/README.md",
                    name, "\(p6BlindGapFails)", p6BlindGapThreshold, p6MinBlindGap))
            }
        }

        // --- #196: checkpoint round trip (the Swift WRITER, not just the reader) ------
        // Writes this fixture's already-loaded model back out with `Checkpoint.save`,
        // reloads it, and checks the result is functionally — and, for tensors,
        // EXACTLY — the same model. This is the Swift half of #196's acceptance
        // criterion ("a checkpoint round-trips without changing the model's function");
        // the Python half is `scripts/check_swift_checkpoint.py` reading whatever this
        // section writes under `--roundtrip-out`. No new checked-in fixture: every
        // comparison below is computed in-process from the fixture already loaded above,
        // and the two SAME-PROCESS exact-tensor comparisons ((a) and (b)) are immune to
        // the cross-machine drift that broke #293's checked-in gradient oracle.
        do {
            let priorFailCount = failures.count
            // When the caller didn't ask to keep the round-trip artifacts
            // (`--roundtrip-out`), we write to a scratch dir under the system temp
            // directory — clean it up on the way out so repeated local runs don't
            // accumulate one throwaway directory per fixture per run.
            let isScratchDir = roundtripOut == nil
            let rtBase = (roundtripOut?.appendingPathComponent(name))
                ?? FileManager.default.temporaryDirectory
                    .appendingPathComponent("monica-parity-roundtrip-\(UUID().uuidString)")
                    .appendingPathComponent(name)
            try FileManager.default.createDirectory(
                at: rtBase, withIntermediateDirectories: true)
            defer {
                if isScratchDir {
                    try? FileManager.default.removeItem(at: rtBase)
                }
            }

            // (a) identity round trip: save this model, reload it, compare everything.
            let rtWeightsA = rtBase.appendingPathComponent("weights.safetensors")
            try Checkpoint.save(model, to: rtWeightsA)
            let (modelB, ckptKeysB) = try Checkpoint.load(weights: rtWeightsA)
            let modelBKeys = Set(modelB.parameters().flattened().map { $0.0 })

            if ckptKeysB != ckptKeys || modelBKeys != modelKeys {
                failures.append("\(name): round trip (a) key-set mismatch — ckpt: "
                                + "\(ckptKeysB) vs \(ckptKeys), model: \(modelBKeys) vs \(modelKeys)")
            }

            // Every parameter tensor must be BIT-IDENTICAL — save/load is lossless, so
            // anything else is a serializer bug, not a numerics question.
            let flatA = Dictionary(
                model.parameters().flattened(), uniquingKeysWith: { a, _ in a })
            let flatB = Dictionary(
                modelB.parameters().flattened(), uniquingKeysWith: { a, _ in a })
            var badTensor: String? = nil
            for (k, a) in flatA {
                guard let b = flatB[k] else { continue }   // key-set mismatch reported above
                MLX.eval([a, b])
                if a.shape != b.shape || !MLX.all(a .== b).item(Bool.self) {
                    badTensor = badTensor ?? k
                }
            }
            if let bad = badTensor {
                failures.append("\(name): round trip (a) tensor '\(bad)' is not "
                                + "bit-identical after save/load")
            }

            // The reloaded sidecar must parse to the SAME config — known fields AND the
            // raw-JSON passthrough (`MambaConfig == ` covers both; see Config.swift).
            if model.config != modelB.config {
                failures.append("\(name): round trip (a) sidecar mismatch — the reloaded "
                                + "config (known fields + raw passthrough) differs from "
                                + "the source — sidecar fidelity is broken")
            }

            // End-to-end: the round-tripped model must still match the Python oracle.
            let rtFwd = modelB.forward(tokens)
            MLX.eval(rtFwd)
            let rtCmp = compare(rtFwd.asType(.float32).asArray(Float.self),
                                refForward.asType(.float32).asArray(Float.self),
                                rtol: fxRtol, atol: fxAtol)
            if !rtCmp.ok {
                failures.append(String(
                    format: "%@: round trip (a) forward logits (post save/load) differ "
                    + "from the Python oracle (max|d| = %.3e)", name, rtCmp.maxAbs))
            }

            // (b) mutation round trip — the anti-file-copy check. Without this, a `save`
            // that just byte-copied the source file would pass (a) too. Mutates modelB
            // (the round-tripped copy from (a), otherwise unused from here on) rather
            // than the shared `model`, which the #168 quant section below still reads.
            //
            // mlx-swift caches MLXArray references via Mirror at the first items() call
            // and never re-scans. Direct property assignment (`model.normF.weight = ...`)
            // replaces the STORED PROPERTY reference but leaves the cache pointing at the
            // old MLXArray — so parameters()/portableStateDict still see the old value.
            // model.update(parameters:) calls _updateInternal on the cached arrays in
            // place: both the stored property and the cache remain the SAME object with
            // the new data, which is what portableStateDict observes.
            let mutKey = "norm_f.weight"
            let preMut = Dictionary(
                modelB.parameters().flattened(), uniquingKeysWith: { a, _ in a })[mutKey]!
            let mutWeight = preMut * MLXArray(Float(2))
            MLX.eval(mutWeight)
            modelB.update(parameters: ModuleParameters.unflattened([(mutKey, mutWeight)]))
            MLX.eval(modelB.normF.weight)
            let rtWeightsB = rtBase.appendingPathComponent("weights-mutated.safetensors")
            try Checkpoint.save(modelB, to: rtWeightsB)
            let (modelC, _) = try Checkpoint.load(weights: rtWeightsB)
            MLX.eval([modelB.normF.weight, modelC.normF.weight])
            if modelB.normF.weight.shape != modelC.normF.weight.shape
                || !MLX.all(modelB.normF.weight .== modelC.normF.weight).item(Bool.self) {
                failures.append("\(name): round trip (b) mutation not preserved exactly "
                                + "— the writer may be copying the source file rather "
                                + "than serializing the model's actual parameters")
            }

            // (c) MoE route bias survives (or correctly does NOT ride) the round trip.
            let savedKeys = Set(try loadArrays(url: rtWeightsA).keys)
            let savedBiasKeys = savedKeys.filter { $0.hasPrefix(Checkpoint.moeBiasPrefix) }
            let expectBias = model.moeBlocks().contains { $0.routeBias != nil }
            if expectBias {
                for (i, layer) in model.layers.enumerated() {
                    guard let moe = layer as? MoEBlock, let bias = moe.routeBias else {
                        continue
                    }
                    guard let moeB = modelB.layers[i] as? MoEBlock,
                          let biasB = moeB.routeBias else {
                        failures.append("\(name): round trip (c) layer \(i) lost its "
                                        + "route bias entirely")
                        continue
                    }
                    if bias != biasB {
                        failures.append("\(name): round trip (c) layer \(i) route bias "
                                        + "changed across the round trip (want \(bias), "
                                        + "got \(biasB))")
                    }
                }
                if savedBiasKeys.isEmpty {
                    failures.append("\(name): round trip (c) model has an active route "
                                    + "bias but the saved checkpoint carries no "
                                    + "\(Checkpoint.moeBiasPrefix)* key")
                }
            } else if !savedBiasKeys.isEmpty {
                failures.append("\(name): round trip (c) model has NO active route bias "
                                + "but the saved checkpoint carries "
                                + "\(savedBiasKeys.sorted()) — an unconditional bias key "
                                + "would break Python's num_parameters()")
            }

            // (d) quantized checkpoints: (a) already bit-compares the packed
            // weight/scales/biases tensors; additionally confirm the `quant` block
            // itself re-decodes to the same mode/group_size/targets.
            if let bits = meta.quant_bits {
                let sidecarURL = URL(fileURLWithPath: weights.path + ".config.json")
                if let origSpec = try QuantSpec.load(sidecar: sidecarURL) {
                    let rtSidecarURL = URL(fileURLWithPath: rtWeightsA.path + ".config.json")
                    if let rtSpec = try QuantSpec.load(sidecar: rtSidecarURL) {
                        if rtSpec.mode != origSpec.mode
                            || rtSpec.groupSize != origSpec.groupSize
                            || rtSpec.targets != origSpec.targets {
                            failures.append("\(name): round trip (d) quant block changed "
                                            + "across the round trip (bits=\(bits))")
                        }
                    } else {
                        failures.append("\(name): round trip (d) the saved sidecar lost "
                                        + "its quant block entirely (bits=\(bits))")
                    }
                }
            }

            if failures.count == priorFailCount {
                print("\(name): round trip (save -> load) OK — tensors bit-identical, "
                      + "sidecar fidelity, mutation, and route-bias checks passed"
                      + (meta.quant_bits != nil ? " (+ quant-block re-decode)" : ""))
            }
        } catch {
            failures.append("\(name): round trip threw — \(error)")
        }

        // --- #168: quantized-fixture-only checks -------------------------------------
        // A fixture whose meta.json declares `quant_bits` MUST carry dequant_ref.safetensors
        // and reference.safetensors's `fp_forward_logits` — a missing artifact here is a
        // FAILURE, never a silently-skipped check (same rule the file already applies to
        // generation.safetensors).
        if let bits = meta.quant_bits {
            let sidecar = URL(fileURLWithPath: weights.path + ".config.json")
            guard let spec = try QuantSpec.load(sidecar: sidecar) else {
                failures.append("\(name): meta.json declares quant_bits=\(bits) but the "
                                + "weights sidecar has no quant block")
                continue
            }

            // (a) EXACT dequant check: what Swift's quantized modules dequantize to must
            // equal what Python packed, at 1e-6 — the tight format-correctness gate, and
            // does not involve the quantized GEMM kernel at all.
            let dequantPath = dir.appendingPathComponent("dequant_ref.safetensors")
            guard FileManager.default.fileExists(atPath: dequantPath.path) else {
                failures.append("\(name): quant_bits set but no dequant_ref.safetensors at "
                                + "\(dequantPath.path)")
                continue
            }
            let dequantRef = try loadArrays(url: dequantPath)
            let flatParams = Dictionary(
                model.parameters().flattened(), uniquingKeysWith: { a, _ in a })
            var worstDequant = 0.0
            for (path, targetBits) in spec.targets.sorted(by: { $0.key < $1.key }) {
                guard let w = flatParams["\(path).weight"],
                      let s = flatParams["\(path).scales"],
                      let b = flatParams["\(path).biases"],
                      let ref = dequantRef["\(path).weight"] else {
                    failures.append("\(name): quantized module '\(path)' missing packed "
                                    + "weight/scales/biases, or dequant_ref has no entry for it")
                    continue
                }
                let deq = dequantized(w, scales: s, biases: b,
                                      groupSize: spec.groupSize, bits: targetBits)
                MLX.eval(deq)
                let c = compare(deq.asType(.float32).asArray(Float.self),
                                ref.asType(.float32).asArray(Float.self),
                                rtol: 1e-6, atol: 1e-6)
                worstDequant = max(worstDequant, c.maxAbs)
                if !c.ok {
                    failures.append(String(
                        format: "%@: dequant mismatch for '%@' (max|d| = %.3e, want <= 1e-6)",
                        name, path, c.maxAbs))
                }
            }

            // (c) top-1 agreement + KL vs the ORIGINAL fp model's logits — the statistical
            // quality gate #168's acceptance criteria ask for.
            guard let fpRef = reference["fp_forward_logits"] else {
                failures.append("\(name): quant_bits set but reference.safetensors has no "
                                + "fp_forward_logits")
                continue
            }
            guard let thr = quantThresholds[bits] else {
                failures.append("\(name): no quantThresholds entry for bits=\(bits)")
                continue
            }
            let vocab = fwd.dim(-1)
            let (top1, kl) = quantParity(
                fp: fpRef.asType(.float32).asArray(Float.self),
                q: fwd.asType(.float32).asArray(Float.self), vocab: vocab)
            let quantOk = top1 >= thr.top1 && kl <= thr.kl
            print(String(
                format: "%@: quant bits=%d top1=%.4f (>=%.2f) kl=%.4f (<=%.1e) dequant "
                + "max|d|=%.3e  %@",
                name, bits, top1, thr.top1, kl, thr.kl, worstDequant, quantOk ? "OK" : "FAIL"))
            if !quantOk {
                failures.append(String(
                    format: "%@: quantized quality gate failed — top1=%.4f (want >= %.2f) "
                    + "kl=%.4f (want <= %.1e)", name, top1, thr.top1, kl, thr.kl))
            }
        }
    } catch {
        failures.append("\(name): threw — \(error)")
    }
}
}

// `MONICA_ENGINE_CPU=1` runs the whole gate on MLX's CPU device. mlx-swift's Metal path
// needs a compiled `default.metallib`, which needs Xcode's `metal` compiler — a machine
// with only the Command Line Tools installed cannot build it and every op fails with
// "Failed to load the default metallib". The parity contract is fp32 at 1e-4, which the
// CPU device satisfies just as well, so this is a real escape hatch and not a weakening.
// CI's `macos-latest` HAS Xcode, so `swift-engine` deliberately does NOT set it: the
// gate's default path is the GPU one it will actually be served on.
if ProcessInfo.processInfo.environment["MONICA_ENGINE_CPU"] == "1" {
    print("(MONICA_ENGINE_CPU=1 — running on the MLX CPU device)")
    Device.withDefaultDevice(.cpu) { runGate() }
} else {
    runGate()
}

if fixtureDirs.isEmpty { failures.append("no fixtures to check") }

if failures.isEmpty {
    // #266: report per-precision counts — a single hardcoded "at rtol=1e-4 / atol=1e-5"
    // would lie now that fixtures run at different bands.
    let perPrecision = precisionCounts.sorted(by: { $0.key < $1.key }).map { p, n in
        let band = precisionBands[p].map { "rtol=\($0.rtol) atol=\($0.atol)" } ?? "?"
        return "\(p): \(n) at \(band)"
    }.joined(separator: ", ")
    print("OK — all \(fixtureDirs.count) fixtures pass forward + step parity (\(perPrecision))")
} else {
    print("\n\(failures.count) FAILURE(S):")
    for f in failures { print("  - \(f)") }
    exit(1)
}
