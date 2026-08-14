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
import MonicaEngine

// The fp32 gate's constants — UNCHANGED by #168. A fixture's `meta.json` MAY carry its
// own `rtol`/`atol` (the #168/#266 hook); absent, every comparison below defaults to
// these exact values, so the four pre-#168 fixtures behave byte-for-byte as before.
let rtol: Double = 1e-4
let atol: Double = 1e-5

// #168: the quantized-vs-fp statistical gate, mirroring
// `src/conformance/quant_parity.QUANT_THRESHOLDS` exactly (int8 nearly lossless, int4
// tolerates real but bounded drift).
let quantThresholds: [Int: (top1: Double, kl: Double)] = [8: (0.99, 1e-3), 4: (0.95, 5e-2)]

var failures: [String] = []

/// A fixture's optional `meta.json` overrides — just the #168/#266 fields this runner
/// reads. Every field is optional so a pre-#168 `meta.json` (no `rtol`/`atol`/
/// `quant_bits`) decodes fine and changes nothing.
struct FixtureMeta: Decodable {
    let rtol: Double?
    let atol: Double?
    let quant_bits: Int?
}

func loadFixtureMeta(_ dir: URL) -> FixtureMeta {
    let empty = FixtureMeta(rtol: nil, atol: nil, quant_bits: nil)
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
    let fxRtol = meta.rtol ?? rtol
    let fxAtol = meta.atol ?? atol
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

        if !fwdCmp.ok || !stepCmp.ok {
            // Localize before reporting the logit diff: a whole-model mismatch is nearly
            // impossible to place by hand across 24 layers, but the exporter checked in each
            // layer's output, so "first divergence at layer 7" is one comparison away.
            let hs = model.hiddenStates(tokens)
            MLX.eval(hs)
            var located = false
            for i in 0..<hs.count {
                guard let refH = reference["hidden.\(i)"] else { continue }
                let c = compare(hs[i].asType(.float32).asArray(Float.self),
                                refH.asType(.float32).asArray(Float.self),
                                rtol: fxRtol, atol: fxAtol)
                if !c.ok {
                    failures.append(String(
                        format: "%@: FIRST DIVERGENCE at hidden state %d (max|d| = %.3e) — "
                        + "that is layer %d's output (state 0 is the embedding)",
                        name, i, c.maxAbs, i - 1))
                    located = true
                    break
                }
            }
            if !located {
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
    print("OK — all \(fixtureDirs.count) fixtures pass forward + step parity "
          + "at rtol=1e-4 / atol=1e-5")
} else {
    print("\n\(failures.count) FAILURE(S):")
    for f in failures { print("  - \(f)") }
    exit(1)
}
