// monica-train — the training-step + optimizer parity gate for the Swift/MLX engine (#195).
//
// Mirrors `monica-parity`'s contract one level up: instead of comparing Swift `forward`/
// `step` against a checked-in Python/MLX logit oracle, this compares Swift's
// `TrainStep.accumulateAndStep` (forward + backward + AdamW update, driven through
// mlx-swift's own autodiff) against `train.safetensors` — a K-step trajectory oracle
// (`scripts/export_parity_fixture.py --train-steps K`) recording loss/grad-norm/the full
// gradient tree/post-step weights for each step.
//
// Style follows `monica-parity`/`monica-bench` exactly: hand-rolled arg parsing, a
// `failures` array, always print achieved numbers (even on success), `exit(1)` on any
// failure. No XCTest (neither Swift package in this repo has a `.testTarget`).
//
// Modes:
//   monica-train --self-test                         # DynamicLossScaler policy only, no MLX math
//   monica-train                                      # fixture gate, default fixtures (toy, toy-hybrid, toy-moe)
//   monica-train --fixtures <dir> [--fixtures <dir> ...]
//   monica-train --overflow-check <fixture>           # forced fp16 overflow -> skip branch, end to end
//   monica-train --train <fixture> --steps N --lr L   # free-running loop (issue's literal AC)

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MonicaEngine

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    exit(1)
}

// --- #195's own tolerance: DELIBERATELY not the fp32 forward/step gate's 1e-4/1e-5. A
// training step accumulates error through the whole backward graph and AdamW moments
// compound it across steps (see `.claude/plans/issue-195.md`'s tolerance table). Read
// from each fixture's `meta.json` `train_rtol`/`train_atol` (new keys — never the
// existing `rtol`/`atol`, which belong to the quantized fixtures' logit gate).
let defaultTrainRtol: Double = 2e-4
let defaultTrainAtol: Double = 1e-6

var failures: [String] = []

/// numpy's `allclose` predicate, elementwise, with `ref` as the reference operand —
/// exactly `monica-parity`'s `compare`, duplicated here (this executable has no
/// dependency on `monica-parity`'s target).
func compare(_ a: [Float], _ ref: [Float], rtol: Double, atol: Double) -> (maxAbs: Double, ok: Bool) {
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

func flatArray(_ arr: MLXArray) -> [Float] {
    arr.asType(.float32).reshaped([-1]).asArray(Float.self)
}

// MARK: - --self-test: the DynamicLossScaler policy (mirrors tests/test_loss_scale.py).
// Pure Swift arithmetic (LossScaler.swift has no `import MLX`), so this runs even on a
// host where mlx-swift cannot execute (no compiled `default.metallib`).

func runSelfTest() -> Never {
    var stFailures: [String] = []
    func check(_ cond: Bool, _ msg: @autoclosure () -> String) {
        if !cond { stFailures.append(msg()) }
    }

    do {   // overflow backs off and resets the good-step counter
        let s = DynamicLossScaler(initScale: 1024, backoff: 0.5, growthInterval: 3)
        s.update(overflow: false)
        s.update(overflow: false)
        check(s.scale == 1024, "backoff: expected 1024 after 2 clean steps, got \(s.scale)")
        s.update(overflow: true)
        check(s.scale == 512, "backoff: expected 512 after overflow, got \(s.scale)")
        s.update(overflow: false)
        s.update(overflow: false)
        check(s.scale == 512, "backoff: good-step counter did not reset, got \(s.scale)")
    }

    do {   // growth fires only after growthInterval consecutive clean steps
        let s = DynamicLossScaler(initScale: 1024, growthFactor: 2.0, growthInterval: 3)
        s.update(overflow: false)
        s.update(overflow: false)
        check(s.scale == 1024, "growth: premature growth, got \(s.scale)")
        s.update(overflow: false)
        check(s.scale == 2048, "growth: expected 2048 after the 3rd clean step, got \(s.scale)")
    }

    do {   // min_scale floor
        let s = DynamicLossScaler(initScale: 2, backoff: 0.5, minScale: 1.0)
        s.update(overflow: true)
        check(s.scale == 1.0, "minScale: expected 1.0, got \(s.scale)")
        s.update(overflow: true)
        check(s.scale == 1.0, "minScale: floor not held, got \(s.scale)")
    }

    do {   // max_scale ceiling
        let s = DynamicLossScaler(
            initScale: pow(2.0, 23), growthFactor: 2.0, growthInterval: 1, maxScale: pow(2.0, 24))
        s.update(overflow: false)
        check(s.scale == pow(2.0, 24), "maxScale: expected 2^24, got \(s.scale)")
        s.update(overflow: false)
        check(s.scale == pow(2.0, 24), "maxScale: ceiling not held, got \(s.scale)")
    }

    do {   // state_dict round trip, including the good-step counter
        let s = DynamicLossScaler(initScale: 1024, growthInterval: 5)
        s.update(overflow: false)
        s.update(overflow: false)
        let restored = DynamicLossScaler(initScale: 1.0, growthInterval: 5)
        restored.loadStateDict(s.stateDict())
        check(restored.scale == s.scale, "stateDict: scale mismatch after restore")
        for _ in 0..<3 {
            s.update(overflow: false)
            restored.update(overflow: false)
        }
        check(restored.scale == s.scale, "stateDict: good-step counter did not survive restore")
    }

    do {   // loading nil is a no-op (resume with no scaler bundle)
        let s = DynamicLossScaler(initScale: 512)
        s.loadStateDict(nil)
        check(s.scale == 512, "loadStateDict(nil): scale changed, got \(s.scale)")
    }

    // precision -> scaler wiring
    check(scalerFor(precision: "fp16")?.scale == pow(2.0, 13), "scalerFor(fp16): wrong default init")
    check(scalerFor(precision: "fp16", initScale: 4096)?.scale == 4096,
         "scalerFor(fp16): custom init not honored")
    check(scalerFor(precision: "fp32") == nil, "scalerFor(fp32) should be nil")
    check(scalerFor(precision: "bf16") == nil, "scalerFor(bf16) should be nil")

    if stFailures.isEmpty {
        print("monica-train --self-test: OK — all checks passed")
        exit(0)
    } else {
        print("monica-train --self-test: \(stFailures.count) FAILURE(S):")
        for f in stFailures { print("  - \(f)") }
        exit(1)
    }
}

// MARK: - fixture resolution (mirrors monica-parity's resolveFixturesDir)

func resolveFixturesDir() -> URL? {
    let fm = FileManager.default
    if let env = ProcessInfo.processInfo.environment["MONICA_ENGINE_FIXTURES"], !env.isEmpty {
        let u = URL(fileURLWithPath: env)
        if fm.fileExists(atPath: u.path) { return u }
    }
    let fromSource = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Sources/monica-train/ -> drop main.swift
        .deletingLastPathComponent()   // -> Sources/
        .deletingLastPathComponent()   // -> swift/engine/
        .appendingPathComponent("Fixtures")
    if fm.fileExists(atPath: fromSource.path) { return fromSource }
    let cwd = URL(fileURLWithPath: "Fixtures")
    if fm.fileExists(atPath: cwd.path) { return cwd }
    return nil
}

// `monica-train` carries its OWN default fixture list — deliberately narrower than
// `monica-parity`'s. Only three of the seven checked-in fixtures carry a `train.safetensors`
// oracle (see `.claude/plans/issue-195.md`'s "Which fixtures get a train oracle" and
// `swift/engine/Fixtures/README.md`); a missing `train.safetensors` in a fixture this tool
// was asked to check is a FAILURE, not a skip.
let defaultTrainFixtureNames = ["toy", "toy-hybrid", "toy-moe"]

struct TrainFixtureMeta: Decodable {
    let train_steps: Int?
    let train_grad_accum: Int?
    let train_grad_clip: Double?
    let train_lrs: [Double]?
    let train_rtol: Double?
    let train_atol: Double?
}

func loadTrainMeta(_ dir: URL) -> TrainFixtureMeta? {
    guard let data = try? Data(contentsOf: dir.appendingPathComponent("meta.json")) else { return nil }
    return try? JSONDecoder().decode(TrainFixtureMeta.self, from: data)
}

func loadMicroBatches(_ arrays: [String: MLXArray], gradAccum: Int, name: String) -> [MicroBatch]? {
    var out: [MicroBatch] = []
    for j in 0..<gradAccum {
        guard let inp = arrays["mb.\(j).inputs"], let tgt = arrays["mb.\(j).targets"] else {
            failures.append("\(name): train.safetensors missing mb.\(j).inputs/mb.\(j).targets")
            return nil
        }
        out.append(MicroBatch(inputs: inp.asType(.int32), targets: tgt.asType(.int32)))
    }
    return out
}

// MARK: - the fixture parity gate (loss / grad-norm / grad-tree / post-step-weights)

func runFixtureGate(_ dirs: [URL]) {
    for dir in dirs {
        let name = dir.lastPathComponent
        do {
            let weightsURL = dir.appendingPathComponent("weights.safetensors")
            guard FileManager.default.fileExists(atPath: weightsURL.path) else {
                failures.append("\(name): no weights.safetensors at \(dir.path)")
                continue
            }
            let trainURL = dir.appendingPathComponent("train.safetensors")
            guard FileManager.default.fileExists(atPath: trainURL.path) else {
                failures.append("\(name): no train.safetensors at \(trainURL.path) — "
                                + "regenerate with scripts/export_parity_fixture.py --train-steps K")
                continue
            }
            guard let meta = loadTrainMeta(dir),
                  let steps = meta.train_steps, let gradAccum = meta.train_grad_accum,
                  let gradClip = meta.train_grad_clip, let lrs = meta.train_lrs,
                  let rtol = meta.train_rtol, let atol = meta.train_atol else {
                failures.append("\(name): meta.json is missing one or more train_* keys")
                continue
            }

            let (model, _) = try Checkpoint.load(weights: weightsURL)
            let optimizer = try makeOptimizer(model, baseLR: Float(lrs[0]))
            let scaler = scalerFor(precision: model.config.precision)
            let lossAndGrad = makePretrainLossAndGrad(model: model, scaler: scaler)

            let arrays = try loadArrays(url: trainURL)
            guard let microBatches = loadMicroBatches(arrays, gradAccum: gradAccum, name: name) else {
                continue
            }

            var worstLoss = 0.0, worstNorm = 0.0, worstGrad = 0.0, worstWeights = 0.0
            var firstBadGrad: String? = nil
            var firstBadWeights: String? = nil
            var anyFail = false

            for k in 0..<steps {
                guard let refLoss = arrays["loss.\(k)"], let refNorm = arrays["grad_norm.\(k)"] else {
                    failures.append("\(name): train.safetensors missing loss.\(k)/grad_norm.\(k)")
                    anyFail = true
                    break
                }
                var captured: ModuleParameters? = nil
                let result = accumulateAndStep(
                    model: model, optimizer: optimizer, lossAndGrad: lossAndGrad,
                    microBatches: microBatches, lr: Float(lrs[k]), gradClip: Float(gradClip),
                    scaler: scaler, captureGrads: { g in captured = g })

                if result.skipped {
                    failures.append("\(name): step \(k) was unexpectedly SKIPPED "
                                    + "(fp16 overflow) — the fixture's own oracle never skips")
                    anyFail = true
                    continue
                }

                let lossCmp = compare([result.loss], flatArray(refLoss), rtol: rtol, atol: atol)
                worstLoss = max(worstLoss, lossCmp.maxAbs)
                if !lossCmp.ok {
                    failures.append(String(format: "%@: step %d loss mismatch (max|d| = %.3e)",
                                           name, k, lossCmp.maxAbs))
                    anyFail = true
                }

                let normCmp = compare([result.gradNorm], flatArray(refNorm), rtol: rtol, atol: atol)
                worstNorm = max(worstNorm, normCmp.maxAbs)
                if !normCmp.ok {
                    failures.append(String(format: "%@: step %d grad_norm mismatch (max|d| = %.3e)",
                                           name, k, normCmp.maxAbs))
                    anyFail = true
                }

                guard let grads = captured else {
                    failures.append("\(name): step \(k) produced no gradient tree to compare")
                    anyFail = true
                    continue
                }
                let gradLeaves = grads.flattened().sorted { $0.0 < $1.0 }
                for (paramName, g) in gradLeaves {
                    guard let ref = arrays["grad.\(k).\(paramName)"] else {
                        failures.append("\(name): train.safetensors missing grad.\(k).\(paramName)")
                        anyFail = true
                        continue
                    }
                    MLX.eval(g)
                    let c = compare(flatArray(g), flatArray(ref), rtol: rtol, atol: atol)
                    worstGrad = max(worstGrad, c.maxAbs)
                    if !c.ok && firstBadGrad == nil {
                        firstBadGrad = "step \(k) \(paramName) (max|d| = \(String(format: "%.3e", c.maxAbs)))"
                    }
                    if !c.ok { anyFail = true }
                }

                let weightLeaves = model.parameters().flattened().sorted { $0.0 < $1.0 }
                for (paramName, w) in weightLeaves {
                    guard let ref = arrays["weights_after.\(k).\(paramName)"] else {
                        failures.append("\(name): train.safetensors missing "
                                        + "weights_after.\(k).\(paramName)")
                        anyFail = true
                        continue
                    }
                    MLX.eval(w)
                    let c = compare(flatArray(w), flatArray(ref), rtol: rtol, atol: atol)
                    worstWeights = max(worstWeights, c.maxAbs)
                    if !c.ok && firstBadWeights == nil {
                        firstBadWeights =
                            "step \(k) \(paramName) (max|d| = \(String(format: "%.3e", c.maxAbs)))"
                    }
                    if !c.ok { anyFail = true }
                }
            }

            // Always print achieved numbers, on success too — the early warning that
            // something moved before it crosses the tolerance (monica-parity's habit).
            print(String(format: "%@: loss max|d|=%.3e  grad_norm max|d|=%.3e  "
                         + "grad max|d|=%.3e  weights_after max|d|=%.3e  "
                         + "(K=%d, rtol=%.1e atol=%.1e) %@",
                         name, worstLoss, worstNorm, worstGrad, worstWeights, steps, rtol, atol,
                         anyFail ? "FAIL" : "OK"))
            if let bad = firstBadGrad {
                failures.append("\(name): FIRST gradient divergence at \(bad) — "
                                + "the training analogue of monica-parity's per-layer localization")
            }
            if let bad = firstBadWeights {
                failures.append("\(name): FIRST post-step-weight divergence at \(bad)")
            }
        } catch {
            failures.append("\(name): threw — \(error)")
        }
    }
}

// MARK: - --overflow-check <fixture>: the fp16 skip branch, end to end, no fp16 fixture needed

func runOverflowCheck(_ dir: URL) {
    let name = dir.lastPathComponent
    do {
        let weightsURL = dir.appendingPathComponent("weights.safetensors")
        let (model, _) = try Checkpoint.load(weights: weightsURL)
        let before = model.parameters().flattened().sorted { $0.0 < $1.0 }
            .map { ($0.0, flatArray($0.1)) }

        // The exact trigger `tests/test_mlx_train_step.py`'s overflow test uses: a scale
        // above fp32-max makes `loss * scale` -> inf, so the gradients are non-finite
        // regardless of per-element magnitude — a robust, fp32-only trigger. No fp16
        // fixture and no new tolerance: this leaves #266's numeric band untouched.
        let scaler = DynamicLossScaler(initScale: 1e40, backoff: 0.5, minScale: 1.0)
        let optimizer = try makeOptimizer(model, baseLR: 1e-3)
        let lossAndGrad = makePretrainLossAndGrad(model: model, scaler: scaler)

        // A synthetic single micro-batch is enough — the overflow is forced by the scale,
        // not by the data.
        let tokens = MLXArray(Int32(0)..<Int32(8)).reshaped([1, 8])
        let mb = MicroBatch(inputs: tokens, targets: tokens)

        let result = accumulateAndStep(
            model: model, optimizer: optimizer, lossAndGrad: lossAndGrad, microBatches: [mb],
            lr: 1e-3, gradClip: 1.0, scaler: scaler)

        if !result.skipped {
            failures.append("\(name): --overflow-check step was NOT skipped (loss=\(result.loss))")
        }
        if scaler.scale != 0.5e40 {
            failures.append("\(name): --overflow-check scale did not back off — expected "
                            + "0.5e40, got \(scaler.scale)")
        }
        let after = model.parameters().flattened().sorted { $0.0 < $1.0 }
            .map { ($0.0, flatArray($0.1)) }
        var worst = 0.0
        for (b, a) in zip(before, after) {
            for (bv, av) in zip(b.1, a.1) {
                worst = max(worst, Double(abs(bv - av)))
            }
        }
        print(String(format: "%@: --overflow-check skipped=%@ scale=%.3e "
                     + "weights max|d|=%.3e (want 0.0, bit-identical)  %@",
                     name, "\(result.skipped)", scaler.scale, worst,
                     (result.skipped && scaler.scale == 0.5e40 && worst == 0.0) ? "OK" : "FAIL"))
        if worst != 0.0 {
            failures.append("\(name): --overflow-check weights were NOT bit-identical "
                            + "after a skipped step (max|d| = \(worst))")
        }
    } catch {
        failures.append("\(name): --overflow-check threw — \(error)")
    }
}

// MARK: - --train <fixture> --steps N --lr L: the free-running loop (the issue's literal AC)

func runFreeTrain(_ dir: URL, steps: Int, lr: Float) {
    let name = dir.lastPathComponent
    do {
        let weightsURL = dir.appendingPathComponent("weights.safetensors")
        let (model, _) = try Checkpoint.load(weights: weightsURL)
        let trainURL = dir.appendingPathComponent("train.safetensors")
        guard FileManager.default.fileExists(atPath: trainURL.path) else {
            failures.append("\(name): --train needs a train.safetensors (for its micro-batches) "
                            + "at \(trainURL.path)")
            return
        }
        let arrays = try loadArrays(url: trainURL)
        guard let meta = loadTrainMeta(dir), let gradAccum = meta.train_grad_accum else {
            failures.append("\(name): meta.json missing train_grad_accum")
            return
        }
        guard let microBatches = loadMicroBatches(arrays, gradAccum: gradAccum, name: name) else {
            return
        }

        let optimizer = try makeOptimizer(model, baseLR: lr)
        let scaler = scalerFor(precision: model.config.precision)
        let trainStep = makeTrainStep(model: model, optimizer: optimizer, gradClip: 1.0, scaler: scaler)

        var firstLoss: Float? = nil
        var lastLoss: Float = .nan
        var anyNaN = false
        for step in 0..<steps {
            let result = trainStep(model, microBatches, lr)
            if result.skipped {
                print("\(name): --train step \(step) skipped (fp16 overflow), scale backed off")
                continue
            }
            if firstLoss == nil { firstLoss = result.loss }
            lastLoss = result.loss
            if result.loss.isNaN { anyNaN = true }
        }
        guard let first = firstLoss else {
            failures.append("\(name): --train — every step was skipped, no loss trajectory to check")
            return
        }
        let ok = !anyNaN && lastLoss < first
        print(String(format: "%@: --train %d steps  loss %.6f -> %.6f  %@",
                     name, steps, first, lastLoss, ok ? "OK (decreasing, finite)" : "FAIL"))
        if anyNaN {
            failures.append("\(name): --train produced a NaN loss")
        }
        if !(lastLoss < first) {
            failures.append("\(name): --train loss did not decrease (\(first) -> \(lastLoss))")
        }
    } catch {
        failures.append("\(name): --train threw — \(error)")
    }
}

// MARK: - argument parsing (hand-rolled; mirrors monica-parity's repeatable --fixtures)

var explicitFixtures: [URL] = []
var overflowCheckFixture: String? = nil
var trainFixture: String? = nil
var trainSteps = 20
var trainLR: Float = 1e-3

var args = Array(CommandLine.arguments.dropFirst())
while let first = args.first {
    args.removeFirst()
    switch first {
    case "--self-test":
        runSelfTest()
    case "--fixtures":
        guard let dir = args.first else { fail("--fixtures needs a directory") }
        args.removeFirst()
        explicitFixtures.append(URL(fileURLWithPath: dir))
    case "--overflow-check":
        guard let f = args.first else { fail("--overflow-check needs a fixture name or path") }
        args.removeFirst()
        overflowCheckFixture = f
    case "--train":
        guard let f = args.first else { fail("--train needs a fixture name or path") }
        args.removeFirst()
        trainFixture = f
    case "--steps":
        guard let s = args.first, let v = Int(s) else { fail("--steps needs an integer") }
        args.removeFirst()
        trainSteps = v
    case "--lr":
        guard let s = args.first, let v = Float(s) else { fail("--lr needs a number") }
        args.removeFirst()
        trainLR = v
    default:
        fail("unknown argument '\(first)'")
    }
}

/// Resolve a bare fixture name (e.g. "toy") against the Fixtures dir, or pass an explicit
/// path through unchanged (e.g. "Fixtures/toy", used by CI — mirrors monica-generate's
/// `--weights` convention of accepting either).
func resolveFixturePath(_ nameOrPath: String) -> URL {
    let asPath = URL(fileURLWithPath: nameOrPath)
    if FileManager.default.fileExists(atPath: asPath.path) { return asPath }
    if let base = resolveFixturesDir() { return base.appendingPathComponent(nameOrPath) }
    return asPath
}

func runOnDevice() {
    if let f = overflowCheckFixture {
        runOverflowCheck(resolveFixturePath(f))
        return
    }
    if let f = trainFixture {
        runFreeTrain(resolveFixturePath(f), steps: trainSteps, lr: trainLR)
        return
    }
    var fixtureDirs = explicitFixtures
    if fixtureDirs.isEmpty {
        guard let base = resolveFixturesDir() else {
            print("FAIL: could not resolve the fixtures directory "
                  + "(set MONICA_ENGINE_FIXTURES, or run from swift/engine/)")
            exit(1)
        }
        fixtureDirs = defaultTrainFixtureNames.map { base.appendingPathComponent($0) }
    }
    runFixtureGate(fixtureDirs)
}

// `MONICA_ENGINE_CPU=1` — same escape hatch monica-parity/monica-bench honor: a host with
// only Command Line Tools (no Xcode) cannot compile `default.metallib`, so Metal ops fail.
// The training gate's tolerance is defined for the CPU device just as well.
if ProcessInfo.processInfo.environment["MONICA_ENGINE_CPU"] == "1" {
    Device.withDefaultDevice(.cpu) { runOnDevice() }
} else {
    runOnDevice()
}

if failures.isEmpty {
    print("OK — monica-train: all checks passed")
} else {
    print("\n\(failures.count) FAILURE(S):")
    for f in failures { print("  - \(f)") }
    exit(1)
}
