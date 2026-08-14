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

let rtol: Double = 1e-4
let atol: Double = 1e-5

var failures: [String] = []

/// numpy's `allclose` predicate, elementwise, with `ref` as the reference operand:
/// `|a - ref| <= atol + rtol * |ref|`. Returns the max absolute difference and whether
/// every element passed.
func compare(_ a: [Float], _ ref: [Float]) -> (maxAbs: Double, ok: Bool) {
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

let defaultFixtureNames = ["toy", "toy-hybrid", "toy-moe", "toy-moe-biased"]

// --- argument parsing (hand-rolled; the runner takes one repeatable flag) ---
var explicitFixtures: [URL] = []
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
                genState = st
                genLogits = lg
            }
            var sampler = Sampler(temperature: 0)
            var actualGreedy: [Int] = []
            actualGreedy.reserveCapacity(expectedGreedy.count)
            for _ in 0..<expectedGreedy.count {
                guard let lg = genLogits else { break }
                MLX.eval(lg)
                let row = lg.asType(.float32).reshaped([-1]).asArray(Float.self)
                let nxt = try sampler.sample(row)
                actualGreedy.append(nxt)
                let (lg2, st2) = try model.step(MLXArray([Int32(nxt)]), genState)
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
                             refForward.asType(.float32).asArray(Float.self))

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
                              refStep.asType(.float32).asArray(Float.self))

        // Always print the numbers, on success too (the smoke gate's habit): a max-abs-diff
        // drifting from 1e-6 to 9e-5 is the early warning that something moved.
        print(String(format: "%@: forward max|d| = %.3e  step max|d| = %.3e  (B=%d, L=%d) %@",
                     name, fwdCmp.maxAbs, stepCmp.maxAbs, batch, seq,
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
                                refH.asType(.float32).asArray(Float.self))
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
