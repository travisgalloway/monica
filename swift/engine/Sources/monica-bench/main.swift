// monica-bench — the benchmark harness CLI for the Swift/MLX engine (#170).
//
// Wraps `MonicaEngine/Bench.swift`'s prefill/decode/memory measurement core with:
// model-source resolution (`--weights` a real checkpoint, or `--config` a random-init
// sidecar for poc-scale shape-only runs), an in-engine `--quantize` path for the
// `--config` arm, a machine-identified `--json` record, and `--baseline` regression
// flagging that refuses to compare across machines.
//
//   monica-bench --weights <w.safetensors> --mode all
//   monica-bench --config Benchmarks/configs/poc.config.json --mode all --json bench.json
//   monica-bench --config Benchmarks/configs/poc.config.json --quantize 8 --mode decode
//   monica-bench --weights <w.safetensors> --mode prefill \
//       --baseline Benchmarks/baselines.json [--tolerance 0.15] [--strict]
//   monica-bench --weights <w.safetensors> --mode spec --gamma 4 [--max-n 8]
//   monica-bench --self-test
//
// `--mode spec` (#172) runs speculative decode against plain greedy on the same prompt and
// asserts the two id arrays are byte-identical (a GATE — `Bench.spec` exits non-zero on a
// mismatch); its speedup/accept-rate numbers are informational, like every other timing
// here. It is deliberately excluded from `--mode all` so `--baseline`/`--strict` keeps
// comparing exactly the metrics it always did.
//
// Style follows `monica-generate`/`monica-parity`: hand-rolled arg parsing (no
// swift-argument-parser — see Package.swift's header on why this target has no
// tokenizer dependency, unlike `monica-generate`), a `failures` array for
// `--self-test`, `exit(1)` on failure. `MONICA_ENGINE_CPU=1` is honored exactly like
// the other two executables.

import Darwin
import Foundation
import MLX
import MLXNN
import MonicaEngine

// MARK: - arg parsing (hand-rolled; mirrors monica-generate's parseFlags)

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    exit(1)
}

func parseFlags(_ args: [String]) -> [String: String] {
    var out: [String: String] = [:]
    var i = 0
    while i < args.count {
        let a = args[i]
        guard a.hasPrefix("--") else { i += 1; continue }
        let key = String(a.dropFirst(2))
        if i + 1 < args.count && !args[i + 1].hasPrefix("--") {
            out[key] = args[i + 1]; i += 2
        } else {
            out[key] = ""; i += 1
        }
    }
    return out
}

func intFlag(_ flags: [String: String], _ name: String, default def: Int) -> Int {
    guard let raw = flags[name], !raw.isEmpty else { return def }
    guard let v = Int(raw) else { fail("--\(name) must be an integer, got '\(raw)'") }
    return v
}

func optIntFlag(_ flags: [String: String], _ name: String) -> Int? {
    guard let raw = flags[name], !raw.isEmpty else { return nil }
    guard let v = Int(raw) else { fail("--\(name) must be an integer, got '\(raw)'") }
    return v
}

func doubleFlag(_ flags: [String: String], _ name: String, default def: Double) -> Double {
    guard let raw = flags[name], !raw.isEmpty else { return def }
    guard let v = Double(raw) else { fail("--\(name) must be a number, got '\(raw)'") }
    return v
}

/// A `TextOutputStream` wrapper so banners/status can target stderr.
struct StandardError: TextOutputStream {
    mutating func write(_ string: String) { FileHandle.standardError.write(Data(string.utf8)) }
}
var standardError = StandardError()

// MARK: - model-source resolution (pure, so --self-test can exercise it without MLX)

enum ModelSource: Equatable {
    case weights(String)
    case config(String)
}

/// A local success/failure enum rather than Swift's `Result` — `Result`'s `Failure`
/// generic requires `Error` conformance, which a plain `String` message does not need.
enum ModelSourceResolution {
    case success(ModelSource)
    case failure(String)
}

/// `--weights` and `--config` are mutually exclusive, and exactly one is required
/// (unless `--self-test`). Pure function so arg-validation is testable without a model.
func resolveModelSourceFlag(_ flags: [String: String]) -> ModelSourceResolution {
    let w = flags["weights"]
    let c = flags["config"]
    switch (w, c) {
    case (.some(let wp), nil): return .success(.weights(wp))
    case (nil, .some(let cp)): return .success(.config(cp))
    case (.some, .some):
        return .failure("--weights and --config are mutually exclusive")
    case (nil, nil):
        return .failure(
            "one of --weights <w.safetensors> or --config <sidecar.json> is required "
            + "(or pass --self-test)")
    }
}

// MARK: - machine identity

func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
    return String(cString: buf)
}

/// Best-effort `swift --version`, run once at startup. Never blocks a bench result on
/// failure — an unreadable toolchain version is recorded as "unknown", not fatal.
func swiftVersionString() -> String {
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    proc.arguments = ["swift", "--version"]
    let pipe = Pipe()
    proc.standardOutput = pipe
    proc.standardError = Pipe()
    do {
        try proc.run()
        proc.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let text = String(data: data, encoding: .utf8) ?? ""
        let firstLine = text.split(separator: "\n").first.map(String.init) ?? ""
        return firstLine.isEmpty ? "unknown" : firstLine
    } catch {
        return "unknown"
    }
}

func gitSha() -> String? {
    if let env = ProcessInfo.processInfo.environment["GITHUB_SHA"], !env.isEmpty { return env }
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    proc.arguments = ["git", "rev-parse", "HEAD"]
    let pipe = Pipe()
    proc.standardOutput = pipe
    proc.standardError = Pipe()
    do {
        try proc.run()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let sha = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (sha?.isEmpty ?? true) ? nil : sha
    } catch {
        return nil
    }
}

func gatherMachineID() -> MachineID {
    let isMetal = Device.defaultDevice().deviceType == .gpu
    let info = GPU.deviceInfo()
    return MachineID(
        architecture: isMetal ? info.architecture : "cpu",
        memorySizeBytes: info.memorySize,
        hwModel: sysctlString("hw.model") ?? "unknown",
        cpuCores: ProcessInfo.processInfo.activeProcessorCount,
        osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
        swiftVersion: swiftVersionString(),
        device: isMetal ? "metal" : "cpu")
}

// MARK: - self-test (no weights/MLX model needed; --self-test)

func runSelfTest() -> Never {
    var failures: [String] = []
    func check(_ cond: Bool, _ msg: String) { if !cond { failures.append(msg) } }
    func eq<T: Equatable>(_ a: T, _ b: T, _ msg: String) {
        if a != b { failures.append("\(msg): \(a) != \(b)") }
    }

    // --- analytic state-byte arithmetic, hand-computed for a known config ---
    // Built via JSON decode (MambaConfig's synthesized memberwise init is
    // module-internal — Decodable is the only public construction path from outside
    // MonicaEngine), mirroring how a real sidecar is loaded.
    // dModel=64 nLayers=4 headDim=16 vocabSize=256 attnEvery=2 -> dInner=128, nHeads=8,
    // nAttnHeadsResolved=max(1,64/64)=1, attnHeadDim=64/1=64.
    func makeConfig(_ json: String) -> MambaConfig {
        guard let data = json.data(using: .utf8),
              let cfg = try? JSONDecoder().decode(MambaConfig.self, from: data) else {
            fail("self-test: could not build a test MambaConfig from JSON: \(json)")
        }
        return cfg
    }
    // `MambaConfig`'s synthesized `Decodable` requires every `CodingKeys` member present
    // (default values do not make a key optional-if-missing) — the same shape every
    // real `<weights>.config.json` sidecar carries (see `src/model/blocks.py:to_dict`).
    let cfgTemplate = #"""
        {"d_model":%d,"n_layers":%d,"d_state":16,"expand":2,"d_conv":4,"head_dim":%d,
         "dt_rank":"auto","vocab_size":%d,"seq_len":64,"tie_embeddings":true,
         "precision":"fp32","chunk_size":null,"long_ctx_factor":1.0,
         "attn_every":%@,"n_attn_heads":null,"moe_every":null,"n_experts":0,"top_k":2,
         "moe_d_ff":null,"dt_min":0.001,"dt_max":0.1,"dt_init_floor":0.0001,
         "optimizer":"adamw","optimizer_8bit":false}
        """#
    func testConfig(dModel: Int, nLayers: Int, headDim: Int, vocabSize: Int,
                    attnEvery: Int?) -> MambaConfig {
        let attnStr = attnEvery.map(String.init) ?? "null"
        let json = String(format: cfgTemplate, dModel, nLayers, headDim, vocabSize, attnStr)
        return makeConfig(json)
    }
    do {
        let cfg = testConfig(dModel: 64, nLayers: 4, headDim: 16, vocabSize: 256, attnEvery: 2)
        // perLayer = (dConv-1)*dInner + nHeads*headDim*dState = 3*128 + 8*16*16 = 384+2048=2432
        // totalFloats = nLayers*perLayer = 4*2432 = 9728; bytes = *4 = 38912
        eq(Bench.analyticStateBytes(cfg), 38912, "analyticStateBytes hand-computed check")
        // nAttnHeadsResolved=max(1,64/64)=1, attnHeadDim=64/1=64; nAttnLayers=4/2=2;
        // kv = 2*2*1*64*10*4 = 10240 at length=10
        eq(Bench.analyticKvBytes(cfg, length: 10), 10240, "analyticKvBytes hand-computed check")
        // A dense config (no attention) has zero KV bytes.
        let dense = testConfig(dModel: 64, nLayers: 4, headDim: 16, vocabSize: 256, attnEvery: nil)
        eq(Bench.analyticKvBytes(dense, length: 1024), 0, "analyticKvBytes is 0 with no attn_every")
    }

    // --- quant target-selection filter ---
    do {
        let leaves: [(path: String, lastAxis: Int)] = [
            ("embedding", 256), ("layers.0.in_proj", 128), ("layers.0.dt_proj", 128),
            ("router", 8), ("layers.1.small", 30),
        ]
        let targets = Bench.quantTargets(leaves: leaves, groupSize: 64, bits: 8, headBits: 4)
        eq(targets["embedding"], 4, "quantTargets: head_bits overrides the embedding path")
        eq(targets["layers.0.in_proj"], 8, "quantTargets: an eligible path gets the default bits")
        check(targets["layers.0.dt_proj"] == nil, "quantTargets: dt_proj is excluded by name")
        check(targets["router"] == nil, "quantTargets: router is excluded by name")
        check(targets["layers.1.small"] == nil,
             "quantTargets: last axis not divisible by group_size is excluded (30 % 64 != 0)")
    }

    // --- baseline comparison: regression detected / not detected / machine mismatch ---
    do {
        let machine = MachineID(architecture: "Apple M1", memorySizeBytes: 17179869184,
                                hwModel: "MacBookPro18,1", cpuCores: 8, osVersion: "14.0",
                                swiftVersion: "swift-6.0", device: "metal")
        let otherMachine = MachineID(architecture: "Apple M4", memorySizeBytes: 34359738368,
                                     hwModel: "Mac16,1", cpuCores: 10, osVersion: "15.0",
                                     swiftVersion: "swift-6.0", device: "metal")
        let baseDecode = DecodeResult(warmupTokens: 32, measuredTokens: 256, tokPerSec: 100.0,
                                      generateTokPerSec: nil, generateTokens: nil)
        let baseline = BenchRecord(
            mode: "decode", machine: machine, ci: false, gitSha: nil, source: "test",
            randomInit: false, quantized: false, quantBits: nil, timestamp: "t",
            prefill: nil, decode: baseDecode, memory: nil)

        func record(machine: MachineID, tokPerSec: Double) -> BenchRecord {
            BenchRecord(mode: "decode", machine: machine, ci: false, gitSha: nil, source: "test",
                       randomInit: false, quantized: false, quantBits: nil, timestamp: "t",
                       prefill: nil,
                       decode: DecodeResult(warmupTokens: 32, measuredTokens: 256,
                                            tokPerSec: tokPerSec, generateTokPerSec: nil,
                                            generateTokens: nil),
                       memory: nil)
        }

        let ok = Bench.compareToBaseline(record(machine: machine, tokPerSec: 100.0),
                                         baselines: [baseline])
        check(ok.contains { $0.status == .ok }, "baseline comparison: unchanged tok/s reads OK")

        let regressed = Bench.compareToBaseline(record(machine: machine, tokPerSec: 50.0),
                                                 baselines: [baseline], tolerance: 0.15)
        check(regressed.contains { $0.status == .regression },
             "baseline comparison: a 50% drop in tok/s is flagged REGRESSION")

        let mismatched = Bench.compareToBaseline(record(machine: otherMachine, tokPerSec: 100.0),
                                                  baselines: [baseline])
        check(mismatched.allSatisfy { $0.status == .skipped },
             "baseline comparison: a different machine id is SKIPPED, never a green OK "
             + "(a comparison that cannot observe its target must not read healthy)")
    }

    // --- arg validation: --weights/--config mutual exclusion and requiredness ---
    do {
        switch resolveModelSourceFlag(["weights": "w.safetensors"]) {
        case .success(.weights("w.safetensors")): break
        default: failures.append("resolveModelSourceFlag: --weights alone should resolve")
        }
        switch resolveModelSourceFlag(["config": "c.json"]) {
        case .success(.config("c.json")): break
        default: failures.append("resolveModelSourceFlag: --config alone should resolve")
        }
        switch resolveModelSourceFlag(["weights": "w", "config": "c"]) {
        case .failure: break
        default: failures.append("resolveModelSourceFlag: both flags should fail")
        }
        switch resolveModelSourceFlag([:]) {
        case .failure: break
        default: failures.append("resolveModelSourceFlag: neither flag should fail")
        }
    }

    // --- #172 criterion 1: PromptLookupDrafter.propose is a faithful port of
    //     src/serve/spec_decode.py:propose (mirrors tests/test_spec_decode.py::test_propose_*)
    do {
        let d8 = PromptLookupDrafter(maxN: 8)
        // Copies the tokens following the most recent EARLIER occurrence of the tail.
        eq(d8.propose(context: [1, 2, 3, 4, 9, 9, 1, 2], gamma: 2), [3, 4],
           "propose: copies after the most recent earlier match")
        // Prefers the LONGER pattern: both [2] and [1,2] recur; [1,2] -> [3] must win.
        eq(d8.propose(context: [1, 2, 3, 7, 2, 5, 1, 2], gamma: 1), [3],
           "propose: prefers the longer pattern")
        // Capped by gamma: tail [1,2,3] recurs at 0, but only 2 tokens are taken.
        eq(d8.propose(context: [1, 2, 3, 1, 2, 3], gamma: 2), [1, 2],
           "propose: result is capped by gamma")
        // No tail recurs at all.
        eq(d8.propose(context: [5, 6, 7, 8], gamma: 2), [],
           "propose: no recurring tail returns []")
        // Respects the maxN window: tail [3,4], earlier [3,4] at index 3.
        eq(d8.propose(context: Array(0..<10) + [3, 4], gamma: 2), [5, 6],
           "propose: respects the maxN window")
        // maxN caps the pattern length: with maxN=1 only the tail [2] is tried, whose most
        // recent earlier occurrence is at index 4 -> [5], NOT the [1,2] -> [3] the
        // maxN=8 case above finds. This is the window doing real work, not a no-op flag.
        eq(PromptLookupDrafter(maxN: 1).propose(context: [1, 2, 3, 7, 2, 5, 1, 2], gamma: 1),
           [5], "propose: maxN=1 restricts matching to the single-token pattern")
        // Edge cases: too-short context, non-positive gamma.
        eq(d8.propose(context: [1], gamma: 2), [], "propose: context.count < 2 returns []")
        eq(d8.propose(context: [], gamma: 2), [], "propose: empty context returns []")
        eq(d8.propose(context: [1, 2, 1, 2], gamma: 0), [], "propose: gamma == 0 returns []")
        eq(d8.propose(context: [1, 2, 1, 2], gamma: -3), [], "propose: gamma < 0 returns []")
        // Fall-through to a SHORTER n when the longer tails have no earlier occurrence
        // (the reachable half of the `break`-vs-`continue` hazard the port calls out):
        // only [2,3] recurs here, at index 1, so the draft is what followed it.
        eq(d8.propose(context: [1, 2, 3, 4, 2, 3], gamma: 2), [4, 2],
           "propose: falls through to a shorter pattern when longer tails do not recur")
    }

    // --- #172 criterion 2: SpecDecode.firstMismatch is a faithful port of the greedy
    //     accept rule (mirrors tests/test_spec_decode.py::test_first_mismatch_*)
    do {
        eq(SpecDecode.firstMismatch([1, 2, 3], [1, 2, 3]), 3, "firstMismatch: all agree")
        eq(SpecDecode.firstMismatch([1, 2, 3], [1, 9, 3]), 1, "firstMismatch: mismatch at 1")
        eq(SpecDecode.firstMismatch([1, 2, 3], [9, 2, 3]), 0, "firstMismatch: immediate mismatch")
        eq(SpecDecode.firstMismatch([], []), 0, "firstMismatch: both empty")
        // Unequal lengths truncate to the shorter, exactly as Python's `zip` does.
        eq(SpecDecode.firstMismatch([1, 2, 3, 4], [1, 2]), 2,
           "firstMismatch: draft longer than preds truncates like zip")
        eq(SpecDecode.firstMismatch([1, 2], [1, 2, 3, 4]), 2,
           "firstMismatch: preds longer than draft truncates like zip")
        eq(SpecDecode.firstMismatch([1, 2, 3], []), 0, "firstMismatch: empty preds accepts none")
    }

    // --- #172 criterion 3: the Drafter seam exists and is consumed through the PROTOCOL,
    //     not the concrete type. `SpecDecode.generate` takes `drafter: Drafter`, so this
    //     records that a Drafter-typed value drives the draft calls (the MLX loop itself is
    //     CI-only: it cannot evaluate an MLXArray on a Command-Line-Tools-only host).
    do {
        /// Records every `propose` call so the seam's call shape is observable without MLX.
        final class RecordingDrafter: Drafter {
            var calls: [(context: [Int], gamma: Int)] = []
            let inner = PromptLookupDrafter(maxN: 8)
            func propose(context: [Int], gamma: Int) -> [Int] {
                calls.append((context, gamma))
                return inner.propose(context: context, gamma: gamma)
            }
        }
        let recorder = RecordingDrafter()
        let seam: Drafter = recorder       // must be usable through the existential
        let out = seam.propose(context: [1, 2, 3, 1, 2], gamma: 3)
        eq(out, [3, 1, 2], "Drafter seam: an existential Drafter proposes through propose(context:gamma:)")
        eq(recorder.calls.count, 1, "Drafter seam: the call reached the recording implementation")
        eq(recorder.calls.first?.gamma, 3, "Drafter seam: gamma is passed through unchanged")
        // A drafter that always abstains is legal — the loop then takes one ordinary step.
        struct NullDrafter: Drafter {
            func propose(context: [Int], gamma: Int) -> [Int] { [] }
        }
        let null: Drafter = NullDrafter()
        eq(null.propose(context: [1, 2, 3], gamma: 4), [],
           "Drafter seam: an abstaining drafter is legal and returns []")
    }

    // --- #172: SpecStats derives its ratios (and never divides by zero) ---
    do {
        let s = SpecStats(rounds: 4, drafted: 10, accepted: 7, generated: 12)
        eq(s.acceptRate, 0.7, "SpecStats: acceptRate = accepted/drafted")
        eq(s.tokensPerRound, 3.0, "SpecStats: tokensPerRound = generated/rounds")
        let empty = SpecStats(rounds: 0, drafted: 0, accepted: 0, generated: 0)
        eq(empty.acceptRate, 0.0, "SpecStats: acceptRate is 0 when nothing was drafted")
        eq(empty.tokensPerRound, 0.0, "SpecStats: tokensPerRound is 0 with no rounds")
    }

    // --- #172: a BenchRecord written before `spec` existed must still decode ---
    do {
        let legacy = #"""
            [{"mode":"decode","machine":{"architecture":"Apple M1","memorySizeBytes":1,
              "hwModel":"m","cpuCores":8,"osVersion":"14","swiftVersion":"6","device":"metal"},
              "ci":false,"randomInit":false,"quantized":false,"timestamp":"t"}]
            """#
        let decoded = try? JSONDecoder().decode([BenchRecord].self,
                                                from: Data(legacy.utf8))
        check(decoded?.count == 1,
             "BenchRecord: JSON without a `spec` key still decodes (old --baseline files)")
        check(decoded?.first?.spec == nil, "BenchRecord: a missing `spec` key decodes as nil")
    }

    // --- flag parsing sanity (no MLX involved) ---
    do {
        let f = parseFlags(["--mode", "decode", "--quantize", "8"])
        eq(f["mode"], "decode", "--mode parses")
        eq(intFlag(f, "quantize", default: 0), 8, "--quantize parses as an int")
        let s = parseFlags(["--mode", "spec", "--gamma", "4", "--max-n", "8"])
        eq(s["mode"], "spec", "--mode spec parses")
        eq(intFlag(s, "gamma", default: 0), 4, "--gamma parses as an int")
        eq(intFlag(s, "max-n", default: 0), 8, "--max-n parses as an int")
        eq(intFlag(parseFlags([]), "gamma", default: 4), 4, "--gamma defaults to 4")
        eq(intFlag(parseFlags([]), "max-n", default: 8), 8, "--max-n defaults to 8")
    }

    if failures.isEmpty {
        print("monica-bench --self-test: OK — all checks passed")
        exit(0)
    } else {
        for f in failures { FileHandle.standardError.write(Data("FAIL: \(f)\n".utf8)) }
        FileHandle.standardError.write(
            Data("monica-bench --self-test: \(failures.count) failure(s)\n".utf8))
        exit(1)
    }
}

// MARK: - model construction

struct BuiltModel {
    let model: MonicaModel
    let source: ModelSource
    let randomInit: Bool
    let quantized: Bool
    let quantBits: Int?
}

/// `BenchRecord.quantBits` is a single `Int?` field, but a checkpoint's quant sidecar or
/// `Bench.quantTargets`'s filter can produce MIXED per-tensor bit-widths (e.g.
/// `--quant-head-bits` giving the embedding a different width than the rest). Recording
/// an arbitrary single value (`targets.values.first`, which is also dictionary-order
/// nondeterministic) would misrepresent a mixed-bit run as uniform — so record the
/// shared value only when every target actually agrees, `nil` otherwise (#170 review).
func uniformQuantBits(_ targets: [String: Int]) -> Int? {
    let distinct = Set(targets.values)
    return distinct.count == 1 ? distinct.first : nil
}

func buildModel(_ flags: [String: String]) -> BuiltModel {
    let src: ModelSource
    switch resolveModelSourceFlag(flags) {
    case .failure(let msg): fail(msg)
    case .success(let s): src = s
    }

    switch src {
    case .weights(let path):
        if flags["quantize"] != nil {
            fail("--quantize only applies to --config random-init runs — a --weights "
                + "checkpoint's quant sidecar (if any) is applied automatically via "
                + "Checkpoint.load")
        }
        do {
            let (model, _) = try Checkpoint.load(weights: URL(fileURLWithPath: path))
            let spec = try? QuantSpec.load(sidecar: URL(fileURLWithPath: path + ".config.json"))
            return BuiltModel(model: model, source: src, randomInit: false,
                              quantized: spec != nil,
                              quantBits: spec.flatMap { uniformQuantBits($0.targets) })
        } catch {
            fail("failed to load weights \(path): \(error)")
        }
    case .config(let path):
        let config: MambaConfig
        let model: MonicaModel
        do {
            config = try MambaConfig.load(sidecar: URL(fileURLWithPath: path))
            model = try MonicaModel(config)
            MLX.eval(model.parameters())
        } catch {
            fail("failed to load config \(path): \(error)")
        }
        print("(random init: timings valid, outputs meaningless)", to: &standardError)

        guard let qStr = flags["quantize"] else {
            return BuiltModel(model: model, source: src, randomInit: true,
                              quantized: false, quantBits: nil)
        }
        guard let bits = Int(qStr), bits == 8 || bits == 4 else {
            fail("--quantize must be 8 or 4, got '\(qStr)'")
        }
        let groupSize = intFlag(flags, "quant-group-size", default: 64)
        let headBits = optIntFlag(flags, "quant-head-bits")
        // #170's throughput/footprint approximation of `quant_targets` — see
        // `Bench.quantTargets`'s doc comment. `--weights` is the authoritative path for
        // real accuracy; this is only for benching a --config-built (random-init) model.
        let leaves: [(path: String, lastAxis: Int)] = model.parameters().flattened()
            .compactMap { name, arr in
                guard name.hasSuffix(".weight"), arr.ndim == 2 else { return nil }
                return (String(name.dropLast(".weight".count)), arr.dim(-1))
            }
        let targets = Bench.quantTargets(leaves: leaves, groupSize: groupSize, bits: bits,
                                         headBits: headBits)
        if targets.isEmpty {
            fail("--quantize \(bits): no quantizable tensors found for "
                + "group_size=\(groupSize) (every eligible tensor's last axis must be "
                + "divisible by group_size)")
        }
        var applied = Set<String>()
        MLXNN.quantize(model: model, filter: { modulePath, module in
            guard let b = targets[modulePath], module is Quantizable else { return nil }
            applied.insert(modulePath)
            return (groupSize: groupSize, bits: b, mode: .affine)
        })
        MLX.eval(model.parameters())
        return BuiltModel(model: model, source: src, randomInit: true,
                          quantized: true, quantBits: uniformQuantBits(targets))
    }
}

// MARK: - dispatch

func sourceName(_ src: ModelSource) -> String {
    switch src {
    case .weights(let p): return (p as NSString).lastPathComponent
    case .config(let p): return (p as NSString).lastPathComponent
    }
}

func runMain(_ flags: [String: String]) {
    let built = buildModel(flags)
    let machine = gatherMachineID()
    let ci = ProcessInfo.processInfo.environment["CI"] != nil
    let sha = gitSha()
    let mode = flags["mode"] ?? "all"
    let modes: [String] = mode == "all" ? ["prefill", "decode", "memory"] : [mode]
    let timestamp = ISO8601DateFormatter().string(from: Date())

    var records: [BenchRecord] = []

    for m in modes {
        var prefillR: PrefillResult? = nil
        var decodeR: DecodeResult? = nil
        var memoryR: MemoryResult? = nil
        var specR: SpecResult? = nil

        switch m {
        case "prefill":
            let promptLen = intFlag(flags, "bench-prompt-len", default: 128)
            let iterations = intFlag(flags, "bench-iterations", default: 5)
            let r = Bench.prefill(model: built.model, promptLen: promptLen, iterations: iterations)
            prefillR = r
            print(String(format: "monica-bench prefill: prompt_len=%d iterations=%d  "
                         + "sequential=%.2fms (%.1f tok/s)  parallel-scan=%.2fms (%.1f tok/s)  "
                         + "speedup=%.2fx", r.promptLen, r.iterations, r.sequentialMs,
                         r.sequentialTokPerSec, r.parallelMs, r.parallelTokPerSec, r.speedup))
            print("monica-bench prefill: sequential argmax=\(r.sequentialArgmax)  "
                + "parallel-scan argmax=\(r.parallelArgmax)")
            if r.sequentialArgmax != r.parallelArgmax {
                fail("monica-bench prefill: argmax DISAGREES between sequential and "
                    + "parallel-scan prefill — this is a correctness failure, not noise")
            }
        case "decode":
            let sampler = Sampler(temperature: 0.8, seed: 0)
            let r = Bench.decode(model: built.model, sampler: sampler)
            decodeR = r
            print(String(format: "monica-bench decode: %.1f tok/s (batch 1, %d tokens, "
                         + "%d warmup)", r.tokPerSec, r.measuredTokens, r.warmupTokens))
            if let g = r.generateTokPerSec {
                print(String(format: "monica-bench decode: Generator.generate end-to-end "
                             + "%.1f tok/s (%d tokens)", g, r.generateTokens ?? 0))
            }
        case "memory":
            let contextLength = intFlag(flags, "context-length", default: 512)
            let decodeTokens = intFlag(flags, "decode-tokens", default: 32)
            let r = Bench.memory(model: built.model, contextLength: contextLength,
                                 decodeTokens: decodeTokens)
            memoryR = r
            let peakStr = r.peakBytes.map { "\($0) bytes" } ?? "unavailable"
            print("monica-bench memory: peak=\(peakStr)  "
                + "analytic-state=\(r.analyticStateBytes) bytes  "
                + "analytic-kv=\(r.analyticKvBytes.map(String.init) ?? "n/a") bytes  "
                + "weights=\(r.totalWeightBytes) bytes  context_length=\(r.contextLength)")
        case "spec":
            // #172. Deliberately NOT part of `--mode all`: `all`'s JSON feeds
            // `--baseline`/`--strict` regression comparison, and adding a new noisy timing
            // section there would be a gratuitous risk to an existing gate. `spec` must be
            // asked for by name.
            let promptLen = intFlag(flags, "bench-prompt-len", default: 32)
            let maxNew = intFlag(flags, "max-new-tokens", default: 64)
            let gamma = intFlag(flags, "gamma", default: 4)
            let maxN = intFlag(flags, "max-n", default: 8)
            if gamma < 1 { fail("--gamma must be >= 1, got \(gamma)") }
            if maxN < 1 { fail("--max-n must be >= 1, got \(maxN)") }
            let r: SpecResult
            do {
                r = try Bench.spec(model: built.model, promptLen: promptLen,
                                   maxNewTokens: maxNew, gamma: gamma, maxN: maxN)
            } catch {
                fail("monica-bench spec: \(error)")
            }
            specR = r
            print(String(format: "monica-bench spec: prompt_len=%d max_new=%d gamma=%d "
                         + "max_n=%d", r.promptLen, r.maxNewTokens, r.gamma, r.maxN))
            print(String(format: "monica-bench spec: plain=%.1f tok/s  speculative=%.1f tok/s"
                         + "  speedup=%.2fx (INFORMATIONAL — no threshold is asserted)",
                         r.plainTokPerSec, r.specTokPerSec, r.speedup))
            print(String(format: "monica-bench spec: accepted %d/%d draft tokens (%.1f%%), "
                         + "%.2f tokens/round over %d rounds",
                         r.accepted, r.drafted, r.acceptRate * 100.0, r.tokensPerRound,
                         r.rounds))
            // `Bench.spec` already fails loud on a mismatch, so reaching here means the
            // ids matched. Print the claim so the CI log carries the gate's verdict, and
            // guard the field anyway rather than trusting a struct we just built.
            if !r.identical {
                fail("monica-bench spec: identical=false — speculative output is NOT "
                    + "byte-identical to plain greedy decode")
            }
            print("monica-bench spec: speculative output is byte-identical to plain greedy "
                + "decode (GATE)")
        default:
            fail("--mode must be one of prefill, decode, memory, spec, all — got '\(m)'")
        }

        records.append(BenchRecord(
            mode: m, machine: machine, ci: ci, gitSha: sha, source: sourceName(built.source),
            randomInit: built.randomInit, quantized: built.quantized, quantBits: built.quantBits,
            timestamp: timestamp, prefill: prefillR, decode: decodeR, memory: memoryR,
            spec: specR))
    }

    if let jsonPath = flags["json"] {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(records)
            try data.write(to: URL(fileURLWithPath: jsonPath))
            print("[monica-bench] wrote \(records.count) record(s) to \(jsonPath)")
        } catch {
            fail("failed to write --json \(jsonPath): \(error)")
        }
    }

    if let baselinePath = flags["baseline"] {
        let tolerance = doubleFlag(flags, "tolerance", default: 0.15)
        let strict = flags["strict"] != nil
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: baselinePath))
            let baselines = try JSONDecoder().decode([BenchRecord].self, from: data)
            var anyRegression = false
            for record in records {
                let comparisons = Bench.compareToBaseline(record, baselines: baselines,
                                                           tolerance: tolerance)
                for c in comparisons {
                    print("[baseline] \(c.status.rawValue): \(c.message)")
                    if c.status == .regression { anyRegression = true }
                }
            }
            if strict && anyRegression {
                fail("--strict: one or more metrics regressed against \(baselinePath)")
            }
        } catch {
            fail("failed to read --baseline \(baselinePath): \(error)")
        }
    }
}

// MARK: - entry point

let argv = Array(CommandLine.arguments.dropFirst())
let flags = parseFlags(argv)

if flags["self-test"] != nil {
    runSelfTest()
}

if ProcessInfo.processInfo.environment["MONICA_ENGINE_CPU"] == "1" {
    print("(MONICA_ENGINE_CPU=1 — running on the MLX CPU device)", to: &standardError)
    Device.withDefaultDevice(.cpu) { runMain(flags) }
} else {
    runMain(flags)
}
