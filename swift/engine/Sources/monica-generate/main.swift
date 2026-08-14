// monica-generate — native completion/chat CLI over the Swift/MLX engine (#167).
//
// Mirrors `scripts/generate.py`'s flag surface so the two are directly comparable. Wires:
// `swift/MonicaTokenizer` (encode/decode) -> `Generator.generate` (the port of
// `src/serve/generate.py`) -> `Sampler` (the port of `src/serve/sampling.py`), streaming
// tokens to stdout.
//
//   monica-generate --weights <w.safetensors> --prompt "..." [--temperature 0.8] [...]
//   monica-generate --weights <w.safetensors> --prompt-ids 1,2,3   # tokenizer-free debug path
//   monica-generate --weights <w.safetensors> --tokenizer <t.json> --chat
//   monica-generate --self-test [--tokenizer <t.json>]
//
// Style follows `monica-parity`/`monica-selfcheck`: hand-rolled arg parsing (no
// swift-argument-parser dependency — see Package.swift's header comment on why the tokenizer
// edge is executable-only), a `failures` array for --self-test, `exit(1)` on failure.
//
// `MONICA_ENGINE_CPU=1` (honored exactly like `monica-parity`) is mandatory on a Command-
// Line-Tools-only Mac: mlx-swift's Metal path needs a compiled `default.metallib`, which
// needs Xcode's `metal` compiler.

import Foundation
import MLX
import MonicaEngine
import MonicaTokenizer

// MARK: - arg parsing (hand-rolled; mirrors monica-tokenize's parseFlags)

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

func doubleFlag(_ flags: [String: String], _ name: String, default def: Double) -> Double {
    guard let raw = flags[name], !raw.isEmpty else { return def }
    guard let v = Double(raw) else { fail("--\(name) must be a number, got '\(raw)'") }
    return v
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

func optDoubleFlag(_ flags: [String: String], _ name: String) -> Double? {
    guard let raw = flags[name], !raw.isEmpty else { return nil }
    guard let v = Double(raw) else { fail("--\(name) must be a number, got '\(raw)'") }
    return v
}

func uint64Flag(_ flags: [String: String], _ name: String, default def: UInt64) -> UInt64 {
    guard let raw = flags[name], !raw.isEmpty else { return def }
    guard let v = UInt64(raw) else { fail("--\(name) must be a non-negative integer, got '\(raw)'") }
    return v
}

// MARK: - chat template (mirrors src/data/instruct_format.py — the single source of truth
// lives in Python; this reproduces its rendering exactly so the two never silently drift on
// the marker strings, which are load-bearing for both sides).

let INSTRUCTION_MARKER = "### Instruction:"
let RESPONSE_MARKER = "### Response:"
let SYSTEM_MARKER = "### System:"

struct ChatTurn { let role: String; let content: String }

/// Marker per role — mirrors `src/data/instruct_format.py`'s `ROLE_MARKERS` dict exactly
/// (system/user/assistant), rather than collapsing every non-assistant role onto
/// `INSTRUCTION_MARKER`. An unrecognized role is a caller bug, not a silent instruction turn.
func renderTurn(_ t: ChatTurn) -> String {
    let marker: String
    switch t.role {
    case "system": marker = SYSTEM_MARKER
    case "user": marker = INSTRUCTION_MARKER
    case "assistant": marker = RESPONSE_MARKER
    default: fail("renderTurn: unknown role '\(t.role)' (expected system/user/assistant)")
    }
    let c = t.content.trimmingCharacters(in: .whitespacesAndNewlines)
    return c.isEmpty ? marker : "\(marker) \(c)"
}

func render(_ turns: [ChatTurn]) -> String {
    turns.map(renderTurn).joined(separator: " ")
}

// MARK: - self-test (no weights needed; --self-test)

func runSelfTest(tokenizerPath: String?) -> Never {
    var failures: [String] = []
    func check(_ cond: Bool, _ msg: String) { if !cond { failures.append(msg) } }
    func eq<T: Equatable>(_ a: T, _ b: T, _ msg: String) {
        if a != b { failures.append("\(msg): \(a) != \(b)") }
    }
    func throwsErr(_ body: () throws -> Void) -> Bool {
        do { try body(); return false } catch { return true }
    }

    // --- greedy argmax + first-tie ---
    do {
        var s = Sampler(temperature: 0)
        let id = (try? s.sample([1.0, 3.0, 3.0, 2.0])) ?? -1
        eq(id, 1, "greedy keeps the FIRST maximum on ties")
    }

    // --- temperature monotonicity: lower temperature concentrates mass on the argmax ---
    do {
        let logits: [Float] = [0.1, 0.2, 5.0, 0.3]
        var hot = Sampler(temperature: 5.0, seed: 1)
        var cold = Sampler(temperature: 0.1, seed: 1)
        var hotArgmaxCount = 0, coldArgmaxCount = 0
        for _ in 0..<200 {
            if (try? hot.sample(logits)) == 2 { hotArgmaxCount += 1 }
            if (try? cold.sample(logits)) == 2 { coldArgmaxCount += 1 }
        }
        check(coldArgmaxCount > hotArgmaxCount,
              "lower temperature draws the argmax more often (\(coldArgmaxCount) vs \(hotArgmaxCount))")
    }

    // --- tie-inclusive top-k: k=1 with a tied max keeps BOTH tied tokens live ---
    do {
        var s = Sampler(temperature: 1.0, topK: 1, seed: 7)
        var seen = Set<Int>()
        for _ in 0..<200 { seen.insert((try? s.sample([0.0, 5.0, 5.0, -1.0])) ?? -1) }
        check(seen.isSubset(of: [1, 2]), "top-k=1 with a tie only draws the tied indices: \(seen)")
        check(seen.count == 2, "top-k=1 with a tie keeps BOTH tied tokens live (tie-inclusive): \(seen)")
    }

    // --- nucleus keeps the crossing token ---
    do {
        // probs after softmax over [huge, 0, 0, 0] are ~[1, 0, 0, 0] before renorm; use a
        // gentler spread so top_p has a real crossing token to include.
        var s = Sampler(temperature: 1.0, topP: 0.5, seed: 3)
        var seen = Set<Int>()
        for _ in 0..<300 { seen.insert((try? s.sample([2.0, 1.0, 0.0, -5.0])) ?? -1) }
        check(seen.contains(0), "nucleus always keeps the top token")
        check(!seen.contains(3), "nucleus at p=0.5 excludes the far tail")
    }

    // --- repetition penalty sign handling (CTRL-style) ---
    do {
        var s = Sampler(temperature: 0, repetitionPenalty: 2.0)
        // positive logit divided by penalty, negative multiplied — both push DOWN. 3.0/2=1.5
        // drops below the unseen 2.9, flipping the argmax.
        let idPos = try? s.sample([3.0, 2.9], previousTokens: [0])
        eq(idPos, 1, "repetition penalty pushes a positive seen logit below an unseen one")
        var s2 = Sampler(temperature: 0, repetitionPenalty: 2.0)
        let idNeg = try? s2.sample([-1.0, -5.0], previousTokens: [0])
        eq(idNeg, 0, "repetition penalty on a negative seen logit still ranks below unseen (sanity)")
        var s3 = Sampler(temperature: 0, repetitionPenalty: 0.5)
        check(throwsErr { _ = try s3.sample([1.0, 2.0], previousTokens: [0]) },
              "repetitionPenalty < 1.0 throws")
    }

    // --- no-repeat-ngram bans ---
    do {
        var s = Sampler(temperature: 0, noRepeatNgramSize: 2)
        // prev = [0, 1, 0]; the trailing 1-gram [0] previously preceded token 1, so token 1
        // is banned as the next draw even though it has the highest logit.
        let id = try? s.sample([1.0, 5.0, 2.0], previousTokens: [0, 1, 0])
        eq(id, 2, "no-repeat-ngram bans the id that would repeat the n-gram")
    }

    // --- allowedIds never escaped, including on the all-banned fallback ---
    do {
        var s = Sampler(temperature: 0)
        for _ in 0..<50 {
            let id = try? s.sample([1.0, 9.0, 2.0, 8.0], allowedIds: [0, 2])
            check(id == 0 || id == 2, "allowedIds constrains greedy draws to the set")
        }
        // Every allowed id gets banned by repetition control -> all-non-finite fallback must
        // still draw only from allowedIds, never escape to the full vocab.
        var s2 = Sampler(temperature: 0, repetitionPenalty: 1.0, noRepeatNgramSize: 1)
        var escaped = false
        for _ in 0..<50 {
            let id = try? s2.sample([1.0, 9.0, 2.0, 8.0], previousTokens: [0, 2],
                                    allowedIds: [0, 2])
            if let id, id != 0 && id != 2 { escaped = true }
        }
        check(!escaped, "all-banned fallback never escapes allowedIds")
        var s3 = Sampler(temperature: 0)
        check(throwsErr { _ = try s3.sample([1.0, 2.0], allowedIds: []) },
              "empty allowedIds throws")
        var s4 = Sampler(temperature: 0)
        check(throwsErr { _ = try s4.sample([1.0, 2.0], allowedIds: [99, 100]) },
              "allowedIds with no in-range ids throws (treated as empty)")
    }

    // --- seeded draws reproducible across two independent runs ---
    do {
        var a = Sampler(temperature: 1.0, seed: 42)
        var b = Sampler(temperature: 1.0, seed: 42)
        let logits: [Float] = [1.0, 2.0, 0.5, 3.0, 1.5]
        var seqA: [Int] = [], seqB: [Int] = []
        for _ in 0..<20 {
            seqA.append((try? a.sample(logits)) ?? -1)
            seqB.append((try? b.sample(logits)) ?? -1)
        }
        eq(seqA, seqB, "same seed reproduces the same draw sequence")
    }

    // --- #169: flag parsing for the new prefill-related flags (no MLX involved, so this
    // runs fully outside CI's mlx-swift gate too) ---
    do {
        let f = parseFlags(["--weights", "w.safetensors", "--no-prefill"])
        check(f["no-prefill"] != nil, "--no-prefill parses as a present (valueless) flag")
        let f2 = parseFlags(["--bench-prefill", "--bench-prompt-len", "256",
                             "--bench-iterations", "3"])
        check(f2["bench-prefill"] != nil, "--bench-prefill parses as a present flag")
        eq(intFlag(f2, "bench-prompt-len", default: 512), 256,
           "--bench-prompt-len parses as an int")
        eq(intFlag(f2, "bench-iterations", default: 5), 3,
           "--bench-iterations parses as an int")
    }

    // --- optional tokenizer round-trip ---
    if let path = tokenizerPath {
        do {
            let tok = try Tokenizer(contentsOf: URL(fileURLWithPath: path))
            for s in ["function add(a, b) { return a + b; }", "", "hello world"] {
                eq(tok.decode(tok.encode(s)), s, "tokenizer round-trip: '\(s)'")
            }
        } catch {
            failures.append("could not load --tokenizer \(path): \(error)")
        }
    }

    if failures.isEmpty {
        print("monica-generate --self-test: OK — all checks passed")
        exit(0)
    } else {
        for f in failures { FileHandle.standardError.write(Data("FAIL: \(f)\n".utf8)) }
        FileHandle.standardError.write(Data("monica-generate --self-test: \(failures.count) failure(s)\n".utf8))
        exit(1)
    }
}

// MARK: - AC3: prefill latency bench (#169)

/// The arrays inside one layer's state, so the bench's `MLX.eval` calls can force the state
/// alongside logits — the same helper `Generator`/`monica-parity` use, duplicated here (not
/// exported by `MonicaEngine`) so lazy evaluation cannot move work outside the timer, the
/// single most likely way to measure nothing at all.
func benchEvalTargets(_ s: LayerState) -> [MLXArray] {
    switch s {
    case .mamba(let conv, let ssm): return [conv, ssm]
    case .attention(let k, let v): return [k, v]
    case .moe: return []
    }
}

/// `--bench-prefill`: sequential per-token `step` prefill vs the one-shot parallel-scan
/// `model.prefill`, timed over N iterations after a warm-up (lazy MLX graphs otherwise let
/// the first call's compile/dispatch cost leak into the measurement). Asserts the two
/// paths' argmax id AGREES — a correctness property that is not timing-sensitive — and
/// prints per-path ms plus the speedup. Informational: CI wires this as a non-gating step
/// (hosted-runner timing is noisy), so this function never calls `exit(1)` on timing alone.
func runBenchPrefill(model: MonicaModel, promptLen: Int, iterations: Int) {
    let vocab = model.config.vocabSize
    // Prompt length is independent of vocab size — do not clamp it against `vocab`, or long
    // prompts silently get truncated on small-vocab fixtures. Only guard against non-positive
    // input; token ids themselves are drawn from `0..<vocab` below.
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
            MLX.eval([lg] + st.flatMap(benchEvalTargets))
            logits = lg
            state = st
        }
        return (logits, state)
    }
    func parallelPrefill() -> (MLXArray, [LayerState]) {
        let (lg, st) = model.prefill(promptArr, lastOnly: true)
        MLX.eval([lg] + st.flatMap(benchEvalTargets))
        return (lg, st)
    }

    // Warm-up (once each) so the first call's graph compile/dispatch is not in the timing.
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
        var bestI = 0
        var bestV = -Float.infinity
        for (i, v) in row.enumerated() where v > bestV { bestV = v; bestI = i }
        return (elapsedMs, bestI)
    }

    let seq = timeMs(sequentialPrefill)
    let par = timeMs(parallelPrefill)
    let speedup = par.ms > 0 ? seq.ms / par.ms : Double.infinity
    print(String(format: "bench-prefill: prompt_len=%d iterations=%d  "
                 + "sequential=%.2fms  parallel-scan=%.2fms  speedup=%.2fx",
                 clampedLen, clampedIterations, seq.ms, par.ms, speedup))
    print("bench-prefill: sequential argmax=\(seq.argmax)  parallel-scan argmax=\(par.argmax)")
    if seq.argmax != par.argmax {
        fail("bench-prefill: argmax DISAGREES between sequential and parallel-scan prefill "
            + "(\(seq.argmax) vs \(par.argmax)) — this is a correctness failure, not noise")
    }
}

// MARK: - generation drivers

/// Streams a completion: incremental delta-decoding (decode the growing id array and emit
/// only the new suffix), because a byte-level BPE id can be a partial UTF-8 sequence and
/// per-token decoding would emit replacement characters mid-multibyte. A real improvement
/// over `scripts/generate.py`'s per-token `tok.decode([t])`.
func streamCompletion(model: MonicaModel, tok: Tokenizer?, promptIds: [Int], sampler: Sampler,
                      maxNewTokens: Int, eosId: Int?, usePrefill: Bool = true) throws {
    var generatedIds: [Int] = []
    var prevDecoded = ""
    _ = try Generator.generate(
        model: model, promptIds: promptIds, sampler: sampler, maxNewTokens: maxNewTokens,
        eosId: eosId,
        onToken: { t in
            generatedIds.append(t)
            guard let tok else {
                // Tokenizer-free debug path (--prompt-ids with no --tokenizer): print raw ids.
                print(t, terminator: " ")
                fflush(stdout)
                return
            }
            let full = tok.decode(generatedIds)
            let delta = String(full.dropFirst(prevDecoded.count))
            prevDecoded = full
            FileHandle.standardOutput.write(Data(delta.utf8))
            fflush(stdout)
        }, usePrefill: usePrefill)
    print()
}

/// Chat mode: buffers and prints once (the stop marker is only detectable after its tokens
/// exist, so streaming would leak it to stdout before it could be truncated) — same trade
/// `scripts/generate.py` documents. The M12 code model has no instruction SFT yet, so this is
/// a shell mirroring the Python CLI's shape, not a claim about the model's actual format.
func runChat(model: MonicaModel, tok: Tokenizer, sampler: Sampler, maxNewTokens: Int,
            eosId: Int?, usePrefill: Bool = true) {
    print("chat mode — type a message (Ctrl-D to exit).", to: &standardError)
    var history: [ChatTurn] = []
    while true {
        FileHandle.standardOutput.write(Data(">>> ".utf8))
        fflush(stdout)
        guard let line = readLine() else { print(); break }
        if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
        history.append(ChatTurn(role: "user", content: line))
        let prompt = render(history + [ChatTurn(role: "assistant", content: "")])
        let promptIds = tok.encode(prompt)
        if promptIds.isEmpty { continue }
        var reply = ""
        do {
            let outIds = try Generator.generate(
                model: model, promptIds: promptIds, sampler: sampler, maxNewTokens: maxNewTokens,
                eosId: eosId,
                stopFn: { gen in tok.decode(gen).contains(INSTRUCTION_MARKER) },
                usePrefill: usePrefill)
            let out = tok.decode(outIds)
            if let range = out.range(of: INSTRUCTION_MARKER) {
                reply = String(out[out.startIndex..<range.lowerBound])
            } else {
                reply = out
            }
        } catch {
            FileHandle.standardError.write(Data("generation error: \(error)\n".utf8))
            continue
        }
        print(reply)
        history.append(ChatTurn(role: "assistant",
                                content: reply.trimmingCharacters(in: .whitespacesAndNewlines)))
    }
}

/// A `TextOutputStream` wrapper so `print(_:to:)` can target stderr, mirroring the
/// `scripts/generate.py` REPL's `file=sys.stderr` banners.
struct StandardError: TextOutputStream {
    mutating func write(_ string: String) { FileHandle.standardError.write(Data(string.utf8)) }
}
var standardError = StandardError()

// MARK: - dispatch

/// The whole checkpoint-load-through-generation path, run under whatever the current default
/// MLX device is. Split out so `MONICA_ENGINE_CPU=1` can scope it with `Device.withDefaultDevice`
/// (the same scoped-not-global convention `monica-parity` uses) rather than the deprecated
/// `Device.setDefault`.
func runMain(_ flags: [String: String]) {
    guard let weightsPath = flags["weights"] else {
        fail("--weights <weights.safetensors> is required (or pass --self-test)")
    }

    let (model, _): (MonicaModel, Set<String>)
    do {
        (model, _) = try Checkpoint.load(weights: URL(fileURLWithPath: weightsPath))
    } catch {
        fail("failed to load weights \(weightsPath): \(error)")
    }

    if flags["bench-prefill"] != nil {
        let promptLen = intFlag(flags, "bench-prompt-len", default: 512)
        let iterations = intFlag(flags, "bench-iterations", default: 5)
        runBenchPrefill(model: model, promptLen: promptLen, iterations: iterations)
        return
    }

    // #169: `--no-prefill` forces the OLD per-token sequential prefill (the AC3 baseline
    // and the AC1 A/B reference); default is the new one-shot parallel-scan prefill.
    let usePrefill = flags["no-prefill"] == nil

    var tokenizer: Tokenizer? = nil
    if let tp = flags["tokenizer"] {
        do {
            let t = try Tokenizer(contentsOf: URL(fileURLWithPath: tp))
            if t.vocabSize > model.config.vocabSize {
                fail("tokenizer vocab \(t.vocabSize) exceeds model vocab \(model.config.vocabSize) "
                    + "(\(weightsPath)) — use a tokenizer trained for this checkpoint")
            }
            tokenizer = t
        } catch {
            fail("failed to load tokenizer \(tp): \(error)")
        }
    }

    let temperature = doubleFlag(flags, "temperature", default: 0.8)
    let maxNewTokens = intFlag(flags, "max-new-tokens", default: 100)
    let sampler = Sampler(
        temperature: temperature,
        topK: optIntFlag(flags, "top-k"),
        topP: optDoubleFlag(flags, "top-p"),
        repetitionPenalty: doubleFlag(flags, "repetition-penalty", default: 1.0),
        noRepeatNgramSize: optIntFlag(flags, "no-repeat-ngram-size"),
        seed: uint64Flag(flags, "seed", default: 0))
    let eosId = tokenizer?.eosTokenId

    if flags["chat"] != nil {
        guard let tok = tokenizer else { fail("--chat requires --tokenizer <tokenizer.json>") }
        runChat(model: model, tok: tok, sampler: sampler, maxNewTokens: maxNewTokens,
               eosId: eosId, usePrefill: usePrefill)
    } else if let idsRaw = flags["prompt-ids"] {
        let promptIds = idsRaw.split(separator: ",").map { s -> Int in
            guard let v = Int(s.trimmingCharacters(in: .whitespaces)) else {
                fail("--prompt-ids must be a comma-separated list of integers, got '\(idsRaw)'")
            }
            return v
        }
        if promptIds.isEmpty { fail("--prompt-ids must be non-empty") }
        do {
            try streamCompletion(model: model, tok: tokenizer, promptIds: promptIds,
                                 sampler: sampler, maxNewTokens: maxNewTokens, eosId: eosId,
                                 usePrefill: usePrefill)
        } catch { fail("generation failed: \(error)") }
    } else if let prompt = flags["prompt"] {
        guard let tok = tokenizer else { fail("--prompt requires --tokenizer <tokenizer.json>") }
        let promptIds = tok.encode(prompt)
        if promptIds.isEmpty { fail("--prompt encoded to zero tokens") }
        FileHandle.standardOutput.write(Data(prompt.utf8))
        do {
            try streamCompletion(model: model, tok: tok, promptIds: promptIds, sampler: sampler,
                                 maxNewTokens: maxNewTokens, eosId: eosId, usePrefill: usePrefill)
        } catch { fail("generation failed: \(error)") }
    } else {
        fail("provide --prompt <text>, --prompt-ids 1,2,3, --chat, or --self-test")
    }
}

let argv = Array(CommandLine.arguments.dropFirst())
let flags = parseFlags(argv)

if flags["self-test"] != nil {
    runSelfTest(tokenizerPath: flags["tokenizer"])
}

// `MONICA_ENGINE_CPU=1` runs generation on the MLX CPU device — mandatory on a Command-Line-
// Tools-only Mac, where mlx-swift's Metal path needs a compiled `default.metallib` that only
// Xcode's `metal` compiler can produce. Exactly the escape hatch `monica-parity` honors.
if ProcessInfo.processInfo.environment["MONICA_ENGINE_CPU"] == "1" {
    print("(MONICA_ENGINE_CPU=1 — running on the MLX CPU device)", to: &standardError)
    Device.withDefaultDevice(.cpu) { runMain(flags) }
} else {
    runMain(flags)
}
