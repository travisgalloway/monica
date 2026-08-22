# Benchmark results ledger

The results record for the two bench harnesses (#170): `monica-bench` (Swift/mlx-swift,
`swift/engine/`) and `scripts/bench_context.py` / `scripts/bench_train_step.py --mode decode`
(Python/MLX). See `docs/design/14-inference-engine.md`'s "#170 — the benchmark harness" section
for the design rationale.

**Read the provenance column before trusting any number here.** CI runners are shared,
virtualized, noisy hardware — useful for "did it run, do the two paths agree, is the shape
right," useless as the local-hardware win CLAUDE.md's POC success criterion asks for. Only a row
tagged **developer Apple Silicon** is a claim about real hardware performance.

## How to reproduce each row

```bash
# Swift engine (requires Xcode's Metal toolchain — default.metallib — not just Command Line
# Tools; see the host-constraint note below):
cd swift/engine
swift run monica-bench --weights Fixtures/toy/weights.safetensors --mode all
swift run monica-bench --config Benchmarks/configs/poc.config.json --mode all --json bench.json
swift run monica-bench --config Benchmarks/configs/poc.config.json --quantize 8 --mode decode
swift run monica-bench --weights Fixtures/toy/weights.safetensors --mode spec --gamma 4
swift run monica-bench --self-test   # deterministic, hardware-independent — runs anywhere

# Python / MLX (runs on Apple Silicon with the project venv; no Xcode needed):
.venv/bin/python scripts/bench_context.py --config config/poc-small.yaml --arms ssm \
    --lengths 512,1024 --decode-tokens 64 --prefill-mode both --json out.json
.venv/bin/python scripts/bench_train_step.py --config config/poc-small.yaml --mode decode
```

## Swift engine (`monica-bench`)

| Mode | Metric | Value | Machine | Provenance |
|---|---|---|---|---|
| prefill | sequential | 332.62 ms | GitHub-hosted `macos-latest` runner | **CI runner** (run [31777284815](https://github.com/travisgalloway/monica/actions/runs/31777284815), `Fixtures/toy`, `prompt_len=128 iterations=5`) |
| prefill | parallel-scan | 6.01 ms | GitHub-hosted `macos-latest` runner | **CI runner** (same run) |
| prefill | speedup | 55.30x | GitHub-hosted `macos-latest` runner | **CI runner** (same run; argmax agreement `186 == 186`) |
| prefill | poc-scale (`Benchmarks/configs/poc.config.json`) | — | — | **not yet measured** — needs a developer Mac with Xcode installed (see host constraint below); command: `swift run monica-bench --config Benchmarks/configs/poc.config.json --mode prefill` |
| decode | poc-scale, fp vs `--quantize 8`/`--quantize 4` | — | — | **not yet measured** — same constraint; command: `swift run monica-bench --config Benchmarks/configs/poc.config.json --mode decode [--quantize 8]` |
| memory | poc-scale peak/analytic | — | — | **not yet measured** — same constraint; command: `swift run monica-bench --config Benchmarks/configs/poc.config.json --mode memory` |
| spec (#172) | speculative vs plain greedy tok/s, speedup, accept rate | — | GitHub-hosted `macos-latest` runner | **CI runner, informational** — emitted per run by `swift-engine`'s `monica-bench --mode spec` step (`Fixtures/toy`, `--gamma 4`) into the `monica-bench-records` artifact (`monica-bench-toy-spec.json`); no figure is promoted into this table, because a hosted-runner timing is not a hardware claim. **What that step DOES gate is correctness, not speed:** speculative output byte-identical to plain greedy (`Bench.spec` exits non-zero naming the first differing index) |
| spec (#172) | speculative decode on developer Apple Silicon | — | — | **no local Apple-Silicon measurement exists, and none can be taken on this host** — see the host constraint below: `SpecDecodeLoop.swift` evaluates `MLXArray`s, so `swift run monica-bench --mode spec` fails here with `Failed to load the default metallib`. A real number needs a developer Mac with Xcode installed; command: `swift run monica-bench --weights Fixtures/toy/weights.safetensors --mode spec --gamma 4` |

The 55.30x row is #169's AC3, carried forward unchanged: `monica-bench`'s prefill mode calls the
same `Bench.prefill` `monica-generate --bench-prefill` calls, so this figure stays the comparable
baseline as the harness grows rather than being orphaned.

### Host constraint (why the "not yet measured" rows exist)

This harness was built and reviewed on a Mac with Command Line Tools only, no Xcode.
`MONICA_ENGINE_CPU=1 swift run monica-bench ...` (and `monica-parity`/`monica-generate` before
it) fails with `MLX error: Failed to load the default metallib` — `default.metallib` is an
Xcode-only build product (see `.github/workflows/ci.yml`'s `swift-engine` job, which works around
it with `xcodebuild`). This is a pre-existing, documented constraint, not specific to #170.
`swift build` (plain SwiftPM) DOES succeed for all three executables including `monica-bench`,
and `monica-bench --self-test` (no MLX array ops — pure Swift arithmetic and Codable) passes on
this box. Every genuinely local Swift-engine timing/memory number therefore needs a developer
machine with Xcode installed to run `swift run monica-bench ...` and fill in the rows above.

## Python / MLX (`scripts/bench_context.py`, `scripts/bench_train_step.py --mode decode`)

Machine: **MacBookPro18,3 (Apple M1 Pro, 32 GB unified memory), macOS 26.5.2** — the same
developer machine CLAUDE.md's `~99 s/step` poc training baseline was measured on. Python MLX has
no metallib constraint (`scripts/smoke_test.py` passes on the MLX backend on this box), so these
are real **developer Apple Silicon** numbers.

| Config | Scale | Metric | Value | Provenance |
|---|---|---|---|---|
| `config/toy.yaml` | ~1M (smoke) | prefill speedup (`--prefill-mode both`, length 8/16) | 4.21x / 7.95x | **developer Apple Silicon** (M1 Pro) — smoke-scale only, not a representative figure |
| `config/poc-small.yaml` | ~97M | sequential prefill | 116.3 tok/s | **developer Apple Silicon** (M1 Pro), `--arms ssm --lengths 512 --decode-tokens 32 --prefill-mode both` |
| `config/poc-small.yaml` | ~97M | parallel-scan prefill | 4,630.7 tok/s | **developer Apple Silicon** (M1 Pro), same run |
| `config/poc-small.yaml` | ~97M | prefill speedup | 39.81x | **developer Apple Silicon** (M1 Pro), same run |
| `config/poc-small.yaml` | ~97M | decode (batch-1 `model.step` loop, length 512) | 134.5 tok/s | **developer Apple Silicon** (M1 Pro), same run |
| `config/poc-small.yaml` | ~97M | peak memory (prefill+decode, length 512) | 0.812 GB | **developer Apple Silicon** (M1 Pro), same run |
| `config/poc-small.yaml` | ~97M | decode, exact M7 protocol (32 warmup, 256 measured, batch 1) | 127.5 tok/s | **developer Apple Silicon** (M1 Pro), `scripts/bench_train_step.py --mode decode` — directly comparable to the cited 94.7 tok/s poc-scale (`config/poc.yaml`, ~127M today, possibly ~205M as measured — see next row) M7 record; poc-small is ~97M, smaller, hence faster |
| `config/poc.yaml` | ~127M | decode, exact M7 protocol | 94.7 tok/s | **developer Apple Silicon** (M1 Pro) — the pre-#170 M7 record cited in `docs/design/14-inference-engine.md`; not re-measured in this run, and predates the poc/poc-qwen split, so the measured config may have carried the larger Qwen vocab (~205M) |
| `config/poc-small.yaml` | ~97M | context-length sweep, `attn` arm / crossover point | — | **not yet measured** — command: `.venv/bin/python scripts/bench_context.py --config config/poc-small.yaml --arms ssm,attn --lengths 512,1024,2048 --decode-tokens 64 --json out.json` |

## Which numbers came from where — summary

- **CI runner (GitHub-hosted `macos-latest`, `xcodebuild`)**: the Swift-engine prefill row
  (55.30x, run 31777284815) and, going forward, every `swift-engine` CI run's `monica-bench
  --mode all` / `--mode decode` / `--mode spec` JSON artifact (uploaded per-run, not yet promoted into this
  table — informational only, no timing threshold gates CI).
- **Developer Apple Silicon (MacBookPro18,3 / Apple M1 Pro / 32 GB / macOS 26.5.2)**: every
  Python/MLX row above. These are the only rows in this document that can honestly be called a
  "local-hardware win" per CLAUDE.md's POC success criterion.
- **Not yet measured**: every poc-scale Swift-engine row (prefill/decode/memory, fp and
  quantized) — blocked on a developer machine with Xcode installed running `swift run
  monica-bench --config Benchmarks/configs/poc.config.json ...`; and the Python attn-arm
  context-length sweep at poc-small scale.

`monica-bench --baseline Benchmarks/baselines.json [--tolerance 0.15] [--strict]` is how a future
run flags a regression against a captured baseline (matching on machine id — architecture, hw
model, memory size, and device; a different machine, including any developer Mac, reads
`SKIPPED` for that comparison, never a false-green `OK`, until its own baseline row is added).

**`Benchmarks/baselines.json`'s checked-in row is a hand-authored placeholder, not a captured
measurement** — its `machine` fields (`hwModel: "github-actions-macos-latest"`, `cpuCores: 0`,
`architecture: "unknown (...)"`, `memorySizeBytes: 0`) are stand-ins and will not equal what
`monica-bench` actually records via `sysctl hw.model` / core count / `GPU.deviceInfo()` on the
real GitHub-hosted runner. Until this row is replaced with a real `monica-bench` JSON record
captured on that runner, every `--baseline` comparison against it reads `SKIPPED` (never a false
`OK` — see `Bench.compareToBaseline`), so `--baseline`/`--strict` do not yet gate CI on a real
regression threshold; they only exercise the comparison machinery end-to-end.
