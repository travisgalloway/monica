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
| `config/poc-small.yaml` | ~97M | context-length sweep, `attn` arm / crossover point | crossover at L=1024 (sequential decode 150.5 vs 110.0 tok/s; parallel prefill 7,982.8 vs 2,041.1 tok/s at L=2048, 3.9x speedup) | **developer Apple Silicon** (M1 Pro), `scripts/bench_context.py --config config/poc-small.yaml --arms ssm,attn --lengths 512,1024,2048 --decode-tokens 64` |

### Context-length scaling: Mamba-2 vs same-size transformer baseline (#104)

Context-length scaling and sustained decode throughput on developer Apple Silicon (Apple M1 Pro, 32 GB unified memory), comparing Mamba-2 architectures against equivalent-sized transformer baselines (`attn_every=1`).

#### Pure Mamba-2 vs transformer (`config/poc-small.yaml`, ~97M)

Measured via `scripts/bench_context.py --config config/poc-small.yaml --arms ssm,attn --lengths 512,1024,2048 --decode-tokens 64 --prefill-mode both`:

| Arm | Context Length | Sequential Prefill (tok/s) | Parallel Prefill (tok/s) | Decode (tok/s) | Peak Memory (GB) | Recurrent State (MB) |
|---|---|---|---|---|---|---|
| `ssm` | 512 | 147.6 | 3,879.7 | 142.5 | 0.760 | 0.891 |
| `ssm` | 1024 | 149.1 | 10,343.1 | 153.0 | 1.016 | 0.891 |
| `ssm` | 2048 | 149.2 | 7,982.8 | 153.8 | 1.551 | 0.891 |
| `attn` | 512 | 160.5 | 12,969.5 | 148.9 | 0.991 | 48.000 |
| `attn` | 1024 | 147.6 | 9,111.7 | 126.3 | 2.123 | 96.000 |
| `attn` | 2048 | 86.6 | 2,041.1 | 45.3 | 1.984 | 192.000 |

Observations and crossover point:
- **Recurrent state scaling**: The SSM state remains constant at 0.891 MB across all context lengths. The transformer key-value cache expands linearly from 48.0 MB at length 512 to 192.0 MB at length 2048 (215 times larger than the SSM state).
- **Throughput crossover**: At context length 512, the transformer achieves competitive decode throughput (148.9 tok/s vs 142.5 tok/s). By context length 1024, transformer decode throughput degrades 15% (126.3 tok/s vs 153.0 tok/s). By context length 2048, transformer decode throughput drops to 45.3 tok/s (a 70% decrease, making Mamba-2 3.4 times faster), while parallel prefill throughput drops to 2,041.1 tok/s (making Mamba-2 3.9 times faster).

#### Mamba-2 hybrid MoE/dense vs transformer (`config/code-small-dense.yaml`, ~232M)

Measured via `scripts/bench_context.py --config config/code-small-dense.yaml --arms ssm,attn --lengths 512,1024,2048 --decode-tokens 32 --prefill-mode parallel`:

| Arm | Context Length | Parallel Prefill (tok/s) | Decode (tok/s) | Peak Memory (GB) | Recurrent State (MB) |
|---|---|---|---|---|---|
| `ssm` (hybrid) | 512 | 3,348.9 | 74.0 | 1.496 | 24.674 |
| `ssm` (hybrid) | 1024 | 3,442.7 | 71.1 | 2.198 | 45.674 |
| `ssm` (hybrid) | 2048 | 2,962.9 | 65.5 | 3.330 | 87.674 |
| `attn` (transformer) | 512 | 4,002.8 | 57.4 | 1.524 | 168.000 |
| `attn` (transformer) | 1024 | 2,784.2 | 43.0 | 3.104 | 336.000 |
| `attn` (transformer) | 2048 | 1,677.1 | 35.8 | 3.395 | 672.000 |

Observations and crossover point:
- **Recurrent state scaling**: The hybrid architecture restricts attention to 12.5% of layers (`attn_every: 8`). At length 2048, hybrid state occupies 87.67 MB, compared to 672.00 MB for the full transformer baseline (a 7.67-fold reduction).
- **Throughput crossover**: Sustained decode throughput on the hybrid model stays above 65 tok/s across 2048 tokens. The transformer baseline falls from 57.4 tok/s at length 512 to 35.8 tok/s at length 2048 (making the hybrid model 1.83 times faster). In parallel prefill, the hybrid model maintains 2,962.9 tok/s at length 2048 versus 1,677.1 tok/s for the transformer (a 1.77-fold throughput lead).

## Scale MoE pretraining runs (M12: #222 small MoE, #223 Large A)

Training throughput targets and cloud execution modeling for the scale MoE milestones.

| Run | Architecture | Scale (Total / Active) | Target Hardware | Projected Throughput | Provenance |
|---|---|---|---|---|---|
| MHM-P4 (#222) | `config/code-small-moe.yaml` | 685M / 345M | 1x NVIDIA A40 (48GB) | ~300,000 tok/s | **RunPod cloud profile** (`scripts/run_small_moe.py --cloud-spec`), single-card baseline |
| MHM-P5 (#223) | `config/code-large-a.yaml` | 3.88B / 710M | 4x NVIDIA A100-SXM4 (80GB) | ~900,000 tok/s | **RunPod cloud profile** (`scripts/run_large_moe.py --cloud-spec`), FSDP2 + EP=4 cluster |
| MHM-P5 (#223) | `config/code-large-a.yaml` | 3.88B / 710M | 4x NVIDIA H100-SXM5 (80GB) | ~2,000,000 tok/s | **RunPod cloud profile** (`scripts/run_large_moe.py --cloud-spec`), Hopper NVLink cluster |

## Which numbers came from where — summary

- **CI runner (GitHub-hosted `macos-latest`, `xcodebuild`)**: the Swift-engine prefill row
  (55.30x, run 31777284815) and, going forward, every `swift-engine` CI run's `monica-bench
  --mode all` / `--mode decode` / `--mode spec` JSON artifact (uploaded per-run, not yet promoted into this
  table — informational only, no timing threshold gates CI).
- **Developer Apple Silicon (MacBookPro18,3 / Apple M1 Pro / 32 GB / macOS 26.5.2)**: every
  Python/MLX row above. These are the only rows in this document that can honestly be called a
  "local-hardware win" per CLAUDE.md's POC success criterion.
- **Not yet measured**: every poc-scale Swift-engine row (prefill/decode/memory, fp and
  quantized), blocked on a developer machine with Xcode installed running `swift run
  monica-bench --config Benchmarks/configs/poc.config.json ...`.
- **RunPod cloud profiles**: Scale MoE pretraining rows (#222, #223) represent analytical throughput models calibrated against datacenter GPU specifications. They are not local measurements.

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
