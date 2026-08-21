# Feature matrix

The single view of what exists and how far along it is. One table per capability area,
matching `docs/design/`. IDs are **stable** — use them in issue titles, test names and
`docs/test-plan.md`, which mirrors this file row for row.

**The per-layer columns are this repo's seam, not the generic UI/API/Data split.**
`docs/design/01-architecture-seam.md` is the architectural rule everything else obeys, so the
layers that make a half-sliced capability visible here are: **Portable** (above the seam —
`src/data`, `src/train`, `src/serve`, `src/eval`, `src/conformance`, `src/lsp`), **MLX** and
**CUDA** (the two backends behind `ModelInterface`), and **Swift** (the native `swift/` and
`swift/engine/` packages, outside the Python seam entirely). A cell is `done`, `none`, or `n/a`.

Status is one of `Planned`, `In progress`, `Shipped`, `Deprecated` — nothing else, and nothing
meaning *partially done*; that state lives in the per-layer columns. A capability reaches
`Shipped` only when its full definition of done passes.

Last audited 2026-08-18 (`/closure-audit`, whole repo).

## Data pipeline

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| DATA-1 | A person can download and clean a raw corpus | done | n/a | n/a | n/a | Shipped | #193 | design/04 |
| DATA-2 | A person can tokenize text to ids | done | n/a | n/a | done | Shipped | #191 | design/04 |
| DATA-3 | A person can pack ids into shards a loader reads | done | n/a | n/a | done | Shipped | #191 | design/04 |
| DATA-4 | A person can split a packed corpus into train/val | done | n/a | n/a | n/a | Shipped | — | design/04 |
| DATA-5 | A person can sync a corpus to and from R2/S3 | done | n/a | n/a | n/a | Shipped | — | infrastructure |
| DATA-6 | A person can build a scale corpus with datatrove | done | n/a | n/a | n/a | Shipped | #193 | design/08 |
| DATA-7 | A person can build the TS LSP-clean Stack-v2 corpus | done | n/a | n/a | n/a | In progress | #252 | design/08 |
| DATA-8 | A person can build an SFT corpus (instruct/reasoning/tool) and train on it | done | n/a | n/a | n/a | Shipped | #306 | design/11 |
| DATA-9 | A person can build a DPO preference set | done | n/a | n/a | n/a | Shipped | — | design/11 |
| DATA-10 | A person can build a decontamination blocklist | done | n/a | n/a | n/a | Shipped | — | design/08 |
| DATA-11 | A person can sweep vocab size against a sample | done | n/a | n/a | n/a | Shipped | #251 | design/08 |
| DATA-12 | A person can train with FIM and a length curriculum | done | done | done | done | Shipped | #216 | design/13 |

## Native tokenizer (`swift/`)

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| TOK-1 | A person can train a byte-level BPE natively, no Python | n/a | n/a | n/a | done | Shipped | #191 | design/13 |
| TOK-2 | A person can encode and pack to the shard layout Python reads | n/a | n/a | n/a | done | Shipped | #191 | design/13 |
| TOK-3 | A person gets bit-identical tokenizer output on macOS and Linux | n/a | n/a | n/a | done | Shipped | #246 | design/13 |
| TOK-4 | A person can ingest Parquet in the pack path | n/a | n/a | n/a | done | Shipped | #247 | design/13 |
| TOK-5 | A person can pack FIM-transformed shards natively | n/a | n/a | n/a | done | Shipped | #215 | design/13 |

## Model and the seam

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| MODEL-1 | A person can build a Mamba-2 hybrid from a config | done | done | done | done | Shipped | — | design/01 |
| MODEL-2 | A person can run the SSD chunked-matmul forward | n/a | done | done | done | Shipped | — | design/02 |
| MODEL-3 | A person can run the matching one-step recurrence | n/a | done | done | done | Shipped | — | design/02 |
| MODEL-4 | A person can run the same model on CUDA | n/a | n/a | done | n/a | Shipped | — | design/01 |
| MODEL-5 | A person can train an aux-loss-free MoE router | done | done | done | done | Shipped | #213 | design/13 |
| MODEL-6 | A person can sparse-upcycle a dense checkpoint to MoE | done | n/a | n/a | n/a | Shipped | #214 | design/13 |
| MODEL-7 | A person can size a config and estimate its train time | done | done | n/a | n/a | Shipped | — | design/07 |
| MODEL-8 | A person can quantize a checkpoint | done | done | n/a | done | Shipped | #196 | design/14 |

## Training

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| TRAIN-1 | A person can run a real training run from a config | done | done | done | done | Shipped | — | design/05 |
| TRAIN-2 | A person can resume a run exactly from a checkpoint | done | done | done | done | In progress | — | design/05 |
| TRAIN-3 | A person can train in fp16 with dynamic loss scaling | done | done | done | n/a | Shipped | — | design/05 |
| TRAIN-4 | A person can stream shards from remote storage | done | n/a | n/a | n/a | Shipped | — | design/05 |
| TRAIN-5 | A person can use a WSD learning-rate schedule | done | done | done | n/a | Shipped | #238 | design/05 |
| TRAIN-6 | A person can train with the Muon/AdamW hybrid | n/a | n/a | done | n/a | Shipped | #237 | design/13 |
| TRAIN-7 | A person can shard a run across GPUs (FSDP2 + expert-parallel) | done | n/a | done | n/a | Shipped | #271 | design/13 |
| TRAIN-8 | A person can train with 8-bit moments and fp8 expert GEMMs | n/a | n/a | done | n/a | In progress | #240 | design/13 |
| TRAIN-9 | A person can read structured progress from a run | done | n/a | n/a | n/a | Shipped | — | design/05 |

## Post-training

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| POST-1 | A person can run supervised fine-tuning | done | done | done | done | Shipped | #195 | design/11 |
| POST-2 | A person can run DPO | done | done | done | n/a | Shipped | — | design/11 |
| POST-3 | A person can run GRPO/RLVR against a verifier reward | done | done | done | n/a | Shipped | #230 | design/11 |
| POST-4 | A person can generate on-policy preference pairs | done | done | n/a | n/a | Shipped | — | design/11 |

## Serving

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| SERVE-1 | A person can generate text from a checkpoint | done | done | done | done | Shipped | — | design/14 |
| SERVE-2 | A person can control sampling (temp/top-p/repetition) | done | done | done | done | Shipped | — | design/14 |
| SERVE-3 | A person can rewind a session to a turn boundary and branch | done | done | none | none | Shipped | #305 | design/14 |
| SERVE-4 | A person can constrain decoding to an allowed id set | done | done | done | done | Shipped | #226 | design/12 |
| SERVE-5 | A person can decode speculatively | done | done | n/a | none | In progress | #172 | design/14 |

## Evaluation

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| EVAL-1 | A person can measure val loss and BPB, overall and per domain | done | done | done | n/a | Shipped | — | design/06 |
| EVAL-2 | A person can run the OLMES benchmark suite | done | done | n/a | n/a | Shipped | — | design/06 |
| EVAL-3 | A person can probe long-context behaviour | done | done | n/a | n/a | Shipped | — | design/06 |
| EVAL-4 | A person can evaluate against the real external code suites | done | n/a | n/a | n/a | Shipped | #304 | design/13 |
| EVAL-5 | A person can evaluate FIM infilling | done | done | n/a | n/a | Shipped | — | design/13 |
| EVAL-6 | A person can measure code recall and cross-file needle retrieval | done | done | n/a | n/a | Shipped | #217 | design/13 |
| EVAL-7 | A person can run synthetic retrieval probes | done | done | n/a | n/a | Shipped | — | design/06 |
| EVAL-8 | A person can inspect MoE routing histograms | done | done | done | done | Shipped | #217 | design/13 |
| EVAL-9 | A person can evaluate tool-calling with BFCL | done | done | n/a | n/a | Shipped | — | design/11 |
| EVAL-10 | A person can check quantization parity | done | done | n/a | done | Shipped | #196 | design/14 |
| EVAL-11 | A person can hold SSI results to the measurement contract | done | n/a | n/a | n/a | Shipped | #225 | design/15 |
| EVAL-12 | A person can evaluate against the TS error-injection set | done | n/a | n/a | n/a | Shipped | #224 | design/15 |

## LSP / structural-signal integration

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| LSP-1 | A person can query a live tsserver oracle over generated code | done | n/a | n/a | done | Shipped | #220 | design/12 |
| LSP-2 | A person can run the diagnostics harness in the generation loop | done | done | n/a | done | Shipped | #197 | design/12 |
| LSP-3 | A person can mask logits to the LSP completion list | done | done | done | done | Shipped | #226 | design/12 |
| LSP-4 | A person can score generated code with an opengrep verifier | done | n/a | n/a | n/a | Shipped | #230 | design/15 |
| LSP-5 | A person can check generated code with the tsc oracle | done | n/a | n/a | n/a | Shipped | — | design/12 |
| LSP-6 | A person can normalize generated code with prettier | done | n/a | n/a | n/a | Shipped | — | design/12 |
| LSP-7 | A person can drive the harness with a chat/LM adapter | done | done | n/a | n/a | Shipped | — | design/12 |
| LSP-8 | A person can execute generated code and capture the result | done | n/a | n/a | n/a | Shipped | — | design/12 |
| LSP-9 | A person can run the LSP harness natively, no Python runtime | n/a | n/a | n/a | done | Shipped | #197 | design/13 |
| LSP-10 | A person can get TS diagnostics without the LSP client debounce | done | n/a | n/a | none | Shipped | #279 | design/12 |

## Swift engine (`swift/engine/`)

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| ENGINE-1 | A person gets Swift logits provably identical to Python/MLX | n/a | done | n/a | done | Shipped | #166 | design/14 |
| ENGINE-2 | A person can benchmark prefill/decode/memory natively | n/a | n/a | n/a | done | Shipped | #170 | benchmarks |
| ENGINE-3 | A person can run an MoE model in the Swift engine | n/a | n/a | n/a | done | Shipped | #265 | design/14 |
| ENGINE-4 | A person gets a stated fp16/bf16 parity contract in Swift | n/a | n/a | n/a | done | Shipped | #266 | design/14 |
| ENGINE-5 | A person can read mixing matrices and hidden states in Swift | n/a | n/a | n/a | done | Shipped | — | design/14 |
| ENGINE-6 | A person gets Swift/MLX parity gated at real poc scale | n/a | done | n/a | done | Shipped | #267 | design/14 |
| ENGINE-7 | A person can round-trip a checkpoint Swift to Python | done | done | n/a | done | Shipped | #196 | design/14 |
| ENGINE-8 | A person gets a fused Metal SSD-scan/conv kernel | n/a | n/a | n/a | none | Planned | #171 | design/14 |
| ENGINE-9 | A person can decode speculatively in the Swift engine | n/a | n/a | n/a | none | Planned | #172 | design/14 |
| ENGINE-10 | A person can trust a checked-in parity oracle was not silently corrupted | done | done | n/a | n/a | Shipped | #298 | design/14 |

## Conformance

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| CONF-1 | A person can prove forward and step agree | done | done | done | done | Shipped | — | design/03 |
| CONF-2 | A person can prove MLX and CUDA agree numerically | done | done | done | n/a | Shipped | #303, #315 | design/03 |
| CONF-3 | A person can prove document boundaries are respected | done | done | done | n/a | Shipped | — | design/03 |
| CONF-4 | A person can prove prefill and decode agree | done | done | done | done | Shipped | — | design/03 |
| CONF-5 | A person can prove a quantized model stays in tolerance | done | done | n/a | done | Shipped | #196 | design/03 |

## Operations

| ID | Capability | Portable | MLX | CUDA | Swift | Status | Issue | Design |
|----|-----------|----------|-----|------|-------|--------|-------|--------|
| OPS-1 | A person can run the M4 smoke gate (resume exactness + eval) | done | done | done | n/a | Shipped | — | design/06 |
| OPS-2 | A person gets every gate run automatically in CI | done | done | done | done | Shipped | #249, #302, #312 | design/06 |
| OPS-3 | A person can read a provenance-tagged benchmark ledger | done | n/a | n/a | done | Shipped | #170 | benchmarks |
