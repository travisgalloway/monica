# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A proof-of-concept **Mamba-2 hybrid** (selective state-space + a few attention layers)
language model, developed and validated on **Apple Silicon with MLX**, architected behind
**one hardware seam** so it migrates to **CUDA** for a larger run with minimal rewrite. The
**active program is M12** — a from-scratch, **TypeScript-first Mamba-2 hybrid Mixture-of-Experts
(MoE) code model** (the "MHM" spine): a mostly Mamba-2/SSD backbone with ~12.5% attention layers
for cross-file recall and Jamba-style MoE on the MLPs, trained on a general multilingual
Essential-Web + Stack-v2 corpus with its own byte-level BPE (a native cross-platform Swift
tokenizer — `swift/`, #191/#245; not Python/MLX) and FIM, at two sizes (small
~120M-active/700M-total; large "Large A" ~700M-active/3.5B-total, sparse-upcycled from the small
dense checkpoint). A **secondary axis (SSI)** studies feeding language-server / static-analysis
signal into the model — a *validated clean-rate tool with a found functional ceiling*, not the
lever for functional correctness (see `docs/design/13-code-model-moe.md`). Tracked in
[issue #198](https://github.com/travisgalloway/monica/issues/198); design record in
`docs/design/13-code-model-moe.md`. **Reserve/history:** the M10 distillation program (distil a
~1B student from a frozen `Qwen/Qwen3-4B-Thinking-2507` teacher, #65) was dropped 2026-07-19 — its
code machinery was **removed from the tree** (#189, recoverable via git history); its design record
is kept under `docs/reserve/`. The original from-scratch
pretrain path (OLMo tokenizer) is complete and is a validated foundation / production reserve.
POC success is a smoothly improving curve plus a local-hardware win (context length + tok/s), with
**BPB** the primary small-model metric — not benchmark scores.

## Commands

```bash
# Install (Apple Silicon — the normal dev environment):
pip install -e ".[dev,data,mlx]"

# Install (Linux/CUDA host — e.g. RunPod):
pip install -e ".[dev,data,cuda]"          # base CUDA backend (pure-PyTorch)
pip install -e ".[dev,data,cuda-fast]"     # + mamba-ssm Triton scan + causal-conv1d (#40)

# Tests (uses the venv at .venv):
.venv/bin/python -m pytest                                   # full suite
.venv/bin/python -m pytest tests/test_mlx_parity.py          # one file
.venv/bin/python -m pytest tests/test_mlx_parity.py::test_forward_step_parity_toy  # one test
.venv/bin/python -m pytest -q -rs                            # quiet, report skips

# The M4 smoke gate — the most important check (resume exactness + eval):
.venv/bin/python scripts/smoke_test.py --data data/split

# Data pipeline offline smoke (no network/tokenizer; uses byte fallback):
python -m src.data.download --dummy --out data/raw --max-docs 2000
python -m src.data.tokenize --in data/raw/dummy.txt --out data/ids.npy --byte-fallback
python -m src.data.pack  --in data/ids.npy --out data/packed.bin
python -m src.data.split --packed data/packed.bin --out data/split --val-tokens 2000
```

There is no separate lint/format/build step — pytest is the gate, and it now runs in CI
(`.github/workflows/ci.yml`, #249): a Linux `portable` job (`pytest -q -rs`, no mlx/torch —
where `test_import_guard.py` is unambiguous), a Linux `smoke-linux` job (CPU-torch,
`scripts/smoke_test.py --backend cuda`), a Linux `cuda-cpu` job (CPU-torch, the CUDA test
suite), and **two** macOS jobs, both on `.[dev,data,mlx,cuda]`: `full-macos` runs the suite
(`pytest -q -rs`) and `parity-macos` runs the cross-backend parity step plus
`scripts/smoke_test.py --backend mlx`. `mlx` is not installable on Linux; on a non-Mac host
the MLX backend simply won't import (by design), and only the portable tests run.
`parity-macos` is the **cross-backend parity gate** (#303, split onto its own job by #315):
it is the **only** job carrying **both** backends, because the macOS-arm64 PyPI `torch` wheel
is CPU-only, which is exactly the surface `src/model/cuda_backend.py` is compared on. Its
dedicated step runs `tests/test_backend_parity.py` with `MONICA_REQUIRE_BOTH_BACKENDS=1` —
under that flag the five MLX↔torch comparisons carry no skip marker, so a missing backend
**errors** instead of skipping (they silently skipped in every job before #303).
`tests/test_ci_backend_matrix.py` is the portable contract over that wiring and runs on
`portable`, where neither backend exists. Note `cuda-cpu` also runs that file but still
skips 5 of 6 — it is not parity coverage. **Why two jobs (#315):** #303's both-backend
install also un-skipped a large torch-gated set and took the combined job from 9m16s to
37m48s against `timeout-minutes: 45`, while the gates it exists for cost 14 seconds. A
timeout kill on the parity job would read as flake, not as a parity regression — CONF-2's
original failure shape again. So the suite job now measures itself against
`MACOS_SUITE_BUDGET_SECONDS` and fails with a legible `::error::`, and
`tests/test_ci_macos_budget.py` (portable) pins the budget, every macOS job's
`timeout-minutes`, and the margin between them. **#322 then cut the suite itself and made
the ceiling mechanical:** #315's `--durations=25` profile showed
`tests/test_cuda_distributed.py` was 1610.94s of a 2086.73s macOS suite (77.2%; 19.06s
locally) because macOS has no `fork` start method for `torch.multiprocessing.spawn`, so
each of its ~26 gloo workers re-imports torch/mlx. That file already runs, with zero skips,
in the 149s Linux `cuda-cpu` job, so `full-macos`'s pytest step now carries
`--ignore=tests/test_backend_parity.py --ignore=tests/test_cuda_distributed.py` (different
reasons — the first would look like coverage while five comparisons skip; the second is a
pure duplicate) and the suite lands at 441s measured (8m30s job wall clock). Budget 2400s→**720s**, `full-macos`
`timeout-minutes` 45→**17**, `swift-engine` 60→**20**. The budget test also pins
`PR_PATH_CEILING_SECONDS = 1200` — #315's stated 20-minute PR-path ceiling — against the
budget and every macOS job's timeout, plus the macOS job *set*, so the stated and enforced
ceilings can no longer disagree (they differed 2× under #315), and
`test_suite_ignores_never_delete_coverage` re-derives each ignore's cover from `cuda-cpu`'s
parsed glob so a "CI speedup" cannot delete a gate. `full-macos` keeps the `cuda` extra:
four torch-gated files (`test_upcycle.py`, `test_parallel.py`, `test_bench_config_export.py`,
`test_bfcl_adapter.py`) fall outside `cuda-cpu`'s glob and have no other coverage. Three more jobs (#246) gate the `swift/` native tokenizer, outside this Python
suite entirely: `swift-macos` and `swift-linux` (the official `swift:latest` container) each
build the package and run `monica-selfcheck` — `swift test` is still a no-op, there is no
`.testTarget` — then train a fixed fixture corpus (`swift/Fixtures/parity-corpus.jsonl`) into
a `tokenizer.json` artifact; `swift-parity` downloads both artifacts and `cmp`s them, turning
#191/#245's bit-identical-output claim into an enforced gate rather than a comment. Two more
jobs (#267), `poc-fixture-oracle` and `poc-parity`, gate the Swift/MLX model port at
**poc scale** (`config/poc.yaml`, d_model 768/24 layers/vocab 50280) rather than only the
checked-in toy fixtures `swift-engine` uses — the first job generates the 571 MB fixture on a
macOS runner and uploads a ~2 KB sha256 manifest, the second regenerates it independently on a
second runner and `cmp`s the two manifests (the #298 cross-process corruption guard) before
running `monica-parity` against it.

Those two poc jobs are **not** in `ci.yml`. Since #302 the repo has **two** workflow files, and
the split is the guarantee: `.github/workflows/ci.yml` holds **9** jobs — the eight named above
plus `swift-engine` (see the seam section) — and triggers on
`pull_request`, `push` to `main`, `workflow_dispatch`, and (since #312) a **monthly `schedule:`**
(09:43 UTC on the 3rd) that runs those same 9 jobs against unchanged `main` as environmental-drift
coverage. No job in it carries an `if:`, so no PR gate can silently stop running.
`.github/workflows/scheduled-parity.yml` holds **4** heavy jobs — `poc-fixture-oracle`/`poc-parity`
(#267) and `train-fixture-oracle`/`train-fixture-oracle-verify` (#195) — and triggers **only** on
`workflow_dispatch` and a weekly `schedule:` (Monday 09:17 UTC). It declares **no
`pull_request`/`push` trigger at all**, so those jobs are structurally unreachable from a PR rather
than merely `if:`-guarded — **a green PR run never implies poc scale was exercised**; see
`swift/engine/Fixtures/README.md` §poc. Note what carries that guarantee: the **file split**, not
the absence of a cron in `ci.yml` — `ci.yml`'s monthly cron fires only `ci.yml`'s own 9 jobs, and
the two crons are independent. A manual full run is therefore two commands
(`gh workflow run ci.yml` and `gh workflow run scheduled-parity.yml`), neither taking inputs. The
whole trigger×job matrix is asserted by `tests/test_workflow_triggers.py`, which runs inside the
`portable` job on every PR — adding a `pull_request:` to `scheduled-parity.yml`, giving a `ci.yml`
job an `if:`, or retiming either cron fails that test.

## The seam — the most important architectural rule

All hardware-specific code lives behind `src/model/interface.py`
(`ModelInterface`). Everything above the seam — `src/data/`, `src/train/`,
`src/serve/`, `src/eval/`, `src/conformance/`, `src/lsp/` — is **portable Python that
must never import `mlx` or `torch`/CUDA**. Exactly seven modules may touch a hardware
library: `src/model/mlx_backend.py`, `src/model/mlx_train_step.py`,
`src/model/cuda_backend.py`, `src/model/cuda_train_step.py`, `src/model/cuda_muon.py`
(#237's Muon/AdamW hybrid), `src/model/cuda_distributed.py` (#271's FSDP2 + expert-
parallel process-group/mesh/collective plumbing for the CUDA backend), and the
LSP-harness's model adapter `src/model/mlx_lm_adapter.py`. Note `src/model/backend.py` is
**not** one of them — it is the portable backend-factory registry and keeps its backend
imports inside the factory closures. `src/train/parallel.py` (#271's dp/ep sizing +
expert-partition policy math) is likewise portable — it computes WHO owns which expert,
never touches a process group, and lives in `tests/test_import_guard.py`'s
`PORTABLE_MODULES` alongside `moe_balance.py`, the precedent it mirrors.

This is enforced by `tests/test_import_guard.py`, which imports every portable module
and asserts no backend leaked into `sys.modules`. **When adding code above the seam,
do not import a backend — and add new portable modules to that test's
`PORTABLE_MODULES` list.** Keep MLX-only imports local (inside functions), as
`scripts/smoke_test.py` does, when a portable-ish entry point needs the backend.
`src/train/upcycle.py` (#214's sparse-upcycle transform — dense checkpoint to MoE init)
is a recent example: numpy-only, no backend import, lives above the seam in
`PORTABLE_MODULES` despite reading/writing the same safetensors weights the two
below-seam backends produce.

**A third category — the `swift/` native toolchain (the repo's first Swift package).** The
code tokenizer (#191, PR #245) is a native, cross-platform Swift package (`swift/Sources/MonicaTokenizer`
+ the `monica-tokenize` CLI) that trains/encodes/packs entirely outside the Python seam — it is
neither above-seam Python nor a hardware backend behind `ModelInterface`, and the import guard
does not cover it. It builds and runs on macOS **and** Linux/CUDA with bit-identical output (Swift
stdlib only in the BPE core; **no MLX** — BPE is CPU/integer work). It emits the same
`src/data/shard.py` shard layout, so Python training reads its shards unchanged. This is the M13
"native, no-Python-runtime" direction (#163/#167) landing first for the tokenizer.

**`swift/engine/` is a SEPARATE, Apple-only SwiftPM package** (#166) — the Swift/MLX port of the
model (`MonicaEngine` + the `monica-parity` runner), depending on `ml-explore/mlx-swift`. It is
deliberately *not* a target in `swift/Package.swift`: platform conditions gate build edges, not
*resolution*, so adding mlx-swift there would make `swift-linux` clone its vendored C++ tree and
would raise the tokenizer's tools-version/platform floor. Keeping them siblings preserves
`swift/`'s zero-dependency, cross-platform property — the thing that makes the #246 bit-identity
gate credible. `cd swift && swift build` ignores `engine/` the way it already ignores `Fixtures/`.
The gate is `cd swift/engine && swift run monica-parity`: the Swift `forward` and stacked-`step`
logits vs checked-in Python/MLX references (`swift/engine/Fixtures/`, regenerated by
`scripts/export_parity_fixture.py`) at fp32 `rtol=1e-4 / atol=1e-5`. CI job `swift-engine`
(macOS only). `monica-bench` (#170, same package, no tokenizer dependency) is the benchmark
harness (prefill/decode/memory, `--baseline` regression flagging); results ledger and
CI-runner-vs-local-Apple-Silicon provenance live in `docs/benchmarks.md`.

Consequences of the seam that shape how code is written:
- The training loop (`src/train/loop.py`) is backend-free and receives the
  backprop/optimizer primitive as an injected `train_step` callable
  (`TrainStepFn = (model, micro_batches, lr) -> {loss, grad_norm, ...}`, where
  `micro_batches` is a list of `(inputs, targets)` of length `grad_accum`). The MLX
  implementation is `make_train_step(...)` in `src/model/mlx_train_step.py`.
- The data loader yields **numpy**; the backend converts to its own array type inside
  `forward`. Eval (`src/eval/val_loss.py`) takes a `to_numpy` converter at the seam.
- Model `State` is opaque (`Any`) above the seam. In the MLX backend it is a per-layer
  list of `(conv_state, ssm_state)` tuples.

## Configuration is the single source of truth

Model dims and run params live in `config/toy.yaml` and `config/poc.yaml`, loaded into
`MambaConfig` (`src/model/blocks.py`). `MambaConfig.validate()` enforces cross-cutting
invariants; **token packing is dtype-aware (#90)** — `vocab < 65536` packs as **uint16**
(the original POC: OLMo-7B-hf), at/above it packs as **uint32** (the reserve distillation
student: Qwen3, vocab 151,669 — see `config/student-1b.yaml` and
`docs/reserve/10-distillation.md`).
The ceiling `validate()` enforces is now uint32 (`2**32`). The YAML **comments are the
decision record** — read them before changing values. Key locked decisions:

- **toy.yaml** (smoke/correctness): tiny, `fp32` for bit-exact fixed-seed resume,
  `vocab_size 256` (byte-fallback tokenizer, offline).
- **poc.yaml** (~127M OLMo scale run, from-scratch/reserve): `vocab_size 50280` (OLMo-7B-hf,
  confirmed `<65536`, uint16), `precision fp16` + (dynamic) loss scaling (~16% faster than bf16
  on Metal per the M1 micro-benchmark — **do not assume bf16**), tied embedding **mandatory**
  (~38M of ~127M params), `grad_checkpoint: true` (required at this depth — see below).
- **poc-qwen.yaml** (the **completed ~205M POC run**, now reserve — val-ppl 75.7): `poc.yaml`'s
  layers retargeted to `vocab_size 151646` (Qwen2.5, uint32) so it trained on the reserve corpus
  (`s3://monica-training/reserve-pretrain`) and mirrored the (reserve) ~1B student's data path.
  Layers are unchanged (~88M); the larger tied embedding (~116M) dominates → **~205M total**,
  embedding-heavy (a deliberate trade for tokenizer alignment, not a clean 100M). Runs on CUDA
  via RunPod (see `config/poc-qwen.yaml`'s header runbook); split the R2 shard corpus into
  train/val with `python -m src.data.split --shards <dir> --out <split> --val-tokens N`.
- **`head_dim`** is the Mamba-2 head width: `d_inner` splits into
  `n_heads = d_inner // head_dim` heads, each with a **scalar** decay A (the SSD
  restriction that makes the scan a matmul). `validate()` requires `head_dim | d_inner`
  (poc `head_dim 64` → 24 heads; toy `head_dim 16` → 8 heads).
- **dt-bias init** (`dt_min`/`dt_max`/`dt_init_floor`) is **load-bearing** — without
  the inverse-softplus init in `SelectiveSSM._init_dt_bias` the model fails to learn
  recall. Now **per-head** (shape `n_heads`). These params are identical across both
  configs by design.

## The SSM: Mamba-2 / SSD (scalar A)

The SSM is **Mamba-2 / SSD** (Dao & Gu, *State Space Duality*): scalar A **per head**,
multi-head with one shared B/C group — migrated from the original diagonal-A Mamba-1
for training throughput/memory (see `docs/design/02-model-ssm.md`). Two separate
implementations must produce identical logits: `forward` (the SSD **chunked-matmul**
scan, training) and `step` (the matching one-step recurrence, inference). The scan
**always chunks** (length Q = `chunk_size`, default **64**) but, unlike the old
diagonal-A cumsum scan, is **overflow-safe by construction** — every decay is `exp` of
a non-positive sum (in `[0,1]`). Conformance (`src/conformance/`) guards train/infer
equivalence: `forward_step_parity` and `backend_parity` (MLX vs CUDA, deferred) both
compare in **fp32 at ~1e-4 rel** — bf16's epsilon is too coarse to be meaningful.

## Training: the scale-run driver and its memory lever

`scripts/train.py` is the real run driver (config → model → data → loop, with resume).
It wires **gradient accumulation** (the loop pulls `grad_accum` micro-batches per step),
**dynamic fp16 loss scaling** (`src/train/loss_scale.py`, a portable policy; the backend
does the inf/nan check and skips overflowing steps), and **gradient checkpointing**
(`grad_checkpoint` config — recompute each layer in backward instead of retaining its
activations). Checkpointing is mandatory at poc depth: without it the 24-layer backward
exceeds the 32 GB unified memory and swaps. Mamba-2/SSD + checkpointing brought the poc
step down from the swapping diagonal-A regime to **~99 s/step** at the standard protocol
(batch 32 × grad_accum 4 × seq 1024 = 131,072 tokens/step, fp16, peak ~24.8 GB of 32 GB
on an M1 Pro) — the measured baseline from `scripts/bench_train_step.py` (issue #31,
posted to #30). Note: an earlier "~3 s/step" figure here was never validated at full
shape; treat ~99 s/step as the real per-step cost (so a 3B-token run is ~26 days of
compute) when planning runs or judging the #30 optimization spike.

## Checkpointing: two deliberately separate concerns

`src/train/checkpoint.py` splits (1) **portable weights** (safetensors + config
sidecar — the cross-backend bridge) from (2) a **within-backend resume bundle** (step
+ RNG + optimizer state, via a backend-supplied serializer). They are not conflated:
weights port across backends; optimizer state does not need to (CUDA trains fresh).
The smoke gate stresses exactly this round-trip.

## Workflow

- The POC core **M1–M8 is done** (tracked in **GitHub issue #2**, now closed): seam + MLX
  model, data pipeline, training loop + smoke gate, the `scripts/train.py` driver, OLMES eval,
  serving/rewind, and the **CUDA backend (M8, A40-verified)**. **M9 post-training is done** —
  SFT/DPO/GRPO machinery on MLX with CUDA step-factory parity. The full 2–5B-token from-scratch
  run is still pending (user-driven).
- The **active program is M12 — the from-scratch Mamba-2 hybrid MoE code model** (**GitHub issue
  #198**, the live tracker; design record `docs/design/13-code-model-moe.md`): the "MHM" spine
  (own BPE #191 **— done: native Swift, PR #245** → Essential-Web + Stack-v2 corpus #193 →
  aux-loss-free MoE router #213 **— done** → CUDA MoE backend #214 **— done: dropless
  grouped-gather routing, shared expert, `src/train/upcycle.py` sparse-upcycle init; fp8
  expert GEMMs (#240) and 8-bit AdamW moments wired but hardware-unverified; FSDP/ZeRO-2 +
  expert parallel split out to **#271**, which blocks #223, not #222** →
  FIM/curriculum/eval build → ablation sweep #219 → small full run #222 → sparse-upcycled
  large run #223), with **SSI**
  (structural-signal integration) as a secondary
  measurement/training-signal axis (completion-list logit masking #226, diagnostic supervision
  #227, RLVR/opengrep verifier reward #230, under the #225 measurement contract + escape-hatch
  gate). The MoE build (#213/#214) is done on both backends, and so are FSDP/ZeRO-2 +
  expert parallel (#271) and #223's `d_model` conflict (#272, resolved to `d_model 768` for
  both rungs). The remaining net-new work ahead of
  the #222/#223 runs is the **corpus (#252)**, plus **#288** — `grad_checkpoint` does not compose
  with FSDP2, which blocks #223 but not #222 (see `docs/design/13-code-model-moe.md`).
- **Reserve/history — M10 distillation (#65) was dropped 2026-07-19; its code was removed from the
  tree (#189).** The teacher/student/distill modules, `scripts/distill.py`/`sweep.py`/
  `precompute_teacher.py`, `distill_manifest`, and the corpus tooling are gone (recoverable via git
  history). Its design record + corpus/decontamination guidance + pod runbooks remain under
  `docs/reserve/` (e.g. `docs/reserve/10-distillation.md`). Do **not** describe distillation as
  active work, and do not assume the machinery is present in the tree.
- `docs/design/` documents the design choices and rationale (start at
  `docs/design/README.md`); `docs/infrastructure.md` is the R2 + RunPod runbook. After completing
  a milestone, tick its box in the relevant tracker (#2 / #198).
- After finishing a milestone or backend change, run the smoke gate, not just pytest.

## Licensing / usage-policy compliance

**Standing rule — flag before crossing the actual boundary:** if any future task would have
Claude *generate* text (synthetic examples, code samples, explanations) that gets fed into the
pretrain/SFT/RL corpus as a training signal, **stop and flag it for review** rather than
proceeding — that is the specific thing Anthropic's Usage Policy restricts. This project keeps
its training corpus entirely third-party/non-Claude, and that must hold for M12 too (including any
SSI reward / SFT corpus, e.g. #230's RLVR data). Anthropic's policy restricts training a model on
**Claude's own** inputs/outputs; it does not restrict using Claude as a coding assistant to build
the pipeline. Re-check the Usage Policy if it changes; this is not a legal ruling and is only as
current as the last check.

**Reserve (M10 distillation, dropped 2026-07-19):** the earlier assessment cleared distilling from
**`Qwen/Qwen3-4B-Thinking-2507`, plain unmodified Apache-2.0** (no distillation/competing-model
restriction — confirmed against the live LICENSE 2026-07-05) using third-party corpora, not
Claude-generated content. Retained under `docs/reserve/` for the record; re-check both the teacher
license and the Usage Policy if that path is ever revived.
