# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A proof-of-concept **Mamba-2 hybrid** (selective state-space + a few attention layers)
language model, developed and validated on **Apple Silicon with MLX**, architected behind
**one hardware seam** so it migrates to **CUDA** for a larger run with minimal rewrite. The
active program is to **distil** a compact **~1B** hybrid student from a larger frozen teacher
(`Qwen/Qwen3-4B-Thinking-2507`, Qwen3 tokenizer ~151,669; thinking-mode CoT is the headline lever),
sweep a few architecture layouts cheaply, then post-train the winner for reasoning — tracked in
[issue #65](https://github.com/travisgalloway/monica/issues/65). The original from-scratch
pretrain path (OLMo tokenizer) is complete and is the validated foundation / production reserve.
POC success is a smoothly decreasing held-out validation-perplexity curve plus a local-hardware
win (context length + tok/s) — not benchmark scores.

## Commands

```bash
# Install (Apple Silicon — the normal dev environment):
pip install -e ".[dev,data,mlx]"

# Install (Linux/CUDA host — e.g. RunPod):
pip install -e ".[dev,data,cuda]"          # base CUDA backend (pure-PyTorch)
pip install -e ".[dev,data,cuda-fast]"     # + mamba-ssm Triton scan + causal-conv1d (#40)

# Tests (uses the venv at .venv):
.venv/bin/python -m pytest                                   # full suite (36 tests)
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

There is no separate lint/format/build step — pytest is the gate. `mlx` is not
installable on Linux; on a non-Mac host the MLX backend simply won't import (by
design), and only the portable tests run.

## The seam — the most important architectural rule

All hardware-specific code lives behind `src/model/interface.py`
(`ModelInterface`). Everything above the seam — `src/data/`, `src/train/`,
`src/serve/`, `src/eval/`, `src/conformance/` — is **portable Python that must never
import `mlx` or `torch`/CUDA**. Only `src/model/mlx_backend.py`,
`src/model/mlx_train_step.py`, and `src/model/cuda_backend.py` may touch a hardware
library.

This is enforced by `tests/test_import_guard.py`, which imports every portable module
and asserts no backend leaked into `sys.modules`. **When adding code above the seam,
do not import a backend — and add new portable modules to that test's
`PORTABLE_MODULES` list.** Keep MLX-only imports local (inside functions), as
`scripts/smoke_test.py` does, when a portable-ish entry point needs the backend.

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
(the original POC: OLMo-7B-hf), at/above it packs as **uint32** (the distillation student:
Qwen3, vocab 151,669 — see `config/student-1b.yaml` and `docs/design/10-distillation.md`).
The ceiling `validate()` enforces is now uint32 (`2**32`). The YAML **comments are the
decision record** — read them before changing values. Key locked decisions:

- **toy.yaml** (smoke/correctness): tiny, `fp32` for bit-exact fixed-seed resume,
  `vocab_size 256` (byte-fallback tokenizer, offline).
- **poc.yaml** (~127M OLMo scale run, from-scratch/reserve): `vocab_size 50280` (OLMo-7B-hf,
  confirmed `<65536`, uint16), `precision fp16` + (dynamic) loss scaling (~16% faster than bf16
  on Metal per the M1 micro-benchmark — **do not assume bf16**), tied embedding **mandatory**
  (~38M of ~127M params), `grad_checkpoint: true` (required at this depth — see below).
- **poc-qwen.yaml** (the **active ~205M POC run**): `poc.yaml`'s layers retargeted to
  `vocab_size 151646` (Qwen2.5, uint32) so it trains on the distillation corpus
  (`s3://monica-training/reserve-pretrain`) and mirrors the ~1B student's tokenizer/data path.
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
- The **active program is M10 — distillation** (**GitHub issue #65**, the live tracker): distil a
  compact Mamba-2 hybrid student from a frozen teacher, sweep layouts, post-train the winner. The
  core machinery is implemented — teacher loader (#93), student init (#99), staged distill loss
  (#100), sweep harness (#98), and the distillation driver `scripts/distill.py` (#81). The
  **remaining work is pod-gated runs**, not builds: the 8k corpus build, the teacher top-k
  precompute (#94), the two-layout sweep (#81), and post-training/eval — per the #65 ordered chain.
- `docs/design/` documents the design choices and rationale (start at
  `docs/design/README.md`); `docs/infrastructure.md` is the R2 + RunPod runbook. After completing
  a milestone, tick its box in the relevant tracker (#2 / #65).
- After finishing a milestone or backend change, run the smoke gate, not just pytest.
- **The corpus-extension domains that chain multiple named sources under one pooled
  `char_budget_cap`** (`conversation`, `reasoning`, `code_problems`, `code_instruct` in
  `src/data/distill_sources.py`) **only exercise later sources in the list once the earlier
  ones exhaust their real data** — a large source (e.g. `ultrachat`, `mot`, `opencodereasoning`,
  `opencodeinstruct`) can single-handedly consume the whole pooled budget and starve its
  siblings entirely (confirmed live, 2026-07-05: the first real corpus-extension build silently
  produced zero docs from `oasst1`/`openthoughts2`/`rosetta-code`/`mceval`/`kodcode`/
  `codefeedback`). To guarantee a specific source contributes, run it **alone** in its own
  `distill_corpus_from_jsonl.py` invocation with its own token budget, then merge (renumber
  shards to continue the existing manifest's sequence, merge `manifest.json`'s `shards` list +
  aggregate counts, merge `provenance.json` — see the merge done for `poc-distill-ext`,
  2026-07-05).
- **Before trusting a new corpus-extension domain, decontamination-check it against the real
  eval sets** (MATH-500, AIME25, LiveCodeBench, IFEval, or whichever this project's actual eval
  axes are), not just against the license/field-shape dry run — a source named after or derived
  from a benchmark (e.g. McEval-Instruct/McEval) is a contamination risk even when its license is
  fine. Checked 2026-07-05 for the `poc-distill-ext` corpus: zero overlap for LiveCodeBench/
  AIME25/IFEval; **5/500 MATH-500 problems (1%) matched `openwebmath`** (verbatim AoPS-forum
  content that any large web crawl picks up — a known, low-severity, expected issue for this
  class of source, not specific to a bad choice here). Accepted as-is; re-check if `openwebmath`
  or the eval suite changes.

## Licensing / usage-policy compliance (M10 distillation)

M10 distils from **`Qwen/Qwen3-4B-Thinking-2507`, licensed plain unmodified Apache-2.0** (no
distillation or competing-model restriction — confirmed against the live LICENSE file
2026-07-05) using third-party corpora (the-stack, open-web-math, Wikipedia, OpenCodeReasoning,
etc.), not Claude-generated content. Anthropic's Usage Policy's "model scraping/distillation"
clause restricts using **Claude's own** inputs/outputs to train another model without
authorization — it does not restrict using Claude as a coding assistant to build an ML pipeline
that distills a *different* (non-Anthropic) model, confirmed against the live Usage Policy the
same date. On that basis M10 does not appear to cross either line.

**Standing rule — flag before crossing the actual boundary:** if any future task would have
Claude *generate* text (synthetic examples, code samples, explanations) that gets fed into the
pretrain/SFT/RL corpus as a training signal for the student model, **stop and flag it for
review** rather than proceeding — that is the specific thing Anthropic's policy restricts, and
this project has otherwise been careful to keep the corpus entirely third-party/non-Claude.
Re-check both the teacher model's license and Anthropic's Usage Policy if either changes, or if
the teacher model itself ever changes — this assessment is not a legal ruling and is only as
current as the sources checked on the date above.
