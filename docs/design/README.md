# Design docs

Why the Mamba POC is built the way it is. These files explain the **design
choices and their rationale** — the *what* and *why*. M1–M4 are implemented and
verified, and M5's infrastructure (training driver + Mamba-2/SSD perf migration) has
landed; the milestone tracking and remaining work (the full M5 run, M6–M8) live in
[GitHub issue #2](https://github.com/travisgalloway/monica/issues/2). For the
project overview, see the root [`README.md`](../../README.md); for end-to-end commands
(install → data → train → serve/chat → eval) see [`../usage.md`](../usage.md).

Every claim here is sourced from a docstring or config comment in the code, with a
`src/...` or `config/...` path so you can jump to the source of truth.

## Topics

1. [Architecture: the hardware seam](01-architecture-seam.md) — one abstraction
   (`ModelInterface`) isolates MLX/CUDA so everything above it stays portable.
2. [Model: the Mamba block + selective SSM](02-model-ssm.md) — block dataflow,
   Mamba-2/SSD scalar-A (per-head) init, load-bearing per-head dt-bias, the SSD
   chunked-matmul scan, recurrence, gradient checkpointing.
3. [Conformance: fp32 parity](03-conformance.md) — forward-vs-step and
   backend-vs-backend equivalence checks, and why they run in fp32.
4. [Data pipeline](04-data-pipeline.md) — uint16 packing, the OLMo tokenizer,
   disjoint val split, the mmap loader.
5. [Training](05-training.md) — backend-free loop, the `scripts/train.py` driver,
   LR schedule, gradient accumulation, grad clipping, dynamic fp16 loss scaling,
   gradient checkpointing, and the two-concern checkpoint split.
6. [Smoke gate & eval](06-smoke-gate-and-eval.md) — why resume-exactness is *the*
   gate, and val perplexity as the success metric.
7. [Configs & locked decisions](07-configs-and-decisions.md) — `toy.yaml` /
   `poc.yaml` in full, plus the precision benchmark and sizing math.
8. [Corpus pipeline](08-corpus-pipeline.md) — the clean-license data flow: `datatrove`
   stages, the common schema, R2 storage layout, RunPod topology. Now the **teacher
   corpus + production-reserve** path (its uint16/StarCoder2 from-scratch framing is
   superseded by the distillation pivot — see 10).
9. [Hybrid architectures](09-hybrid-architectures.md) — why the model is a Mamba-2
   hybrid (config-gated attention) and how it sizes.
10. [Distillation (teacher → hybrid student)](10-distillation.md) — the **distillation-first
    pivot**: distil a compact (~1–1.5B) hybrid from a frozen, fully-open `open-r1/OpenR1-Distill-7B`
    teacher (Qwen2.5 tokenizer → uint32 packing, #90), precompute teacher artifacts once,
    sweep student layouts cheaply (#98).
11. [Post-training](11-post-training.md) — instruct SFT → reasoning-trace SFT → optional
    tool-use → GRPO, the Qwen `<|im_end|>` chat-template invariant, shared with production.

> Topics 8–11 are the **scale-up / distillation** design record (epic
> [#65](https://github.com/travisgalloway/monica/issues/65)) — forward-looking decisions,
> partly implemented (the corpus stages, hybrid attention, post-training machinery, and the
> #90 uint32 packing exist; the distillation pipeline #92–#104 is pending), unlike the fully
> verified POC in topics 1–7. The current plan is **distillation-first** (10/11); the
> from-scratch pretraining in 08 is deferred to a production reserve (#75).

## Locked decisions at a glance

| Decision | Choice | Why | Source |
|---|---|---|---|
| Hardware isolation | one seam (`ModelInterface`) | clean MLX→CUDA migration | `src/model/interface.py` |
| Token storage | uint16/uint32 packing (per vocab, #90) | uint16 when vocab < 65536 (POC), uint32 for Qwen2.5 (151,646) | `src/data/pack.py` |
| Tokenizer (POC) | `allenai/OLMo-7B-hf` (vocab 50280) | fits uint16; matches AI2 for comparison | `src/data/tokenize.py` |
| Embedding | tied (input = output) | ~38M of ~100M budget at POC scale | `config/poc.yaml` |
| dt-bias init | inverse-softplus, log-uniform (per head) | **load-bearing** — model can't learn recall without it | `src/model/mlx_backend.py` |
| Selective SSM | Mamba-2 / SSD, scalar A per head | matmul scan; ~62× faster than diagonal-A at poc scale | `src/model/mlx_backend.py` |
| Selective scan | SSD chunked matmul (default 64) | scalar-A matmul form; overflow-safe by construction | `src/model/mlx_backend.py` |
| Memory at depth | gradient checkpointing | recompute layers in backward; fits the 24-layer poc backward in 32 GB | `config/poc.yaml` |
| Precision (poc) | fp16 + loss scaling | ~18% faster than bf16 on Metal (M1 benchmark) | `config/poc.yaml` |
| Precision (toy/smoke) | fp32 | exact, reproducible resume | `config/toy.yaml` |
| Conformance | compare in fp32, ~1e-4 rel | bf16 epsilon (~8e-3) too large to be meaningful | `src/conformance/` |
| Checkpoints | portable weights + separate resume bundle | weights port across backends; optimizer state doesn't need to | `src/train/checkpoint.py` |
| Success metric | held-out val perplexity (Tier-1) | a smoothly decreasing curve *is* the POC goal | `src/eval/val_loss.py` |
| OLMES / lm-eval | deferred (Tier-2) | its own milestone-sized task; not needed for the POC | `src/eval/olmes_adapter.py` |
| Build method | **distillation** from a frozen teacher (not pretrain) | reaches capability at <1% of from-scratch tokens; cheap layout sweep | `docs/design/10-distillation.md` |
| Scale-up tokenizer | **Qwen2.5** (vocab 151,646) | fixed by the conversion teacher for logit/hidden matching; uint32 packing (#90). POC stays OLMo. StarCoder2 (the old uint16 pick) superseded. | `docs/design/10-distillation.md` |
| Conversion teacher | `open-r1/OpenR1-Distill-7B` (Apache-2.0) | fully open (open R1 traces + recipe), reasoning-ready, Qwen2.5 tokenizer; 7B→~1B size gap bridged by adaptive init | `docs/design/10-distillation.md` |
| Scale-up model | compact ~1–1.5B Mamba-2 hybrid reasoning student | attention layers close the SSM retrieval gap; no-KV-cache local inference | `docs/design/10-distillation.md` |
| Data framework | `datatrove` + R2 + RunPod | builds the teacher corpus + production-reserve from-scratch data (#75) | `docs/design/08-corpus-pipeline.md` |
