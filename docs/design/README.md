# Design docs

Why the Mamba-2 Hybrid POC is built the way it is. These files explain the **design
choices and their rationale** — the *what* and *why*. The POC core (M1–M7), the **CUDA
backend (M8, A40-verified)**, and the **post-training machinery (M9, SFT/DPO/GRPO)** are
implemented and verified; the M1–M8 milestones were tracked in
[issue #2](https://github.com/travisgalloway/monica/issues/2). The **active program is M12 — the
from-scratch Mamba-2 hybrid MoE code model** ([issue #198](https://github.com/travisgalloway/monica/issues/198));
see topic 13 below. (The earlier M10 distillation program, #65, was dropped 2026-07-19 — reserve
under [`../reserve/`](../reserve/10-distillation.md).) For the
project overview, see the root [`README.md`](../../README.md); for end-to-end commands
(install → data → train → serve/chat → eval) see [`../usage.md`](../usage.md), for the
cloud (R2 + RunPod) runbook see [`../infrastructure.md`](../infrastructure.md), and for the
local Apple-Silicon dev loop (one-command validation, small training configs) see
[`../local-development.md`](../local-development.md).

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
   stages, the common schema, R2 storage layout, RunPod topology. The reusable foundation for
   the M12 corpus (Essential-Web + Stack-v2, #193).
9. [Hybrid architectures](09-hybrid-architectures.md) — why the model is a Mamba-2
   hybrid (config-gated attention) and how it sizes.
10. **[Distillation → moved to reserve](../reserve/10-distillation.md)** — the M10 distillation
    design record (distil a ~1B hybrid from a frozen `Qwen/Qwen3-4B-Thinking-2507` teacher).
    **Dropped 2026-07-19**; kept under [`../reserve/`](../reserve/10-distillation.md) as history.
11. [Post-training](11-post-training.md) — instruct SFT → reasoning-trace SFT → optional
    tool-use → GRPO, the chat-template invariant. Design record; the M12 arms are #101/#103.
12. [LSP-in-the-loop (measurement + assessment)](12-lsp-in-the-loop.md) — the `tsc`-in-the-loop
    repair harness (#199), its measurement, and the **arc-level assessment**: a validated
    clean-rate tool (0.887→0.962) that leaves functional pass@1 flat (0.503) — the recorded
    rationale for demoting the LSP signal to secondary.
13. **[The M12 code model — Mamba-2 hybrid MoE + SSI](13-code-model-moe.md)** — the **live
    program** (#198): the MHM spine (own BPE → Essential-Web + Stack-v2 → aux-loss-free MoE →
    CUDA MoE backend → FIM/curriculum/evals → ablation sweep → sparse-upcycled large run) and the
    secondary SSI structural-signal fold (#225/#226/#227/#230).

> **The live program is topic 13** (M12, [#198](https://github.com/travisgalloway/monica/issues/198)):
> a from-scratch Mamba-2 hybrid **MoE code model** with the structural-signal (SSI) fold as a
> secondary axis (topic 12 is its measurement record + assessment). Topics 8/9/11 are reusable
> **foundation** (corpus pipeline, hybrid-architecture sizing, post-training design). Topic 10 (the
> M10 distillation design record, epic #65) was **dropped 2026-07-19** and moved to
> [`../reserve/`](../reserve/10-distillation.md); its machinery still exists but the program is not
> active. The from-scratch pretraining in 08 remains a production reserve (#75).

## Locked decisions at a glance

| Decision | Choice | Why | Source |
|---|---|---|---|
| Hardware isolation | one seam (`ModelInterface`) | clean MLX→CUDA migration | `src/model/interface.py` |
| Token storage | uint16/uint32 packing (per vocab, #90) | uint16 when vocab < 65536, uint32 at/above it (e.g. the M12 code BPE, or the reserve Qwen3 vocab 151,669) | `src/data/pack.py` |
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
| OLMES / lm-eval | implemented (Tier-2); scores near chance at POC scale | loglikelihood + generative tasks run end-to-end | `src/eval/olmes_adapter.py` |
| Build method (M12) | **from-scratch** Mamba-2 hybrid **MoE** code model (not distillation) | model quality is the primary axis; MoE = capability per active param | `docs/design/13-code-model-moe.md` |
| Tokenizer (M12) | **own byte-level BPE** on the final mixture (#191) | code-first vocab, no external-teacher alignment constraint; uint32 packing (#90) | `docs/design/13-code-model-moe.md` |
| Model sizes (M12) | small ~120M-act/700M-tot; large "Large A" ~700M-act/3.5B-tot | large is **sparse-upcycled** from the small dense ckpt; ablation sweep picks the layout (#219) | `docs/design/13-code-model-moe.md` |
| Structural signal (M12, secondary) | LSP/opengrep as measurement + training signal (SSI) | validated clean-rate tool, functional ceiling found; #225/#226/#227/#230 | `docs/design/13-code-model-moe.md` |
| Data framework | `datatrove` + R2 + RunPod | builds the M12 corpus (Essential-Web + Stack-v2, #193) + reserve data | `docs/design/08-corpus-pipeline.md` |
| Build method (reserve) | **distillation** from a frozen teacher — Qwen3 vocab, `Qwen/Qwen3-4B-Thinking-2507`, ~1B student | M10 program, **dropped 2026-07-19**; machinery built, kept as history | `docs/reserve/10-distillation.md` |
