# The M12 code model — Mamba-2 hybrid MoE (MHM) + structural signal (SSI)

[← Index](README.md)

This is the **live program** ([issue #198](https://github.com/travisgalloway/monica/issues/198),
the 2026-07-18 "MHM fold"). It supersedes the M10 distillation program (#65), whose design record
now lives under [`../reserve/`](../reserve/10-distillation.md). Where the earlier docs describe
distilling a ~1B student from a frozen Qwen teacher, that path is **reserve**; the active plan is a
from-scratch code model, described below.

## What this is

A from-scratch, **TypeScript-first Mamba-2 hybrid Mixture-of-Experts (MoE) code model**. The
backbone is mostly Mamba-2/SSD state-space layers with a **minority (~12.5%) of full-attention
layers** for the cross-file symbol recall that pure SSMs are weak at, and **MoE on the MLP layers**
(Jamba-style: fine-grained experts, top-k routing, one shared expert, aux-loss-free load
balancing). It targets two sizes — a **small** rung (~120M active / ~700M total) and a **large**
rung ("Large A", ~700M active / ~3.5B total, the default), the large one **sparse-upcycled** from
the small dense checkpoint. It trains on a general multilingual **Essential-Web + Stack-v2**
mixture with its **own byte-level BPE** and **fill-in-the-middle (FIM)**. Success stays the POC
bar: a smoothly improving curve plus a local-hardware win (context length + tok/s), with **BPB**
elevated to the primary small-model metric — not leaderboard scores.

## The MHM spine (the program's phases)

Namespaced **MHM-P#** to avoid colliding with backlog priority tiers (P0/P1/P2):

- **MHM-P0 — Decisions.** From-scratch, own BPE, RunPod (CUDA) + M1 (MLX) dev. Carried decisions:
  D4 Jamba-vs-Routing-Mamba, D5 Large A vs Large B, vocab size.
- **MHM-P1 — Corpus** (#193): general multilingual Essential-Web + Stack-v2 mixture, repo-context
  packing, decontamination blocklist. (Rescopes the earlier FineWeb-Edu + Stack-v1 corpus.)
- **MHM-P1b — Tokenizer** (#191): own byte-level BPE trained on the final mixture. Blocks
  everything downstream.
- **MHM-P2 — Architecture & harness build** (the large net-new engineering): aux-loss-free
  balancing router (#213, *land first*) → CUDA MoE backend (#214: dropless routing, shared expert,
  FSDP, sparse-upcycle init) → FIM collator (#215), length curriculum + dataloader-state resume
  (#216), routing instrumentation (#217), pure-PyTorch Mamba-2 reference for laptop parity (#218).
- **MHM-P2e — Evals** (build first): code eval suite (#221), BPB (#192, primary).
- **MHM-P3 — Small-model ablation sweep** (#219, ~$80–120 each): attention ratio 8/12/16%,
  d_state 128 vs 256, Jamba vs Routing-Mamba.
- **MHM-P4 — Small-model full run** (#222, 50–70B tokens → the dense checkpoint to upcycle).
- **MHM-P5 — Large-model run** (#223): sparse-upcycle from the dense ckpt, Large A default,
  ~150B tokens; gated on P4 + D5.
- **Cross-cutting:** rented-pod ops runbook (#224). **Parked:** post-training SFT (#101) / RLVR
  (#103).

Today the MoE is **MLX-only** (the MLX router is a toy dense softmax-top-k; the CUDA backend
explicitly rejects MoE), so scaling on RunPod requires the #213/#214 build — this is the bulk of
the net-new work.

## The SSI fold (structural signal integration — secondary)

"Does feeding language-server / static-analysis signal into the model help?" — retained as a
**secondary measurement-and-training-signal** axis riding on the MoE model, under a formal
measurement contract:

- **SSI-M — measurement contract** (#225): one variable per arm, ≥3 seeds + paired
  Wilcoxon/McNemar, repo-level contamination split, availability-vs-use null arms, and a shared
  **escape-hatch lint gate** (extends `SUPPRESSION_RE` in `src/lsp/diagnostics.py` with
  `as unknown as`, `@ts-nocheck`, non-null `!`, empty bodies, `throw …not implemented`, `declare`
  stubs, deletion-of-target).
- **Surviving arms:** completion-list logit masking / constrained decode (#226), diagnostic
  supervision — rejection-sampled FT + contrastive hard negatives (#227), and **RLVR/GRPO with an
  LSP/opengrep verifier reward** (#230).
- **Dropped arms:** two-clock "slow-clock structural state" (conflicts with the MoE spine) and the
  diffusion path.

### Why SSI is secondary — the recorded assessment

The LSP-in-the-loop experiment (design record + measurement in
[`12-lsp-in-the-loop.md`](12-lsp-in-the-loop.md)) reached a partly-negative but useful conclusion:

- **Validated clean-rate tool.** Diagnostic-guided rollback/regeneration is a reliable
  *type-cleanliness* improver — the persistent-LSP swap moved clean-rate **0.887 → 0.962
  (p=0.0005)**, robust and well-instrumented, with the over-repair failure mode understood and
  mostly neutralized (#212 final-segment gate, forward-resolvable TS2xxx deferral, #211 confirmed
  the oracle isn't dropping diagnostics).
- **But not the lever for the functional gap.** The project's target gap is clean-but-wrong —
  bodies are **88.7% type-clean but only 50.3% functionally correct**. Persistent LSP leaves
  **pass@1 flat (0.503, p=0.69, ns)**: the failures are *algorithmic*, which a type/lint checker
  structurally cannot see. opengrep catches a genuine but far-too-sparse corner (4/159, 4/4
  precise), and over-repair on multi-statement code is **trajectory-bound, not trigger-bound** —
  structural to incremental repair, not tunable away.

So: inference-time type/lint-guided rollback on a **frozen** model is a validated clean-rate tool,
**not** the lever for functional correctness. The correctness-bearing signal lives in
semantics/execution — which is why the MHM fold makes model quality the primary axis and holds the
structural signal as a secondary program.

### The open fork (stated, not resolved)

With M10 off the plan, the #198 gate faces a real choice, in evidence-to-cost order:

1. **Resolve the tsc-vs-LSP pass@1 divergence** — batch `tsc` moved pass@1 **0.491 → 0.560
   (p=0.001)** where open-document LSP did not; the doc attributes this to whole-program vs
   open-document diagnostics. Cheapest, highest-information; could flip the conclusion. Do first.
2. **Put the signal in training** — the oracle as a reward (#230 / the parked #103) and/or the
   fast/slow/both ablation on a *trained* model instead of the frozen base coder. Tests whether a
   signal that barely moves a frozen model at inference can shape one toward correctness.
3. **Swap in a semantic/execution oracle** — execution against tests/spec as the `DiagnoseFn`,
   which naturally pairs with (2) as a train-time reward (expensive, test-gated).
4. **Exploratory** — the #203 diffusion discriminator (diagnostic-guided denoising); #211 cleared
   its prerequisite.
5. **The honest gate call** — "validated clean-rate tool, functional ceiling found, functional
   signal needs semantics-as-training" — and shelve, a legitimate well-earned outcome.

## See also

- [`12-lsp-in-the-loop.md`](12-lsp-in-the-loop.md) — the LSP harness design record + its
  assessment/conclusion (the source of the numbers above).
- [`09-hybrid-architectures.md`](09-hybrid-architectures.md) — why the backbone is a Mamba-2
  hybrid and how attention placement sizes it.
- [`../reserve/10-distillation.md`](../reserve/10-distillation.md) — the superseded M10
  distillation design record (reserve).
