# The M12 code model — Mamba-2 hybrid MoE (MHM) + structural signal (SSI)

[← Index](README.md)

This is the **live program** ([issue #198](https://github.com/travisgalloway/monica/issues/198),
the 2026-07-18 "MHM fold"). It supersedes the M10 distillation program (#65), whose design record
now lives under [`../reserve/`](../reserve/10-distillation.md). Where the earlier docs describe
distilling a ~1B student from a frozen Qwen teacher, that path is **reserve**; the active plan is a
from-scratch code model, described below.

## What this is

> **Read this section as the target design, not as shipped state.** As of 2026-08-03 the MHM
> components that are **built** are the tokenizer (MHM-P1b, #191/#245) and aux-loss-free load
> balancing on MLX (MHM-P2-T0, #213). `MambaConfig` carries the MoE and hybrid-attention knobs
> (`moe_every`, `n_experts`, `top_k`, `attn_every`, `fp8_experts`, `moe_balance_rate`) and the MLX
> backend's `MoEBlock` now has a DeepSeek-V3 Loss-Free-Balancing router: a per-expert bias steers
> top-k **selection** only, updated outside the gradient from per-expert load counts, with no aux
> loss on the objective (portable policy in `src/train/moe_balance.py`; `moe_balance_rate: null`
> keeps the pre-#213 router byte-identical). It still has **no shared expert**, and it densely
> evaluates every expert (sparse *combination*, not sparse compute) — a capacity experiment, not a
> production kernel. The CUDA backend still raises `NotImplementedError` for MoE (#214), which will
> reuse `src/train/moe_balance.py` unchanged. The shared expert, FIM, the Essential-Web + Stack-v2
> mixture, and repo-context packing described below are **designed, not implemented**. Per-item
> status is in the MHM-P# list that follows; when the two disagree, the P# list wins.

A from-scratch, **TypeScript-first Mamba-2 hybrid Mixture-of-Experts (MoE) code model**. The
backbone is mostly Mamba-2/SSD state-space layers with a **minority (~12.5%) of full-attention
layers** for the cross-file symbol recall that pure SSMs are weak at, and **MoE on the MLP layers**
(Jamba-style: fine-grained experts, top-k routing, one shared expert, aux-loss-free load
balancing). It targets two sizes — a **small** rung (~120M active / ~700M total) and a **large**
rung ("Large A", ~700M active / ~3.5B total, the default), the large one **sparse-upcycled** from
the small dense checkpoint. It trains on a general multilingual **Essential-Web + Stack-v2**
mixture with its **own byte-level BPE** (a native cross-platform Swift tokenizer, #191/#245 — see
MHM-P1b) and **fill-in-the-middle (FIM)**. Success stays the POC
bar: a smoothly improving curve plus a local-hardware win (context length + tok/s), with **BPB**
elevated to the primary small-model metric — not leaderboard scores.

## The MHM spine (the program's phases)

Namespaced **MHM-P#** to avoid colliding with backlog priority tiers (P0/P1/P2):

- **MHM-P0 — Decisions.** From-scratch, own BPE (shipped as a **native cross-platform Swift**
  tokenizer, not Python/MLX — see MHM-P1b), RunPod (CUDA) + M1 (MLX) dev. Carried decisions:
  D4 Jamba-vs-Routing-Mamba, D5 Large A vs Large B.

  > **Vocab size — 49152, ratified 2026-08-04 (#251).** `DEFAULT_VOCAB_SIZE` was 16384, sized when
  > #193 was scoped as *"TS LSP-clean pipeline (Stack v2 **TS subset**)"* — reasonable for
  > TypeScript alone. #198 rescoped the corpus to general multilingual Essential-Web + Stack-v2 and
  > the vocab never moved. Measured on the rescoped distribution, 16384 costs **7.6% of overall
  > compression** and **11.4% on Markdown**; 49152 is the best of the three candidates swept and is
  > under the 65536 uint16 packing cap.
  >
  > **bytes/token** (higher is better — more raw bytes carried per token):
  >
  > | Language | sample bytes | 16384 | 32768 | 49152 | Δ 32768 | Δ 49152 |
  > |---|---:|---:|---:|---:|---:|---:|
  > | cpp | 1.05 MB | 2.9998 | 3.2195 | 3.3177 | +7.33% | +10.60% |
  > | en | 10.58 MB | 3.4368 | 3.7451 | 3.8893 | +8.97% | +13.17% |
  > | go | 1.05 MB | 3.0427 | 3.2193 | 3.3134 | +5.80% | +8.90% |
  > | java | 1.58 MB | 3.6480 | 3.9145 | 4.0703 | +7.31% | +11.58% |
  > | javascript | 2.10 MB | 3.3452 | 3.5518 | 3.6518 | +6.18% | +9.17% |
  > | markdown | 1.05 MB | 2.8511 | 3.1751 | 3.3509 | +11.37% | +17.53% |
  > | python | 2.12 MB | 3.5047 | 3.7173 | 3.8233 | +6.07% | +9.09% |
  > | rust | 1.06 MB | 3.5151 | 3.6811 | 3.7628 | +4.72% | +7.05% |
  > | typescript | 4.26 MB | 3.5703 | 3.7943 | 3.9201 | +6.28% | +9.80% |
  > | **overall** | 24.85 MB | 3.4029 | 3.6630 | 3.7920 | +7.64% | +11.43% |
  >
  > **Per-language rows are blend-independent** — a language's bytes/token is computed only over
  > its own documents. **Only the "overall" row is blend-weighted**, so the decision does not
  > silently depend on mixture weights nobody ratified.
  >
  > **Two limitations, so this is not read as better-founded than it is:**
  > 1. **The tested range did not bracket the knee.** bytes/token is monotonic non-decreasing in
  >    vocab size, and 32768→49152 was still yielding a lot (+3.5% overall, +5.25% Markdown).
  >    Returns never flattened inside 16k–49k, so the rule returned the **largest candidate
  >    tested** — which is not the same claim as "49152 is optimal". The honest statement is
  >    **≥49152 on this evidence; the ceiling was not located.** The uint16 packing cap (65536)
  >    bounds where it can be, leaving roughly **49k–64k unexplored**.
  > 2. **The rule prices the benefit, not the cost.** Since bytes/token improves monotonically,
  >    this metric alone has no interior optimum. The counterweight is the **tied-embedding
  >    parameter cost** — at `d_model 768` each token costs 768 params in the embedding *and* the
  >    output head, and 16384→49152 roughly triples that matrix (for scale: `config/poc.yaml`'s
  >    tied embedding is ~38M of ~127M params, and M12-small is ~120M-active). The sweep measured
  >    **compression only**; the parameter-cost side was **not** evaluated. Revisit if the small
  >    rung turns out embedding-heavy.
  >
  > **Sample.** 24.85 MB / 5,420 docs, built by `src/data/vocab_sample.py`: 10.58 MB Essential-Web
  > prose + 14.27 MB Stack-v2 code across 8 languages (TypeScript over-weighted at 4.26 MB — the
  > program is TypeScript-first). Filters, applied to Stack-v2 *metadata columns before* any S3
  > fetch: permissive license (`filters.is_permissive`; ~67% of rows carry an empty
  > `detected_licenses`), not vendored, not generated, 1 KiB ≤ `length_bytes` ≤ 128 KiB. Carries
  > **240,073 unique pre-tokens (122,111 appearing ≥2×)** — 4.9× / 2.5× the 49,152 merges
  > requested, so the tail merges come from real repeated structure, not the exhaustion regime that
  > caps an undersized sample. The run produced all 48,890 requested merges without exhausting.
  >
  > **Method.** One 49,152 train (2,035 s), with 32,768 / 16,384 derived by **truncating `merges`**
  > — `Trainer.train` is greedy with no lookahead, so a k-vocab merge list is a strict prefix of
  > any larger run's. That invariant is now guarded by `monica-selfcheck`, not assumed. `tokens`
  > counts the EOS `pack` appends, so bytes/token is the real packed ratio. Reproduce with
  > `scripts/vocab_sweep.py`.
  >
  > **Invariants that held:** FIM sentinel ids identical across all three sizes; pre-token counts
  > identical (5,884,475 — the pre-tokenizer never sees the merges); `max_token_id` 49,151 < 65,536,
  > and `monica-tokenize pack` confirmed the uint16 shard path.
  >
  > **BPB is unaffected by this choice.** BPB is byte-normalized, so a larger vocab does not
  > "cheat" it the way it deflates raw perplexity — the table measures genuine compression, and
  > BPB remains a fair primary metric across vocab sizes.
- **MHM-P1 — Corpus** (#193): general multilingual Essential-Web + Stack-v2 mixture, repo-context
  packing, decontamination blocklist. (Rescopes the earlier FineWeb-Edu + Stack-v1 corpus.)
- **MHM-P1b — Tokenizer** (#191, **done** — PR #245): own byte-level BPE, shipped as a **native
  Swift package** (`swift/MonicaTokenizer` + the `monica-tokenize` CLI) rather than a Python build.
  Its own **tiktoken-style JSON format**, **raw-byte** BPE (no GPT-2 `bytes_to_unicode` remap), and
  an **o200k-style pretokenizer** (digit runs split ≤3; whitespace/indentation runs grouped so BPE
  learns indentation merges). It **builds and runs on macOS *and* Linux/CUDA with bit-identical
  trained vocab and token ids** (Swift stdlib only in the BPE core; no regex engine). This makes it
  the M13 native engine's tokenizer directly (#163/#167) and the corpus tokenizer for training.
  **MLX is deliberately not used here** — BPE is branchy integer/hash work, not tensor math, so a
  GPU buys nothing and would threaten the cross-platform bit-exact guarantee; MLX's role stays the
  *model* (#163). The Python code-tokenizer path was retired with this switch
  (`src/data/tokenizer_train.py` deleted; `CodeTokenizer`/`load_code_tokenizer`/`--tokenizer code`
  removed). New corpus split: **Python cleans → Swift tokenizes+packs.** `monica-tokenize pack`
  emits the exact `src/data/shard.py` shard layout (uint16 `.bin` + `.bounds` + `manifest.json`),
  so the Python training loop reads Swift-produced shards unchanged. Since #247,
  `monica-tokenize` reads the corpus pipeline's Parquet shards **directly** (a minimal
  pure-Swift reader, `swift/Sources/MonicaTokenizer/Parquet/` — UNCOMPRESSED/SNAPPY only, no
  zstd), so a full corpus build no longer round-trips through `cleaned.jsonl`; shards must be
  written `compression="snappy"` (`src/data/corpus.py --compression snappy`) for the Swift
  packer to read them.
- **MHM-P2 — Architecture & harness build** (the large net-new engineering): aux-loss-free
  balancing router (#213, **done on MLX** — portable `MoEBalancer` policy above the seam +
  selection-only route bias in `MoEBlock`; the bias rides in the portable safetensors so a served
  or ported checkpoint routes as trained) → CUDA MoE backend (#214: dropless routing, shared expert,
  FSDP, sparse-upcycle init) → FIM (#215, **done** — see the resolved note below), length
  curriculum + dataloader-state resume
  (#216, **done** — `--curriculum "0.25:2048,0.5:4096,1.0:16384"` on `scripts/train.py`, plus
  explicit `MicroBatchStream` state committed inside the checkpoint slot; see
  [05-training.md](05-training.md#length-curriculum--dataloader-state-resume-216)),
  routing instrumentation (#217), pure-PyTorch Mamba-2 reference for laptop parity (#218).
  **Training-efficiency levers** (folded 2026-07-20 from the efficiency-survey review): hybrid
  Muon+AdamW optimizer at the `make_optimizer` seam (#237, **done** — `3b02e6b`), WSD
  warmup-stable-decay LR schedule (#238, **done** — `8fe62f7`), and `torch.compile` default-on for
  real CUDA runs (#239, **done** — `7a71073`) — all three landed ahead of the #219 sweep, so it
  and the #222/#223 runs carry them; plus fp8 MoE-expert linears
  (Transformer Engine / Hopper, #240 — **design-only**, `c57a8e6`: the probe landed but the code is
  never called), *gated on #214* and ahead of the #223 large run. See
  [05-training.md](05-training.md#efficiency-levers-m12-landed-2026-07-2021). (The repo
  is already mature on the survey's biggest axes — data dedup/filtering, fused AdamW, SDPA, the
  mamba-ssm kernels, grad-checkpoint — so these four are the net-new levers; #216 is the
  length-curriculum lever.)
> **Resolved 2026-08-06 (#215) — FIM insertion is in the Swift `pack` path, not a Python
> collator.** An earlier draft left this open ("*may* live in the Swift pack path"). It is settled:
> the transform is `swift/Sources/MonicaTokenizer/FIM.swift`, driven by `monica-tokenize pack
> --fim-rate <0..1> --fim-seed <n>`.
>
> *Why.* FIM needs **document boundaries**, and pack time is the only place they still exist as
> objects. `src/data/split.py` drops the `.bounds` sidecars, so the trainer reads a flat
> `train.bin` through `PackedLoader`, which cuts fixed `seq_len` windows and cannot see doc
> structure. Reconstructing boundaries in the loader to feed a collator is a far larger change
> than doing the transform where the documents are still whole.
>
> *Trade-offs, accepted.* Changing the FIM rate requires a **re-pack** of the corpus, not a config
> edit. The transform sits outside the pytest gate because it is Swift — `monica-selfcheck` is its
> test runner (the package declares no `.testTarget`), backed by `tests/test_swift_fim.py`, which
> shells out to the built binary.
>
> *Shape.* PSM only (SPM is out of scope): `[fim_prefix] prefix [fim_suffix] suffix [fim_middle]
> middle`. The **rate band is 0.4–0.5** — explicitly *not* the 0.5–0.9 reported elsewhere; a rate
> outside the band warns rather than fails so an ablation is not blocked. Cuts are on UTF-8 **byte**
> offsets of the document text, not on token ids: slicing ids would make every middle begin exactly
> on a token boundary, the classic silent FIM distribution bug (StarCoder2 shipped one).
>
> *Determinism is a CI gate.* Insertion is seeded per document with SplitMix64 (integer-only, no
> stdlib random conveniences, no floats, no hashed-collection iteration, no grapheme indexing), and
> `.github/workflows/ci.yml`'s `swift-parity` job now `cmp`s macOS-packed FIM shards against
> Linux-packed ones. `--fim-rate` defaults to 0, so pre-#215 pack output is byte-identical.
>
> *Architectural consequence — configuration guidance, not a validator rule.* In PSM the suffix is
> consumed **before** the middle is generated, so every middle token depends on recalling the suffix
> out of state — the fixed-width-state bottleneck the hybrid exists to patch. Therefore **the final
> block should be an attention block**: `n_layers % attn_every == 0`, from `layer i is attention iff
> (i+1) % attn_every == 0` (`src/model/blocks.py:192`). `config/toy-hybrid.yaml` satisfies it
> (4 layers, `attn_every 2` → `[Mamba, Attn, Mamba, Attn]`); `config/student-1b.yaml` (28 layers,
> `attn_every 8` → attention at 7/15/23, top block Mamba) does **not**, so a future MHM config must
> not copy that shape. This is deliberately *not* a `MambaConfig.validate()` rule — that validator
> is shared with reserve configs that legitimately violate it. `src/eval/fim_eval.py`'s
> `attention_after_suffix()` is the advisory check.

- **MHM-P2e — Evals** (build first): code eval suite (#221), BPB (#192, primary).
  Distance-bucketed FIM eval ships with #215 as `src/eval/fim_eval.py` (portable, pure numpy,
  teacher-forced loss over the middle span with per-instance records); **#215 owns that module and
  #221 extends it** rather than writing a second one.

> **What #221 shipped.** Instrumentation, not numbers — there is no trained MHM checkpoint yet
> (#200/#222 are downstream), so every acceptance check is a fixture/determinism check. All of it
> is above the seam (numpy + stdlib, in `tests/test_import_guard.py`'s `PORTABLE_MODULES`) and all
> of it is **teacher-forced — no pass@1 gating anywhere**, because at the small rung a generative
> gate is noise (the LSP-in-the-loop assessment: functional pass@1 flat at 0.503 while clean-rate
> moved 0.887 → 0.962).
>
> | Module | Probe |
> |---|---|
> | `src/eval/code_suite.py` | shared per-instance record schema, canonical JSONL writer, bucketed aggregator, the batched causal span scorer, `StubCausalModel` |
> | `src/eval/code_recall.py` | cross-file TS symbol resolution by token distance — CE **plus a discriminative rank** against near-miss exports (the actual recall signal) |
> | `src/eval/code_needle.py` | RULER-over-code on a `context_len × depth` grid, `single` + `multikey` |
> | `src/eval/fim_eval.py` | `evaluate_fim_multi_key` — prefix-length **and** recall-distance keyings from one forward pass |
> | `src/eval/domain_bpb.py` | held-out BPB per domain, byte-weighted overall |
> | `src/eval/external_sets.py` | pinned loaders + normalizing adapters for the seven named suites |
> | `scripts/eval_code_suite.py` | the driver: shared-schema JSONL transcript + results JSON, `--stub-model` for offline runs |
> | `scripts/build_domain_val_sets.py`, `scripts/build_decontam_blocklist.py` | the two artifacts the above consume |
>
> Type-aware completion (`tsc`) was **not** rebuilt — `TscRunner`/`CompositeOracle` already
> implement it, and `--suites tsc` only surfaces their verdicts in the shared schema.
>
> **Two gaps stated rather than papered over.**
>
> 1. **The seven external revisions are unpinned.** An HF commit SHA cannot be resolved without
>    network access, and inventing one is worse than not having one — a wrong pin silently loads a
>    different revision or errors far from its cause. So the table ships `revision=None` with a
>    `# TODO(pin):` per entry, a live pull **raises** while unpinned, and every `None` is echoed
>    into the driver's results JSON. Only the MultiPL-E repo identifiers were confirmable offline
>    (`repo_verified`); the rest must be checked at pin time. Fixtures are synthetic,
>    schema-shaped rows — **no third-party benchmark data is checked in**.
> 2. **Packed shards carry no domain/language field**, so per-domain BPB comes from purpose-built
>    per-domain val sets (`scripts/build_domain_val_sets.py`, reading the cleaned Parquet where
>    `source`/`lang`/`meta` still exist) rather than from the training corpus. That is a different
>    measurement from "BPB per domain over the actual training mix", and the module docstring says
>    so. **Follow-up:** a `.domains` uint8 sidecar written next to `.bounds` by both
>    `src/data/shard.py` and `swift/Sources/MonicaTokenizer/Packing.swift` would let the number be
>    read off the training corpus directly — a data-pipeline change requiring a re-pack, out of
>    scope for an eval issue. `src/data/split.py` writing `n_bytes` (it currently does not, so
>    `val_bpb` is silently omitted on the shard path) belongs with it.
- **MHM-P3 — Small-model ablation sweep** (#219, ~$80–120 each): attention ratio 8/12/16%,
  d_state 128 vs 256, Jamba vs Routing-Mamba.
- **MHM-P4 — Small-model full run** (#222, 50–70B tokens): the small **MoE** rung
  (~120M-active/700M-total), sparse-upcycled from #200's dense checkpoint.
- **MHM-P5 — Large-model run** (#223): sparse-upcycle **from #200's dense checkpoint** (not from
  #222), Large A default, ~150B tokens; gated on P4 + D5.

> **Resolved 2026-07-25 — where the dense checkpoint comes from.** An earlier draft had #222
> producing "the dense checkpoint to upcycle" while also describing the small rung as
> ~120M-active/700M-total. Those cannot both hold: sparse upcycling initializes MoE experts as
> copies of a **dense** FFN, so an already-MoE checkpoint is not a valid source. The roles are now:
>
> | Issue | Produces | Role |
> |---|---|---|
> | **#200** | the **dense** small checkpoint, at the small rung's backbone dims | the single upcycle source for both MoE runs |
> | **#222** | small **MoE** (~120M-active/700M-total), upcycled from #200 | validates the whole #213/#214 MoE build cheaply, before the expensive run |
> | **#223** | **Large A** (~700M-active/3.5B-total), upcycled from **#200** | the headline model |
>
> This costs one extra dense run but buys the thing that matters: the MoE machinery is proven at
> ~$100s rather than first being exercised on the ~$1.2k large run. It also means **#200 must be
> re-dimensioned** off `poc-small.yaml`'s 97M shape to match the small rung's non-expert backbone
> — otherwise the upcycle won't line up.
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
