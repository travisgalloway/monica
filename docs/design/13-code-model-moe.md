# The M12 code model — Mamba-2 hybrid MoE (MHM) + structural signal (SSI)

[← Index](README.md)

This is the **live program** ([issue #198](https://github.com/travisgalloway/monica/issues/198),
the 2026-07-18 "MHM fold"). It supersedes the M10 distillation program (#65), whose design record
now lives under [`../reserve/`](../reserve/10-distillation.md). Where the earlier docs describe
distilling a ~1B student from a frozen Qwen teacher, that path is **reserve**; the active plan is a
from-scratch code model, described below.

## What this is

> **Read this section as the target design, not as shipped state.** As of 2026-08-08 the MHM
> components that are **built** are the tokenizer (MHM-P1b, #191/#245), the ratified vocab
> (#251, 49152), aux-loss-free load balancing (MHM-P2-T0, #213, portable — shared by both
> backends), the **CUDA MoE backend** (MHM-P2-T1, #214), FIM training (#215), the length
> curriculum + dataloader-state resume (#216), the code eval suite (#221), the pure-PyTorch
> Mamba-2 reference (#218), and the three training-efficiency levers — hybrid Muon+AdamW (#237),
> the WSD schedule (#238), and `torch.compile` default-on (#239). `MambaConfig` carries the MoE
> and hybrid-attention knobs (`moe_every`,
> `n_experts`, `top_k`, `n_shared_experts`, `moe_impl`, `attn_every`, `fp8_experts`,
> `optimizer_8bit`, `moe_balance_rate`). Both the MLX and CUDA `MoEBlock`s now have a shared
> expert (DeepSeek-V2/V3 form, additive, outside the router softmax/top-k renormalization) and
> the Loss-Free-Balancing router: a per-expert bias steers top-k **selection** only, updated
> outside the gradient from per-expert load counts, with no aux loss on the objective (portable
> policy in `src/train/moe_balance.py`; `moe_balance_rate: null` keeps the pre-#213 router
> byte-identical). MLX still densely evaluates every expert (sparse *combination*, not sparse
> compute — a capacity experiment, not a production kernel); the CUDA backend adds a real
> **grouped-gather dispatch** (`moe_impl: gather`, dispatch-only-to-chosen-experts, with a
> deterministic backward so it stays inside the bit-exact-fp32-resume smoke gate), alongside the
> dense path as its always-kept oracle. fp8 expert GEMMs (Transformer Engine, Hopper+) and an
> 8-bit AdamW-moments switch (bitsandbytes) are wired but **hardware-unverified** — see
> docs/infrastructure.md's manual-verification checklist. `src/train/upcycle.py` +
> `scripts/upcycle.py` (#214) turn a dense (degenerate `n_experts: 1`) checkpoint into a sparse-MoE
> init that matches the dense forward at step 0. What is **designed, not implemented** is the
> Essential-Web + Stack-v2 mixture and repo-context packing described below — the pipeline is
> #193 (closed) but the corpus itself is #252, and nothing has been built from it yet. (FIM used
> to be listed here as unimplemented; it shipped with #215.) Per-item status is in the MHM-P#
> list that follows; when the two disagree, the P# list wins.

A from-scratch, **TypeScript-first Mamba-2 hybrid Mixture-of-Experts (MoE) code model**. The
backbone is mostly Mamba-2/SSD state-space layers with a **minority (~12.5%) of full-attention
layers** for the cross-file symbol recall that pure SSMs are weak at, and **MoE on the MLP layers**
(Jamba-style: fine-grained experts, top-k routing, one shared expert, aux-loss-free load
balancing). It targets two sizes — a **small** rung (~120M active / ~700M total) and a **large**
rung ("Large A", ~700M active / ~3.5B total, the default), the large one **sparse-upcycled** from
the small dense checkpoint. **Both rungs share `d_model 768`** (resolved 2026-08-09, #272 — see
the resolved note under MHM-P5), which is what lets a single dense run (#200) seed both. It trains on a general multilingual **Essential-Web + Stack-v2**
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
  Swift package** (`swift/Sources/MonicaTokenizer` + the `monica-tokenize` CLI) rather than a Python build.
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
  or ported checkpoint routes as trained) → CUDA MoE backend (#214, **done** — dropless
  grouped-gather routing, shared expert, `src/train/upcycle.py` sparse-upcycle init, 8-bit AdamW
  moments and fp8 expert GEMMs wired but hardware-unverified; **FSDP/ZeRO-2 + expert parallel
  split out to #271, blocking #223 but not #222**) → FIM (#215, **done** — see the
  resolved note below), length curriculum + dataloader-state resume
  (#216, **done** — `--curriculum "0.25:2048,0.5:4096,1.0:16384"` on `scripts/train.py`, plus
  explicit `MicroBatchStream` state committed inside the checkpoint slot; see
  [05-training.md](05-training.md#length-curriculum--dataloader-state-resume-216)),
  routing instrumentation (#217), pure-PyTorch Mamba-2 reference for laptop parity (#218).
  **Training-efficiency levers** (folded 2026-07-20 from the efficiency-survey review): hybrid
  Muon+AdamW optimizer at the `make_optimizer` seam (#237, **done** — `3b02e6b`), WSD
  warmup-stable-decay LR schedule (#238, **done** — `8fe62f7`), and `torch.compile` default-on for
  real CUDA runs (#239, **done** — `7a71073`) — all three landed ahead of the #219 sweep, so it
  and the #222/#223 runs carry them; plus fp8 MoE-expert linears
  (Transformer Engine / Hopper, #240 — **wired, hardware-unverified**: `_expert_linear` builds
  `te.Linear` on Hopper+, `MoEBlock._fp8_ctx` wraps the expert-compute region in `fp8_autocast`,
  and `_layer_forward` selects `transformer_engine.pytorch.checkpoint` for MoE layers — landed with
  #214 since it shares `_Expert`/`MoEBlock`), *ahead of the #223 large run*. See
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
> block should be an attention block**: `n_layers % attn_every == 0`, from
> `MambaConfig.is_attention_layer` (`src/model/blocks.py`) — `(i + 1) % attn_every == 0`.
> `config/toy-hybrid.yaml` satisfies it
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
> **One gap stated rather than papered over, and one closed (#304).**
>
> 1. **The seven external suites are pinned and all seven load (#304).** Every entry carries a real
>    40-hex HF commit SHA, resolved against the live hub on 2026-08-20 unauthenticated; none of the
>    seven is gated, so no credential is needed and none is committed. All seven were **pulled
>    live** and normalized end-to-end (row counts and SHAs in `eval_sets/external/README.md`).
>    Filling the pins was the smaller half of the job: **five of the seven adapters were wrong the
>    first time they met real data** — SAFIM has no `suffix` column (the hole is `{{completion}}`
>    inside `eval_prompt`), McEval has no `language` column, RepoBench v1.1's `context` is a *list
>    of structs* rather than a string, and two `hf_repo` ids were 401s. `_norm` now type-checks the
>    normalized row, which is what stops a struct-valued column passing a presence-only check and
>    emitting a non-string prompt. Fixtures were re-authored to the real schemas and stay synthetic
>    — **no third-party benchmark data is checked in**. Four identifiers changed and each records
>    why; two measurement caveats are recorded rather than smoothed over: **CrossCodeEval has no
>    official HF dataset**, so a third-party mirror is pinned (verified by file-tree match against
>    the official release layout and by row count, 3356 = the published TypeScript instance count),
>    and **RepoBench v1.1 is Python/Java only**, so that row measures cross-file recall in Python,
>    not in M12's target language. The unpinned-pull guard stays and is proven against a synthetic
>    unpinned entry; a *stale* pin now fails by name rather than as an opaque `datasets` traceback.
>    Live pulls are covered by an **opt-in** test (`MONICA_EXTERNAL_LIVE=1`), deliberately not
>    wired into any CI job — CI's contract is the offline fixture path.
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
> ~$100s rather than first being exercised on the ~$1.2k large run.

> **What shape the "dense checkpoint" actually has (#214).** "Dense" here is a statement about
> routing, not about block type, and reading it as "a plain non-MoE config" produces a checkpoint
> the upcycle cannot use. Three constraints, all enforced in code:
>
> 1. **#200 is trained as a degenerate `n_experts: 1, top_k: 1` MoE**, not as a `poc-small.yaml`-shaped
>    dense model. `scripts/upcycle.py` replicates `layers.{i}.experts.0.{gate,up,down}` into the
>    target's expert slots, and only a one-expert MoE block has those keys — a plain dense model has
>    nothing to copy. At E=1 the router's softmax over a single logit is exactly `1.0` and
>    `MoEBlock._moe` takes the `k == E` branch, so the block *is* a plain SwiGLU FFN: same math, right
>    key layout. `MambaConfig.validate()` carries an explicit carve-out for E=1 (and requires
>    `moe_balance_rate: null` there, since balancing one expert is `sign(0) == 0` forever).
> 2. **`moe_d_ff` must equal the target's per-expert width — for Large A that is `d_inner` (1536).**
>    This is the constraint most likely to be missed, because a dense config would normally leave it
>    null, and `null → d_inner` only *happens* to be right here. It is not a default to rely on:
>    `moe_d_ff_resolved` is one of the 15 `_MUST_MATCH` fields, so if the sweep (#219) moves the
>    target's expert width, #200 moves with it.
>
>    > *An earlier revision of this section said `moe_d_ff ≈ d_inner/8`, reasoning that 64 full-width
>    > experts at `d_model 1536` would cost ~906M params per MoE layer. The per-layer arithmetic was
>    > right; the conclusion was not. Narrowing experts cuts **total** params but barely moves
>    > **active** ones — active is dominated by the backbone plus `top_k` experts — so the
>    > `d_inner/8` shape lands near ~255M active against a ~700M target. Resolved by #272 (below):
>    > every shape that hits both targets wants `d_inner` or `d_inner/2`.*
> 3. **Matching the non-expert backbone is necessary but NOT sufficient.**
>    `src/train/upcycle.py:48-52` (`_MUST_MATCH`) is the authority — **15 fields** must agree,
>    including `moe_every`, `moe_d_ff_resolved`, `attn_every` and `n_attn_heads_resolved`, not just
>    `d_model`. `check_upcycle_compatible` reports every mismatch at once and refuses to transform.
>    Treat that tuple as the spec rather than re-deriving the list here, where it would drift.
>
> Consequence for #200: it must be **re-dimensioned** off `poc-small.yaml`'s 97M shape to match the
> small rung's non-expert backbone *and* carry the source shape above. `config/toy-moe-dense.yaml`
> and `config/toy-moe-fine.yaml` are the worked source/target pair; `scripts/upcycle.py --dry-run`
> checks compatibility in seconds, which is the cheap way to find this out rather than after a run.

> **Resolved 2026-08-09 (#272) — Large A is `d_model 768`, deep-and-narrow.** The conflict:
> #200 is the small rung (`d_model 768`) and was named the single upcycle source for both MoE runs,
> but Large A was dimensioned at `d_model 1536`, and
> `src/train/upcycle.py::check_upcycle_compatible` hard-**raises `UpcycleError`** on any `d_model`
> mismatch (#214, by design — expert replication copies a dense FFN into E expert slots; a `.copy()`
> cannot *widen* a tensor). Of the four exits considered, **exit 4 — re-shape Large A to 768** was
> chosen, because it is the only one that keeps a single dense pretrain:
>
> | | Large A |
> |---|---|
> | `d_model` | **768** (was 1536) |
> | `n_layers` | **56** (`56 % attn_every == 0`, so the final block is attention — the FIM constraint above) |
> | `attn_every` | 8 → 7 attention layers (12.5%) |
> | `moe_every` | **3** → 16 MoE layers, **33 Mamba layers survive** |
> | `moe_d_ff` | **`d_inner` (1536)** — full width, *not* `d_inner/8` |
> | experts | 64 routed, **top-8**, + 1 shared |
> | **total / active** | **3.88B / 710M** (target ~3.5B / ~700M) |
>
> Two things this rules out, both verified against `MambaConfig`'s own accounting rather than
> estimated:
>
> - **`moe_every: 1` is disqualified outright.** MoE blocks *replace* Mamba blocks, so routing every
>   non-attention layer leaves **zero** state-space layers — an attention+MoE transformer, not a
>   Mamba-2 hybrid. Two otherwise-attractive shapes were eliminated on this alone.
> - **Narrow experts cannot buy active params.** Shrinking `moe_d_ff` cuts total but not active, so
>   no amount of fine-graining reaches ~700M active; depth and expert width do. This is what
>   invalidated the earlier `d_inner/8` guidance.
>
> Cost of the choice: Large A is deeper and narrower than a 1536-wide model of the same budget (56
> layers vs ~24), so it carries more sequential depth per token — a real consideration for the
> local-hardware tok/s headline. **#219's sweep should confirm the shape before the #223 budget is
> committed**; these numbers establish that the shape *exists*, not that it is optimal.

- **Cross-cutting:** rented-pod ops runbook (#224). **Parked:** post-training SFT (#101) / RLVR
  (#103).

The MoE backend build (#213/#214, **done**) scales on both MLX and CUDA now — the CUDA backend
builds dropless grouped-gather MoE with a shared expert and a sparse-upcycle init path. What
remains before a real RunPod run is FSDP/ZeRO-2 + expert parallel (#271, blocking #223 but not
#222) — the `d_model` conflict above (#272) is resolved.

### #271 — FSDP2 + in-node expert parallel: decisions and open gaps

Built (`src/model/cuda_distributed.py`, `src/train/parallel.py`, `scripts/train_dist.py`) and
gloo/CPU-provable (`tests/test_cuda_distributed.py`); NCCL/real-GPU verification is still open —
see `docs/infrastructure.md`'s "Multi-GPU training pod" checklist, which this record feeds.

- **ZeRO-2 by default, ZeRO-3 (`--reshard-after-forward`) for Large A.** `reshard_after_forward`
  is the dial (`cuda_distributed.wrap_backbone`): keeping params materialized after forward
  (ZeRO-2) costs more peak memory but less communication than resharding every layer (ZeRO-3).
  #222 (small, one card, doesn't even need #271) never exercises this; #223 (Large A) is the
  first real user and should default to ZeRO-3 given the memory pressure that motivated #271 in
  the first place — #219's sweep / the actual #223 bring-up should confirm.
- **Expert-parallel degree is a run-time flag, not a config field.** `--ep-size` (must divide
  both `--nproc_per_node` and the config's `n_experts` — `ParallelConfig` raises loudly
  otherwise) rather than a `MambaConfig` field: EP shard layout is a *deployment* decision (how
  many GPUs, how they're grouped), not a model-architecture one, and the portable weight format
  (`experts.<global_id>.*`, unchanged from pre-#271) is identical regardless of `ep_size`.
- **`fully_shard` cannot wrap `CUDAMambaModel` itself.** Discovered on the planning/exec host,
  not anticipated by the original plan: `CUDAMambaModel(ModelInterface, nn.Module)` inherits
  `ModelInterface(ABC)`, and `fully_shard`'s dynamic `module.__class__` swap raises
  `TypeError: ... object layout differs` on any ABC-rooted class. `wrap_backbone` works around it
  by wrapping each layer plus the top-level leaves (`embedding`, and `lm_head` when untied)
  individually rather than a final root `fully_shard(model, ...)` call — same practical
  parameter coverage, no root wrap.
- **`fully_shard`'s auto-unshard hooks never fire — a second, deeper finding.** FSDP2 unshards a
  wrapped module's params via a forward pre-hook on `nn.Module.__call__`. This codebase's block
  API (`forward_seq`/`step`/`forward_prefill`, the whole train/infer-parity design — see "The
  SSM" in `CLAUDE.md`) is invoked by DIRECT method call, never through `layer(...)`, specifically
  so `forward` and `step` share identical compute without an `nn.Module` dispatch layer between
  them. That design choice means FSDP2's hook never fires, and a sharded `DTensor` param hits an
  op against a plain activation tensor (`mixed torch.Tensor and DTensor`). Fixed with an explicit
  `CUDAMambaModel._fsdp_unshard(layer)` call before every direct block invocation — see its
  docstring for the tradeoff this forces: no paired `reshard()` call (so the actual per-layer
  memory-reclaim timing is UNVERIFIED — a CUDA-host follow-up), and **`grad_checkpoint` does not
  yet compose with this workaround at all** (the checkpointed recompute in backward calls
  `layer.forward_seq` directly, bypassing `_layer_forward`'s unshard entirely) — `grad_checkpoint`
  must stay `false` under FSDP2 until that gap is closed.
- **Muon under FSDP2: one explicit gather, not ~10 implicit ones.** `Muon.step` orthogonalizes a
  `DTensor` momentum buffer by gathering it to a full tensor ONCE (`buf.full_tensor()`),
  Newton-Schulz in fp32 on the full matrix (bit-identical math to the single-process path), then
  redistributing back to the local shard — see `cuda_muon.py`'s module docstring. Without this,
  `X @ X.T` etc. inside Newton-Schulz silently redistribute on every one of the 5 iterations
  (~10 hidden collectives per Muon param per step) instead of crashing, which is the more
  dangerous failure mode (correct-looking, invisible cost).
- **The load-reduce fix is the correctness-critical piece, not an optimization.**
  `MoEBalancer.update` must see per-expert counts summed across every rank BEFORE it runs — see
  `src/train/moe_balance.py`'s module docstring and `cuda_train_step.py`'s `load_reduce` hook.
  Un-reduced, N ranks silently train N divergent routers, and rank 0's copy is the one that ships
  in the checkpoint.
- **EP-sharded expert state cannot reshard across a changed `ep_size`.** Unlike FSDP2's
  DTensor-backed state (which `torch.distributed.checkpoint` reshards across a world-size change
  for free — V8 in the plan's Verification section), each EP rank's local experts are DISTINCT
  parameters, not shards of one tensor. `cuda_distributed.load_resume_dcp` checks a small
  `dcp_meta.json` sidecar and raises before touching any collective if `ep_size` changed, rather
  than silently dropping or duplicating expert state.

### #217 — routing diagnostics + kill-criterion

Before #217 the only routing signal reaching `metrics.jsonl` was `moe_util_var` (a single scalar,
`max` over layers), and only when a `MoEBalancer` is attached (`moe_balance_rate` set) — which is
`null` in every config in the tree. There was no way to tell mid-run whether the router was
*specializing*, the thing the #222/#223 runs turn on. #217 adds three signals, all keyed off the
same per-block `_count_loads` gate the balancer already uses:

- **`moe_router_entropy`** (+ `moe_router_entropy_per_layer`) — mean per-token entropy (nats) of
  the UNBIASED gate distribution, accumulated in `MoEBlock._moe` on both backends, **outside** the
  `top_k < n_experts` guard (loads are meaningless at `top_k == n_experts`, but the gate
  distribution is still real). `MoEBlock.pop_routing_stats()` / the model-level
  `pop_moe_routing_stats()` drain both the entropy accumulator and the load counter in ONE pop —
  the train step (`_accumulate_and_step`, both backends) calls it exactly once per step and shares
  the result between the balancer and the diagnostics, which are no longer gated on
  `balancer is not None`.
- **`moe_util_var`** (+ `moe_util_var_per_layer`) — unchanged key/meaning, now reported on every
  step that observed any routed load, independent of whether a balancer is attached.
- **Per-domain expert histograms** (`src/eval/moe_routing.py`) — a periodic diagnostic eval pass
  (`--moe-diag-every`) over the per-domain val sets `scripts/build_domain_val_sets.py` builds (the
  same `domains.json` input `src/eval/domain_bpb.py` consumes). **Packed training shards carry no
  domain tag** (documented at `src/eval/domain_bpb.py:15-31`; #252, which would add one, is open
  and blocked on #251), so this measures expert usage on held-out samples *drawn from* each
  domain, not over the live training mix. `histogram_overlap` / `specialization_report` compute
  pairwise routing overlap (`1 - TV`, 1.0 = identical routing) across every measured domain pair,
  written to `metrics.jsonl` as `moe_domain_overlap` (mean), `moe_domain_overlap_max`, and the raw
  `moe_expert_hist`.

**The kill-criterion (`kill_check`) reports; it does not abort.** `--moe-kill-pair` /
`--moe-kill-overlap` (default 0.90) name two domains and a routing-overlap threshold; when the
measured overlap is `>= threshold` the run prints a loud `[moe-kill]` line and a `moe_kill` event
in `metrics.jsonl` (`moe_kill_triggered`, echoed live), but **never stops the run** — killing a
multi-day run from a threshold is the operator's call, not this module's. Early in training
routing is near-uniform by construction, so a high overlap there is *expected*, not evidence of a
broken router; a threshold crossed at step 50 does not mean what the same crossing means at step
50,000.

Standing repo rule applied throughout: a check that cannot observe its target reports BLIND,
never healthy. An unmeasured domain, an all-zero histogram, or counting-off entropy never reads
as "the router is fine" — each case raises or is recorded as an explicit `None`.

## The SSI fold (structural signal integration — secondary)

"Does feeding language-server / static-analysis signal into the model help?" — retained as a
**secondary measurement-and-training-signal** axis riding on the MoE model, under a formal
measurement contract:

- **SSI-M — measurement contract** (#225, **done**): one variable per arm, ≥3 seeds + paired
  Wilcoxon/McNemar, repo-level contamination split, availability-vs-use null arms, and a shared
  **escape-hatch lint gate** (extends `SUPPRESSION_RE`'s hatch set — the constant itself stays
  byte-identical — with `as unknown as`, `@ts-nocheck`, non-null `!`, empty bodies, `throw …not
  implemented`, `declare` stubs, deletion-of-target). See
  [`15-ssi-measurement-contract.md`](15-ssi-measurement-contract.md) and
  `src/eval/ssi_contract.py`.
- **Surviving arms:** completion-list logit masking / constrained decode (**#226, built** — the
  trie mechanism, the 5-arm #225-compliant driver, and the decode-loop seam are in; wiring verified
  live end-to-end against a real model + `typescript-language-server`; the full multi-seed
  resolve-rate/clean-rate/latency measurement is a follow-up run, see
  [`12-lsp-in-the-loop.md`](12-lsp-in-the-loop.md)'s "#226" section), diagnostic
  supervision — rejection-sampled FT + contrastive hard negatives (#227), and **RLVR/GRPO with an
  LSP/opengrep verifier reward** (#230).
- **Dropped arms:** two-clock "slow-clock structural state" (conflicts with the MoE spine) and the
  diffusion path.

### #230 — RLVR/GRPO verifier reward (Phase 1 shipped)

Promotes the LSP/opengrep oracle from a measurement-only harness to a **verifier reward**:
GRPO's within-group standardization (`group_advantages`) is itself the diagnostic-cleanliness
reranker, so no new sampling/training plumbing was needed — the whole net-new surface is one
reward function (`src/train/verifiers.py`'s `diagnostics_to_reward` + `LspVerifier`) wired into
`scripts/rlvr.py` via `--reward lsp`.

- **Reward shape.** `reward = max(base_clean − Σ severity_weight(d), diag_floor)` over the
  oracle's findings on the FULL artifact (`prompt + completion` — a completion fragment alone
  isn't standalone TypeScript); clean → `+1.0`, each surviving Error → `−0.30` (the density term
  that keeps an all-clean/all-dirty group from producing zero GRPO advantage), floor for an
  honest-but-dirty attempt `−0.5`.
- **Two anti-Goodhart guards, both dominant over the density term.** `hacked` (any #225 M5
  escape hatch — `@ts-ignore`/`as any`/etc. — present in the completion) and `degenerate` (empty /
  whitespace-only / comment-only / near-empty output) each force the reward to `−1.0`,
  strictly below `diag_floor`. Without that ordering, a many-diagnostic honest attempt could
  score *below* a suppression hack (inverting the intended ranking) or a single 20-diagnostic
  outlier could squash an entire group's advantages toward zero after standardization — both
  silent training failures the floor exists to prevent. Guards are evaluated on the
  **completion alone** (the prompt may legitimately contain `as any` or an unfinished body);
  only the model's own text is penalized.
- **`SUPPRESSION_RE`-vs-superset — deliberately the superset.** Like #226/#227, this arm imports
  `src.lsp.diagnostics.find_escape_hatches` (the #225 M5 ten-hatch superset), not the narrow
  `SUPPRESSION_RE` that stays reserved for `harness.py`'s in-loop rollback control (see that
  module's decision-record comment). `--lsp-hatches directives` restricts to the five explicit
  directives for an ablation that isolates the two known-noisy superset members (`empty_body`,
  `non_null_assertion`); `--lsp-hatches none` disables the hack floor entirely (control arm only).
- **Cost model.** #278 established a ~350ms client-side debounce floor per `didChange` on the
  pinned `typescript-language-server` 5.3.0 that no project-size reduction removes, so
  `oracle.wall_s ≈ K × prompts_per_step × steps × per-call latency` dominates step time at
  scale — e.g. K=8 × 200 steps ≈ 1600 calls ≈ 9–10 minutes of pure oracle wall. `--oracle ts`
  (not `both`) is the default for this reason; `scripts/rlvr.py` logs `oracle_wall_frac`
  (oracle wall ÷ elapsed wall) every `--log-every` step and writes `telemetry.json` so a run's
  real cost is measured, not assumed.
- **No prompt-baseline subtraction.** The prompt's own diagnostics are constant across all K
  samples of a GRPO group and cancel exactly under `group_advantages`' standardization, so
  `LspVerifier` scores `prompt + completion` without a second oracle call to subtract a
  per-prompt baseline — it shifts `mean_reward` in the logs but not the gradient.
- **Scope.** This is Phase 1 only (diagnostic-cleanliness + anti-hack + anti-degenerate +
  telemetry + the `scripts/build_rlvr_prompts.py` train/val split over the checked-in
  `eval_sets/ts_error_injection` + `humaneval_ts` sets). Phase 2 (a reference-based over-repair
  penalty on `clean_control` rows) is deferred to a follow-up issue — it needs a
  `{prompt, reference, kind:"clean_control"}` data-format change the issue itself sequences
  second. The RLVR *accept* criteria (monotone `error_avoidance_rate` gain vs. a math/random
  control, with no hatch-rate rise) need a real GPU run under the #225 M4 null-arm control and
  are not claimed here — this PR ships the mechanism and the telemetry that makes that run
  measurable.

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
