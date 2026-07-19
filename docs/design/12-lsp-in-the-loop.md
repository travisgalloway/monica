# LSP-in-the-loop: no-training validation (#199)

[← Index](README.md)

M12 ([issue #198](https://github.com/travisgalloway/monica/issues/198)) asks whether feeding
language-server feedback *into* generation (roll back and retry, or inject a diagnostic and
regenerate) beats re-reading diagnostics as tool-call tokens. **#199 is the gate on that whole
program**: build a `tsc`-in-the-loop autoregressive harness on an off-the-shelf model (no
training) and measure it against the [#194 eval set](../../eval_sets/ts_error_injection/README.md)
before funding the P1 tier (#191/#192/#193/#200/#201/#101/#103/#104). This doc is the design
record and the measurement.

**Model**: `mlx-community/Qwen2.5-Coder-1.5B-bf16` (base, not instruct — see risks below).
**Locked decisions** (user): build and run locally on MLX-LM; implement both hard and soft
repair; defer the fast loop (tree-sitter + tsserver) to #201.

## What was built

- `src/lsp/tsc.py` / `diagnostics.py` — `TscRunner` (one persistent scratch dir per run, not a
  `TemporaryDirectory` per call), `parse_tsc_output` (the real `tsc --pretty false`
  parenthesized format — `snippet.ts(3,15): error TS2339: ...`, not the `file:line:col:` form
  the issue text assumed, which matches zero real diagnostics), `filter_diagnostics`
  (frontier + `generation_start` clamping), `close_open_delimiters` / `statement_boundary`
  (string/template/comment-aware scanner), `strip_suggestion`, `SUPPRESSION_RE`.
- `src/lsp/lm.py` + `src/model/mlx_lm_adapter.py` — the `LMAdapter` Protocol
  (`encode/decode/reset/step/rollback` + forward-token cost counters) and its `mlx_lm` backend,
  with exact-rollback via `trim_prompt_cache`. `tests/test_mlx_lm_adapter.py` is the gate that
  matters here: reset+step and reset+rollback+re-step reproduce identical logits to 1e-4 in fp32.
- `src/lsp/harness.py` — `generate_baseline`, `generate_slow_loop` (`repair="hard"|"soft"|"both"`,
  `budget="stmt"|"block"`), `generate_toolcall`.
- `src/eval/lsp_eval.py` — `score_record`/`summarize`/`compare` (exact McNemar).
- `scripts/eval_lsp_harness.py` — the driver; `docs/design/12-lsp-in-the-loop.md` (this file).

## The two-string invariant and frontier semantics

**CONTEXT** is what the model conditions on (may contain an injected `// tsc: ...` comment).
**ARTIFACT** (`prompt + completion`) is what gets scored and never contains a comment — soft
repair edits the context, never the artifact, which is what keeps the score honest.

**Frontier**: an error is only real once the model has committed to it by emitting at least one
more token (`nam` is not wrong; it may become `name`) — diagnostics at or past the frontier are
dropped. One refinement found by running against the real model (not caught by the FakeLM suite,
which never modeled multi-character tokens): the frontier is normally "the start of the last
emitted token," but once a segment has reached a **genuine statement boundary** (the model itself
emitted an explicit terminator — no more tokens are ever coming for that attempt), the *entire*
segment counts as committed, last token included. Withholding the last token in that case creates
a permanent blind spot for single-token segments, where the last token is the *only* token —
confirmed as a real bug: `console.log(u.` followed by a single BPE token decoding to `);` produced
`TS1003 "Identifier expected"` anchored exactly at the segment's start, and the un-fixed frontier
rule dropped it every time, silently accepting a broken completion as clean.

**TS1xxx filtering** — `filter_diagnostics` drops the whole TS1xxx family, since an in-progress
autoregressive prefix can trigger any of ~200 such codes for the trivial reason that generation
hasn't finished the statement yet. That is correct while a segment is still mid-generation, but a
**deliberate, scoped deviation from the plan's filter rule** was needed for the same boundary
case above: once a segment has reached its statement boundary, a surviving TS1xxx (e.g. the
`u.)` case) is rescued back in, because it is a real defect in finished text, not incompleteness
noise. This only fires when `hit_boundary` is true and the frontier-filtered result was otherwise
empty; it never applies to a still-forming mid-generation check.

## Metrics (fixed before the run)

| metric | definition |
|---|---|
| `diagnostic_clean_rate` | raw `tsc` on the ARTIFACT, unfiltered. Truncated or a suppression hack (`@ts-ignore`/`as any`) forces not-clean |
| `error_avoidance_rate` | **pre-registered primary.** 84 error rows: `expected_diagnostic not in codes(tsc(artifact))` |
| `exact_gold_rate` | secondary/informational only |
| `over_repair_rate` | fraction of the 12 `clean_control` rows with >= 1 rollback |
| `no_progress_rate` | soft-repair rounds where the regenerated statement was byte-identical to the last attempt |
| cost | `n_forward_tokens` (cached) / `_nocache`, `n_tsc_calls`, `n_rollbacks`, wall-clock, `tsc` wall-clock share |

Go/no-go was committed in advance: **GO** iff, on the 84 error rows, the best slow-loop variant
beats baseline on `error_avoidance_rate` with McNemar p < 0.05, regression_rate <= 2%, and the
slow loop beats or matches tool-call. **NO HEADROOM** (distinct from NO-GO) if baseline
`diagnostic_clean_rate` > 90% at both temperatures.

## Step 0 — the headroom spike (1.5B model, all 96 records)

| | greedy | temp 0.8 |
|---|---|---|
| `diagnostic_clean_rate` | 0.854 | 0.771 |
| `error_avoidance_rate` | 0.976 | 0.988 |

Both well under the 90% no-headroom threshold on `diagnostic_clean_rate` — real headroom exists,
so the run proceeded to the full measurement. Note the split that turned out to matter: headroom
on *clean-rate* is real, but `error_avoidance_rate` (the pre-registered primary) was already
saturated at baseline even in this first look.

## Tables A / B / C

All three tables report **two McNemar comparisons per strategy vs. baseline**: the
**pre-registered** one (`error_avoidance_rate`, the 84 error rows) and a **post-hoc, NOT
pre-registered** one (`diagnostic_clean_rate`, all rows) — reported because the pre-registered
metric turned out to be near-saturated at baseline on every table, which is itself a finding, not
grounds to quietly swap the primary metric after seeing the data.

### Table A — `stmt` budget, greedy, all 96 records

| strategy | clean | avoid | over_repair | no_progress | mean tsc share |
|---|---|---|---|---|---|
| baseline | 0.854 | 0.976 | 0.000 | — | 0% |
| slow-hard | 1.000 | 1.000 | 0.083 | — | 65% |
| slow-soft | 0.854 | 0.952 | 0.000 | 1.000 | 56% |
| slow-both | 0.990 | 1.000 | 0.083 | 0.000 | 65% |
| toolcall-k1 | 0.854 | 0.952 | 0.000 | 0.357 | 60% |
| toolcall-k2 | 0.854 | 0.952 | 0.000 | 0.714 | 57% |

McNemar, pre-registered (`avoided`, n=84): every strategy p >= 0.50 (slow-hard/slow-both: 2
discordant pairs, 0 regressions — the best possible p at that count is 0.50). McNemar, post-hoc
(`clean`, n=96): **slow-hard p=0.000122** (14 fixed / 0 regressed), **slow-both p=0.000244** (13
fixed / 1 regressed); slow-soft and both toolcall variants p=1.0 (no discordant pairs at all —
`no_progress` is why).

### Table B — `stmt` budget, temp 0.8 (seed 0), all 96 records

| strategy | clean | avoid | over_repair | no_progress | mean tsc share |
|---|---|---|---|---|---|
| baseline | 0.771 | 0.988 | 0.000 | — | 0% |
| slow-hard | 0.958 | 1.000 | 0.250 | — | 65% |
| slow-soft | 0.812 | 0.952 | 0.000 | 0.286 | 48% |
| slow-both | 0.958 | 1.000 | 0.167 | 0.000 | 65% |
| toolcall-k1 | 0.750 | 0.929 | 0.000 | 0.083 | 58% |
| toolcall-k2 | 0.792 | 0.929 | 0.000 | 0.143 | 57% |

Confirms avoidance saturation is not a greedy-decoding artifact (baseline avoid = 0.988 here
too). **toolcall-k1's clean-rate (0.750) is strictly below baseline's (0.771)** — feeding the
diagnostic back as text made the model measurably worse here, not just no-better. Post-hoc
`clean` McNemar: **slow-hard/slow-both p=0.000040** (19 fixed / 1 regressed each); slow-soft
p=0.39, toolcall-k1 p=0.73, toolcall-k2 p=0.79 (no significant effect either direction).
`over_repair_rate` for slow-hard/slow-both rises to 0.250/0.167 (vs. 0.083 greedy) — a real cost
of sampling, not just a greedy-decoding curiosity, and it's reported in the table rather than the
appendix on purpose.

### Table C — `block` budget (96 tokens), greedy, 32-row stratified subsample

The only condition that exercises the checkpoint stack (multiple statements per run) and virtual
delimiter closing (budget exhausted mid-statement). Mean generated tokens confirm the comparison
is fair after the bug fix below: baseline 96.0 (fixed length by construction), slow-hard 96.0,
slow-both 93.1, slow-soft 60.9, toolcall-k1/k2 60.5/60.9 (soft-family strategies stop early on
`no_progress`, not because they were budget-starved).

| strategy | clean | avoid | over_repair | no_progress | rollbacks | tsc share |
|---|---|---|---|---|---|---|
| baseline | 0.312 | 0.929 | 0.000 | — | 0 | 0% |
| slow-hard | 0.688 | 1.000 | 0.500 | — | 69 | 52% |
| slow-soft | 0.469 | 0.929 | 0.000 | 0.765 | 0 | 48% |
| slow-both | 0.688 | 1.000 | 0.500 | 0.000 | 68 | 52% |
| toolcall-k1 | 0.469 | 0.929 | 0.000 | 0.529 | 0 | 49% |
| toolcall-k2 | 0.469 | 0.929 | 0.000 | 0.706 | 0 | 49% |

Post-hoc `clean` McNemar (n=32): **slow-hard/slow-both p=0.000488** (12 fixed / 0 regressed each);
slow-soft/toolcall-k1/toolcall-k2 all p=0.0625 (5 fixed / 0 regressed — a real but smaller effect,
under-powered at n=32 to clear 0.05). Pre-registered `avoided` (n=28 error rows): all p >= 0.50.

**This is the headline table.** The free-running block budget breaks the stmt budget's ceiling
effect — baseline clean-rate drops to 0.312 (vs. 0.854 at `stmt`), giving the loop real room to
show an effect: **slow-hard more than doubles clean-rate over baseline (0.312 -> 0.688) with zero
regressions on `avoided` and zero net regressions on `clean`.** And the two predicted-in-advance
findings both confirmed empirically:
1. **toolcall-k1 and slow-soft land on the identical clean-rate (0.469).** The design doc's risks
   section predicted this before running anything: on a base model with no chat template,
   "diagnostic as a chat message" and "diagnostic as an injected comment" are the same algorithm
   at different granularity. Exact agreement here is evidence the harness measures what it claims
   to, not a coincidence.
2. **slow-hard (0.688) beats toolcall (0.469) by 22 points on a baseline of 0.312** — roughly
   doubling clean-rate where tool-call-style feedback does not move it beyond what soft repair
   alone gets. This is the M12 thesis, measured: the hard-ban condition is the only one that tests
   something a tool-call baseline structurally cannot express (see risks), and it is the one that
   wins.

## Bug found mid-measurement: `generate_toolcall`'s budget was hardcoded

`generate_toolcall` called `generate_slow_loop(..., budget="stmt", ...)` with no `budget`
parameter of its own. Tables A and B (`--budget stmt`) were unaffected — the hardcoded value
happened to be correct there — but the first Table C run silently gave every toolcall variant a
1-statement budget while every other strategy free-ran to 96 tokens: toolcall's completions
averaged **10 characters** against baseline's 281, making its apparent "win" an artifact of
generating almost nothing. Caught by comparing mean generated length across strategies before
trusting the table (see `git log` on `src/lsp/harness.py` for the fix commit). Fixed by threading
`budget`/`block_size`/`max_gen_tokens` through; pinned with a regression test
(`test_toolcall_respects_block_budget_not_hardcoded_to_stmt`) asserting toolcall's generated
length matches baseline's under `budget="block"`. Table C above is the **re-run, post-fix** data.

## Qualitative findings from the transcripts

- **Hard repair sometimes "fixes" by inserting a comment token, not just banning the bad one.**
  10/192 slow-hard-clean rows across Tables A+B contain a hard-repair-induced comment where
  banning pushed the model into opening one instead of a plain identifier — e.g.
  `console.log(u./*age*/age);`, `p./*error*/x;`, `truck./* ERROR */wheels;`, and one non-ASCII
  garbage token `/*■*/number;` (temp 0.8 only). The resulting code genuinely type-checks and
  references a real member — it is not the `@ts-ignore`/`as any` suppression hack the harness
  explicitly guards against — but "100% clean" overstates code *quality* in these cases, and it's
  reported here rather than left implicit in the aggregate number.
- **The baseline sometimes hallucinates a diagnostic instead of writing code.** 6/96 Table-A
  baseline completions respond to a prompt ending mid-member-access with a comment like
  `// error: Property 'gate' does not exist on type 'Flight'.` instead of any code at all — the
  1.5B base model pattern-matching toward "this looks like a place an error message goes,"
  presumably from code-review-comment-heavy pretraining data. It's a small but real illustration
  of why the text-mode feedback paths (soft repair, tool-call) struggle on this model: it isn't
  reliably distinguishing "write code" from "describe an error."
- **`tsc` dominates wall-clock**: 48-65% of every repair strategy's runtime across all three
  tables, while the GPU sits idle during each call. Not a flaw to hide — it is the empirical case
  for #201's tsserver daemon, directly from this measurement's own cost accounting.

## Ablations

Both planned ablations were run (neither was dropped).

### 1. `--strip-suggestions` — does the loop only work because `tsc` hands over the answer?

`tsc` volunteers spelling suggestions (`Cannot find name 'visitorName'. Did you mean 'userName'?`),
and 40 of the 84 error rows have an injected name spelling-close to the gold. That is a confound
about what a GO *means*: "the model uses diagnostics" vs. "the model copies the compiler's answer."
Re-running the `stmt`/greedy table with the `Did you mean …?` clause stripped from every message:

| strategy | clean | avoid | no_progress | vs. Table A (suggestions kept) |
|---|---|---|---|---|
| baseline | 0.854 | 0.976 | — | identical |
| slow-soft | 0.854 | 0.952 | 1.000 | identical |
| toolcall-k1 | 0.854 | 0.952 | 0.357 | identical |

**No effect whatsoever** — every number matches Table A exactly. The confound is fully defused, for
a slightly deflating reason: the suggestion-consuming strategies never worked *at all* (soft repair's
`no_progress_rate` is 1.000 with or without the hint), so there was nothing for a leaked answer to
help. And the headline strategy, hard-ban, never reads the message text — it bans a token id — so it
is structurally incapable of benefiting from suggestion leakage. **The result owes nothing to the
compiler giving away the answer.**

### 2. Qwen2.5-Coder-0.5B — does a weaker model widen the gap?

M12's plan rests on the premise that *"a deliberately weak ~100M model is the point: it makes many
errors, so the LSP has more to correct and the measured delta is larger and cleaner."* This ablation
tests that premise directly, and **contradicts it** on this eval set:

| model / temp | strategy | clean | avoid |
|---|---|---|---|
| 0.5B, greedy | baseline | **0.875** | **1.000** |
| 0.5B, greedy | slow-hard | 0.969 | 1.000 |
| 0.5B, temp 0.8 | baseline | 0.792 | 1.000 |
| *1.5B, greedy (Table A)* | *baseline* | *0.854* | *0.976* |

The 0.5B model is **not worse** than the 1.5B here — its baseline clean-rate is *higher* (0.875 vs.
0.854) and its `error_avoidance_rate` is a perfect **1.000**, i.e. it never once reproduces the
injected error. Shrinking the model does not create headroom on the pre-registered axis; it removes
what little remained.

This matters beyond this issue: the saturation documented in the Verdict is a property of the
**task**, not of model capacity, so it **cannot be fixed by reaching for a weaker student** — which
is precisely the lever M12 planned to pull. Budget shape (`block` vs. `stmt`) moved the baseline by
54 points (0.854 → 0.312); halving the model moved it by −2. Harder records or a `block`-shaped
eval set is the only route to a measurable `error_avoidance_rate`.

The mechanism itself generalizes across scale: hard-ban lifts the 0.5B from 0.875 → 0.969
(McNemar **p=0.0039**, 9 fixed / 0 regressed), mirroring its effect on the 1.5B.

## Verdict

The **pre-registered go/no-go criterion (`error_avoidance_rate`, McNemar p < 0.05) is NOT met on
any of the three tables** — not because the repair loop failed, but because the metric was
saturated at baseline (82-83/84 error rows already avoided on `stmt`, 26-27/28 on `block`),
leaving too few discordant pairs (0-5) for any test to reach significance regardless of effect
size. This is a genuine finding about **#194**, not a harness failure: the error-injection eval
set's `stmt`-budget baseline is too easy for a 1.5B coder model to leave the pre-registered metric
measurable.

On the **post-hoc, clearly-labeled secondary metric** (`diagnostic_clean_rate`, not
pre-registered), the hard-ban repair strategy shows a strong, consistent, statistically
significant effect with near-zero regressions on every table: **Table A p=0.000122 (14 fixed / 0
regressed), Table B p=0.000040 (19 fixed / 1 regressed), Table C p=0.000488 (12 fixed / 0
regressed)** — and on Table C, the one budget with real baseline headroom (0.312), it roughly
doubles clean-rate (-> 0.688) while clearly beating the tool-call/soft-repair family (0.469),
confirming the M12 thesis on the one axis this measurement could actually move.

**GO on the P1 tier's hard-ban mechanism specifically; the pre-registered `error_avoidance_rate`
axis is undecided and needs a harder #194 (or a `block`-budget-shaped variant of it) before it can
gate anything**, per the compound verdict:

> Pre-registered GO criterion NOT met (`error_avoidance_rate` saturated at baseline on every
> table); strong, significant effect on the post-hoc `diagnostic_clean_rate` metric across all
> three tables (p ranging 0.000040-0.000488, near-zero regressions), most decisively on the
> `block` budget (0.312 -> 0.688, beating tool-call's 0.469); the base model essentially cannot
> use a diagnostic delivered as tokens (soft-repair/tool-call `no_progress_rate` up to 1.000 at
> greedy) while the logit-level hard ban fixes nearly everything it touches. Fund the hard-ban
> mechanism (#191/#192/#193 as applicable) and #201's tsserver daemon (tsc is 48-65% of every
> repair run's wall-clock); before funding anything gated on `error_avoidance_rate` specifically,
> #194 needs harder records or #199's `block`-budget framing needs to become the eval set's
> native shape.

---

# Follow-up validation (E1–E4): attacking the Phase-0 conclusion

Phase 0 concluded *"the mechanism works, but only at the logit level, not as text."* That claim
had three load-bearing weaknesses, and this section attacks each one rather than defending it.

> **Framing correction.** Hard-ban is not "feedback" at all. It masks a token id and forces
> resampling; the model never learns that `gorblak` isn't a property of `User`, it just finds
> `age` because the door closed on `gorblak`. What works is **constrained search with a
> verifier**, not communication. The comment-insertion pathology (§ Qualitative findings) is the
> tell: banned, the model escapes into `/*…*/` and routes *around* the constraint.

## E1 — a fair tool-call opponent

Phase 0's tool-call baseline ran on a **base** model, which cannot follow instructions by
construction. That made "diagnostics-as-text doesn't work" close to unfalsifiable — every
production coding agent feeds compiler errors back as text and it demonstrably works. E1 replaces
it with `Qwen2.5-Coder-1.5B-**Instruct**` (deliberately the *same size*, so the comparison isolates
instruction-tuning rather than scale) receiving the `tsc` error in a genuine chat turn.

**The design had to be fixed before it could answer anything.** Comparing chat-tool-call against
completion-baseline confounds *two* variables — the feedback mechanism **and** the task format. An
instruct model in chat mode writes different code than the same model continuing a prefix,
feedback or no feedback. So each mechanism is measured against **its own format's control**:

| | control (no feedback) | with feedback | Δ |
|---|---|---|---|
| **completion mode** — logit-mask feedback | baseline **0.948** | slow-hard **1.000** | **+0.052** |
| **chat mode** — text feedback, 1 round | chat-k0 **0.802** | chat-k1 **0.802** | **+0.000** |
| **chat mode** — text feedback, 2 rounds | chat-k0 **0.802** | chat-k2 **0.802** | **+0.000** |

*(stmt budget, 96 records, greedy, length-matched: every arm generates 11–12 chars.)*

**Feeding a real compiler error to a competent instruct model, in a real chat turn, changed the
clean-rate by exactly zero — at one round and at two.** The mechanism underneath that zero:

> Of the **19** records where the instruct model wrote broken code, it was shown the actual `tsc`
> diagnostic and **re-emitted byte-identical wrong code 18 times (95%)**. The single time it did
> change its answer, it made things *worse* — a type error (`TS7022`/`TS2448`) became two syntax
> errors (`TS1005`).

The Phase-0 thesis therefore **survives its strongest available challenge**. The pre-registered
decision rule said a fair tool-caller matching hard-ban would scope the thesis to weak models; it
did not match. It did not move at all.

**Two measurement bugs were found and fixed en route, and both biased toward the conclusion we
already believed** — the direction a bug is most dangerous:
1. The chat extractor rejected valid short completions (`.title`, `zx.x`) as "prose" (33% bogus
   extraction-failure rate), and
2. the join concatenated `books[0].` + `.title` → `books[0]..title`, manufacturing `TS1003`s.

Both were our bugs being scored as the model's. Fixed and pinned by tests before any number here
was recorded.

**A third bug was in the metric itself:** `no_progress` was only wired into the completion-mode
path, so chat mode reported `0.000` while the model was in fact stuck 95% of the time. A metric
that cannot see the phenomenon it exists to measure is worse than no metric.

### The `block` budget is not a fair arena (and that is a finding)

The first E1 run said chat-tool-call *beat* hard-ban on the block budget (0.781 vs 0.500), which
would have overturned everything. It is an artifact: the chat model generates **10 characters**
while every other strategy free-runs **~340**, and scores well for writing almost nothing. This is
the *same* length-mismatch that faked the tool-call win in Phase-0's Table C — mirrored, now
flattering the opposite conclusion.

Asking the instruct model for a multi-statement continuation (`BLOCK_SYSTEM_PROMPT`) moved it from
10 to only ~26 characters. An instruct model writes correct, concise code and stops; the
completion-mode strategies are *forced* to keep emitting for 96 tokens. **So the block budget
measures how well a model survives being made to keep typing — not how much feedback helps.** Only
the length-matched `stmt` comparison above is fair, and it is the one reported.

## E2 — the capability ladder: is hard-ban a crutch that better training subsumes?

If hard-ban only patches errors a better-trained model wouldn't make, then **#193** (LSP-clean
corpus) and **#103** (verifier GRPO) would *subsume* it, and the inference harness would be the
wrong place to spend. Committed decision rule: **Δclean ≤ 3 points at 7B ⇒ crutch.**

| model | baseline | slow-hard | Δclean | **headroom captured** |
|---|---|---|---|---|
| `mamba2-130m` (SSM) | 0.625 | 0.781 | +15.6 | 42% |
| Qwen2.5-Coder-0.5B base | 0.875 | 0.969 | +9.4 | 75% |
| Qwen2.5-Coder-1.5B base | 0.854 | 1.000 | +14.6 | 100% |
| Qwen2.5-Coder-1.5B instruct | 0.948 | 1.000 | +5.2 | 100% |
| Qwen2.5-Coder-7B instruct (4-bit) | 0.833 | 0.979 | +14.6 | 87% |
| Mamba-Codestral-7B (SSM, 8-bit, n=32) | 0.938 | 0.969 | +3.1 | 50% |

**Verdict against the pre-registered rule first: NOT a crutch.** The rule was stated on **raw
Δclean ≤ 3 at 7B**; the 7B instruct rung is **+14.6**, nearly 5× the threshold, so the conclusion
holds on the metric committed to *before* seeing the data. Hard-ban is an orthogonal lever and
*composes* with #193/#103 rather than being replaced by them.

The "headroom captured" column is a **post-hoc** normalization (chosen after seeing that raw Δ is
noisy because it tracks available headroom — a model at 0.948 cannot gain 15 points). It is a
better statistic and tells the same story — hard-ban closes 42–100% of whatever gap remains, at
every scale and both architectures — but it is presentation, not the verdict, and is labeled
post-hoc for the same reason Phase 0's `diagnostic_clean_rate` was.

**Two caveats that genuinely weaken this experiment:** the ladder is **not monotone in capability**
— the 4-bit 7B's baseline (0.833) is *below* the bf16 1.5B's (0.948), so quantization is a
confound and the top rung is a weaker test than it looks; and the whole thing is a **proxy** — a
stronger model is not a model trained on LSP-clean data, which only #193 can supply.

**Caveats, stated rather than buried.** (a) The 7B is 4-bit quantized and its baseline (0.833) is
*below* the 1.5B bf16's (0.948) — quantization degraded it, so the ladder is **not monotone in
capability** and the 7B rung is a weaker test than it looks. (b) This whole experiment is a
**proxy**: a stronger model is not the same thing as a model trained on LSP-clean data. Only #193
can answer that; this is the best stand-in until it exists.

## E3 — the SSM arm and #202: is the mechanism affordable on *our* architecture?

The one mechanism that works needs **rollback**, and monica is an SSM. Verified against the real
library:

| cache | `is_trimmable()` | rollback |
|---|---|---|
| `KVCache` (transformer) | **True** | O(1) trim — but the cache grows linearly with context |
| `ArraysCache` (Mamba SSM) | **False** | no per-token history to trim → **full re-prefill** |

This looked like bad news for the architecture the whole project rests on. **It is the opposite.**
An SSM's state is **fixed-size**, so while it cannot be *trimmed*, it can be *checkpointed* — at a
cost that is **constant in context length**, where a transformer's is linear. That is #202, and it
is now implemented (`MLXLMAdapter.checkpoint()/restore()`) and measured on **Mamba-Codestral-7B**
(Mamba-2 *and* code-specialized — the closest public analogue of monica's own architecture):

| context | #202 snapshot+restore | naive re-prefill | ratio |
|---|---|---|---|
| 39 tok | 0.4 ms | 307 ms | 840× |
| 159 tok | 0.4 ms | 759 ms | 2,081× |
| 639 tok | 1.1 ms | 5,499 ms | 4,981× |
| 1,839 tok | 4.4 ms | 116,482 ms | 26,692× |

Snapshot/restore is **bit-exact** (max\|Δ\| = 0.0 vs a fresh re-prefill, fp32 — pinned by a parity
gate, because an inexact restore would silently corrupt every rollback while still *looking* fine).
Cost: **272 MB per snapshot**, constant.

**Do not read those ratios as an architectural property.** A 116-*second* re-prefill means MLX's
Mamba-2 prefills sequentially (~9–21 ms/token, no chunked-scan kernel), so the absolute re-prefill
figures — and hence the ratios — are **implementation-dependent**. What is intrinsic is the
*shape*: **snapshot is O(1) in context, re-prefill is O(context).** A production Mamba kernel would
shrink the constant, not the asymptote.

**Verdict, stated as measured-then-extrapolated so the two are not conflated:**

- **Measured (trust high):** #202 is correct (bit-exact, test-pinned) and cheap in absolute terms
  (sub-millisecond at our eval's ~40-token contexts). At *short* context the SSM is nonetheless
  **worse off than a transformer** — a KV trim is free, a 272 MB snapshot is not. That is the honest
  short-context picture and it is not flattering.
- **Extrapolated (trust lower — not measured end-to-end):** because snapshot is O(1) in context and
  the KV cache is O(context), the ordering *must* invert at long context, which is the regime this
  project targets. This is an inference from the complexity shape, not a measured long-context
  rollback benchmark, and the 840×–26,692× ratios above are additionally inflated by MLX's sequential
  Mamba-2 prefill — so they are illustrative of the *shape*, not a claimed speedup.

On that basis **#202 should still be promoted out of the "optional P2 extension" tier** — it is a
precondition for the one mechanism that works, and correct + cheap is already enough to justify it —
but the "long-context advantage" is a hypothesis to confirm with a real long-context benchmark, not
a result this experiment delivered.

## E4 — over-repair on real code: **resolved (#201)**

The intent was to replace Phase-0's 4-of-8-row over-repair anecdote with a real denominator: ~200
statement-boundary prefixes from real TypeScript (`bigcode/the-stack-smol`), filtered to files that
already compile clean, so a rollback on one is an interruption of code that was heading toward clean.
The first attempt was **inconclusive by construction** — three stacked confounds. #201 fixed all
three and re-ran; the number below is trustworthy.

The three confounds and their fixes:
1. **Module resolution.** The model continues a real file with plausible `import`s that don't resolve
   under the isolated tsconfig (`TS2307` etc.). These were filtered out of *file selection* but not
   the *in-loop diagnose* or *scoring* — a real bug. **Fixed:** `MODULE_RESOLUTION_CODES` is a shared
   set (`src/lsp/diagnostics.py`) applied consistently everywhere via `--ignore-module-resolution`
   (`drop_codes` wraps the diagnose fn for both the loop and scoring).
2. **Cuts inside multi-line constructs.** A brace-depth-0 newline *within a cut fragment* is not a
   true top-level boundary, so `prompt+completion` was often an unclosed `interface`/object fragment
   (`TS1005`/`TS1109`). **Fixed:** the probe set is now cut with a real parser
   (`src/lsp/ts_boundaries.py`, tree-sitter-typescript) that returns only true top-level statement
   ends — no boundary can land inside an unclosed construct (proved in `tests/test_ts_boundaries.py`).
3. **Special-token leakage.** `<|endoftext|>`/`<|fim_prefix|>` decoded into completions and scored as
   TS. **Already fixed** by the F1 EOS-stop (`_eos_ids`, `harness.py`) under `budget=block`; **verified
   0 leaks across all 600 transcript rows** of the re-run.

**The measured real-code over-repair** (`results/e4_overrepair_realcode.json`; 200 tree-sitter-cut
prefixes, `mlx-community/Qwen2.5-Coder-1.5B-bf16`, greedy, `budget=block --block-size 256`,
`--ignore-module-resolution --oracle ts`):

| metric | slow-hard | slow-both |
|---|---|---|
| `over_repair_rate` (any rollback on a clean_control row) | **0.260** | 0.260 |
| true over-repair (rollback \| the model's continuation was *actually* clean) | **13/36 = 0.36** | 0.36 |
| final clean-rate (vs baseline 0.180) | 0.235 | 0.240 |
| mean rollbacks / row | 1.09 | 1.07 |

Read carefully, because the raw rate and the honest rate differ:

- **The raw `over_repair_rate` (0.26) is 2–3× the synthetic-control estimate** (Phase-0 0.083 greedy /
  0.25 temp 0.8; E1/E2 0.08–0.17; #199 Stage A ts/both 0.101/0.120). The synthetic controls
  **understated real-code over-repair.**
- But a 256-token block continuation of real code is *usually itself broken* — the model's raw
  continuation is diagnostic-clean only **18%** of the time — so most of the 52 rollbacks are
  *legitimate* repairs (the loop nets clean-rate **0.180 → 0.235**), not over-repair. The metric as
  defined conflates the two.
- **The honest over-repair signal** is the rollback rate *conditioned on the continuation being
  clean*: of the 36 rows where the model's greedy continuation was already clean, the loop still
  rolled back **13 (36%)**. On code that was genuinely fine, the loop interrupts it about a third of
  the time.

**Mechanism** (from the transcripts): the spurious rollbacks are the frontier logic reinstating a
*committed* `TS1xxx` syntax-incompleteness code (e.g. `TS1146` "Declaration expected") as "real" at a
segment boundary when `is_source_balanced` — but on multi-segment block generation that code is
transient and resolves as generation continues (the same run's baseline reaches a clean final
artifact). The `is_source_balanced` gate (a Phase-0 Stage-A heuristic) is not tight enough for real
multi-segment real-code bodies; tightening it is the concrete follow-up.

**Verdict:** over-repair on real code is **real and non-trivial** — ~26% of rows, ~36% of the rows
where the model was already right — materially higher than the synthetic controls implied, though
partly offset because the loop also fixes genuinely-broken continuations. This is a cost the
trained-model ablation (#201's blocked half) and any reward design (#103) must weigh, and it points
at the frontier/`is_source_balanced` transient-diagnostic handling as the first thing to harden.

### The final-segment gate — first hardening pass (#201)

The mechanism above says the reinstatement fires at *every* balanced boundary with no notion of
whether more segments are still coming. The fix: reinstate a committed `TS1xxx` **only on the
genuinely final segment of the block** (`is_final_segment` = `budget=="stmt"`, or EOS, or this
segment exhausts the block budget, or a stop string fired), threading a new `eos_hit` out of
`_extend_to_boundary_or_budget`. A transient `TS1xxx` and a genuine one are indistinguishable *at*
the prefix — the only discriminator is whether generation continues — so the final-segment condition
is not a heuristic but the actual invariant.

Re-measured (same protocol; `results/e4_overrepair_realcode_fix.json`, `results/f1_ts_fix.json`):

| | before | after gate |
|---|---|---|
| #201 slow-hard `over_repair_rate` (raw) | 0.260 | **0.215** |
| #201 conditioned true over-repair (rollback \| continuation clean) | 13/36 = 0.36 | **7/36 = 0.194** |
| #199 F1 slow-hard clean-rate / pass@1 | 0.962 / 0.503 | **0.962 / 0.503 (byte-identical)** |

The conditioned over-repair — the honest signal — **nearly halves**, at **zero F1 cost**: the gate
never fires on HumanEval-TS (single open-function bodies are unbalanced at intermediate boundaries,
so the reinstatement never fired there to begin with), so the whole F1 run is byte-identical. The
change is confined to multi-top-level-statement real code, exactly the over-repair case.

**Residual (honest):** the gate cuts the `TS1xxx`-transient rollbacks, but a *second* source
surfaces — transient committed **`TS2xxx`** semantic codes (e.g. `TS2395` "merged declaration must be
all exported or all local") on incomplete multi-segment code, which flow through the normal filter
(not the `TS1xxx`-only reinstatement) and still trigger rollbacks. On the exemplar `clean-prefix-0020`
the `TS1146` chasing is gone (5→0) but the trajectory now chases `TS2395`. Extending the same
"final-segment / transient" reasoning to committed `TS2xxx` is the next hardening step; raw
over-repair (0.215) won't fall to zero until it's addressed.

## Risks realized (see the original plan for the full list)

- **Tool-call-as-base-model risk, confirmed as predicted**: toolcall-k1 and slow-soft land on
  identical clean-rate on Table C (0.469 both) — the same algorithm at different granularity on
  this one-statement-shaped eval set, exactly as anticipated before running anything.
- **`tsc` leaks the answer** (`--strip-suggestions`): **not realized** — stripping the `Did you
  mean …?` clause changes nothing (see Ablations), and hard-ban never reads the message text at all.
- **A weaker model would widen the gap** (M12's stated premise): **contradicted** — the 0.5B model
  has a *higher* baseline clean-rate and a perfect `error_avoidance_rate` (see Ablations). The
  saturation is a property of the task, not of model capacity.
- **Reward hacking**: `SUPPRESSION_RE` (`@ts-ignore`/`as any`) forces not-clean regardless of
  `tsc`'s verdict; zero suppression hacks were observed in any table's transcript.

# Stage A — the real-analysis oracle: persistent TS-LSP + opengrep (#199 follow-up)

Everything above draws its diagnostic signal from `TscRunner` — a fresh `tsc -p` **batch compile per
check**. Two things motivated replacing it. First, the F1 measurement on real HumanEval-TS (159
records, functional pass@1 via the reference test suites) sharpened the gap the whole project is
about: model bodies are **88.7% type-clean but only 50.3% correct** (`results/f1_base.json`,
baseline) — the failures are logic errors a type checker structurally cannot see. Second, a batch
compile per check is the wrong model of "LSP-in-the-loop". Stage A (PR #208) replaced the oracle with
a **persistent TypeScript language server** and an **opengrep** arm carrying a frozen, pre-registered
12-rule correctness ruleset, behind the same `DiagnoseFn` seam, then re-measured. All three runs
below are the same protocol: 159 records, `block` budget 256, greedy (temp 0, seed 0),
`baseline` + `slow-hard`.

## The oracle swap moves the win from the functional axis to the clean axis

| slow-hard vs baseline | `f1_base` (batch `tsc`) | `f1_ts` (persistent LSP) | `f1_both` (LSP + opengrep) |
|---|---|---|---|
| clean-rate | 0.887 → 0.906, p=**0.375** (ns) | 0.887 → **0.962**, p=**0.0005** | 0.862 → 0.937, p=**0.0005** |
| pass@1 | 0.491 → **0.560**, p=**0.001** | 0.491 → 0.503, p=**0.69** (ns) | 0.491 → 0.528, p=**0.109** (ns) |
| over-repair (synthetic controls) | 0.094 | 0.101 | 0.120 |
| mean diagnose calls / record | 8.18 | 6.75 | 7.25 |

The batch-`tsc` slow loop delivered a **real functional lift** (pass@1 +6.9 pts, p=0.001, 11 records
flipped to passing and 0 to failing) but no significant clean-rate lift. Swapping in the persistent
LSP **inverts this**: a strong clean-rate lift (+7.5 pts, p=0.0005) but the functional lift **vanishes**
(p=0.69) and the loop now flips 2 correct records to failing. This is the plan's "retune, don't port
blind" risk, now measured: a language server's diagnostics on an *open/incomplete* document are not
the batch whole-program diagnostics the `tsc`-tuned loop was calibrated against. The loop cleans more
aggressively under the LSP and that extra cleaning is not correctness-bearing. (Baseline pass@1 is
identical — 0.491 — across all three runs, as it must be: baseline generation is greedy and
oracle-independent; the oracle only affects clean-rate scoring and the slow loop.)

## Does opengrep find a syntactic correctness signal for the clean-but-wrong class?

**Yes — narrow, but with perfect precision on this eval.** On the raw (`baseline`) outputs, the
12-rule set fired on exactly **4 of 159 records, and all 4 are functionally wrong** (0 false
positives). That is the direct answer to the question the arm exists for — it is measured as the
baseline clean-rate dropping from 0.887 (`ts`) to 0.862 (`both`): those 4 records are type-clean but
rule-flagged, and every one fails its tests.

| Record | Rule | Functionally correct? |
|---|---|---|
| `HumanEval_14_all_prefixes` | `loop-bound-off-by-one` | ✗ wrong |
| `HumanEval_67_fruit_distribution` | `parseint-no-radix` | ✗ wrong |
| `HumanEval_95_check_dict_case` | `for-in-over-array` | ✗ wrong |
| `HumanEval_111_histogram` | `for-in-over-array` | ✗ wrong |

Across the whole `both` run only three rules ever fired (`for-in-over-array` ×4, `parseint-no-radix`
×2, `loop-bound-off-by-one` ×1). The rules the plan flagged as noisy (`sort-without-comparator`,
etc.) **never fired**, so the feared false-positive tax did not materialize on this set. This exceeds
the pre-registered calibrated expectation (a near-zero hit-rate was the honest prior); the hit-rate
*is* low (4/159 ≈ 2.5%), but it is a genuine, precise signal, not noise.

Folding opengrep into the slow loop **partially recovers the pass@1 the LSP swap lost** (0.503 →
0.528; 8 records flipped to passing vs baseline) but does **not** reach significance (p=0.109) and
costs more over-repair (0.101 → 0.120; 2 correct records broken). Net: acting on opengrep's findings
fixes a few real bugs, but the idiom-rule signal is too sparse — HumanEval failures are mostly
*algorithmic* misunderstanding, which a syntactic AST matcher structurally cannot see — to move the
functional metric on its own.

## The measurement is trustworthy: the opengrep stall did not corrupt it

A prior stress test had seen a ~10% full-timeout stall under sustained single-file rescanning of one
long-lived `opengrep lsp` process — a *live-but-unresponsive* process that the dead-process liveness
check never caught, so a stalled scan was silently counted as a timeout and returned `[]`, **dropping
findings**. At ~10% that would corrupt any `both` measurement. Since each scan overwrites the whole
candidate file and is independent of prior calls, respawning changes reliability but never the finding
set, so `OpengrepOracle` now proactively recycles the process every 32 calls and reactively restarts +
retries once on a no-response scan (`src/lsp/opengrep.py`; `scripts/opengrep_soak.py` is the soak
harness). The `both` run's self-reported counters: **`n_timeouts = 0` over 1471 opengrep calls**, 0
stall recoveries, 45 proactive recycles (~68 s overhead). No findings were silently dropped — and
notably the stall did not reproduce on this host at all (0 timeouts across 160 rotating + 200
single-file soak calls with the mitigation off), so the recycler is a cheap, unit-tested safety net
rather than a demonstrated before/after fix.

## The TS-LSP oracle's first-push-wins is complete — no multi-push race in practice (#211)

`TsLspOracle.diagnostics()` arms a single `threading.Event` and returns on the **first**
`textDocument/publishDiagnostics` push for the candidate URI, with **no settle/quiescence window** —
unlike `OpengrepOracle._rescan`'s `_SETTLE_S`. The open concern (#211): a push-model server that
publishes a fast **syntactic** pass then a slower **semantic** pass would have the oracle capture the
early syntactic-only push and **miss the semantic `TS2xxx` errors**, biasing measured LSP lift toward
"finds nothing" and making the set timing-dependent. That would silently corrupt the phases that trust
the oracle's diagnostic set (the #201 ablation, and especially #203, LSP-as-diffusion-discriminator).

**Probed, not assumed** (`scripts/probe_ts_lsp_multipush.py`, `results/ts_lsp_multipush_probe.json`).
The probe re-opens each candidate the oracle sees but registers a *list-append* callback that records
**every** push for the URI — timestamp + codes — and captures the server's advertised
`capabilities.diagnosticProvider`. Run over the whole #194 set (96 records, all 4 error classes) plus a
**crafted candidate carrying both a syntactic (`TS1109`) and a distinct semantic (`TS2322`) defect**,
with a settle window (3.5 s) wide enough to catch late pushes:

| Signal | Result |
|---|---|
| Candidates where the **first** push ≠ the final union of all pushes | **0 / 97** |
| Candidates where a **semantic** code appeared only in a later push (the race) | **0 / 97** |
| Candidates where a later push added *any* code | **0 / 97** |
| Crafted syntax+semantic candidate | **1 push**, carrying `TS1109` **and** `TS2322` together |
| `capabilities.diagnosticProvider` (pull diagnostics / LSP 3.17) | **absent** on `typescript-language-server` 5.3.0 |

The pinned server *does* emit up to **3** pushes per document (median inter-push gap ~1.2 s), but the
**first push is always semantically complete** — pushes 2–3 are **idempotent re-publishes** carrying
the identical code set (tsserver re-emitting after the project settles). The crafted candidate — which
deliberately mixes a syntactic and a semantic defect, the exact split the race hypothesis needs —
**coalesces both into one push**, directly refuting a syntactic-then-semantic split for this server.

**Resolution: no production change.** First-push-wins captures the complete set on every candidate and
every error class, so the current `diagnostics()` is correct as written. The pull-diagnostics fix path
is moot anyway (5.3.0 advertises no `diagnosticProvider`). The multi-push race remains a **latent**
risk for a *different* server or config that front-loads a syntactic-only push — the probe script is
retained so it can be re-run if the pinned toolchain changes — but it is not a live bug, and **#211 is
closed** on this evidence.

## Verdict

The persistent-LSP swap is **not a free upgrade over batch `tsc`**: it trades the slow loop's
functional benefit for a clean-rate benefit, because open-document diagnostics differ from
whole-program ones and the `tsc`-tuned gating does not transfer. The opengrep arm delivers a **real
but narrow** correctness signal (4/4 precision on raw outputs, sparse), which nudges pass@1 up inside
the slow loop without reaching significance and at a small over-repair cost. The honest read for the
program gate (#198): the clean-but-wrong gap is real and a syntactic idiom-matcher can *see a corner
of it precisely* but not enough of it to carry the functional metric — the correctness-bearing signal
for this class lives in semantics (tests, execution, type-aware analysis), which is where the AR
harness ablation (#201) and the LSP-verifier reward work (#103) should aim.
