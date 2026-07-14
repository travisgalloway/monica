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
