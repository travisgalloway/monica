# The SSI measurement contract (#225)

[← Index](README.md)

The SSI axis ([topic 12](12-lsp-in-the-loop.md)'s arc-level assessment; [topic 13](13-code-model-moe.md)'s
"SSI fold") is a *validated clean-rate tool with a found functional ceiling* — models get
type-clean (0.887 → 0.962) without getting more correct (pass@1 flat at 0.503). That is
precisely the situation where an under-specified measurement lets a reward-hacked or confounded
result read as a win: a small, secondary axis with three still-unbuilt arms (#226 completion-list
logit masking, #227 diagnostic supervision, #230 RLVR/opengrep verifier reward) is exactly where
a shortcut is cheapest to take and hardest to notice. This doc encodes the contract those three
arms must satisfy — **once**, in code, so compliance is checkable rather than asserted in a PR
description. The module is `src/eval/ssi_contract.py`; every claim below carries a `src/...`
path.

Two concrete failure shapes motivated this (from the #225 issue body):

1. **Confounded arms** — an arm that changes the signal *and* the prompt shape *and* the token
   budget, so a delta can't be attributed to the signal; single-seed results reported as
   effects; file-level (not repo-level) contamination splits leaking a near-duplicate sibling
   file into eval.
2. **Reward hacking that reads as clean** — the model makes `tsc` silent without fixing
   anything. `SUPPRESSION_RE` (`src/lsp/diagnostics.py:49`) already catches three moves
   (`@ts-ignore`, `@ts-expect-error`, `as any`); seven more (`as unknown as`, `@ts-nocheck`,
   non-null `!`, empty bodies, `throw …not implemented`, `declare` stubs, deletion-of-target)
   were invisible before this issue, so a #230 verifier reward would have paid for them.

## M1 — one variable per arm

**Rule.** Each declared arm changes exactly one thing relative to its baseline arm.

**Why.** An arm that simultaneously changes the structural signal *and* the prompt shape *and*
the token budget produces a delta that can't be attributed to any one of those — the classic
confound this whole contract exists to prevent.

**Mechanism.** `ArmSpec` (`src/eval/ssi_contract.py`) makes the variable a field, not a
convention: `name`, `variable` (the one thing that changed), `baseline` (the arm it's compared
against — a root/control arm self-references), `signal_available`, `signal_used` (see M4),
`seeds` (see M2), `notes`. `validate_arms(arms)` raises `ContractViolation` when an arm declares
an empty `variable`, when its `baseline` doesn't resolve to a declared arm, or when two arms
compared against the *same* baseline declare the *same* `variable` **and** the same
`signal_used` value — that last qualifier matters: an M4 null/treatment pair is *required* to
share `(baseline, variable)` and differ only in `signal_used`, so the collision check is scoped
to exclude exactly that deliberate pairing while still catching a genuine relabeling confound
(two arms that are supposed to test different things sharing a variable name by mistake).
`validate_arms` never returns a silent "looks fine" for an **empty** arm set — it raises, per the
repo's standing rule that a check that cannot observe its target reports BLIND, never healthy.

**How a run proves it.** `validate_arms(arms)` is called once, at the top of the arm's driver
script, before any generation happens. A passing run's `contract_report(...)` output includes
every arm's full `ArmSpec` (see the reporting block below), so the "one variable" claim is an
artifact, not a sentence in a PR description.

## M2 — ≥3 seeds + paired stats

**Rule.** Every arm reports ≥3 seeds, and the arm-vs-baseline comparison uses paired statistics:
per-seed McNemar (binary metrics), a cross-seed sign test, and — for continuous per-record
metrics — exact Wilcoxon signed-rank.

**Why.** A single-seed delta reported as "the effect" is exactly the kind of result this contract
exists to prevent from reading as a win; the existing `src.eval.lsp_eval.compare`/
`mcnemar_p_value` machinery (`src/eval/lsp_eval.py:139,154`) is strictly single-seed,
single-pair, and `scripts/eval_lsp_harness.py` takes one `--seed` — neither aggregates.

**Mechanism.** `ssi_contract.py` wraps, never reimplements, `lsp_eval.compare`:

- `validate_arms` rejects any arm with `len(set(seeds)) < 3`.
- `per_seed_compare(baseline_by_seed, other_by_seed, *, key)` runs one `compare(...)` per seed.
  Paired means paired: it raises `ContractViolation` if the two arms' seed sets differ, or if any
  seed's baseline/other record `id` sequence differs — `compare`'s own docstring puts the
  sort/align burden on the caller, so this wrapper is what actually enforces it.
- `pooled_compare(...)` concatenates every seed's aligned records into one `compare(...)` call,
  reported *alongside*, never instead of, the per-seed table.
- `sign_test_p(n_positive, n_negative)` is an exact two-sided binomial on the seed-level effect
  *directions* (same sign in every seed?) — the same exact-enumeration style as
  `mcnemar_p_value`, deliberately mirrored: `sign_test_p(3, 0) == 0.25`, `sign_test_p(0, 0) ==
  1.0`, symmetric in its two arguments.
- `wilcoxon_signed_rank_p(deltas)` is **exact** (enumerates all `2**n` sign assignments over the
  ranks of the nonzero `|delta|`s) for continuous per-record metrics (BPB, pass@k rates) where
  McNemar's binary table doesn't apply. `n > 20` raises `ValueError` rather than silently falling
  back to a normal approximation — the same failure mode `mcnemar_p_value`'s docstring already
  rejects (an approximation that's wrong at small `n`).
- `summarize_arm(per_seed, sign_p, pooled)` is the reportable block: per-seed rates, mean ±
  spread, the sign test, the pooled McNemar, and an explicit `"consistent_direction"` bool — a
  reader sees directly whether every seed agreed on direction rather than inferring it from a
  p-value.

**How a run proves it.** The `stats` block of `contract_report(...)` carries `summarize_arm`'s
output for every arm-vs-baseline pair — `n_seeds >= 3` and `consistent_direction` are visible
without recomputing anything.

## M3 — repo-level contamination split with a logged manifest

**Rule.** Train/eval contamination splitting for any SSI corpus is done at repo granularity, not
file granularity, and the split is logged in a manifest that can be checked, not just trusted.

**Why.** `src/data/dedup.py`'s `Decontaminator` and `scripts/build_decontam_blocklist.py` do
*benchmark-text* decontamination (n-gram matching against a fixed blocklist) — a different
problem from partitioning a *training* corpus by repo so a near-duplicate sibling file from the
same repo can't leak from train into eval. Repo identity already exists in the corpus:
`src/data/stack_v2.py:93` sets `meta["repo"] = row["repo_name"]`, carried through
`Record.meta` (`src/data/corpus.py:44-57`).

**Mechanism.**

- `repo_of(record)` reads `meta["repo"]` off either a `Record`-like object or a plain dict.
  Returns `None` when absent — it never invents an identity.
- `split_by_repo(records, *, eval_fraction, seed, on_unknown="raise")` assigns **whole repos** to
  train/eval by a stable hash: `sha256(f"{seed}:{repo}")`'s leading 8 bytes, as a fraction of
  `2**64`, compared against `eval_fraction`. Deliberately **not** Python's `hash()` — salted per
  interpreter (the same gotcha already called out at `scripts/eval_code_suite.py:404`), so a
  `hash()`-based split isn't reproducible across runs. A record with no repo identity is, per
  `on_unknown`: `"raise"` (default) — because silently repo-splitting a corpus with no repo
  identity would produce a *file-level* split wearing a repo-level label, the exact defect M3
  exists to prevent — or `"quarantine"` — routed to a bucket in **neither** side, counted in the
  manifest, never silently folded into train.
- `assert_disjoint(train, eval)` raises `ContractViolation` if any repo appears on both sides —
  the file-vs-repo bug this rule exists to catch.
- `split_manifest(split)` returns `{n_train, n_eval, n_quarantine, n_train_repos, n_eval_repos,
  eval_fraction, seed, eval_repos_sha256, manifest_sha256}`, following the byte-reproducible
  manifest pattern of `scripts/build_decontam_blocklist.py`'s `blocklist.manifest.json`.
  `eval_repos_sha256` hashes the sorted, newline-joined eval repo list, so two runs claiming the
  same split can be *checked* against each other rather than trusted.

**How a run proves it.** `contract_report(...)`'s `splits` block is `{name: split_manifest(...)}`
for every corpus split the arm used — `n_quarantine` and `eval_repos_sha256` are visible without
re-deriving the split.

## M4 — availability-vs-use null arms

**Rule.** Any arm with `signal_used=True` has a sibling null arm: same `variable`, same
`baseline`, `signal_available=True`, `signal_used=False`.

**Why.** Without the null arm, "the signal helped" is indistinguishable from "the arm got extra
tokens / extra compute / a different prompt shape because the signal's plumbing was present" —
an availability confound wearing an effect's clothes.

**Mechanism.** `validate_arms` checks, for every `signal_used=True` arm, that some *other*
declared arm shares its `(variable, baseline)`, has `signal_available=True`, and
`signal_used=False`. No such sibling → `ContractViolation`.

**How a run proves it.** The arm list in `contract_report(...)` is the `ArmSpec` for every arm
that ran — a null arm with `signal_available=True, signal_used=False` sitting next to its
treatment sibling is visible by inspection, and `validate_arms` having not raised is itself the
proof the pairing exists.

## M5 — the shared escape-hatch lint gate

**Rule.** Every SSI arm gates its generated artifacts through the same escape-hatch detector
before scoring "clean".

**Why.** `SUPPRESSION_RE` (`src/lsp/diagnostics.py:49`) catches three reward-hacking moves
(`@ts-ignore`, `@ts-expect-error`, `as any`) but not the other seven a model can use to make
`tsc` silent without fixing anything.

**Decision — two tiers, not an in-place widening of `SUPPRESSION_RE`.** `SUPPRESSION_RE` stays
byte-identical and narrow. It is a *live in-loop control*, not just a metric:
`src/lsp/harness.py:321` (`_is_clean`) uses it to force a mid-generation rollback, and
`src/lsp/harness.py:501` (`generate_slow_loop`) uses it to set `reward_hack_detected`. Widening it
in place would (a) silently redefine what the already-published #199 numbers (clean-rate
0.887→0.962, pass@1 0.503) measured, making old and new runs non-comparable, and (b) put a
non-null-`!` false positive — logical negation is the single most common operator in the
language — on a hard rollback path. So the ten-hatch superset lives in a **new** constant,
`ESCAPE_HATCH_PATTERNS`, that every SSI arm uses instead.

**Mechanism** (`src/lsp/diagnostics.py`):

- `mask_strings_and_comments(text)` blanks every character inside a string/template literal or a
  comment to a space, preserving `len(text)` and every newline — built on the same
  string/template/comment-aware `_scan` walk that `close_open_delimiters`/`statement_boundary`
  already use, so all three agree on what counts as "inside a string/comment". `${...}`
  interpolations stay live code (`_scan`'s own stack push takes them back to code context), which
  is correct: a hatch written inside an interpolation is real code, not string content.
- `ESCAPE_HATCH_PATTERNS: Dict[str, re.Pattern]` — nine named regex hatches (a tenth,
  `deletion_of_target`, is structural, see below):

  | Name | Text matched | Notes |
  |---|---|---|
  | `ts_ignore` / `ts_expect_error` / `ts_nocheck` | raw | directives live inside comments by definition |
  | `as_any` / `as_unknown_as` | masked | prose ("`// use as any`") must not fire |
  | `non_null_assertion` | masked | postfix `!` only — lookbehind + `(?!=)` excludes `!=`/`!==`/prefix `!x` |
  | `empty_body` | masked | `)…{}`  or `=>…{}` |
  | `declare_stub` | masked | `^\s*declare (function\|const\|…)` |
  | `throw_not_implemented` | raw, guarded | see below |

  `throw_not_implemented` is the one hatch that must see real, unmasked string content — the
  thing it's checking (`"not implemented"`/`"unimplemented"`/`"TODO"`) **is** the string content,
  so masking it away would make the hatch permanently dead. It matches raw text, but a candidate
  match whose `throw` keyword itself falls inside a masked span (the statement is commented out,
  or sits inside an unrelated string) is discarded — checking whether `masked[start:start+5]`
  still reads `"throw"` is exactly that signal.
- `find_escape_hatches(text) -> List[str]` returns the sorted names present;
  `has_escape_hatch(text) -> bool` is the go/no-go form.
- `deletion_of_target(before, after, *, anchors)` is the **tenth**, structural hatch — the
  degenerate "delete the thing the diagnostic was about" move. Returns the `anchors` present in
  `before` and absent from `mask_strings_and_comments(after)` (so commenting the target out
  counts as deletion too). Anchors are **injected by the calling arm** (the symbol under test,
  its signature, the call site) — never guessed from the text, because guessing is exactly the
  silent-mismatch failure this whole contract exists to prevent.
- `src/eval/ssi_contract.py` re-exports `find_escape_hatches`/`has_escape_hatch`/
  `deletion_of_target` so an arm has one import surface
  (`from src.eval.ssi_contract import ...`) — "applied identically across arms" becomes a
  property of the import, not of discipline.

**How a run proves it.** Every scored record an arm reports should carry
`find_escape_hatches(artifact)`; a nonempty list forces the record to score not-clean regardless
of `tsc`'s own verdict, mirroring `score_record`'s existing `suppression_hack` handling
(`src/eval/lsp_eval.py:64`).

## The reporting block

`contract_report(arms, splits, stats)` (`src/eval/ssi_contract.py`) assembles the JSON block a
run writes next to its results — the artifact that makes M1–M5 compliance auditable after the
fact rather than asserted in a PR description:

```json
{
  "arms": [
    {"name": "...", "variable": "...", "baseline": "...",
     "signal_available": true, "signal_used": false, "seeds": [1, 2, 3], "notes": "..."}
  ],
  "splits": {"main": {"n_train": 0, "n_eval": 0, "n_quarantine": 0,
                       "n_train_repos": 0, "n_eval_repos": 0,
                       "eval_fraction": 0.0, "seed": 0,
                       "eval_repos_sha256": "...", "manifest_sha256": "..."}},
  "stats": {"treatment_vs_baseline": {"n_seeds": 3, "mean_delta": 0.0,
                                        "delta_spread": 0.0, "sign_test_p": 0.0,
                                        "pooled": {}, "consistent_direction": true}}
}
```

## Known limits

- **`non_null_assertion` false positives.** `a! = b` (a space before `=` defeats the `(?!=)`
  negative lookahead) is a known, accepted residual false positive. Kept out of `SUPPRESSION_RE`
  on purpose (see M5) so it never reaches the hard-rollback path.
- **`empty_body` false positives.** A legitimately empty `catch {}` or `constructor() {}` fires.
  Accepted — the gate is conservative by design (a hatch present scores not-clean), and
  `find_escape_hatches` names *which* hatch fired so a caller can see this rather than eat an
  opaque bool.
- **`mask_strings_and_comments` residual imprecision.** `_scan` yields only one index for a
  two-character token (`\x` escape, `${`, `*/`, opening `//`); the *second* character of such a
  token is never independently visited and can survive unmasked as stray punctuation. This
  cannot itself spell a whole hatch pattern except in a contrived, invalid-TypeScript corner case
  (an empty `${}` interpolation) — documented, not fixed.
- **Exact Wilcoxon's `n <= 20` ceiling.** `wilcoxon_signed_rank_p` raises past 20 nonzero deltas
  rather than falling back to a normal approximation. An arm with a larger continuous-metric
  record count needs a different (not-yet-built) exact or resampling method; using the normal
  approximation silently would reintroduce the small-`n` unreliability `mcnemar_p_value`'s
  docstring already rejects.
- **`deletion_of_target` anchors are arm-supplied**, never inferred. A caller that skips this or
  supplies a vacuous anchor set gets a silent `[]` — the function has no way to detect "the arm
  forgot to pass real anchors" versus "the target genuinely survived".
- **`meta["repo"]` availability.** Only `src/data/stack_v2.py` populates it today. `on_unknown=
  "raise"` is the default specifically so a corpus without repo identity fails loud instead of
  producing a file-level split wearing a repo-level label.
