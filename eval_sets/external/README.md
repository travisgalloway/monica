# External code-eval suites — pin table and fixtures (#221, #304)

Loader specs live in `src/eval/external_sets.py`; this directory holds one **synthetic**
fixture per suite so the adapters can be tested with no network.

## The rule: no third-party rows are checked in

Every `fixture.jsonl` here is **hand-authored synthetic content whose field names match the
upstream schema**. No benchmark rows are copied into this repo. Two reasons, both binding:

1. **Licensing** — nothing upstream is redistributed, so no upstream licence attaches.
2. **Contamination** — a benchmark's real rows sitting in the repo are a leak risk into the
   training corpus. The fixtures are additionally fed into the decontamination blocklist by
   `scripts/build_decontam_blocklist.py`.

The fixtures therefore test **the adapter**, which is the only thing that can be tested
offline. They are not a substitute for the real suites and must never be reported as one —
in particular, **a green fixture test is not evidence that a live pull works.** That is what
the opt-in live test and the observed row counts below are for.

## Pin table

Every SHA below was resolved against the live hub on **2026-08-20** with no token, and every
row was then **pulled live and normalized end to end** — `n_rows` is the observed row count
of that pull, not an estimate. **All seven were load-verified; none is schema-verified-only.**

| name | `hf_repo` | config | split | revision (SHA) | n_rows (live) | identifier confirmed |
|---|---|---|---|---|---|---|
| `multipl-e-humaneval-ts` | `nuprl/MultiPL-E` | `humaneval-ts` | `test` | `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2` | 159 | yes — hub API + already used by `scripts/build_humaneval_ts_set.py` |
| `multipl-e-mbpp-ts` | `nuprl/MultiPL-E` | `mbpp-ts` | `test` | `28441b6024e71d4a1c1c0f6bf171c935cd5a43f2` | 390 | yes — hub API, same repo/commit |
| `safim` | `gonglinyuan/safim` | `block` | `test` | `be132cc15372e90b6f03a608e77f2d940e384edb` | 8781 | yes — hub API |
| `real-fim-eval` | `gonglinyuan/real_fim_eval` | `add` | `test` | `a36062544c7ed6c4e5ffb8dad8536fc7777e1f36` | 17879 | yes — hub API |
| `crosscodeeval` | `Vincentvmt/CrossCodeEval` | — (`data_files`, see below) | `train` | `41f916e35cc48bcca5dc369664f931afd9ffa22f` | 3356 | yes — hub API, **third-party mirror** (see Provenance) |
| `repobench` | `tianyang/repobench_python_v1.1` | `default` | `cross_file_first` | `8a7cf0c8942cc1aa066bf261839650ac55a2ff79` | 8033 | yes — hub API |
| `mceval` | `Multilingual-Multimodal-NLP/McEval` | `generation` | `test` | `c4767fec950de7decd487a40a4dab2795df4a094` | 2007 | yes — hub API |

CrossCodeEval's mirror declares no configs, so its file is named explicitly:
`data_files="crosscodeeval_data/typescript/line_completion_rg1_bm25.jsonl"`, which lands in
the default `train` split. `external_sets_manifest()` echoes `data_files` alongside
`config`/`revision` into `scripts/eval_code_suite.py`'s results JSON, so the exact pin a run
measured against is visible in that run's own output.

**Credentials: none.** All seven are public and ungated; every SHA above and every live pull
was done unauthenticated. No token is committed anywhere in this repo, and none is needed.
*Operator gotcha:* an **expired** token cached in `~/.cache/huggingface/token` is worse than
no token — the hub answers an authenticated request for a public repo with `401 Repository
Not Found`, which reads as "the dataset was withdrawn". Point `HF_TOKEN_PATH` at a
nonexistent file (or log out) to force an anonymous pull before believing a pin has gone
stale.

## Four identifiers were wrong, and why they changed (#304)

The table #221 shipped was written offline, and it did not survive contact with the hub.

- **`JetBrains-Research/real-fim-eval` → HTTP 401**, it does not exist. The dataset that does
  is `gonglinyuan/real_fim_eval` (same author as SAFIM), configs `add`/`edit`. We take
  **`add`**: pure insertion, exactly the `(prefix, suffix, middle)` shape. `edit` carries a
  `to_remove` column, making it a *replacement* task that does not map onto the normalized
  infill row without inventing semantics.
- **`Salesforce/CrossCodeEval` → HTTP 401.** See Provenance below.
- **`tianyang/repobench-c` exists but cannot be loaded.** Its only files are
  `.gitattributes`, `README.md` and `repobench-c.py` — it is a loading-script dataset, and
  the installed `datasets` (4.8.5) removed script execution. Pinning it would have produced
  a pin that only fails when someone runs a live pull.
  `tianyang/repobench_python_v1.1` is the maintained parquet re-release. It has **no `test`
  split** — the splits are `cross_file_first` / `cross_file_random` / `in_file`.
- **`safim` and `mceval` had `config=None`, which is invalid.** SAFIM declares
  `block / control / api / block_v2 / control_fixed` and McEval declares
  `generation / explanation / completion`, neither with a default, so `load_dataset(..., None)`
  would have failed at the first live pull.

## Provenance and caveats — read before quoting a number from these suites

**1. CrossCodeEval is a third-party mirror.** CrossCodeEval has no official Hugging Face
dataset; upstream (`amazon-science/cceval`) distributes it as a GitHub/Drive release. The
mirror `Vincentvmt/CrossCodeEval` was checked, not trusted, in two ways: its file tree is a
verbatim copy of the official `crosscodeeval_data/` release layout (`LICENSES/`, `README`,
`{python,java,csharp,typescript}/line_completion*.jsonl`), and the pinned TypeScript file
holds **3356** rows, matching the published CrossCodeEval TypeScript instance count. No
upstream checksum exists, so that is the whole of the verification. `zijwang/CrossCodeEval`
(by a CCEval author) was rejected — it holds only `README.md` and `.gitattributes`, no data.

The RG-1/BM25 file is pinned rather than the plain `line_completion.jsonl` because only it
carries the retrieved `crossfile_context` this suite exists to measure. Both files hold the
same 3356 instances.

**2. RepoBench v1.1 is Python/Java only — there is no TypeScript.** This row therefore
measures cross-file recall in *Python*, not in M12's target language. It is still the right
instrument for "does the SSM backbone retrieve across files", but it is not a TypeScript
number and must not be reported as one.

## Running a live pull

```bash
# One set, all rows, normalized:
.venv/bin/python -c "from src.eval.external_sets import load_external; \
    print(len(load_external('safim', fixture_only=False)))"

# The opt-in test over all seven:
MONICA_EXTERNAL_LIVE=1 .venv/bin/python -m pytest tests/test_external_sets.py -q -k live
```

`MONICA_EXTERNAL_LIVE` is **not** wired into any CI job, deliberately. The macOS job runs
37m48s against a 45-minute cap (#315), a network-dependent job is flaky by construction, and
caching benchmark rows onto a runner cuts against the contamination rule at the top of this
file. **CI's contract is the offline fixture path**; the live path is an operator command.
The live tests SKIP with an explicit reason when the variable is unset — they never silently
pass.

## When a pin goes stale

A withdrawn revision, a renamed repo or a newly-gated dataset raises `SystemExit` naming the
set, repo, config, `data_files`, split and revision, with the original `datasets` error
chained — not an opaque traceback. Re-resolve the SHA against the hub, update the entry in
`src/eval/external_sets.py`, and update the table above **including its `n_rows`**, which is
an observation of that pull and not a constant.

An entry added *without* a revision is refused outright: `load_external(..., fixture_only=False)`
raises before it imports `datasets`, because an unpinned pull is not a reproducible
measurement. Fixture mode (the default) needs no pin.

## Normalized row shape

Every adapter maps an upstream row to:

```json
{"id": "...", "kind": "completion|infill|cross_file",
 "prompt": "...", "suffix": null, "answer": "...", "meta": {}}
```

`suffix` is non-`null` only for infill sets (SAFIM, Real-FIM-Eval). Cross-file context
(`crossfile_context`, `crossfile_snippets`, `import_statement`) rides in `meta` rather than
being concatenated by the adapter — how it is laid into the context window is an experimental
choice belonging to the caller, not to the loader.

`_norm` enforces the shape by **type**, not just by presence. This matters: RepoBench's
`context` column exists and is a *list of `{identifier, path, snippet}` structs*, so a
presence-only check would have let `prompt=row["context"]` through and emitted a non-string
prompt — a guard that accepts bad data instead of rejecting it. An empty prompt is likewise
rejected, except for an infill row whose suffix carries the context (56 of Real-FIM-Eval
`add`'s 17879 rows are genuine top-of-file insertions with an empty prefix).
