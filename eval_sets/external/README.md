# External code-eval suites — pin table and fixtures (#221)

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
offline. They are not a substitute for the real suites and must never be reported as one.

## Pin table

`revision` is the Hugging Face commit SHA a live pull would be pinned to.
**Every entry is currently unpinned.** A SHA cannot be resolved without network access, and
inventing one is worse than not having one — a wrong pin silently loads a different revision
or errors far from its cause. So `load_external(..., fixture_only=False)` **raises** while
the pin is missing, naming the set and the fix. Fixture mode (the default) needs no pin.

| name | `hf_repo` | config | split | revision | repo id verified? |
|---|---|---|---|---|---|
| `multipl-e-humaneval-ts` | `nuprl/MultiPL-E` | `humaneval-ts` | `test` | **unpinned** | yes — matches `scripts/build_humaneval_ts_set.py` |
| `multipl-e-mbpp-ts` | `nuprl/MultiPL-E` | `mbpp-ts` | `test` | **unpinned** | yes |
| `safim` | `gonglinyuan/safim` | — | `test` | **unpinned** | no |
| `real-fim-eval` | `JetBrains-Research/real-fim-eval` | — | `test` | **unpinned** | no |
| `crosscodeeval` | `Salesforce/CrossCodeEval` | — | `test` | **unpinned** | no |
| `repobench` | `tianyang/repobench-c` | — | `test` | **unpinned** | no |
| `mceval` | `Multilingual-Multimodal-NLP/McEval` | — | `test` | **unpinned** | no |

"repo id verified?" is separate from the revision pin on purpose: an unverified `hf_repo`
is a *second* way a live pull could quietly measure the wrong thing. Only the MultiPL-E
identifiers were confirmable here (they are already used by an in-repo script that has run
against the live hub). The loader refuses live pulls regardless of this column, so an
unconfirmed identifier can never be hit silently.

## Filling a pin

1. Resolve the commit SHA on the hub for the exact revision you intend to measure against,
   and confirm the `hf_repo`/`config`/`split` while you are there.
2. Set `revision="<sha>"` on that entry in `src/eval/external_sets.py` and drop its
   `# TODO(pin):` comment.
3. Update the row above.

`external_sets_manifest()` echoes every `revision` (including the `None`s) into
`scripts/eval_code_suite.py`'s results JSON, so an unpinned run is visible in its own
output, not only in this file.

## Normalized row shape

Every adapter maps an upstream row to:

```json
{"id": "...", "kind": "completion|infill|cross_file",
 "prompt": "...", "suffix": null, "answer": "...", "meta": {}}
```

`suffix` is non-`null` only for infill sets (SAFIM, Real-FIM-Eval). Cross-file context
(`crossfile_context`, `import_statement`) rides in `meta` rather than being concatenated by
the adapter — how it is laid into the context window is an experimental choice belonging to
the caller, not to the loader.
