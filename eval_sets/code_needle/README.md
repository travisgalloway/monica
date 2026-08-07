# `code_needle` haystack filler (#221)

**Synthetic** needle-free TypeScript modules used as haystack padding for the
RULER-over-code retrieval probe (`src/eval/code_needle.py`,
`scripts/eval_code_suite.py --suites needle`).

## Schema

`haystack.jsonl` — one JSON object per line, canonical (`sort_keys`, no spaces):

| field | type | meaning |
|---|---|---|
| `path` | string | module path (cosmetic here — the haystack is concatenated, not resolved) |
| `text` | string | the file's full TypeScript source |

## Provenance

**Hand-authored synthetic code, generated from one template with a varying index.** Nothing
is copied from a third-party corpus or benchmark. Eval-only, and fed into the
decontamination blocklist by `scripts/build_decontam_blocklist.py`.

## How it is used

The probe plants

```ts
export const MONICA_NEEDLE_<KEY> = "<VALUE>";
```

at a fractional `depth` inside a haystack tiled from these files up to an exact token
budget, then queries `<VALUE>` at the end and scores its token span teacher-forced. The
filler is deliberately *ordinary* — the needle must not be findable by formatting alone.

Two consequences of the design worth knowing when reading results:

* The haystack is truncated at a **token** boundary to hit `context_len` exactly, so the
  last filler may be cut mid-statement. It is a distractor and is never scored.
* Filler modules repeat cyclically for long contexts. That is intentional: a repetitive
  haystack is the *harder* case for retrieval, not the easier one.
