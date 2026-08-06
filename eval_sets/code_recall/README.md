# `code_recall` fixture repo (#221)

A small **synthetic** multi-file TypeScript "repo" used to build cross-file
symbol-resolution instances offline. Consumed by `src/eval/code_recall.py` and
`scripts/eval_code_suite.py --suites recall`.

## Schema

`fixture_repo.jsonl` — one JSON object per line, canonical (`sort_keys`, no spaces):

| field | type | meaning |
|---|---|---|
| `path` | string | POSIX-style module path; used to resolve relative `import` specifiers |
| `text` | string | the file's full TypeScript source |

## Provenance

**Hand-authored synthetic code. Nothing here is copied from a third-party corpus or
benchmark**, so no upstream licence is redistributed. It is eval-only and is fed into the
decontamination blocklist by `scripts/build_decontam_blocklist.py` — i.e. it is explicitly
*excluded* from the training corpus, the opposite of a training signal.

## What each file is for

* **Definers** — `src/geometry.ts`, `src/strings.ts`, `src/units.ts`, `src/matrix.ts`:
  export the symbols that get recalled.
* **Distractor definers** — `src/shapes.ts`, `src/text.ts`, `src/measures.ts`,
  `src/vectors.ts`: never imported, but their exported names (`areaOfSquare` vs
  `areaOfCircle`, `slugifyPath` vs `slugify`, …) are the near-miss candidates the
  discriminative rank has to reject. Without them a "recall" score is just a measure of how
  cheap the identifier is.
* **Consumers** — `src/report.ts`, `src/labels.ts`, `src/convert.ts`, `src/linalg.ts`:
  import a definer's symbol and use it, which is where the scored span sits.
* **Ambiguity cases** — `src/reexport.ts` (`export *`, `export { x as y } from …`) and
  `src/anonymous.ts` (`export default function ()`): the extractor must **skip** these, not
  guess. They exist so the fail-closed path is exercised by the tests rather than merely
  asserted in a docstring.

## Regenerating / extending

The file is checked in and edited directly. Keep it canonical JSONL (`json.dumps(row,
sort_keys=True, separators=(",", ":"))`, one row per line, sorted by `path`) — the suite's
determinism check diffs generated transcripts, and a reordered fixture moves every
instance id.
