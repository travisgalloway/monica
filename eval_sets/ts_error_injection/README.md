# TS error-injection eval set (#194)

A labeled, held-out set of TypeScript "error-injected completion" examples. Each item is a
compilable prompt paired with a correct completion (`gold_completion`) and a wrong reference
completion (`error_completion`) that deliberately triggers a specific, real `tsc` diagnostic.
It exists to give **#199** (the LSP-harness no-training validation) a ground truth for measuring
diagnostic-clean rate and error-induced pass rate — not to score models itself.

## Schema (`eval.jsonl`, one JSON object per line)

| field                 | meaning                                                              |
|-----------------------|-----------------------------------------------------------------------|
| `id`                  | unique record id                                                     |
| `error_class`         | one of the classes below                                             |
| `expected_diagnostic` | the `tsc` code the class maps to (`""` for `clean_control`)          |
| `prompt`              | self-contained TS prefix (types/decls included) ending at the completion point |
| `gold_completion`     | correct completion; `prompt + gold_completion` compiles with **zero** diagnostics |
| `error_completion`    | reference wrong completion; `prompt + error_completion` produces `expected_diagnostic` (empty for `clean_control`) |
| `notes`               | what the trap is / why the label holds                               |

Every record is self-contained: `prompt` includes any interfaces/classes/decls the completion
references, so `tsc` alone (no external symbol table) can catch or clear the labeled error.

## Error families

| `error_class`               | `tsc` code | what it tests                                             |
|------------------------------|------------|------------------------------------------------------------|
| `unfamiliar_member_access`   | `TS2339`   | accessing a property/method that doesn't exist on a typed object |
| `undefined_name`             | `TS2304`   | referencing an identifier that was never declared          |
| `arity_mismatch`             | `TS2554`   | calling a function/method/constructor with the wrong argument count |
| `clean_control`              | *(none)*   | already-correct completion — tests over-repair / false positives |

28 items each for the 3 error families + 12 `clean_control` items = 96 total.

## Provenance

Hand-authored for this issue, released **CC0** (no third-party code, no scraped snippets).
Each label is mechanically verified against a real, pinned TypeScript compiler — see below.

## Validating the labels

```bash
cd eval_sets/ts_error_injection
npm install                 # installs the pinned compiler (package.json / package-lock.json)
cd ../..
python scripts/validate_ts_error_set.py
```

For every record the script asserts `prompt + gold_completion` produces **zero** `tsc`
diagnostics, and (for non-`clean_control` rows) `prompt + error_completion` produces a
diagnostic whose code matches `expected_diagnostic`. It prints a per-record pass/fail table and
exits non-zero on any mismatch. If no node/npm/tsc toolchain is found on the host, it prints a
message and exits 0 rather than failing a host that isn't meant to run it.

The compiler is pinned via `package.json`/`package-lock.json` (TypeScript 5.9.3 — the last
pre-native-port classic release; TypeScript 7's rewrite dropped the `moduleResolution: "node"`
option this set's `tsconfig.json` uses) and `@types/node` (so ambient globals like `console`
resolve without pulling in the DOM lib, which would otherwise collide with common variable names
like `length`/`target`/`name`/`origin`).

## Using this set (for #199)

`src/eval/ts_error_eval.py` is the portable loader: `load_ts_error_set(path)` reads and validates
the JSONL schema (raises `ValueError` on a malformed record), returning a list of dicts. It does
not run `tsc` — that stays `scripts/validate_ts_error_set.py`'s `tsc_diagnostics()` function,
which #199's LSP-harness can reuse as its `diagnose_fn` (same `prompt + completion -> list[str]
of TSxxxx codes` shape).
