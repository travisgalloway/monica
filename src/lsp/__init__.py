"""LSP-in-the-loop generation harness (#199).

Everything in this package is above the seam: stdlib + numpy only, no `mlx`/`torch`
import anywhere (guarded by `tests/test_import_guard.py`). It answers whether feeding
`tsc` diagnostics *into* generation (roll back and retry, or inject the diagnostic as
context and regenerate) beats an off-the-shelf model's unaided completion, on the #194
labeled TypeScript error-injection eval set.

- `tsc.py` — shells out to a pinned `tsc`, one persistent scratch dir per run.
- `diagnostics.py` — parses `tsc`'s real (parenthesized) output into `Diagnostic`s,
  and the string-level primitives the repair loop needs (frontier filtering,
  delimiter closing, statement-boundary detection).
- `lm.py` — the `LMAdapter` Protocol the harness generates against; backends
  (`src/model/mlx_lm_adapter.py`) implement it below the seam.
- `harness.py` — the baseline / slow-loop / tool-call generation strategies.

See `docs/design/12-lsp-in-the-loop.md` for the design writeup and measurement.
"""
