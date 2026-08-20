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

**Stage A (#199)** replaces the batch-`tsc` oracle with real analysis tooling behind
the same `DiagnoseFn` seam, split into a fake-transport-tested framing layer plus one
module per arm:

- `jsonrpc.py` — pure LSP framing + the async request/notification demux, tested
  with a fake `os.pipe()` transport (no subprocess) since the demux is the hard part.
- `ts_lsp.py` — a persistent `typescript-language-server` oracle (incremental,
  open-document checking, replacing per-call batch `tsc` compiles).
- `ts_server_direct.py` — a SECOND transport to the same pinned TypeScript (#279),
  speaking **tsserver's own protocol** (`semanticDiagnosticsSync`) instead of LSP, so
  the ~350ms client-side debounce inside `typescript-language-server` 5.3.0 (#278) is
  not on the path: 19.7ms median vs 377ms, bit-identical diagnostics over the whole
  #194 set. Diagnostics only; `ts_service.py` stays the LSP-conformant client.
- `opengrep.py` — a persistent `opengrep lsp` oracle carrying a custom, frozen
  correctness ruleset (`eval_sets/opengrep_rules/`) targeting syntactic bug idioms a
  type checker can't see; experimental (see its module docstring's residual-risk
  note) and not Stage A's default.
- `oracle.py` — `CompositeOracle`, the `TscRunner`-contract-compatible drop-in that
  wires one or both arms behind `--oracle {ts,opengrep,both}`; `"ts"` is the default.

See `docs/design/12-lsp-in-the-loop.md` for the design writeup and measurement.
"""
