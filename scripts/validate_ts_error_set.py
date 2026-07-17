#!/usr/bin/env python3
"""Proves every label in `eval_sets/ts_error_injection/eval.jsonl` is real.

For each record: `prompt + gold_completion` must compile with **zero** `tsc`
diagnostics, and (for non-`clean_control` rows) `prompt + error_completion` must
produce a diagnostic whose code equals the record's `expected_diagnostic`. This is
the mechanism that makes the eval set trustworthy — a label is only as good as this
script's ability to reproduce it against a real compiler.

Shells out to the TypeScript compiler pinned in
`eval_sets/ts_error_injection/package.json` (installed to that directory's
`node_modules/`). If node/npm/tsc genuinely cannot be found (e.g. a CI runner with
no node toolchain), this script prints a message and exits 0 rather than failing a
host that was never meant to run it — the hard gate belongs on machines that do have
node, not on every host that happens to run the portable test suite.

`resolve_tsc` / `tsc_diagnostics` now live in `src/lsp/tsc.py` (#199's LSP-harness
reuses them — same prompt+completion -> diagnostic-codes shape its repair loop
needs, plus the persistent-scratch-dir `TscRunner` for its much hotter call rate).
This module re-exports them so existing callers (`tests/test_ts_error_eval.py`)
keep working unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from src.lsp.tsc import (DEFAULT_TSCONFIG_PATH, SET_DIR, resolve_tsc,  # noqa: F401 (re-export)
                          tsc_diagnostics)

DEFAULT_SET_PATH = SET_DIR / "eval.jsonl"


def _load_records(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", type=Path, default=DEFAULT_SET_PATH,
                         help="path to eval.jsonl")
    parser.add_argument("--tsconfig", type=Path, default=DEFAULT_TSCONFIG_PATH,
                         help="path to the pinned tsconfig.json")
    args = parser.parse_args()

    tsc_argv = resolve_tsc()
    if tsc_argv is None:
        print("validate_ts_error_set: no node/npm/tsc toolchain found on this host — "
              "skipping data-correctness validation (this is not a failure).")
        return 0

    records = _load_records(args.set)
    if not records:
        print(f"no records found in {args.set}", file=sys.stderr)
        return 1

    n_fail = 0
    for rec in records:
        rid = rec.get("id", "<no id>")
        error_class = rec.get("error_class")
        expected = rec.get("expected_diagnostic", "")

        missing = [k for k in ("prompt", "gold_completion", "error_completion") if k not in rec]
        if missing:
            print(f"FAIL {rid}: record missing keys {missing}")
            n_fail += 1
            continue

        gold_source = rec["prompt"] + rec["gold_completion"]
        gold_codes = tsc_diagnostics(gold_source, args.tsconfig, tsc_argv)
        if gold_codes:
            print(f"FAIL {rid}: gold_completion produced diagnostics {gold_codes} "
                  f"(expected zero)")
            n_fail += 1
            continue

        if error_class == "clean_control":
            print(f"PASS {rid} (clean_control, gold zero-diagnostic)")
            continue

        error_source = rec["prompt"] + rec["error_completion"]
        error_codes = tsc_diagnostics(error_source, args.tsconfig, tsc_argv)
        if expected not in error_codes:
            print(f"FAIL {rid}: error_completion produced {error_codes}, "
                  f"expected {expected} among them")
            n_fail += 1
            continue

        print(f"PASS {rid} ({error_class}, gold clean / error -> {expected})")

    total = len(records)
    print(f"\n{total - n_fail}/{total} records validated")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
