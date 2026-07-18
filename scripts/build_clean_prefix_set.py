#!/usr/bin/env python3
"""Build the over-repair probe set (#199 E4) from real third-party TypeScript.

WHY THIS EXISTS. `over_repair_rate` — how often the repair loop interrupts code that was
already correct — is currently measured on **12 hand-authored `clean_control` rows**, and
only 8 of them appear in the block-budget subset. The alarming "0.50 over-repair" in the
Phase-0 tables is therefore literally *4 out of 8*. That is an anecdote, not a rate, and
over-repair is the one metric that could disqualify the whole mechanism: a repair loop
that corrupts working code is worse than no repair loop.

So: sample real TypeScript, keep only files that **already compile clean** under the eval
set's pinned `tsconfig.json`, and cut them at statement boundaries. Every resulting prefix
has a known-good natural continuation, so *any* rollback the loop fires on one is by
construction an interruption of correct code.

TWO THINGS THAT MAKE OR BREAK THIS SET:

1. **Module resolution.** Real TS files `import` things. Under the isolated eval tsconfig
   those raise `TS2307: Cannot find module` etc., which would mark essentially every file
   "dirty" and leave an empty set. So we KEEP import-bearing files and ignore the
   module-resolution family (`MODULE_RESOLUTION_CODES`) **consistently** — here at file
   selection, and (the #201 fix) also in the eval's in-loop diagnose and scoring
   (`--ignore-module-resolution`). The tradeoff is explicit: an unresolved import makes its
   symbols `any`, so checking of code that touches them is weaker — these prefixes therefore
   under-detect errors rather than inventing them, the safe direction for an OVER-repair
   probe whose question is "does the loop interrupt code tsc considers fine?".
2. **True top-level cuts.** A prefix must end after a *complete* top-level statement, or
   `prompt+completion` is a fragment of an unclosed construct that can't compile in isolation
   (`TS1005`/`TS1109`) and every rollback on it is spurious (E4 confound (b)). We cut with a
   real parser (`src.lsp.ts_boundaries`, tree-sitter), never a brace-depth newline scan.
3. **Self-containment.** `declare`/`/// <reference>` and ambient DOM types (`document`,
   `window`) — the pinned tsconfig is `lib: ["ES2020"]`, no DOM.

PROVENANCE / POLICY. Source is `bigcode/the-stack-smol` (permissively-licensed third-party
code), *not* Claude-generated. This repo's standing rule is that Claude-generated text must
never become training signal for the student; these are eval prefixes rather than training
data, but the rule's spirit is worth honoring and the third-party source honors it outright.

    .venv/bin/python scripts/build_clean_prefix_set.py --n 200 \
        --out eval_sets/ts_error_injection/clean_prefixes.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lsp.diagnostics import MODULE_RESOLUTION_CODES  # noqa: E402
from src.lsp.ts_boundaries import first_boundary_in_range, tree_sitter_available  # noqa: E402
from src.lsp.tsc import TscRunner, resolve_tsc  # noqa: E402

# Still un-compilable in isolation regardless of imports: ambient DOM types (the pinned
# tsconfig is lib: ["ES2020"], no DOM) and triple-slash references.
_NEEDS_DOM = re.compile(r"\b(document|window|localStorage|HTMLElement|navigator)\b")
_TRIPLE_SLASH = re.compile(r"^\s*///\s*<reference", re.M)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="how many prefixes to emit")
    ap.add_argument("--out", type=Path,
                    default=Path("eval_sets/ts_error_injection/clean_prefixes.jsonl"))
    ap.add_argument("--max-files", type=int, default=1500,
                    help="how many source files to scan for clean candidates")
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=1200)
    args = ap.parse_args()

    tsc_argv = resolve_tsc()
    if tsc_argv is None:
        raise SystemExit("no node/tsc toolchain — run `npm install` in "
                         "eval_sets/ts_error_injection first.")
    if not tree_sitter_available():
        raise SystemExit("tree-sitter not installed — run `pip install -e \".[tree-sitter]\"` "
                         "(needed for true top-level statement boundaries).")

    from datasets import load_dataset

    print("streaming bigcode/the-stack-smol (TypeScript)...")
    ds = load_dataset("bigcode/the-stack-smol", data_dir="data/typescript",
                      split="train", streaming=True)

    records, n_seen, n_skipped_shape, n_skipped_dirty, n_skipped_no_cut = [], 0, 0, 0, 0
    with TscRunner(tsc_argv) as tsc:
        for row in ds:
            if len(records) >= args.n or n_seen >= args.max_files:
                break
            n_seen += 1
            src = row["content"]

            if (_TRIPLE_SLASH.search(src) or _NEEDS_DOM.search(src)
                    or not src.isascii() or len(src) > 6000):
                n_skipped_shape += 1
                continue

            # The load-bearing filter: only files that ALREADY compile clean (ignoring
            # unresolved-module noise) can serve as an over-repair probe. If the file is
            # genuinely broken to begin with, a rollback on it isn't over-repair — it
            # might be a correct repair, and the metric would mean nothing.
            if [c for c in tsc.codes(src) if c not in MODULE_RESOLUTION_CODES]:
                n_skipped_dirty += 1
                continue

            # One prefix per file, cut at a TRUE top-level statement boundary (tree-sitter)
            # so the continuation is a fresh statement, never a fragment of an unclosed
            # construct. Keeps the set diverse across repos.
            b = first_boundary_in_range(src, args.min_chars, args.max_chars)
            if b is None:
                n_skipped_no_cut += 1
                continue
            records.append({
                "id": f"clean-prefix-{len(records):04d}",
                "error_class": "clean_control",   # reuse the #194 schema
                "expected_diagnostic": "",
                "prompt": src[:b],
                "gold_completion": "\n",          # any tsc-clean continuation is correct
                "error_completion": "",
                "notes": f"real TS, {row.get('max_stars_repo_name', '?')}; "
                         f"file compiles clean under the pinned tsconfig; "
                         f"cut at a tree-sitter top-level boundary",
            })
            if len(records) % 25 == 0 and records:
                print(f"  {len(records)}/{args.n} prefixes ({n_seen} files scanned)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nscanned {n_seen} files: {n_skipped_shape} not self-contained, "
          f"{n_skipped_dirty} already had diagnostics, "
          f"{n_skipped_no_cut} had no top-level boundary in range")
    print(f"wrote {len(records)} clean prefixes -> {args.out}")
    if len(records) < args.n:
        print(f"NOTE: fell short of --n {args.n}; raise --max-files to scan more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
