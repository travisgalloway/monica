#!/usr/bin/env python3
"""Build the real-code generalization set (#199 F1) from MultiPL-E HumanEval-TS.

Every result in PR #207 is on the #194 error-injection set: single-statement, hand-authored
completions where the error is one local token at the generation frontier — the case that most
favours hard-ban. This set is the opposite: 159 real function-generation problems (HumanEval
translated to TypeScript by MultiPL-E), where the model writes a whole multi-line body and makes
its own natural errors, many of them *non-local* (a wrong return type flagged far from its cause,
which a frontier-only logit ban cannot reach). It exists to answer the one question that decides
whether M12 funds training: does the hard-ban lift survive off the toy eval?

WHY HUMANEVAL-TS SPECIFICALLY. It is **self-contained** — a completed body compiles clean under
the pinned tsconfig with no imports, so none of E4's `TS2307` module-resolution confound — and it
ships per-problem `tests` (for the functional `pass@1` guard) and `stop_tokens` (to cut the body).

SCHEMA. Records satisfy the #194 7-key schema so `src/eval/lsp_eval.py::score_record` scores them
unchanged: `error_class="clean_control"` (so `error_avoidance_rate` ignores them — there is no
injected error here, only natural ones), a placeholder `gold_completion`, empty `error_completion`.
Three extra keys ride along for F1's driver — `name`, `tests`, `stop_tokens` — and pass straight
through both the scorer (reads only the 7) and `validate_record` (checks presence, not absence).

    .venv/bin/python scripts/build_humaneval_ts_set.py \
        --out eval_sets/humaneval_ts/humaneval_ts.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path("eval_sets/humaneval_ts/humaneval_ts.jsonl"))
    ap.add_argument("--config", default="humaneval-ts",
                    help="MultiPL-E config (default: humaneval-ts; mbpp-ts also valid)")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"loading nuprl/MultiPL-E {args.config} ...")
    ds = load_dataset("nuprl/MultiPL-E", args.config, split="test")

    records = []
    for row in ds:
        records.append({
            # #194 schema (load-bearing for score_record / validate_record)
            "id": row["name"],
            "error_class": "clean_control",        # no injected error; natural errors only
            "expected_diagnostic": "",
            "prompt": row["prompt"],               # signature + docstring, open body
            "gold_completion": "\n",               # placeholder — no gold body ships; clean-rate
                                                    # + pass@1 are the metrics, not gold-match
            "error_completion": "",
            "notes": f"MultiPL-E {args.config} {row['name']}; self-contained, real generation",
            # F1 extras (ignored by the scorer, used by the driver)
            "name": row["name"],
            "tests": row["tests"],
            "stop_tokens": row["stop_tokens"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
