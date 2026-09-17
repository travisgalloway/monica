"""#201 AR harness ablation across 4 cells (fast / slow / both loops).

Part of #198. Evaluates the trained ~100M model (default: `mlx-community/mamba2-130m`)
on the #194 TS error-injection eval set across the 4 cells:
  1. Baseline (no LSP feedback)
  2. + fast-loop constrained decode (tree-sitter grammar mask + cached tsserver completion bias)
  3. + slow-loop validate-and-rollback (checkpoint stack, tsc check, hard repair)
  4. Both loops together

Emits:
  - Markdown ablation table on stdout (diagnostic-clean rate, error-induced pass rate)
  - `--output` JSON with per-cell summaries, pairwise comparisons against baseline,
    rollback path counts, and mask step counters.
  - `--transcript` JSONL with per-record contexts, artifacts, and repair events.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/mamba2-130m",
                    help="model path or HF repo (default: trained ~100M mamba2-130m)")
    ap.add_argument("--dtype", default=None,
                    help="upcast dtype (default None = model native bf16)")
    ap.add_argument("--set", type=Path, default=Path("eval_sets/ts_error_injection/eval.jsonl"),
                    help="path to eval.jsonl (default: the #194 eval set)")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of records")
    ap.add_argument("--budget", choices=("stmt", "block"), default="stmt")
    ap.add_argument("--block-size", type=int, default=96)
    ap.add_argument("--max-gen-tokens", type=int, default=200)
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--rollback-strategy", choices=("auto", "trim", "reprefill", "snapshot"),
                    default="auto")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-scope", choices=("member", "identifier"), default="member")
    ap.add_argument("--output", type=Path, default=None, help="results JSON destination")
    ap.add_argument("--transcript", type=Path, default=None, help="results JSONL destination")
    args = ap.parse_args()

    try:
        import mlx.core as mx  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit("mlx not found -- run with project venv on Apple Silicon.") from e

    from src.eval.lsp_eval import compare, score_record, summarize
    from src.eval.ts_error_eval import load_ts_error_set
    from src.lsp.completion_mask import CompletionMasker, LspLabels
    from src.lsp.harness import generate_slow_loop
    from src.lsp.oracle import CompositeOracle, resolve_oracle
    from src.lsp.ts_service import TsLspService
    from src.model.mlx_lm_adapter import MLXLMAdapter

    if not resolve_oracle("ts"):
        raise SystemExit("No TypeScript toolchain found. Ensure node and tsc are available.")

    records = load_ts_error_set(args.set)
    if args.limit is not None:
        records = records[: args.limit]

    print(f"model={args.model} records={len(records)} budget={args.budget} temp={args.temperature}")

    t0 = time.monotonic()
    lm = MLXLMAdapter(args.model, dtype=args.dtype, rollback_strategy=args.rollback_strategy)
    oracle = CompositeOracle("ts")
    service = TsLspService()
    service.open_project({f"rec_{i}.ts": rec["prompt"] for i, rec in enumerate(records)})
    print(f"model + oracle + mask service ready in {time.monotonic() - t0:.1f}s")

    rng = np.random.default_rng(args.seed) if args.temperature > 0 else None

    transcript_f = None
    if args.transcript:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript_f = open(args.transcript, "w", encoding="utf-8")

    cells = [
        ("baseline", "none", False),
        ("fast", "none", True),
        ("slow", "hard", False),
        ("both", "hard", True),
    ]

    gen_kwargs = dict(
        budget=args.budget,
        block_size=args.block_size,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        rng=rng,
    )

    scored_by_cell: Dict[str, List[dict]] = {}
    summaries: Dict[str, dict] = {}
    t_run = time.monotonic()

    for cell_name, repair_mode, use_masker in cells:
        print(f"Running cell: {cell_name} (repair={repair_mode}, mask={use_masker})")
        scored: List[dict] = []
        for i, rec in enumerate(records):
            path = f"rec_{i}.ts"
            masker = None
            if use_masker:
                labels_source = LspLabels(service, path)
                masker = CompletionMasker(labels_source, path, lm.decode,
                                          mask_scope=args.mask_scope, encode=lm.encode)

            result = generate_slow_loop(
                lm,
                oracle.diagnostics,
                rec["prompt"],
                repair=repair_mode,
                masker=masker,
                strategy=cell_name,
                max_retries=args.max_retries,
                **gen_kwargs,
            )

            s = score_record(rec, result, oracle.diagnostics)
            scored.append(s)

            if transcript_f:
                transcript_f.write(json.dumps({
                    "cell": cell_name,
                    "id": rec["id"],
                    "error_class": rec["error_class"],
                    "prompt": rec["prompt"],
                    "artifact": result.artifact,
                    "completion": result.completion,
                    "codes": s["codes"],
                    "clean": s["clean"],
                    "avoided": s["avoided"],
                    "n_rollbacks": result.n_rollbacks,
                    "n_mask_steps": result.n_mask_steps,
                    "events": result.events,
                }) + "\n")

            if (i + 1) % 16 == 0 or (i + 1) == len(records):
                print(f"  {cell_name}: {i + 1}/{len(records)}")

        scored_by_cell[cell_name] = scored
        summary = summarize(scored)
        summary["cell"] = cell_name
        summary["repair"] = repair_mode
        summary["masker_active"] = use_masker
        summary["rollback_paths"] = {
            "trim": lm.n_trim_rollbacks,
            "reprefill": lm.n_reprefill_rollbacks,
            "snapshot": lm.n_snapshot_rollbacks,
            "reprefill_tokens": lm.n_reprefill_tokens,
        }
        summaries[cell_name] = summary
        print(f"[{cell_name}] clean={summary['diagnostic_clean_rate']:.3f} "
              f"avoid={summary['error_avoidance_rate']:.3f} "
              f"exact_gold={summary['exact_gold_rate']:.3f} "
              f"over_repair={summary['over_repair_rate']:.3f} "
              f"mean_fwd_tok={summary['mean_n_forward_tokens']:.1f}")

    if transcript_f:
        transcript_f.close()

    baseline_scored = scored_by_cell["baseline"]
    comparisons = {}
    for cell_name in ["fast", "slow", "both"]:
        cell_scored = scored_by_cell[cell_name]
        err_base = [s for s in baseline_scored if s["is_error_row"]]
        err_cell = [s for s in cell_scored if s["is_error_row"]]
        comparisons[cell_name] = {
            "avoided_vs_baseline": compare(err_base, err_cell, key="avoided"),
            "clean_vs_baseline": compare(baseline_scored, cell_scored, key="clean"),
        }

    results = {
        "model": args.model,
        "dtype": args.dtype,
        "set": str(args.set),
        "n_records": len(records),
        "budget": args.budget,
        "temperature": args.temperature,
        "seed": args.seed,
        "rollback_strategy": args.rollback_strategy,
        "summaries": summaries,
        "comparisons": comparisons,
        "wall_s_total": time.monotonic() - t_run,
    }

    # Print Acceptance Ablation Table
    print("\n### #201 AR Harness Ablation Table (4 Cells)")
    print(f"Model: `{args.model}` | Eval Set: `{args.set}` ({len(records)} records)\n")
    print("| Cell | Strategy | Diagnostic-Clean Rate | Error-Induced Pass (Avoid) | Exact Gold | Over-Repair | Mean Fwd Tokens | McNemar p (Clean) | McNemar p (Avoid) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cell_name in ["baseline", "fast", "slow", "both"]:
        summ = summaries[cell_name]
        if cell_name in comparisons:
            p_clean = comparisons[cell_name]["clean_vs_baseline"]["mcnemar_p"]
            p_avoid = comparisons[cell_name]["avoided_vs_baseline"]["mcnemar_p"]
            p_clean_str = f"{p_clean:.4f}"
            p_avoid_str = f"{p_avoid:.4f}"
        else:
            p_clean_str = "--"
            p_avoid_str = "--"

        strat_desc = {
            "baseline": "Baseline (no LSP feedback)",
            "fast": "+ fast-loop constrained decode",
            "slow": "+ slow-loop validate-and-rollback",
            "both": "Both loops together",
        }[cell_name]

        print(f"| {cell_name} | {strat_desc} | "
              f"{summ['diagnostic_clean_rate']:.3f} | "
              f"{summ['error_avoidance_rate']:.3f} | "
              f"{summ['exact_gold_rate']:.3f} | "
              f"{summ['over_repair_rate']:.3f} | "
              f"{summ['mean_n_forward_tokens']:.1f} | "
              f"{p_clean_str} | {p_avoid_str} |")

    service.close()
    oracle.close()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nresults saved to {args.output}")


if __name__ == "__main__":
    main()
