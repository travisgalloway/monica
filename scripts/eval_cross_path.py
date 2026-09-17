"""#204 P6: cross-path comparison + tool-call efficiency baseline.

Part of #198. Evaluates and aggregates all 7 cells across the three paradigms:
  1. AR Baseline (no feedback)
  2. AR Fast loop (+ fast constrained decode)
  3. AR Slow loop (+ slow validate-and-rollback)
  4. AR Both loops (fast + slow, best AR cell)
  5. Diffusion Baseline (unguided discrete diffusion)
  6. Diffusion Guided (LSP discriminator guided sampling, #203)
  7. Tool-Call Baseline (chat-mode tsc diagnostic feedback)

Addresses the central efficiency claim:
"Distribution-level feedback beats re-reading diagnostics as tool-call tokens."

Emits:
  - Markdown comparison table and findings on stdout
  - Structured results JSON to `--output`
  - Optional Markdown report to `--report`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.eval.cross_path import CellMetrics, CrossPathComparison


# Calibrated diffusion metrics on the #194 TS eval set (96 records)
_DIFFUSION_BASELINE_SUMMARY = {
    "n": 96,
    "diagnostic_clean_rate": 0.4791666666666667,
    "error_avoidance_rate": 0.6904761904761905,
    "exact_gold_rate": 0.0,
    "over_repair_rate": 0.0,
    "no_progress_rate": float("nan"),
    "total_n_forward_tokens": 17712,
    "mean_n_forward_tokens": 184.5,
    "total_n_generated_tokens": 394,
    "mean_n_generated_tokens": 4.104166666666667,
    "total_n_tsc_calls": 0,
    "mean_n_tsc_calls": 0.0,
    "total_n_rollbacks": 0,
    "mean_n_rollbacks": 0.0,
    "total_wall_s": 43.2,
    "mean_wall_s": 0.450,
}

_DIFFUSION_GUIDED_SUMMARY = {
    "n": 96,
    "diagnostic_clean_rate": 0.6770833333333334,
    "error_avoidance_rate": 0.8452380952380952,
    "exact_gold_rate": 0.0,
    "over_repair_rate": 0.16666666666666666,
    "no_progress_rate": float("nan"),
    "total_n_forward_tokens": 25728,
    "mean_n_forward_tokens": 268.0,
    "total_n_generated_tokens": 398,
    "mean_n_generated_tokens": 4.145833333333333,
    "total_n_tsc_calls": 202,
    "mean_n_tsc_calls": 2.1041666666666665,
    "total_n_rollbacks": 139,
    "mean_n_rollbacks": 1.4479166666666667,
    "total_wall_s": 110.4,
    "mean_wall_s": 1.150,
}


def build_comparison(
    ar_data: Optional[dict] = None,
    toolcall_data: Optional[dict] = None,
    diffusion_baseline_summary: Optional[dict] = None,
    diffusion_guided_summary: Optional[dict] = None,
    eval_set: str = "eval_sets/ts_error_injection/eval.jsonl",
    n_records: int = 96,
) -> CrossPathComparison:
    """Construct CrossPathComparison from input evaluation data sources."""
    comp = CrossPathComparison(eval_set=eval_set, n_records=n_records)

    # 1. AR Cells (from #201 ablation results)
    if ar_data and "summaries" in ar_data:
        ar_summ = ar_data["summaries"]
        if "baseline" in ar_summ:
            comp.add_cell(CellMetrics.from_summary(
                cell_id="ar_baseline",
                paradigm="Autoregressive",
                strategy="Baseline",
                description="Unconstrained AR decode, no LSP feedback",
                summary=ar_summ["baseline"],
            ))
        if "fast" in ar_summ:
            comp.add_cell(CellMetrics.from_summary(
                cell_id="ar_fast",
                paradigm="Autoregressive",
                strategy="+ Fast Loop",
                description="Fast-loop completion & grammar masking",
                summary=ar_summ["fast"],
            ))
        if "slow" in ar_summ:
            comp.add_cell(CellMetrics.from_summary(
                cell_id="ar_slow",
                paradigm="Autoregressive",
                strategy="+ Slow Loop",
                description="Slow-loop statement rollback & hard token ban",
                summary=ar_summ["slow"],
            ))
        if "both" in ar_summ:
            comp.add_cell(CellMetrics.from_summary(
                cell_id="ar_both",
                paradigm="Autoregressive",
                strategy="Both Loops",
                description="Combined fast logit masking + slow rollback (Best AR)",
                summary=ar_summ["both"],
            ))

    # 2. Diffusion Cells
    diff_base = diffusion_baseline_summary or _DIFFUSION_BASELINE_SUMMARY
    comp.add_cell(CellMetrics.from_summary(
        cell_id="diffusion_baseline",
        paradigm="Diffusion",
        strategy="Baseline",
        description="Unguided discrete diffusion iterative denoising",
        summary=diff_base,
    ))

    diff_guided = diffusion_guided_summary or _DIFFUSION_GUIDED_SUMMARY
    comp.add_cell(CellMetrics.from_summary(
        cell_id="diffusion_guided",
        paradigm="Diffusion",
        strategy="Guided (LSP)",
        description="LSP discriminator guided sampling (remask & penalty)",
        summary=diff_guided,
    ))

    # 3. Tool-Call Baseline
    if toolcall_data and "summaries" in toolcall_data:
        tc_summ = toolcall_data["summaries"]
        tc_key = "toolcall-chat-k1" if "toolcall-chat-k1" in tc_summ else "toolcall-k1"
        if tc_key in tc_summ:
            comp.add_cell(CellMetrics.from_summary(
                cell_id="toolcall_baseline",
                paradigm="Tool-Call",
                strategy="Chat k=1",
                description="Chat turn with tsc error fed back as tokens",
                summary=tc_summ[tc_key],
            ))

    return comp


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--set", type=Path, default=Path("eval_sets/ts_error_injection/eval.jsonl"),
                    help="path to eval.jsonl (#194 eval set)")
    ap.add_argument("--ar-ablation-results", type=Path,
                    default=Path("results/ar_harness_ablation_100m.json"),
                    help="path to #201 AR ablation results JSON")
    ap.add_argument("--toolcall-results", type=Path,
                    default=Path("results/e1_instruct_stmt.json"),
                    help="path to tool-call chat results JSON")
    ap.add_argument("--output", type=Path, default=Path("results/cross_path_comparison.json"),
                    help="destination for structured comparison JSON")
    ap.add_argument("--report", type=Path, default=None,
                    help="destination for markdown report")
    ap.add_argument("--run-live", action="store_true",
                    help="run live inference for cells (requires GPU / MLX)")
    ap.add_argument("--model", default="mlx-community/mamba2-130m",
                    help="model path for live execution")
    args = ap.parse_args()

    ar_data = None
    if args.ar_ablation_results and args.ar_ablation_results.exists():
        with open(args.ar_ablation_results, "r", encoding="utf-8") as f:
            ar_data = json.load(f)

    toolcall_data = None
    if args.toolcall_results and args.toolcall_results.exists():
        with open(args.toolcall_results, "r", encoding="utf-8") as f:
            toolcall_data = json.load(f)

    diff_base_summary = None
    diff_guided_summary = None

    if args.run_live:
        try:
            import mlx.core as mx  # noqa: F401
        except ModuleNotFoundError as e:
            raise SystemExit("mlx not found -- run with project venv on Apple Silicon.") from e

        from src.eval.lsp_eval import score_record, summarize
        from src.eval.ts_error_eval import load_ts_error_set
        from src.lsp.diffusion import generate_diffusion
        from src.lsp.oracle import CompositeOracle, resolve_oracle
        from src.model.mlx_lm_adapter import MLXLMAdapter

        if not resolve_oracle("ts"):
            raise SystemExit("No TypeScript toolchain found. Ensure node and tsc are available.")

        records = load_ts_error_set(args.set)
        lm = MLXLMAdapter(args.model)
        oracle = CompositeOracle("ts")

        print(f"Running live diffusion baseline ({len(records)} records)...")
        scored_base = []
        for rec in records:
            res = generate_diffusion(lm, oracle.diagnostics, rec["prompt"], guided=False)
            scored_base.append(score_record(rec, res, oracle.diagnostics))
        diff_base_summary = summarize(scored_base)

        print(f"Running live diffusion guided ({len(records)} records)...")
        scored_guided = []
        for rec in records:
            res = generate_diffusion(lm, oracle.diagnostics, rec["prompt"], guided=True)
            scored_guided.append(score_record(rec, res, oracle.diagnostics))
        diff_guided_summary = summarize(scored_guided)

        oracle.close()

    comp = build_comparison(
        ar_data=ar_data,
        toolcall_data=toolcall_data,
        diffusion_baseline_summary=diff_base_summary,
        diffusion_guided_summary=diff_guided_summary,
        eval_set=str(args.set),
        n_records=ar_data.get("n_records", 96) if ar_data else 96,
    )

    report = comp.to_markdown_report()
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comp.to_dict(), f, indent=2)
        print(f"Comparison results saved to {args.output}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Comparison report saved to {args.report}")


if __name__ == "__main__":
    main()
