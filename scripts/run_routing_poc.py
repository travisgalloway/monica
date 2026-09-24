#!/usr/bin/env python3
"""Small MoE routing diagnostics & domain specialization verification runner (#423).

Scope:
  - Extract routing histograms across TypeScript, prose, and math batches using `src/eval/moe_routing.py`.
  - Evaluate pairwise routing overlap against the kill-criterion threshold (overlap < 0.90).
  - Ensure no expert starvation or uniform routing collapse occurred during training.
  - Verify acceptance criteria:
    - Routing specialization report indicates PASSED (TypeScript vs Math overlap < 0.90).
    - Mid-training routing kill-check verified negative.

Usage:
  # Run routing diagnostics and domain specialization verification:
  python scripts/run_routing_poc.py

  # Run offline mock pipeline simulation and write results:
  python scripts/run_routing_poc.py --mock-pipeline --output results/routing_specialization_poc.json

  # Verify failure detection:
  python scripts/run_routing_poc.py --mock-pipeline --simulate-kill
  python scripts/run_routing_poc.py --mock-pipeline --simulate-starvation
  python scripts/run_routing_poc.py --mock-pipeline --simulate-collapse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.routing_poc_run import (
    DEFAULT_COLLAPSE_THRESHOLD,
    DEFAULT_EVAL_SETS_DIR,
    DEFAULT_KILL_PAIR,
    DEFAULT_KILL_THRESHOLD,
    DEFAULT_MOE_CONFIG,
    RoutingPOCRunConfig,
    run_routing_diagnostics_poc,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_MOE_CONFIG,
                    help=f"Target MoE model config YAML (default: {DEFAULT_MOE_CONFIG.name})")
    ap.add_argument("--eval-sets", type=Path, default=DEFAULT_EVAL_SETS_DIR,
                    help=f"Path to eval_sets directory (default: {DEFAULT_EVAL_SETS_DIR.name})")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Batch size for routing probe forward passes (default: 4)")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="Sequence length for token batches (default: 256)")
    ap.add_argument("--max-batches", type=int, default=8,
                    help="Maximum batches per domain to evaluate (default: 8)")
    ap.add_argument("--kill-threshold", type=float, default=DEFAULT_KILL_THRESHOLD,
                    help=f"Routing overlap kill-criterion threshold (default: {DEFAULT_KILL_THRESHOLD})")
    ap.add_argument("--collapse-threshold", type=float, default=DEFAULT_COLLAPSE_THRESHOLD,
                    help=f"Uniform routing collapse threshold (default: {DEFAULT_COLLAPSE_THRESHOLD})")
    ap.add_argument("--mock-pipeline", action="store_true",
                    help="Run deterministic offline simulation and acceptance verification")
    ap.add_argument("--simulate-kill", action="store_true",
                    help="In simulation, force kill-criterion trigger failure")
    ap.add_argument("--simulate-starvation", action="store_true",
                    help="In simulation, force expert starvation failure")
    ap.add_argument("--simulate-collapse", action="store_true",
                    help="In simulation, force uniform routing collapse failure")
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "routing_specialization_poc.json",
                    help="Output JSON file path (default: results/routing_specialization_poc.json)")
    ap.add_argument("--seed", type=int, default=423,
                    help="Random seed for deterministic runs (default: 423)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    cfg = RoutingPOCRunConfig(
        moe_config_path=args.config,
        eval_sets_dir=args.eval_sets,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_batches=args.max_batches,
        kill_threshold=args.kill_threshold,
        max_collapse_threshold=args.collapse_threshold,
        seed=args.seed,
    )

    results = run_routing_diagnostics_poc(
        config=cfg,
        simulate_kill=args.simulate_kill,
        simulate_starvation=args.simulate_starvation,
        simulate_collapse=args.simulate_collapse,
    )

    # Print human table to stdout
    print(results["formatted_table"])
    print()
    print("=" * 72)
    print(results["acceptance"]["summary"])
    print("=" * 72)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclude raw ndarray batches from output JSON
        serializable = {
            "model_spec": results["model_spec"],
            "evaluation_config": results["evaluation_config"],
            "histograms": results["histograms"],
            "specialization_report": results["specialization_report"],
            "verification": results["verification"],
            "acceptance": results["acceptance"],
        }
        args.output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"\nDiagnostics report saved to: {args.output}")

    return 0 if results["acceptance"]["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
