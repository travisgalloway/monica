#!/usr/bin/env python3
"""Small-model ablation sweep runner (#219).

Executes the ablation sweep grid across:
  - attention ratio: 8%, 12%, 16% (via arbitrary-depth placement)
  - d_state: 128 vs 256
  - mixture variant: Jamba-style vs Routing-Mamba

Evaluates each candidate on recall, applies the routing kill-criterion,
and outputs the winning configuration.

Usage:
  # List all 12 configurations in the grid:
  python scripts/ablation_sweep.py --list-grid

  # Run deterministic offline sweep:
  python scripts/ablation_sweep.py --eval-mode mock

  # Export all candidate YAML configs to a directory:
  python scripts/ablation_sweep.py --export-dir config/ablations/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
from src.model.blocks import load_config
from src.eval.ablation_sweep import (
    AblationSweepRunner,
    default_small_base_config,
    format_sweep_table,
    generate_ablation_grid,
    mock_evaluator,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", type=Path, default=None,
                    help="Base model config YAML (default: config/code-small-dense.yaml if present)")
    ap.add_argument("--list-grid", action="store_true",
                    help="Print all candidate configurations in the grid and exit")
    ap.add_argument("--export-dir", type=Path, default=None,
                    help="Directory to dump candidate YAML configurations into")
    ap.add_argument("--eval-mode", choices=("mock",), default="mock",
                    help="Evaluation backend mode (default: mock deterministic fixture)")
    ap.add_argument("--kill-threshold", type=float, default=0.90,
                    help="Routing kill-criterion overlap threshold (default: 0.90)")
    ap.add_argument("--simulate-kill", type=str, default=None,
                    help="Comma-separated candidate names to trigger routing kill for")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional JSON file path to write results into")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for evaluation")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    base_path = args.base_config or (REPO_ROOT / "config" / "code-small-moe.yaml")
    if base_path.exists():
        base_cfg = load_config(base_path)
    else:
        base_cfg = default_small_base_config()

    if args.list_grid:
        candidates = generate_ablation_grid(base_cfg)
        print(f"Ablation Grid ({len(candidates)} configurations):\n")
        for i, c in enumerate(candidates, 1):
            cfg = c.config
            print(f"{i:2d}. {c.name:<32} variant={c.variant:<14} "
                  f"attn={c.attn_ratio*100:4.1f}% ({cfg.n_attention_layers}L) "
                  f"d_state={c.d_state:<4} params={cfg.num_parameters()/1e6:6.1f}M "
                  f"active={cfg.active_num_parameters()/1e6:6.1f}M")
            print(f"    attn_layers = {cfg.attn_layers}")
        return

    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        candidates = generate_ablation_grid(base_cfg)
        for c in candidates:
            out_file = args.export_dir / f"{c.name}.yaml"
            with open(out_file, "w") as f:
                yaml.safe_dump(c.config.to_dict(), f, sort_keys=False)
        print(f"Exported {len(candidates)} configuration files to {args.export_dir}")
        return

    runner = AblationSweepRunner(
        base_cfg=base_cfg,
        kill_threshold=args.kill_threshold,
    )

    kill_list = [k.strip() for k in args.simulate_kill.split(",")] if args.simulate_kill else None

    def _eval(cand):
        return mock_evaluator(cand, seed=args.seed, simulate_kill_for=kill_list)

    results, winner = runner.run(_eval)

    print("\n" + "=" * 70)
    print("SMALL-MODEL ABLATION SWEEP RESULTS (#219)")
    print("=" * 70 + "\n")
    print(format_sweep_table(results, winner))
    print("\n" + "-" * 70)
    print(f"WINNING CONFIGURATION: {winner.candidate.name}")
    print(f"  Variant:          {winner.candidate.variant}")
    print(f"  Attention Ratio:  {winner.candidate.attn_ratio*100:.1f}% ({winner.candidate.config.n_attention_layers} layers)")
    print(f"  d_state:          {winner.candidate.d_state}")
    print(f"  Recall Score:     {winner.recall_score:.4f}")
    if winner.kill_result:
        print(f"  Routing Overlap:  {winner.kill_result['overlap']:.4f} (threshold {winner.kill_result['threshold']:.2f})")
    print("-" * 70 + "\n")

    if args.output:
        out_data = {
            "winner": {
                "name": winner.candidate.name,
                "variant": winner.candidate.variant,
                "attn_ratio": winner.candidate.attn_ratio,
                "d_state": winner.candidate.d_state,
                "recall_score": winner.recall_score,
                "config": winner.candidate.config.to_dict(),
            },
            "results": [
                {
                    "rank": r.rank,
                    "name": r.candidate.name,
                    "variant": r.candidate.variant,
                    "attn_ratio": r.candidate.attn_ratio,
                    "d_state": r.candidate.d_state,
                    "recall_score": r.recall_score,
                    "killed": r.killed,
                    "kill_result": r.kill_result,
                }
                for r in results
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
