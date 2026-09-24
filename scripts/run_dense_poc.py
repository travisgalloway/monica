#!/usr/bin/env python3
"""Dense baseline pretraining run (2B tokens) runner & upcycle verification (#421).

Coordinates:
  - Dense baseline model pretraining (config/code-small-dense.yaml, ~232.1M parameters)
  - 2B token tranche training budget, WSD schedule, and double-buffered checkpointing
  - Cloud compute / RunPod specifications, hardware tiers, and cost estimates
  - Acceptance verification on decreasing BPB curve, Slot-A and Slot-B checkpoint bundles,
    and 15/15 upcycle compatibility dry-run against config/code-small-moe.yaml

Usage:
  # 1. Print Cloud Compute / RunPod launch specification and cost breakdown:
  python scripts/run_dense_poc.py --cloud-spec

  # 2. Print canonical scripts/train.py training command:
  python scripts/run_dense_poc.py --print-train-cmd

  # 3. Dry-run upcycle compatibility against Small MoE target:
  python scripts/run_dense_poc.py --dry-run

  # 4. Run deterministic offline mock pipeline simulation & verify acceptance:
  python scripts/run_dense_poc.py --mock-pipeline

  # 5. Verify an existing run directory:
  python scripts/run_dense_poc.py --verify-run runs/dense_poc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.dense_poc_run import (
    CLOUD_HARDWARE_TEMPLATES,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_DENSE_CONFIG,
    DEFAULT_MOE_CONFIG,
    DensePOCRunConfig,
    get_cloud_run_spec,
    simulate_mock_pipeline,
    verify_dense_poc_acceptance,
    verify_upcycle_dry_run,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense-config", type=Path, default=DEFAULT_DENSE_CONFIG,
                    help=f"Source dense config YAML (default: {DEFAULT_DENSE_CONFIG.name})")
    ap.add_argument("--moe-config", type=Path, default=DEFAULT_MOE_CONFIG,
                    help=f"Target MoE model config YAML (default: {DEFAULT_MOE_CONFIG.name})")
    ap.add_argument("--total-tokens", type=int, default=DEFAULT_BUDGET_TOKENS,
                    help=f"Total training token budget (default: {DEFAULT_BUDGET_TOKENS:,})")
    ap.add_argument("--hardware-tier", choices=list(CLOUD_HARDWARE_TEMPLATES.keys()), default="a40",
                    help="Hardware tier for cloud compute modeling (default: a40)")
    ap.add_argument("--cloud-spec", action="store_true",
                    help="Print Cloud Compute / RunPod specifications and cost estimates, then exit")
    ap.add_argument("--print-train-cmd", action="store_true",
                    help="Print the canonical scripts/train.py training command, then exit")
    ap.add_argument("--data-dir", type=str, default="data/split",
                    help="Training packed shard directory (default: data/split)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Verify configs and upcycle compatibility without executing paid runs")
    ap.add_argument("--mock-pipeline", action="store_true",
                    help="Run deterministic offline simulation of 2B training plan, checkpoints, and acceptance")
    ap.add_argument("--verify-run", type=Path, default=None,
                    help="Verify an existing run output directory against acceptance criteria")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "dense_poc",
                    help="Output directory for runs and checkpoints (default: runs/dense_poc)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional JSON file path to write results into")
    ap.add_argument("--seed", type=int, default=421,
                    help="Random seed for deterministic runs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    run_cfg = DensePOCRunConfig(
        dense_config_path=args.dense_config,
        moe_config_path=args.moe_config,
        output_dir=args.out_dir,
        total_tokens=args.total_tokens,
        seed=args.seed,
    )

    if args.cloud_spec:
        spec = get_cloud_run_spec(total_tokens=args.total_tokens, hardware=args.hardware_tier)
        print("=" * 70)
        print("DENSE BASELINE RUN (#421) — CLOUD COMPUTE SPECIFICATION")
        print("=" * 70)
        print(f"Intended Use:       {spec['intended_use']}\n")
        print(f"Hardware Tier:      {spec['hardware_configuration']['gpu_model']} "
              f"({spec['hardware_configuration']['vram_gb']}GB VRAM, {spec['hardware_configuration']['cloud_type']})")
        print(f"Single GPU Fit:     {spec['hardware_configuration']['single_gpu']} (no FSDP required)")
        print(f"Estimated Runtime:  {spec['estimated_runtime']}")
        print(f"Hourly Rate:        ${spec['approx_hourly_rate_usd']:.2f}/hr")
        print(f"Estimated Cost:     {spec['estimated_cost']}")
        print("-" * 70)
        print("RunPod Launch Command:")
        print(f"  {spec['runpod_launch_command']}")
        print("-" * 70)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(spec, indent=2))
        return

    if args.print_train_cmd:
        cmd = run_cfg.generate_train_command(args.data_dir)
        print(" ".join(cmd))
        return

    if args.dry_run:
        from src.model.blocks import load_config
        from src.train.upcycle import check_upcycle_compatible
        src = load_config(str(args.dense_config))
        dst = load_config(str(args.moe_config))
        src.validate()
        dst.validate()
        check_upcycle_compatible(src, dst)
        print(f"[dry-run] Upcycle compatibility confirmed between {args.dense_config.name} and {args.moe_config.name} (15/15 fields match).")
        return

    if args.verify_run:
        res = verify_dense_poc_acceptance(args.verify_run, moe_config_path=args.moe_config)
        print("=" * 70)
        print(f"VERIFY RUN: {args.verify_run}")
        print("=" * 70)
        print(f"Status: {res['summary']}")
        print(f"BPB Check: {res['bpb_check']['message']}")
        print(f"Bundles:   {res['bundle_check']['message']}")
        print(f"Upcycle:   {res['upcycle_check']['message']}")
        print("=" * 70)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(res, indent=2))
        if not res["accepted"]:
            sys.exit(1)
        return

    if args.mock_pipeline:
        print("=" * 70)
        print("DENSE BASELINE PRETRAINING RUN (#421) — DETERMINISTIC MOCK PIPELINE")
        print("=" * 70)
        res = simulate_mock_pipeline(
            output_dir=args.out_dir,
            total_tokens=args.total_tokens,
            seed=args.seed,
        )
        acc = res["acceptance"]
        print(f"Token Budget:     {res['run_config']['total_tokens']:,} tokens")
        print(f"Total Steps:      {res['run_config']['total_steps']} steps (tokens/step={res['run_config']['tokens_per_step']:,})")
        print(f"Step Breakdown:   warmup={res['run_config']['step_breakdown']['warmup_steps']}, "
              f"stable={res['run_config']['step_breakdown']['stable_steps']}, "
              f"decay={res['run_config']['step_breakdown']['decay_steps']}")
        print(f"BPB Reduction:    {res['metrics_summary']['initial_bpb']:.4f} -> {res['metrics_summary']['final_bpb']:.4f} "
              f"({res['metrics_summary']['eval_points']} eval points, monotonic)")
        print(f"Bundles:          {acc['bundle_check']['message']}")
        print(f"Upcycle Dry-Run:  {acc['upcycle_check']['message']}")
        print("-" * 70)
        print(f"Acceptance Summary: {acc['summary']}")
        print("=" * 70)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(res, indent=2))
        if not acc["accepted"]:
            sys.exit(1)
        return

    print("Please specify an action: --cloud-spec, --print-train-cmd, --dry-run, --mock-pipeline, or --verify-run <path>")
    sys.exit(1)


if __name__ == "__main__":
    main()
