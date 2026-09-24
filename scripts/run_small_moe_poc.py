#!/usr/bin/env python3
"""Small MoE model (code-small-moe.yaml) sparse upcycle & initial training runner (#422).

Coordinates:
  - Sparse upcycle from dense POC checkpoint to Small MoE target (685.1M params, 8 routed experts,
    top-2 dropless routing, 1 shared expert with zero-down init, moe_balance_rate: 0.001)
  - Initial training budget calculation, WSD schedule, and double-buffered checkpointing
  - Cloud compute / RunPod specifications, hardware tiers (NVIDIA A40), and cost estimates
  - Acceptance verification on bit-exact resume at step 50 and router stability without expert collapse

Usage:
  # 1. Print Cloud Compute / RunPod launch specification and cost breakdown:
  python scripts/run_small_moe_poc.py --cloud-spec

  # 2. Print canonical scripts/train.py training command:
  python scripts/run_small_moe_poc.py --print-train-cmd

  # 3. Dry-run upcycle compatibility against Small MoE target:
  python scripts/run_small_moe_poc.py --dry-run

  # 4. Perform upcycle from dense checkpoint:
  python scripts/run_small_moe_poc.py --upcycle --dense-checkpoint runs/dense_poc/weights.safetensors \
      --upcycle-out runs/small_moe_poc/small_moe_init.safetensors

  # 5. Run deterministic offline mock pipeline simulation & verify acceptance:
  python scripts/run_small_moe_poc.py --mock-pipeline

  # 6. Verify an existing run directory:
  python scripts/run_small_moe_poc.py --verify-run runs/small_moe_poc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.small_moe_poc_run import (
    CLOUD_HARDWARE_TEMPLATES,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_DENSE_CONFIG,
    DEFAULT_MOE_CONFIG,
    DEFAULT_POC_RESUME_STEP,
    SmallMoEPOCRunConfig,
    execute_sparse_upcycle,
    get_cloud_run_spec,
    simulate_mock_pipeline,
    verify_small_moe_poc_acceptance,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_MOE_CONFIG,
                    help=f"Target MoE model config YAML (default: {DEFAULT_MOE_CONFIG.name})")
    ap.add_argument("--dense-config", type=Path, default=DEFAULT_DENSE_CONFIG,
                    help=f"Source dense config YAML (default: {DEFAULT_DENSE_CONFIG.name})")
    ap.add_argument("--dense-checkpoint", type=Path, default=None,
                    help="Path to source dense checkpoint (.safetensors) to upcycle from")
    ap.add_argument("--upcycle-out", type=Path, default=None,
                    help="Path to save upcycled MoE checkpoint (.safetensors)")
    ap.add_argument("--upcycle", action="store_true",
                    help="Execute upcycle transform from --dense-checkpoint to --upcycle-out")
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
                    help="Run deterministic offline simulation of upcycle, training plan, step-50 resume, and acceptance gate")
    ap.add_argument("--simulate-collapse", action="store_true",
                    help="In --mock-pipeline, simulate router collapse failure")
    ap.add_argument("--verify-run", type=Path, default=None,
                    help="Verify an existing run output directory against acceptance criteria")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "small_moe_poc",
                    help="Output directory for runs and checkpoints (default: runs/small_moe_poc)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional JSON file path to write results into")
    ap.add_argument("--seed", type=int, default=422,
                    help="Random seed for deterministic runs (default: 422)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    run_cfg = SmallMoEPOCRunConfig(
        moe_config_path=args.config,
        dense_config_path=args.dense_config,
        dense_checkpoint=args.dense_checkpoint,
        output_dir=args.out_dir,
        total_tokens=args.total_tokens,
        seed=args.seed,
    )

    if args.cloud_spec:
        spec = get_cloud_run_spec(total_tokens=args.total_tokens, hardware=args.hardware_tier)
        print("=" * 70)
        print("SMALL MOE POC RUN (#422) — CLOUD COMPUTE SPECIFICATION")
        print("=" * 70)
        print(f"Intended Use:       {spec['intended_use']}\n")
        print(f"Hardware Tier:      {spec['hardware_configuration']['gpu_model']} "
              f"({spec['hardware_configuration']['vram_gb']}GB VRAM, {spec['hardware_configuration']['cloud_type']})")
        print(f"Single GPU Fit:     {spec['hardware_configuration']['single_gpu']} (no FSDP/multi-node required)")
        print(f"Estimated Runtime:  {spec['estimated_runtime']}")
        print(f"Hourly Rate:        ${spec['approx_hourly_rate_usd']:.2f}/hr")
        print(f"Estimated Cost:     {spec['estimated_cost']}")
        print("-" * 70)
        print("RunPod Launch Command:")
        print(f"  {spec['runpod_launch_command']}")
        print("=" * 70)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(spec, indent=2))
        return

    if args.print_train_cmd:
        init_ckpt = args.upcycle_out or (args.out_dir / "small_moe_init.safetensors")
        cmd = run_cfg.generate_train_command(args.data_dir, init_ckpt)
        print(" ".join(cmd))
        return

    if args.verify_run:
        print("=" * 70)
        print(f"SMALL MOE POC RUN (#422) — VERIFY RUN: {args.verify_run}")
        print("=" * 70)
        acc = verify_small_moe_poc_acceptance(args.verify_run, moe_config_path=args.config)
        print(f"Upcycle Check:      {acc['upcycle_check']['status']} ({acc['upcycle_check'].get('message', '')})")
        print(f"Bundle Check:       {acc['bundle_check']['status']} ({acc['bundle_check'].get('message', '')})")
        print(f"Resume Step 50:     {acc['resume_check']['status']} ({acc['resume_check'].get('message', '')})")
        print(f"Router Stability:   {acc['router_stability_check']['status']} ({acc['router_stability_check'].get('message', '')})")
        print("-" * 70)
        print(f"OVERALL VERDICT:    {acc['summary']}")
        print("=" * 70)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(acc, indent=2, default=str))
        if not acc["accepted"]:
            sys.exit(1)
        return

    if args.mock_pipeline:
        print("=" * 70)
        print("SMALL MOE POC RUN (#422) — DETERMINISTIC MOCK PIPELINE")
        print("=" * 70)
        res = simulate_mock_pipeline(
            output_dir=args.out_dir,
            total_tokens=args.total_tokens,
            seed=args.seed,
            resume_step=DEFAULT_POC_RESUME_STEP,
            simulate_collapse=args.simulate_collapse,
        )

        acc = res["acceptance"]
        u_chk = acc["upcycle_check"]
        b_chk = acc["bundle_check"]
        r_chk = acc["resume_check"]
        s_chk = acc["router_stability_check"]

        print(f"Upcycle Status:         {u_chk['status'].upper()}")
        print(f"Total Parameters:       {res['upcycle']['dst_config']['num_parameters']/1e6:.1f}M "
              f"({res['upcycle']['dst_config']['active_num_parameters']/1e6:.1f}M active)")
        print(f"Router & Experts:       8 routed experts (top-2), 1 shared expert (zero-down), "
              f"balance_rate={res['upcycle']['dst_config']['moe_balance_rate']}")
        print("-" * 70)
        print(f"Bit-Exact Resume:       {r_chk['status']} ({r_chk['message']})")
        print(f"Router Stability:       {s_chk['status']} ({s_chk['message']})")
        print(f"Checkpoint Bundles:     {b_chk['status']} ({b_chk['message']})")
        print("-" * 70)
        print(f"OVERALL VERDICT:        {acc['summary']}")
        print("=" * 70)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(res, indent=2, default=str))

        if not acc["accepted"]:
            sys.exit(1)
        return

    if args.dry_run or args.upcycle:
        if args.upcycle and not args.dense_checkpoint:
            sys.exit("Error: --upcycle requires --dense-checkpoint")
        dense_src = args.dense_checkpoint or args.dense_config
        out_dst = args.upcycle_out or (args.out_dir / "small_moe_init.safetensors")

        print("=" * 70)
        print(f"SMALL MOE POC RUN (#422) — {'DRY RUN' if args.dry_run else 'UPCYCLE'}")
        print("=" * 70)
        res = execute_sparse_upcycle(
            dense_ckpt_path=dense_src,
            moe_config_path=args.config,
            out_path=out_dst if not args.dry_run else None,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        print(f"Status:             {res['status'].upper()}")
        print(f"Source Model:       {res['src_config']['num_parameters']/1e6:.1f}M params "
              f"(E={res['src_config']['n_experts']}, top_k={res['src_config']['top_k']})")
        print(f"Target MoE Model:   {res['dst_config']['num_parameters']/1e6:.1f}M total "
              f"({res['dst_config']['active_num_parameters']/1e6:.1f}M active, E={res['dst_config']['n_experts']})")
        print(f"Expert Expansion:   {res['experts_expansion']}")
        print(f"Shared Expert:      {res['shared_expert']}")
        print(f"MoE Balance Rate:   {res['moe_balance_rate']}")

        steps = run_cfg.step_breakdown()
        print("-" * 70)
        print(f"Training Plan for {run_cfg.total_tokens/1e6:.1f}M Tokens:")
        print(f"  Tokens/Step:      {run_cfg.tokens_per_step():,} (B={run_cfg.batch_size}, L={run_cfg.seq_len}, accum={run_cfg.grad_accum})")
        print(f"  Total Steps:      {steps['total_steps']:,}")
        print(f"  Resume Step:      {run_cfg.resume_step}")
        print("-" * 70)
        print("Canonical Training Invocation:")
        print("  " + " ".join(run_cfg.generate_train_command(args.data_dir, out_dst)))
        print("=" * 70)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(res, indent=2))
        return

    # Default action: print help / cloud spec
    print("Please specify an action: --cloud-spec, --print-train-cmd, --dry-run, --upcycle, --mock-pipeline, or --verify-run.")


if __name__ == "__main__":
    main()
