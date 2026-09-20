#!/usr/bin/env python3
"""Small-model MoE full run (50–70B tokens) pipeline runner (#222).

Coordinates:
  - Sparse upcycle from #200's dense checkpoint to #222 small MoE target
  - Training budget calculation (50–70B tokens, WSD schedule, Muon optimizer)
  - RunPod / Cloud compute specifications, hardware tiers, and cost estimates
  - Acceptance verification on recall evals (#221) and routing diagnostics (#217)

Usage:
  # 1. Print Cloud Compute / RunPod launch specification and cost breakdown:
  python scripts/run_small_moe.py --cloud-spec

  # 2. Dry-run upcycle compatibility & print canonical training command:
  python scripts/run_small_moe.py --dry-run

  # 3. Perform upcycle from dense checkpoint:
  python scripts/run_small_moe.py --upcycle --dense-checkpoint runs/dense/weights.safetensors \
      --upcycle-out runs/small_moe_init.safetensors

  # 4. Run deterministic offline mock pipeline simulation & verify acceptance:
  python scripts/run_small_moe.py --mock-pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.small_moe_run import (
    CLOUD_HARDWARE_TEMPLATES,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_DENSE_CONFIG,
    DEFAULT_MOE_CONFIG,
    SmallMoERunConfig,
    execute_upcycle,
    get_cloud_run_spec,
    simulate_mock_pipeline,
    verify_routing_specialization,
    verify_small_moe_acceptance,
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
    ap.add_argument("--data-dir", type=str, default="data/shards",
                    help="Training packed shard directory (default: data/shards)")
    ap.add_argument("--domains-json", type=Path, default=None,
                    help="Path to domains.json for mid-training MoE routing diagnostics (#217)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Verify configs and upcycle compatibility without executing paid runs")
    ap.add_argument("--mock-pipeline", action="store_true",
                    help="Run deterministic offline simulation of upcycle, training plan, and acceptance evals")
    ap.add_argument("--simulate-kill", action="store_true",
                    help="In --mock-pipeline, simulate routing kill-criterion failure")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional JSON file path to write results into")
    ap.add_argument("--seed", type=int, default=222,
                    help="Random seed for upcycle and diagnostics")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    run_cfg = SmallMoERunConfig(
        moe_config_path=args.config,
        dense_config_path=args.dense_config,
        dense_checkpoint=args.dense_checkpoint,
        total_tokens=args.total_tokens,
        seed=args.seed,
    )

    if args.cloud_spec:
        spec = get_cloud_run_spec(total_tokens=args.total_tokens, hardware=args.hardware_tier)
        print("=" * 70)
        print("M12 SMALL-MODEL FULL RUN (#222) — CLOUD COMPUTE SPECIFICATION")
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
        print("-" * 70)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(spec, indent=2))
        return

    if args.print_train_cmd:
        init_ckpt = args.upcycle_out or Path("runs/small_moe_init.safetensors")
        cmd = run_cfg.generate_train_command(args.data_dir, init_ckpt, domains_json=args.domains_json)
        print(" ".join(cmd))
        return

    if args.mock_pipeline:
        print("=" * 70)
        print("SMALL-MODEL MOE FULL RUN (#222) — DETERMINISTIC MOCK PIPELINE")
        print("=" * 70)
        res = simulate_mock_pipeline(
            total_tokens=args.total_tokens,
            seed=args.seed,
            simulate_kill=args.simulate_kill,
        )

        acc = res["acceptance"]
        r_check = acc["routing_check"]
        c_check = acc["recall_check"]

        print(f"Upcycle Compatibility:  {res['upcycle']['status'].upper()}")
        print(f"Total Parameters:       {res['upcycle']['dst_config']['num_parameters']/1e6:.1f}M "
              f"({res['upcycle']['dst_config']['active_num_parameters']/1e6:.1f}M active)")
        print(f"Total Steps (WSD):      {res['training_plan']['total_steps']:,} steps")
        print(f"  Warmup:               {res['training_plan']['step_breakdown']['warmup_steps']:,} steps")
        print(f"  Stable:               {res['training_plan']['step_breakdown']['stable_steps']:,} steps")
        print(f"  Decay (20%):          {res['training_plan']['step_breakdown']['decay_steps']:,} steps")
        print("-" * 70)
        print("Acceptance Evaluation (#221 recall + #217 routing):")
        print(f"  Routing Specialization: {r_check['status']} ({r_check['message']})")
        print(f"  Recall Probes:          {c_check['status']} ({c_check['message']})")
        print("-" * 70)
        print(f"OVERALL VERDICT:          {acc['summary']}")
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
        out_dst = args.upcycle_out or (REPO_ROOT / "runs" / "upcycled_small_moe.safetensors")

        print("=" * 70)
        print(f"SMALL-MODEL MOE RUN (#222) — {'DRY RUN' if args.dry_run else 'UPCYCLE'}")
        print("=" * 70)
        res = execute_upcycle(
            dense_ckpt_path=dense_src,
            moe_cfg_path=args.config,
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

        steps = run_cfg.step_breakdown()
        print("-" * 70)
        print(f"Training Plan for {run_cfg.total_tokens/1e9:.1f}B Tokens:")
        print(f"  Tokens/Step:      {run_cfg.tokens_per_step():,} (B={run_cfg.batch_size}, L={run_cfg.seq_len}, accum={run_cfg.grad_accum})")
        print(f"  Total Steps:      {steps['total_steps']:,}")
        print(f"  Schedule:         WSD (warmup={steps['warmup_steps']:,}, stable={steps['stable_steps']:,}, decay={steps['decay_steps']:,})")
        print("-" * 70)
        print("Canonical Training Invocation:")
        print("  " + " ".join(run_cfg.generate_train_command(args.data_dir, out_dst, domains_json=args.domains_json)))
        print("=" * 70)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(res, indent=2))
        return

    # Default action: print help / cloud spec
    print("Please specify an action: --cloud-spec, --dry-run, --mock-pipeline, or --upcycle.")


if __name__ == "__main__":
    main()
