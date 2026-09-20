#!/usr/bin/env python3
"""Large-model MoE full run (150B tokens, Large A shape) pipeline runner (#223).

Coordinates:
  - Sparse upcycle from #200's dense checkpoint to #223 Large A MoE target
  - Distributed multi-GPU training budget calculation (150B tokens, WSD schedule, Muon optimizer)
  - Routing bias freeze during final 10–15% anneal (prevents router thrash per #223 spec)
  - RunPod / Cloud compute multi-GPU specifications, hardware tiers, and cost estimates
  - Long-context KV cache memory ceilings under KV8 and MLA (M12 review)
  - Acceptance verification on recall evals (#221), routing diagnostics (#217), and resume stability

Usage:
  # 1. Print Cloud Compute / RunPod launch specification and cost breakdown:
  python scripts/run_large_moe.py --cloud-spec

  # 2. Dry-run upcycle compatibility & print canonical distributed training command:
  python scripts/run_large_moe.py --dry-run

  # 3. Check KV cache memory ceilings:
  python scripts/run_large_moe.py --check-kv-cache

  # 4. Perform upcycle from dense checkpoint:
  python scripts/run_large_moe.py --upcycle --dense-checkpoint runs/dense/weights.safetensors \
      --upcycle-out runs/large_a_init.safetensors

  # 5. Run deterministic offline mock pipeline simulation & verify acceptance:
  python scripts/run_large_moe.py --mock-pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.train.large_moe_run import (
    CLOUD_HARDWARE_TEMPLATES,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_DENSE_CONFIG,
    DEFAULT_MOE_CONFIG,
    LargeMoERunConfig,
    execute_upcycle,
    get_cloud_run_spec,
    simulate_mock_pipeline,
    verify_kv_cache_ceilings,
    verify_large_moe_acceptance,
    verify_routing_specialization,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_MOE_CONFIG,
                    help=f"Target Large A MoE config YAML (default: {DEFAULT_MOE_CONFIG.name})")
    ap.add_argument("--dense-config", type=Path, default=DEFAULT_DENSE_CONFIG,
                    help=f"Source dense config YAML (default: {DEFAULT_DENSE_CONFIG.name})")
    ap.add_argument("--dense-checkpoint", type=Path, default=None,
                    help="Path to source dense checkpoint (.safetensors) to upcycle from")
    ap.add_argument("--upcycle-out", type=Path, default=None,
                    help="Path to save upcycled Large A checkpoint (.safetensors)")
    ap.add_argument("--upcycle", action="store_true",
                    help="Execute upcycle transform from --dense-checkpoint to --upcycle-out")
    ap.add_argument("--total-tokens", type=int, default=DEFAULT_BUDGET_TOKENS,
                    help=f"Total training token budget (default: {DEFAULT_BUDGET_TOKENS:,})")
    ap.add_argument("--hardware-tier", choices=list(CLOUD_HARDWARE_TEMPLATES.keys()), default="4x-a100-80gb",
                    help="Hardware tier for cloud compute modeling (default: 4x-a100-80gb)")
    ap.add_argument("--cloud-spec", action="store_true",
                    help="Print Cloud Compute / RunPod specifications and cost estimates, then exit")
    ap.add_argument("--check-kv-cache", action="store_true",
                    help="Verify long-context KV cache memory ceilings under KV8 and MLA, then exit")
    ap.add_argument("--print-train-cmd", action="store_true",
                    help="Print the canonical distributed torchrun training command, then exit")
    ap.add_argument("--data-dir", type=str, default="data/shards",
                    help="Training packed shard directory (default: data/shards)")
    ap.add_argument("--domains-json", type=Path, default=None,
                    help="Path to domains.json for mid-training MoE routing diagnostics (#217)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Verify configs and upcycle compatibility without executing paid runs")
    ap.add_argument("--mock-pipeline", action="store_true",
                    help="Run deterministic offline simulation of upcycle, training plan, KV ceilings, and acceptance")
    ap.add_argument("--simulate-kill", action="store_true",
                    help="In --mock-pipeline, simulate routing kill-criterion failure")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional JSON file path to write results into")
    ap.add_argument("--seed", type=int, default=223,
                    help="Random seed for upcycle and diagnostics")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    run_cfg = LargeMoERunConfig(
        moe_config_path=args.config,
        dense_config_path=args.dense_config,
        dense_checkpoint=args.dense_checkpoint,
        total_tokens=args.total_tokens,
        seed=args.seed,
    )

    if args.cloud_spec:
        spec = get_cloud_run_spec(total_tokens=args.total_tokens, hardware=args.hardware_tier)
        print("=" * 70)
        print("M12 LARGE-MODEL FULL RUN (#223) — CLOUD COMPUTE SPECIFICATION")
        print("=" * 70)
        print(f"Intended Use:       {spec['intended_use']}\n")
        hw = spec["hardware_configuration"]
        print(f"Hardware Tier:      {hw['gpu_model']} ({hw['vram_gb_total']}GB VRAM aggregate, {hw['cloud_type']})")
        print(f"Multi-GPU Sharding: Required (Model+Optimizer ~61.7GB exceeds single 80GB card)")
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

    if args.check_kv_cache:
        print("=" * 70)
        print("LARGE-MODEL MOE (LARGE A, #223) — KV CACHE MEMORY CEILINGS")
        print("=" * 70)
        kv_res = verify_kv_cache_ceilings()
        print(f"Status:             {kv_res['status']}")
        print(f"128k context (KV8): {kv_res['kv8_128k']['gb']:.2f} GB (Ceiling: <= {kv_res['kv8_128k']['ceiling_gb']} GB) -> {'PASS' if kv_res['kv8_128k']['passed'] else 'FAIL'}")
        print(f"256k context (KV8): {kv_res['kv8_256k']['gb']:.2f} GB (Ceiling: <= {kv_res['kv8_256k']['ceiling_gb']} GB) -> {'PASS' if kv_res['kv8_256k']['passed'] else 'FAIL'}")
        print(f"256k context (MLA): {kv_res['mla_256k']['gb']:.2f} GB (Ceiling: < {kv_res['mla_256k']['ceiling_gb']} GB) -> {'PASS' if kv_res['mla_256k']['passed'] else 'FAIL'}")
        print("-" * 70)
        print(f"Verdict:            {kv_res['message']}")
        print("=" * 70)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(kv_res, indent=2))
        if not kv_res["passed"]:
            sys.exit(1)
        return

    if args.print_train_cmd:
        init_ckpt = args.upcycle_out or Path("runs/large_a_init.safetensors")
        cmd = run_cfg.generate_train_command(args.data_dir, init_ckpt, domains_json=args.domains_json)
        print(" ".join(cmd))
        return

    if args.mock_pipeline:
        print("=" * 70)
        print("LARGE-MODEL MOE FULL RUN (#223) — DETERMINISTIC MOCK PIPELINE")
        print("=" * 70)
        res = simulate_mock_pipeline(
            total_tokens=args.total_tokens,
            seed=args.seed,
            simulate_kill=args.simulate_kill,
        )

        acc = res["acceptance"]
        r_check = acc["routing_check"]
        c_check = acc["recall_check"]
        k_check = acc["kv_cache_check"]
        res_check = acc["resume_check"]

        print(f"Upcycle Compatibility:  {res['upcycle']['status'].upper()}")
        print(f"Total Parameters:       {res['upcycle']['dst_config']['num_parameters']/1e9:.2f}B "
              f"({res['upcycle']['dst_config']['active_num_parameters']/1e6:.1f}M active)")
        print(f"Total Steps (WSD):      {res['training_plan']['total_steps']:,} steps")
        breakdown = res['training_plan']['step_breakdown']
        print(f"  Warmup:               {breakdown['warmup_steps']:,} steps")
        print(f"  Stable:               {breakdown['stable_steps']:,} steps")
        print(f"  Decay (10–15%):       {breakdown['decay_steps']:,} steps (from step {breakdown['anneal_start_step']:,})")
        print(f"  Router Bias Freeze:   Step {breakdown['freeze_router_bias_step']:,} (anneal start)")
        print("-" * 70)
        print("Acceptance Evaluation (#221 recall + #217 routing + KV ceilings + resume):")
        print(f"  Routing Specialization: {r_check['status']} ({r_check['message']})")
        print(f"  Recall Probes:          {c_check['status']} ({c_check['message']})")
        print(f"  KV Cache Ceilings:      {k_check['status']} ({k_check['message']})")
        print(f"  Resume Stability:       {res_check['status']} ({res_check['message']})")
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
        out_dst = args.upcycle_out or (REPO_ROOT / "runs" / "upcycled_large_a.safetensors")

        print("=" * 70)
        print(f"LARGE-MODEL MOE RUN (#223) — {'DRY RUN' if args.dry_run else 'UPCYCLE'}")
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
        print(f"Target MoE Model:   {res['dst_config']['num_parameters']/1e9:.2f}B total "
              f"({res['dst_config']['active_num_parameters']/1e6:.1f}M active, E={res['dst_config']['n_experts']})")
        print(f"Expert Expansion:   {res['experts_expansion']}")
        print(f"Shared Expert:      {res['shared_expert']}")

        steps = run_cfg.step_breakdown()
        print("-" * 70)
        print(f"Training Plan for {run_cfg.total_tokens/1e9:.1f}B Tokens:")
        print(f"  Tokens/Step:      {run_cfg.tokens_per_step():,} (B={run_cfg.batch_size}, L={run_cfg.seq_len}, accum={run_cfg.grad_accum}, DP={run_cfg.dp_size})")
        print(f"  Total Steps:      {steps['total_steps']:,}")
        print(f"  Schedule:         WSD (warmup={steps['warmup_steps']:,}, stable={steps['stable_steps']:,}, decay={steps['decay_steps']:,})")
        print(f"  Anneal Freeze:    Router bias frozen at step {steps['freeze_router_bias_step']:,}")
        print("-" * 70)
        print("Canonical Training Invocation:")
        print("  " + " ".join(run_cfg.generate_train_command(args.data_dir, out_dst, domains_json=args.domains_json)))
        print("=" * 70)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(res, indent=2))
        return

    # Default action: print help / cloud spec
    print("Please specify an action: --cloud-spec, --check-kv-cache, --dry-run, --mock-pipeline, or --upcycle.")


if __name__ == "__main__":
    main()
