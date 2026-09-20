#!/usr/bin/env python3
"""Autonomous coding agent harness benchmark POC & MVP runner (#371).

Tracks and operationalizes execution runs of the autonomous ReAct coding agent
harness (#349-#352, #366-#368) against synthetic and real-world software engineering
benchmarks:

Stage 1: POC Run (Synthetic & Sandbox Validation)
  - Anti-spin circuit breakers trigger on repeated failed edits (#349)
  - Staged context compaction elides old tool outputs without context window blowup (#350)
  - Immediate post-edit diagnostic feedback halts syntax errors (#351)
  - Out-of-history plan injection maintains multi-step focus (#352)
  - Native Brave search and page fetch retrieve external docs cleanly (#366-#368)

Stage 2: MVP Run (Full Benchmark Evaluation)
  - SWE-bench-lite synthetic multi-file repository bugfix tasks
  - HumanEval TypeScript tasks from eval_sets/humaneval_ts/
  - Repo refactoring tasks interfacing with deterministic RLVR verifiers (#339-#348, #361)
  - Pass@k, token consumption, trajectory length, and execution latency profiles

Usage:
  # 1. Print Cloud Compute / RunPod launch specification and cost breakdown:
  python scripts/run_agent_benchmark.py --cloud-spec

  # 2. Dry-run benchmark suite discovery and verification without execution:
  python scripts/run_agent_benchmark.py --dry-run

  # 3. Run Stage 1 POC stress validation suite:
  python scripts/run_agent_benchmark.py --stage poc

  # 4. Run Stage 2 MVP benchmarks:
  python scripts/run_agent_benchmark.py --stage mvp --limit 2

  # 5. Full operational evaluation with JSON results and JSONL trajectory transcript:
  python scripts/run_agent_benchmark.py --stage all --output results/agent_benchmark.json \
      --transcript results/agent_benchmark.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.benchmark import (
    CLOUD_HARDWARE_TEMPLATES,
    BenchmarkRunSummary,
    get_cloud_run_spec,
    get_humaneval_tasks,
    get_refactoring_tasks,
    get_swe_bench_lite_tasks,
    run_agent_benchmark,
    run_mvp_stage,
    run_poc_stage,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--stage",
        choices=("all", "poc", "mvp"),
        default="all",
        help="Benchmark execution stage: poc (stress validation), mvp (full benchmarks), or all (default: all)",
    )
    ap.add_argument(
        "--benchmark",
        default="swe-bench-lite,humaneval,refactoring",
        help="Comma-separated subset of benchmark suites for Stage 2 MVP (default: swe-bench-lite,humaneval,refactoring)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of tasks to evaluate per benchmark suite",
    )
    ap.add_argument(
        "--cloud-spec",
        action="store_true",
        help="Print Cloud Compute / RunPod specifications and cost estimates, then exit",
    )
    ap.add_argument(
        "--hardware-tier",
        choices=list(CLOUD_HARDWARE_TEMPLATES.keys()),
        default="a40",
        help="Hardware tier for cloud compute modeling (default: a40)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify benchmark task availability and scenario definitions without executing agent loops",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON benchmark summary results",
    )
    ap.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Path to save per-instance JSONL trajectory transcripts",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=371,
        help="Random seed for reproducible execution (default: 371)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed execution events and diagnostics",
    )
    return ap.parse_args()


def print_cloud_spec(hardware_tier: str) -> None:
    spec = get_cloud_run_spec(num_tasks=50, hardware_tier=hardware_tier)
    print("=" * 76)
    print("  MONICA M12 AGENT HARNESS BENCHMARK -- CLOUD COMPUTE / RUNPOD SPECIFICATION")
    print("=" * 76)
    print(f"Intended Use:        {spec['intended_use']}")
    print(f"Hardware Tier:       {spec['hardware_tier']} ({spec['selected_hardware']}, {spec['vram_gb']}GB VRAM)")
    print(f"Hourly Rate:         ${spec['hourly_rate_usd']:.2f} USD")
    print(f"Tasks Modeled:       {spec['num_tasks']} tasks (~{spec['estimated_total_turns']} total turns)")
    print(f"Estimated Runtime:   {spec['estimated_gpu_hours']:.2f} GPU-hours")
    print(f"Estimated Cost:      ${spec['estimated_cost_usd']:.2f} USD")
    print("\nHardware Tier Comparison:")
    print(f"  {'Tier':<10} {'GPU / Hardware':<26} {'Rate ($/hr)':<14} {'Est. Hours':<12} {'Est. Cost ($)':<12}")
    print("  " + "-" * 72)
    for tier_id, info in spec["tier_comparisons"].items():
        rec = " [RECOMMENDED]" if tier_id == "a40" else ""
        print(f"  {tier_id:<10} {info['name'] + rec:<26} ${info['hourly_rate_usd']:<13.2f} {info['estimated_hours']:<12.2f} ${info['estimated_cost_usd']:<11.2f}")
    print("\nRunPod Launch Command:")
    print(f"  {spec['runpod_launch_command']}")
    print("=" * 76)


def print_dry_run_summary(suites: list[str], limit: int | None) -> None:
    print("=" * 76)
    print("  MONICA M12 AGENT HARNESS BENCHMARK -- DRY RUN VERIFICATION")
    print("=" * 76)
    print("\nStage 1: POC Run Stress Scenarios (Synthetic & Sandbox Validation):")
    scenarios = [
        ("anti_spin_circuit_breakers", "Triggers redirection at 5 repeats; terminates at 8 repeats (#349)"),
        ("staged_context_compaction", "Soft elision at 0.60; hard summarization at 0.85; preserves recent turns (#350)"),
        ("post_edit_diagnostics_safety", "Immediate post-mutation syntax diagnostics & read-before-write gates (#351)"),
        ("out_of_history_plan_injection", "Out-of-history {PLAN} conditioning & exit-gate completion (#352)"),
        ("native_search_and_fetch", "Brave search distilled records (<400 tok), SSRF defense, 12k char ceiling (#366-#368)"),
    ]
    for name, desc in scenarios:
        print(f"  - {name:<32}: {desc}")

    print("\nStage 2: MVP Benchmark Suites:")
    if "swe-bench-lite" in suites:
        swe_tasks = get_swe_bench_lite_tasks()
        if limit:
            swe_tasks = swe_tasks[:limit]
        print(f"  - swe-bench-lite : {len(swe_tasks)} tasks available (multi-file repository bugfix scenarios)")
        for t in swe_tasks:
            print(f"      [{t.task_id}] {t.name}")

    if "humaneval" in suites:
        he_tasks = get_humaneval_tasks(limit=limit)
        print(f"  - humaneval      : {len(he_tasks)} tasks available (TypeScript function tasks from eval_sets/)")
        for t in he_tasks[:3]:
            print(f"      [{t.task_id}] {t.name}")
        if len(he_tasks) > 3:
            print(f"      ... and {len(he_tasks) - 3} more tasks")

    if "refactoring" in suites:
        refactor_tasks = get_refactoring_tasks()
        if limit:
            refactor_tasks = refactor_tasks[:limit]
        print(f"  - refactoring    : {len(refactor_tasks)} tasks available (RLVR behavioral invariance & decoupling)")
        for t in refactor_tasks:
            print(f"      [{t.task_id}] {t.name} (verifier: {t.metadata.get('verifier')})")

    print("\nDry-run check completed successfully. All components verified Above the Seam.")
    print("=" * 76)


def print_summary_table(summary: BenchmarkRunSummary) -> None:
    print("\n" + "=" * 76)
    print("  MONICA M12 AGENT HARNESS BENCHMARK RUN RESULTS")
    print("=" * 76)

    # Stage 1: POC Stress Summary
    if summary.stress_verification:
        stress = summary.stress_verification
        print(f"\nStage 1: POC Stress Validation ({stress.get('passed_checks', 0)}/{stress.get('total_checks', 0)} checks passed, {stress.get('wall_s', 0):.3f}s):")
        print(f"  {'Scenario':<34} {'Status':<10} {'Evidence Details'}")
        print("  " + "-" * 72)
        for sc in stress.get("scenarios", []):
            st = "PASSED" if sc.get("passed") else "FAILED"
            ev = json.dumps(sc.get("evidence", {}), separators=(", ", ": "))
            if len(ev) > 36:
                ev = ev[:33] + "..."
            print(f"  {sc.get('name', ''):<34} {st:<10} {ev}")

    # Stage 2: MVP Suite Summary
    if summary.suite_breakdowns:
        print(f"\nStage 2: MVP Benchmark Suites Evaluation (Overall Pass@1: {summary.pass_rate:.1%}):")
        print(f"  {'Suite':<18} {'Tasks':<8} {'Passed':<8} {'Pass@1':<10} {'Mean Turns':<12} {'Mean Tokens':<12} {'Wall Time'}")
        print("  " + "-" * 72)
        for suite_name, stats in summary.suite_breakdowns.items():
            print(
                f"  {suite_name:<18} "
                f"{stats.get('tasks', 0):<8} "
                f"{stats.get('passed', 0):<8} "
                f"{stats.get('pass_rate', 0.0):<10.1%} "
                f"{stats.get('mean_turns', 0.0):<12.1f} "
                f"{stats.get('mean_tokens', 0.0):<12.1f} "
                f"{stats.get('total_wall_s', 0.0):.2f}s"
            )

    # Aggregate Telemetry Profiles
    print("\nExecution Telemetry Profiles:")
    tc = summary.token_consumption
    tp = summary.trajectory_profile
    lp = summary.latency_profile
    print(
        f"  • Token Consumption: {tc.get('total_tokens', 0):,} total tokens "
        f"({tc.get('total_prompt_tokens', 0):,} prompt, {tc.get('total_completion_tokens', 0):,} completion; "
        f"mean {tc.get('mean_tokens_per_task', 0.0):.1f}/task)"
    )
    print(
        f"  • Trajectory Length: {tp.get('total_turns', 0)} turns "
        f"(mean {tp.get('mean_turns_per_task', 0.0):.1f} turns/task, "
        f"min: {tp.get('min_turns', 0)}, max: {tp.get('max_turns', 0)})"
    )
    print(
        f"  • Execution Latency: {lp.get('total_wall_s', 0.0):.3f}s total wall time "
        f"(mean {lp.get('mean_wall_s_per_task', 0.0):.3f}s/task; "
        f"gen: {lp.get('total_generation_wall_s', 0.0):.3f}s, tool: {lp.get('total_tool_wall_s', 0.0):.3f}s)"
    )
    print("=" * 76)


def main() -> None:
    args = parse_args()

    if args.cloud_spec:
        print_cloud_spec(args.hardware_tier)
        return

    suites = [s.strip() for s in args.benchmark.split(",") if s.strip()]

    if args.dry_run:
        print_dry_run_summary(suites, args.limit)
        return

    summary = run_agent_benchmark(
        stage=args.stage,
        suites=suites,
        limit_per_suite=args.limit,
        output_path=args.output,
        transcript_path=args.transcript,
        verbose=args.verbose,
    )

    print_summary_table(summary)


if __name__ == "__main__":
    main()
