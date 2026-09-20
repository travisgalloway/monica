"""Unit and integration tests for autonomous coding agent harness benchmark POC & MVP runs (#371)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.agent.benchmark import (
    CLOUD_HARDWARE_TEMPLATES,
    BenchmarkRunSummary,
    BenchmarkTask,
    DeterministicBenchmarkAgent,
    TaskExecutionResult,
    execute_single_benchmark_task,
    get_cloud_run_spec,
    get_humaneval_tasks,
    get_refactoring_tasks,
    get_swe_bench_lite_tasks,
    run_agent_benchmark,
    run_mvp_stage,
    run_poc_anti_spin_stress,
    run_poc_compaction_stress,
    run_poc_diagnostics_safety_stress,
    run_poc_planning_injection_stress,
    run_poc_stage,
    run_poc_web_search_fetch_stress,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Cloud Compute & RunPod Specification Tests
# --------------------------------------------------------------------------- #

def test_cloud_hardware_templates_definition():
    """Verify hardware templates include expected tiers, rates, and recommendations."""
    assert "a40" in CLOUD_HARDWARE_TEMPLATES
    assert "rtx4090" in CLOUD_HARDWARE_TEMPLATES
    assert "a100" in CLOUD_HARDWARE_TEMPLATES
    assert "h100" in CLOUD_HARDWARE_TEMPLATES

    a40 = CLOUD_HARDWARE_TEMPLATES["a40"]
    assert a40["recommended"] is True
    assert a40["vram_gb"] == 48
    assert a40["hourly_rate_usd"] == 0.40


def test_get_cloud_run_spec_calculation():
    """Verify calculation of runtime hours, costs, and launch command in cloud spec."""
    spec = get_cloud_run_spec(num_tasks=50, hardware_tier="a40")
    assert "intended_use" in spec
    assert spec["hardware_tier"] == "a40"
    assert spec["selected_hardware"] == "NVIDIA A40"
    assert spec["num_tasks"] == 50
    assert spec["estimated_gpu_hours"] > 0
    assert spec["estimated_cost_usd"] > 0
    assert "runpod_launch_command" in spec
    assert "cloud_pod.py launch --template a40" in spec["runpod_launch_command"]
    assert len(spec["tier_comparisons"]) == 4


# --------------------------------------------------------------------------- #
# Stage 1: POC Run Operational Stress Tests
# --------------------------------------------------------------------------- #

def test_run_poc_anti_spin_stress(tmp_path: Path):
    """Verify POC stress test: anti-spin circuit breaker redirection at 5, trip at 8 (#349)."""
    res = run_poc_anti_spin_stress(tmp_path)
    assert res["passed"] is True
    evidence = res["evidence"]
    assert evidence["redirection_observed"] is True
    assert evidence["termination_observed"] is True
    assert evidence["tripped_turn"] == 8


def test_run_poc_compaction_stress():
    """Verify POC stress test: staged context compaction elides bulky outputs without blowup (#350)."""
    res = run_poc_compaction_stress()
    assert res["passed"] is True
    evidence = res["evidence"]
    assert evidence["elision_applied"] is True
    assert evidence["preamble_preserved"] is True
    assert evidence["recent_preserved"] is True
    assert evidence["final_tokens"] < evidence["initial_tokens"]


def test_run_poc_diagnostics_safety_stress(tmp_path: Path):
    """Verify POC stress test: post-edit diagnostics and read-before-write safety gates (#351)."""
    res = run_poc_diagnostics_safety_stress(tmp_path)
    assert res["passed"] is True
    evidence = res["evidence"]
    assert evidence["unread_rejected"] is True
    assert evidence["read_registered"] is True
    assert evidence["has_diagnostic_feedback"] is True
    assert evidence["jailbreak_caught"] is True


def test_run_poc_planning_injection_stress():
    """Verify POC stress test: out-of-history plan injection and exit-gate completion (#352)."""
    res = run_poc_planning_injection_stress()
    assert res["passed"] is True
    evidence = res["evidence"]
    assert evidence["has_plan_block"] is True
    assert evidence["turns_unchanged"] is True
    assert evidence["all_completed"] is True
    assert evidence["exit_gate_triggered"] is True


def test_run_poc_web_search_fetch_stress():
    """Verify POC stress test: native search & fetch with SSRF blocking and truncation (#366-#368)."""
    res = run_poc_web_search_fetch_stress()
    assert res["passed"] is True
    evidence = res["evidence"]
    assert evidence["compact_payload"] is True
    assert evidence["ssrf_blocked_localhost"] is True
    assert evidence["ssrf_blocked_metadata"] is True
    assert evidence["truncation_applied"] is True


def test_run_poc_stage_orchestration(tmp_path: Path):
    """Verify full Stage 1 POC run executes all 5 stress scenarios cleanly."""
    res = run_poc_stage(sandbox_dir=tmp_path)
    assert res["stage"] == "poc"
    assert res["total_checks"] == 5
    assert res["passed_checks"] == 5
    assert res["all_passed"] is True
    assert len(res["scenarios"]) == 5


# --------------------------------------------------------------------------- #
# Stage 2: MVP Benchmark Task Definitions and Execution Tests
# --------------------------------------------------------------------------- #

def test_swe_bench_lite_tasks_definition():
    """Verify SWE-bench-lite task definitions contain required files and instructions."""
    tasks = get_swe_bench_lite_tasks()
    assert len(tasks) >= 2
    for t in tasks:
        assert t.suite == "swe-bench-lite"
        assert t.task_id.startswith("swe-lite-")
        assert len(t.files) > 0
        assert len(t.test_files) > 0
        assert t.instruction


def test_humaneval_tasks_loader():
    """Verify HumanEval task loader imports records from eval_sets or falls back safely."""
    tasks = get_humaneval_tasks(limit=3)
    assert len(tasks) == 3
    for t in tasks:
        assert t.suite == "humaneval"
        assert "HumanEval" in t.task_id or "solution" in str(t.files)


def test_refactoring_tasks_definition():
    """Verify refactoring tasks define Strategy and Dependency Injection patterns."""
    tasks = get_refactoring_tasks()
    assert len(tasks) >= 2
    patterns = {t.metadata.get("pattern") for t in tasks}
    assert "strategy" in patterns
    assert "dependency_injection" in patterns


def test_execute_single_benchmark_task(tmp_path: Path):
    """Verify single benchmark task execution with DeterministicBenchmarkAgent."""
    task = get_swe_bench_lite_tasks()[0]
    result = execute_single_benchmark_task(task=task, sandbox_root=tmp_path)

    assert result.task_id == task.task_id
    assert result.suite == "swe-bench-lite"
    assert result.success is True
    assert result.turns == 3
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.total_tokens > 0
    assert result.total_wall_s > 0


def test_run_mvp_stage_execution(tmp_path: Path):
    """Verify MVP benchmark evaluation across all suites with telemetry profiling."""
    res = run_mvp_stage(
        suites=("swe-bench-lite", "humaneval", "refactoring"),
        limit_per_suite=1,
        sandbox_root=tmp_path,
    )

    assert res["stage"] == "mvp"
    assert res["total_tasks"] == 3
    assert res["passed_tasks"] == 3
    assert res["pass_rate"] == 1.0
    assert "pass@1" in res["pass_at_k"]
    assert res["pass_at_k"]["pass@1"] == 1.0

    # Token profile
    tc = res["token_consumption"]
    assert tc["total_tokens"] > 0
    assert tc["mean_tokens_per_task"] > 0

    # Trajectory profile
    tp = res["trajectory_profile"]
    assert tp["total_turns"] == 9
    assert tp["mean_turns_per_task"] == 3.0

    # Latency profile
    lp = res["latency_profile"]
    assert lp["total_wall_s"] > 0
    assert lp["mean_wall_s_per_task"] > 0

    # Suite breakdowns
    sb = res["suite_breakdowns"]
    assert "swe-bench-lite" in sb
    assert "humaneval" in sb
    assert "refactoring" in sb


# --------------------------------------------------------------------------- #
# Full Benchmark Orchestrator & Persistence Tests
# --------------------------------------------------------------------------- #

def test_run_agent_benchmark_full_persisted(tmp_path: Path):
    """Verify run_agent_benchmark runs POC & MVP and writes JSON results and JSONL transcript."""
    out_json = tmp_path / "results" / "summary.json"
    out_jsonl = tmp_path / "results" / "transcript.jsonl"

    summary = run_agent_benchmark(
        stage="all",
        suites=("swe-bench-lite", "refactoring"),
        limit_per_suite=1,
        sandbox_dir=tmp_path / "sandbox",
        output_path=out_json,
        transcript_path=out_jsonl,
    )

    assert summary.stage == "all"
    assert summary.total_tasks == 2
    assert summary.passed_tasks == 2
    assert summary.pass_rate == 1.0
    assert summary.stress_verification["all_passed"] is True

    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["stage"] == "all"
    assert data["total_tasks"] == 2
    assert data["stress_verification"]["passed_checks"] == 5

    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert "task_id" in rec1
    assert "telemetry" in rec1


# --------------------------------------------------------------------------- #
# CLI Driver Integration Tests
# --------------------------------------------------------------------------- #

def test_cli_driver_cloud_spec():
    """Verify CLI --cloud-spec outputs complete hardware specifications."""
    cmd = [sys.executable, "scripts/run_agent_benchmark.py", "--cloud-spec"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "CLOUD COMPUTE / RUNPOD SPECIFICATION" in res.stdout
    assert "NVIDIA A40" in res.stdout
    assert "Hourly Rate:" in res.stdout
    assert "RunPod Launch Command:" in res.stdout


def test_cli_driver_dry_run():
    """Verify CLI --dry-run validates scenarios and benchmark tasks."""
    cmd = [sys.executable, "scripts/run_agent_benchmark.py", "--dry-run"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "DRY RUN VERIFICATION" in res.stdout
    assert "Stage 1: POC Run Stress Scenarios" in res.stdout
    assert "Stage 2: MVP Benchmark Suites:" in res.stdout
    assert "swe-bench-lite" in res.stdout
    assert "humaneval" in res.stdout
    assert "refactoring" in res.stdout


def test_cli_driver_stage_poc():
    """Verify CLI --stage poc executes Stage 1 POC stress checks."""
    cmd = [sys.executable, "scripts/run_agent_benchmark.py", "--stage", "poc"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Stage 1: POC Stress Validation (5/5 checks passed" in res.stdout
    assert "anti_spin_circuit_breakers" in res.stdout
    assert "PASSED" in res.stdout


def test_cli_driver_stage_mvp_with_output(tmp_path: Path):
    """Verify CLI --stage mvp writes output files and prints summary tables."""
    out_json = tmp_path / "mvp_out.json"
    out_jsonl = tmp_path / "mvp_out.jsonl"

    cmd = [
        sys.executable,
        "scripts/run_agent_benchmark.py",
        "--stage",
        "mvp",
        "--limit",
        "1",
        "--benchmark",
        "swe-bench-lite,refactoring",
        "--output",
        str(out_json),
        "--transcript",
        str(out_jsonl),
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Stage 2: MVP Benchmark Suites Evaluation" in res.stdout
    assert "Overall Pass@1: 100.0%" in res.stdout
    assert out_json.exists()
    assert out_jsonl.exists()
