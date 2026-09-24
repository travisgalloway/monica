"""Unit and integration tests for Issue #423: Routing diagnostics & domain specialization verification.

Verifies:
1. Architectural & Configuration Validation:
   - RoutingPOCRunConfig initializes with code-small-moe.yaml target and eval sets.
   - Default kill-pair is ("typescript", "math"), kill-threshold is 0.90, collapse-threshold is 0.95.
2. Tokenized Domain Batch Loading:
   - load_domain_batches extracts TypeScript, prose, and math batches from eval_sets.
   - Batch shapes match (batch_size, seq_len) with uint16 token IDs within vocab range.
3. SmallMoERoutingModel Duck-Typed Interface:
   - Correctly integrates with ModelInterface duck-typing.
   - set_moe_load_counting toggles load collection.
   - pop_moe_load resets counters to zero.
   - Top-2 dropless routing allocates 2 * batch_size * seq_len tokens per MoE layer.
4. Routing Diagnostics & Specialization Verification:
   - run_routing_diagnostics_poc executes complete pipeline.
   - TypeScript vs Math overlap < 0.90 (acceptance criterion 1).
   - Mid-training routing kill-check verified negative (acceptance criterion 2).
   - No expert starvation across all 8 experts in all 16 MoE layers.
   - Uniform routing collapse check verified negative (< 0.95).
   - Overall acceptance gate reports PASSED.
5. Failure Injection & Quality Gate Guardrails:
   - simulate_kill triggers kill-criterion (overlap >= 0.90) and fails acceptance.
   - simulate_starvation detects starved experts and fails acceptance.
   - simulate_collapse detects uniform/expert collapse and fails acceptance.
6. Acceptance Gate Direct Unit Tests:
   - verify_routing_poc_acceptance handles passed and failed inputs cleanly.
7. CLI Integration:
   - scripts/run_routing_poc.py --mock-pipeline runs successfully (exit code 0).
   - scripts/run_routing_poc.py with failure simulation exits with non-zero code.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.model.blocks import load_config
from src.eval.routing_poc_run import (
    DEFAULT_COLLAPSE_THRESHOLD,
    DEFAULT_EVAL_SETS_DIR,
    DEFAULT_KILL_PAIR,
    DEFAULT_KILL_THRESHOLD,
    DEFAULT_MOE_CONFIG,
    RoutingPOCRunConfig,
    SmallMoERoutingModel,
    load_domain_batches,
    run_routing_diagnostics_poc,
    verify_routing_poc_acceptance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. Configuration Validation
# ==============================================================================

def test_routing_poc_config_defaults():
    cfg = RoutingPOCRunConfig()
    assert cfg.moe_config_path == DEFAULT_MOE_CONFIG
    assert cfg.kill_pair == ("typescript", "math")
    assert cfg.kill_threshold == 0.90
    assert cfg.max_collapse_threshold == 0.95
    assert cfg.batch_size == 4
    assert cfg.seq_len == 256
    assert cfg.max_batches == 8


# ==============================================================================
# 2. Tokenized Domain Batch Loading
# ==============================================================================

def test_load_domain_batches_structure():
    batches = load_domain_batches(batch_size=2, seq_len=128, max_batches=3, seed=423)
    assert set(batches.keys()) == {"typescript", "prose", "math"}
    for domain, domain_b in batches.items():
        assert len(domain_b) == 3
        for b in domain_b:
            assert isinstance(b, np.ndarray)
            assert b.shape == (2, 128)
            assert b.dtype == np.uint16
            assert np.all(b < 49152)


def test_load_domain_batches_domain_ranges_separated():
    batches = load_domain_batches(batch_size=2, seq_len=64, max_batches=2, seed=423)
    # TypeScript tokens should average in lower range (< 10000)
    ts_mean = float(np.mean(batches["typescript"][0]))
    assert ts_mean < 10000

    # Prose tokens should average in middle range (10000..25000)
    prose_mean = float(np.mean(batches["prose"][0]))
    assert 10000 <= prose_mean < 25000

    # Math tokens should average in high range (>= 28000)
    math_mean = float(np.mean(batches["math"][0]))
    assert math_mean >= 28000


# ==============================================================================
# 3. SmallMoERoutingModel Duck-Typed Interface
# ==============================================================================

def test_small_moe_routing_model_counting_toggle():
    model = SmallMoERoutingModel(seed=423)
    assert model.n_moe_layers == 16
    assert model.n_experts == 8
    assert model.top_k == 2

    inp = np.zeros((4, 256), dtype=np.uint16)

    # Counting off: counts remain zero
    model.set_moe_load_counting(False)
    model.forward(inp)
    counts = model.pop_moe_load()
    assert sum(sum(l) for l in counts) == 0

    # Counting on: counts accumulate
    model.set_moe_load_counting(True)
    model.forward(inp)
    counts = model.pop_moe_load()
    # 4 * 256 * 2 tokens routed per layer = 2048 tokens per layer
    for layer_counts in counts:
        assert sum(layer_counts) == 4 * 256 * 2

    # Pop drains the counts
    drained = model.pop_moe_load()
    assert sum(sum(l) for l in drained) == 0


# ==============================================================================
# 4. Routing Diagnostics & Specialization Verification (#423 Acceptance)
# ==============================================================================

def test_run_routing_diagnostics_poc_end_to_end_passes_acceptance():
    cfg = RoutingPOCRunConfig(batch_size=4, seq_len=256, max_batches=4, seed=423)
    res = run_routing_diagnostics_poc(config=cfg)

    # Model spec
    assert res["model_spec"]["n_experts"] == 8
    assert res["model_spec"]["top_k"] == 2
    assert res["model_spec"]["n_moe_layers"] == 16

    # Acceptance gate
    acceptance = res["acceptance"]
    assert acceptance["accepted"] is True
    assert acceptance["status"] == "PASSED"

    # Acceptance Criterion 1: TypeScript vs Math overlap < 0.90
    spec_check = acceptance["specialization_check"]
    assert spec_check["passed"] is True
    assert spec_check["overlap"] < 0.90
    assert spec_check["pair"] == "math|typescript"

    # Acceptance Criterion 2: Mid-training routing kill-check verified negative
    kill_check = acceptance["kill_check"]
    assert kill_check["negative"] is True
    assert kill_check["triggered"] is False

    # Starvation check
    starv_check = acceptance["starvation_check"]
    assert starv_check["passed"] is True

    # Collapse check
    collapse_check = acceptance["collapse_check"]
    assert collapse_check["passed"] is True

    # Report structure
    report = res["specialization_report"]
    assert "math|typescript" in report["by_pair"]
    assert "math|prose" in report["by_pair"]
    assert "prose|typescript" in report["by_pair"]
    assert report["code_vs_noncode_overlap"] < 0.95


# ==============================================================================
# 5. Failure Injection & Quality Gate Guardrails
# ==============================================================================

def test_failure_injection_simulate_kill():
    cfg = RoutingPOCRunConfig(batch_size=4, seq_len=256, max_batches=2, seed=423)
    res = run_routing_diagnostics_poc(config=cfg, simulate_kill=True)

    assert res["acceptance"]["accepted"] is False
    assert res["acceptance"]["kill_check"]["negative"] is False
    assert res["acceptance"]["kill_check"]["triggered"] is True
    assert res["acceptance"]["specialization_check"]["passed"] is False
    assert res["verification"]["kill_triggered"] is True


def test_failure_injection_simulate_starvation():
    cfg = RoutingPOCRunConfig(batch_size=4, seq_len=256, max_batches=2, seed=423)
    res = run_routing_diagnostics_poc(config=cfg, simulate_starvation=True)

    assert res["acceptance"]["accepted"] is False
    assert res["acceptance"]["starvation_check"]["passed"] is False
    assert res["verification"]["starvation_check"]["passed"] is False
    assert len(res["verification"]["starvation_check"]["starved_experts"]) > 0


def test_failure_injection_simulate_collapse():
    cfg = RoutingPOCRunConfig(batch_size=4, seq_len=256, max_batches=2, seed=423)
    res = run_routing_diagnostics_poc(config=cfg, simulate_collapse=True)

    assert res["acceptance"]["accepted"] is False
    assert res["acceptance"]["collapse_check"]["passed"] is False
    assert res["verification"]["collapse_check"]["passed"] is False
    assert res["verification"]["collapse_check"]["collapsed"] is True


# ==============================================================================
# 6. Direct Unit Tests for Acceptance Gate
# ==============================================================================

def test_verify_routing_poc_acceptance_blind():
    gate = verify_routing_poc_acceptance({})
    assert gate["accepted"] is False
    assert gate["status"] == "FAILED"


def test_verify_routing_poc_acceptance_passed():
    fake_results = {
        "verification": {
            "passed": True,
            "pair": "math|typescript",
            "overlap": 0.62,
            "threshold": 0.90,
            "kill_check": {"specializing": True, "triggered": False},
            "starvation_check": {"passed": True, "status": "PASSED", "message": "No starvation"},
            "collapse_check": {"passed": True, "status": "PASSED", "message": "No collapse"},
        }
    }
    gate = verify_routing_poc_acceptance(fake_results)
    assert gate["accepted"] is True
    assert gate["status"] == "PASSED"
    assert gate["specialization_check"]["passed"] is True
    assert gate["kill_check"]["negative"] is True


# ==============================================================================
# 7. CLI Integration
# ==============================================================================

def test_cli_mock_pipeline_run(tmp_path):
    out_file = tmp_path / "routing_report.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_routing_poc.py"),
        "--mock-pipeline",
        "--batch-size", "2",
        "--seq-len", "128",
        "--max-batches", "2",
        "--output", str(out_file),
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI run failed: {res.stderr} {res.stdout}"
    assert "ACCEPTANCE PASSED" in res.stdout
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["acceptance"]["accepted"] is True
    assert data["verification"]["overlap"] < 0.90


def test_cli_simulate_kill_fails():
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_routing_poc.py"),
        "--mock-pipeline",
        "--simulate-kill",
        "--batch-size", "2",
        "--seq-len", "128",
        "--max-batches", "2",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 1
    assert "ACCEPTANCE FAILED" in res.stdout
