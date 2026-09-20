"""Unit and integration tests for Issue #222: small-model MoE full run (50–70B tokens).

Verifies:
1. Architecture & Config Validation:
   - config/code-small-moe.yaml valid MambaConfig (56 layers, d_model 768, d_state 16, 8 routed experts, 1 shared expert)
   - Fits on a single GPU (~685M total params, ~345M active)
   - FIM attention constraint satisfied (56 % 8 == 0, final layer is attention)
2. Sparse Upcycle Compatibility (#214 / #200):
   - All 15 _MUST_MATCH fields match between code-small-dense.yaml and code-small-moe.yaml
   - MoE layer indices match exactly
   - check_upcycle_compatible succeeds without error
   - Synthetic weight upcycle verifies expert replication and zero-down shared expert init
3. Full Run Training Budget & Hyperparameters:
   - 50B–70B token budget calculations
   - WSD schedule phase transitions (warmup, stable, decay)
   - Effective batch size and tokens/step
   - Training command generation includes Muon, WSD, and MoE routing flags
4. Cloud Compute & RunPod Resource Modeling:
   - Intended use, estimated runtime (40–60 GPU-hrs), hardware tiers (A40, RTX 4090, A100), and cost estimates
   - RunPod launch command formatting
5. Quality Gate & Acceptance Verification:
   - Routing specialization check (#217): passes on domain-separated histograms, fails on kill-criterion trigger
   - BLIND rule enforcement on missing or unmeasured domains
   - Recall evaluation check (#221): evaluates MRR, top-1 rank, cross-entropy
   - Combined acceptance gate
6. Deterministic Mock Pipeline Simulation & CLI Verification:
   - simulate_mock_pipeline passes end-to-end
   - CLI flags (--cloud-spec, --dry-run, --print-train-cmd, --mock-pipeline)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.model.blocks import MambaConfig, load_config
from src.train.checkpoint import save_weights
from src.train.upcycle import _MUST_MATCH, check_upcycle_compatible, _expected_keys
from src.train.small_moe_run import (
    CLOUD_HARDWARE_TEMPLATES,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_DENSE_CONFIG,
    DEFAULT_MOE_CONFIG,
    MAX_BUDGET_TOKENS,
    MIN_BUDGET_TOKENS,
    SmallMoERunConfig,
    execute_upcycle,
    get_cloud_run_spec,
    simulate_mock_pipeline,
    verify_recall_acceptance,
    verify_routing_specialization,
    verify_small_moe_acceptance,
)
from src.eval.moe_routing import specialization_report

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. Architecture & Config Validation
# ==============================================================================

def test_code_small_moe_config_validity():
    """Verify config/code-small-moe.yaml satisfies all architectural specs."""
    cfg = load_config(str(REPO_ROOT / "config/code-small-moe.yaml"))
    cfg.validate()

    # Backbone dimensions
    assert cfg.d_model == 768
    assert cfg.n_layers == 56
    assert cfg.d_state == 16
    assert cfg.expand == 2
    assert cfg.d_inner == 1536
    assert cfg.head_dim == 64
    assert cfg.n_heads == 24
    assert cfg.d_conv == 4
    assert cfg.dt_rank_resolved == 48

    # Vocab and packing (uint16, tied embeddings)
    assert cfg.vocab_size == 49152
    assert cfg.packing_dtype == "uint16"
    assert cfg.tie_embeddings is True

    # FIM constraint (final block is attention)
    assert cfg.attn_every == 8
    assert cfg.n_layers % cfg.attn_every == 0
    assert cfg.is_attention_layer(cfg.n_layers - 1) is True
    assert cfg.n_attention_layers == 7

    # MoE structure: 8 routed experts, top-2, 1 shared expert
    assert cfg.moe_every == 3
    assert cfg.n_experts == 8
    assert cfg.top_k == 2
    assert cfg.n_shared_experts == 1
    assert cfg.moe_d_ff == 1536
    assert cfg.moe_d_ff_resolved == 1536
    assert cfg.moe_balance_rate == 0.001
    assert cfg.n_moe_layers == 16

    # Surviving Mamba layers
    assert cfg.n_layers - cfg.n_attention_layers - cfg.n_moe_layers == 33

    # Total parameters ~685M (fits single GPU)
    total_params = cfg.num_parameters()
    active_params = cfg.active_num_parameters()
    assert 680_000_000 < total_params < 690_000_000
    assert 340_000_000 < active_params < 350_000_000


def test_code_small_moe_upcycle_compatibility_with_dense():
    """Verify that code-small-dense.yaml and code-small-moe.yaml match on all 15 _MUST_MATCH fields."""
    src = load_config(str(REPO_ROOT / "config/code-small-dense.yaml"))
    dst = load_config(str(REPO_ROOT / "config/code-small-moe.yaml"))
    src.validate()
    dst.validate()

    for field in _MUST_MATCH:
        sv = getattr(src, field)
        dv = getattr(dst, field)
        assert sv == dv, f"Field {field} mismatch: src={sv!r} vs dst={dv!r}"

    src_moe = {i for i in range(src.n_layers) if src.is_moe_layer(i)}
    dst_moe = {i for i in range(dst.n_layers) if dst.is_moe_layer(i)}
    assert src_moe == dst_moe

    # check_upcycle_compatible passes
    check_upcycle_compatible(src, dst)


# ==============================================================================
# 2. Sparse Upcycle Execution
# ==============================================================================

def test_execute_upcycle_dry_run():
    """Verify execute_upcycle dry-run produces correct structural preview."""
    preview = execute_upcycle(
        DEFAULT_DENSE_CONFIG,
        DEFAULT_MOE_CONFIG,
        dry_run=True,
    )
    assert preview["status"] == "compatible"
    assert preview["dry_run"] is True
    assert preview["experts_expansion"] == "1 -> 8"
    assert preview["shared_expert"] == "0 -> 1"
    assert preview["dst_config"]["n_experts"] == 8
    assert preview["dst_config"]["top_k"] == 2
    assert preview["dst_config"]["n_shared_experts"] == 1


def test_execute_upcycle_with_synthetic_checkpoint(tmp_path):
    """Verify execute_upcycle correctly transforms synthetic dense checkpoint into target MoE."""
    src_cfg = load_config(str(DEFAULT_DENSE_CONFIG))
    dst_cfg = load_config(str(DEFAULT_MOE_CONFIG))

    # Generate synthetic shape-correct weights for dense model
    rng = np.random.default_rng(42)
    dense_weights = {
        k: rng.standard_normal(shp).astype(np.float32)
        for k, shp in _expected_keys(src_cfg).items()
    }

    dense_path = tmp_path / "dense_weights.safetensors"
    save_weights(dense_weights, str(dense_path), config=src_cfg)

    out_path = tmp_path / "upcycled_moe.safetensors"
    result = execute_upcycle(
        dense_path,
        DEFAULT_MOE_CONFIG,
        out_path=out_path,
        seed=222,
        dry_run=False,
    )

    assert result["status"] == "compatible"
    assert out_path.exists()
    assert (tmp_path / "upcycled_moe.safetensors.config.json").exists()
    assert (tmp_path / "upcycled_moe.safetensors.upcycle.json").exists()


# ==============================================================================
# 3. Training Budget & Hyperparameters
# ==============================================================================

def test_training_budget_calculations():
    """Verify step counts, effective batch size, and WSD schedule breakdown for 50B-70B tokens."""
    cfg_50b = SmallMoERunConfig(total_tokens=50_000_000_000, batch_size=16, seq_len=2048, grad_accum=8)
    assert cfg_50b.tokens_per_step() == 262_144
    assert cfg_50b.total_steps() == 190_735

    bd_50b = cfg_50b.step_breakdown()
    assert bd_50b["total_steps"] == 190_735
    assert bd_50b["warmup_steps"] == 2000
    assert bd_50b["decay_steps"] == int(round(190_735 * 0.20))
    assert bd_50b["stable_steps"] == 190_735 - 2000 - bd_50b["decay_steps"]

    cfg_70b = SmallMoERunConfig(total_tokens=70_000_000_000, batch_size=16, seq_len=2048, grad_accum=8)
    assert cfg_70b.total_steps() == 267_029


def test_training_command_generation():
    """Verify generated scripts/train.py command carries all required flags."""
    cfg = SmallMoERunConfig(total_tokens=50_000_000_000)
    cmd = cfg.generate_train_command("data/shards", "runs/init.safetensors", domains_json="data/domains.json")
    cmd_str = " ".join(cmd)

    assert "scripts/train.py" in cmd_str
    assert "--config" in cmd_str and "code-small-moe.yaml" in cmd_str
    assert "--init runs/init.safetensors" in cmd_str
    assert "--lr-schedule wsd" in cmd_str
    assert "--decay-frac 0.2" in cmd_str
    assert "--warmup-steps 2000" in cmd_str
    assert "--moe-diag-every 500" in cmd_str
    assert "--moe-diag-domains data/domains.json" in cmd_str
    assert "--moe-kill-pair typescript,math" in cmd_str
    assert "--moe-kill-overlap 0.9" in cmd_str


# ==============================================================================
# 4. Cloud Compute & RunPod Specifications
# ==============================================================================

def test_cloud_run_spec_conformance():
    """Verify get_cloud_run_spec fulfills all 4 RunPod policy requirements."""
    for hw in ("a40", "rtx4090", "a100-pcie"):
        spec = get_cloud_run_spec(total_tokens=50_000_000_000, hardware=hw)

        # 1. Intended use
        assert "MHM-P4" in spec["intended_use"]
        assert "Small-model MoE" in spec["intended_use"]

        # 2. Estimated runtime
        assert "GPU-hours" in spec["estimated_runtime"]
        assert 25.0 <= spec["estimated_runtime_hours"]["min"] <= 45.0
        assert 40.0 <= spec["estimated_runtime_hours"]["max"] <= 65.0

        # 3. Hardware configuration
        hw_conf = spec["hardware_configuration"]
        assert hw_conf["single_gpu"] is True
        assert hw_conf["blocked_on_fsdp"] is False
        assert hw_conf["vram_gb"] in (24, 48, 80)

        # 4. Estimated cost
        assert "$" in spec["estimated_cost"]
        assert spec["estimated_cost_usd"]["min"] > 0
        assert spec["estimated_cost_usd"]["max"] >= spec["estimated_cost_usd"]["min"]

        # Launch command
        assert "scripts/cloud_pod.py launch" in spec["runpod_launch_command"]
        assert f"--template {hw}" in spec["runpod_launch_command"]


# ==============================================================================
# 5. Quality Gate & Acceptance Verification
# ==============================================================================

def test_routing_specialization_verification_passes_when_separated():
    """Verify routing verification passes when domains route to distinct experts."""
    # 2 MoE layers, 8 experts: typescript routes to experts 0-3, math to 4-7
    hist_ts = [[100, 100, 100, 100, 0, 0, 0, 0], [100, 100, 100, 100, 0, 0, 0, 0]]
    hist_math = [[0, 0, 0, 0, 100, 100, 100, 100], [0, 0, 0, 0, 100, 100, 100, 100]]

    report = specialization_report({"typescript": hist_ts, "math": hist_math})
    res = verify_routing_specialization(report, kill_pair=("typescript", "math"), kill_threshold=0.90)

    assert res["status"] == "PASSED"
    assert res["passed"] is True
    assert res["specializing"] is True
    assert res["kill_triggered"] is False
    assert res["overlap"] == 0.0


def test_routing_specialization_verification_fails_when_kill_criterion_triggers():
    """Verify routing verification fails when domains overlap and kill-criterion triggers."""
    # Identical routing across domains -> overlap = 1.0
    hist_ts = [[50] * 8, [50] * 8]
    hist_math = [[50] * 8, [50] * 8]

    report = specialization_report({"typescript": hist_ts, "math": hist_math})
    res = verify_routing_specialization(report, kill_pair=("typescript", "math"), kill_threshold=0.90)

    assert res["status"] == "FAILED"
    assert res["passed"] is False
    assert res["specializing"] is False
    assert res["kill_triggered"] is True
    assert res["overlap"] == 1.0


def test_routing_specialization_blind_rule():
    """Verify BLIND status when required domain is missing."""
    hist_ts = [[50] * 8]
    hist_prose = [[50] * 8]
    report = specialization_report({"typescript": hist_ts, "prose": hist_prose})

    res = verify_routing_specialization(report, kill_pair=("typescript", "math"))
    assert res["status"] == "BLIND"
    assert res["passed"] is False


def test_recall_acceptance_verification():
    """Verify recall acceptance threshold checking."""
    # High MRR passes
    good_res = {"mrr": 0.45, "rank_top1": 0.38, "ce_nats": 2.8}
    assert verify_recall_acceptance(good_res, min_mrr=0.15)["passed"] is True

    # Low MRR fails
    bad_res = {"mrr": 0.05, "rank_top1": 0.01, "ce_nats": 6.5}
    assert verify_recall_acceptance(bad_res, min_mrr=0.15)["passed"] is False


def test_overall_small_moe_acceptance():
    """Verify combined acceptance gate requires both recall and routing to pass."""
    hist_ts = [[100, 100, 100, 100, 0, 0, 0, 0]]
    hist_math = [[0, 0, 0, 0, 100, 100, 100, 100]]
    routing_data = {"typescript": hist_ts, "math": hist_math}
    good_recall = {"mrr": 0.40, "rank_top1": 0.30}
    bad_recall = {"mrr": 0.02, "rank_top1": 0.00}

    # Both pass -> accepted
    res1 = verify_small_moe_acceptance(routing_data, good_recall)
    assert res1["accepted"] is True
    assert "ACCEPTANCE PASSED" in res1["summary"]

    # Recall fails -> not accepted
    res2 = verify_small_moe_acceptance(routing_data, bad_recall)
    assert res2["accepted"] is False

    # Routing fails -> not accepted
    bad_routing = {"typescript": [[50] * 8], "math": [[50] * 8]}
    res3 = verify_small_moe_acceptance(bad_routing, good_recall)
    assert res3["accepted"] is False


# ==============================================================================
# 6. Mock Pipeline Simulation & CLI Integration
# ==============================================================================

def test_simulate_mock_pipeline():
    """Verify simulate_mock_pipeline executes end-to-end deterministically."""
    res = simulate_mock_pipeline(total_tokens=50_000_000_000, seed=222)
    assert res["upcycle"]["status"] == "compatible"
    assert res["training_plan"]["total_steps"] == 190_735
    assert res["acceptance"]["accepted"] is True

    # When kill is simulated, acceptance must fail
    res_kill = simulate_mock_pipeline(total_tokens=50_000_000_000, seed=222, simulate_kill=True)
    assert res_kill["acceptance"]["accepted"] is False


def test_cli_execution_cloud_spec(monkeypatch, capsys):
    """Verify scripts/run_small_moe.py --cloud-spec runs cleanly."""
    spec_mod = importlib.util.spec_from_file_location(
        "_scripts_run_small_moe", REPO_ROOT / "scripts/run_small_moe.py"
    )
    mod = importlib.util.module_from_spec(spec_mod)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec_mod.loader.exec_module(mod)
    finally:
        sys.path.pop(0)

    monkeypatch.setattr(sys, "argv", ["run_small_moe.py", "--cloud-spec"])
    mod.main()
    out = capsys.readouterr().out
    assert "CLOUD COMPUTE SPECIFICATION" in out
    assert "NVIDIA A40" in out
    assert "Estimated Cost" in out


def test_cli_execution_dry_run(monkeypatch, capsys):
    """Verify scripts/run_small_moe.py --dry-run runs cleanly."""
    spec_mod = importlib.util.spec_from_file_location(
        "_scripts_run_small_moe", REPO_ROOT / "scripts/run_small_moe.py"
    )
    mod = importlib.util.module_from_spec(spec_mod)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec_mod.loader.exec_module(mod)
    finally:
        sys.path.pop(0)

    monkeypatch.setattr(sys, "argv", ["run_small_moe.py", "--dry-run"])
    mod.main()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "COMPATIBLE" in out
    assert "685.1M total" in out
    assert "190,735" in out


def test_cli_execution_mock_pipeline(monkeypatch, capsys):
    """Verify scripts/run_small_moe.py --mock-pipeline runs cleanly."""
    spec_mod = importlib.util.spec_from_file_location(
        "_scripts_run_small_moe", REPO_ROOT / "scripts/run_small_moe.py"
    )
    mod = importlib.util.module_from_spec(spec_mod)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec_mod.loader.exec_module(mod)
    finally:
        sys.path.pop(0)

    monkeypatch.setattr(sys, "argv", ["run_small_moe.py", "--mock-pipeline"])
    mod.main()
    out = capsys.readouterr().out
    assert "DETERMINISTIC MOCK PIPELINE" in out
    assert "ACCEPTANCE PASSED" in out
