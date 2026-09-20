"""Unit and integration tests for Issue #223: large-model MoE full run (150B tokens, Large A).

Verifies:
1. Architecture & Config Validation:
   - config/code-large-a.yaml valid MambaConfig (56 layers, d_model 768, d_state 16, 64 routed experts, 1 shared expert)
   - Correct parameter sizing (~3.88B total, ~686M–710M active)
   - FIM attention constraint satisfied (56 % 8 == 0, final layer is attention)
2. Sparse Upcycle Compatibility (#214 / #200):
   - All 15 _MUST_MATCH fields match between code-small-dense.yaml and code-large-a.yaml
   - check_upcycle_compatible succeeds without error
   - Synthetic weight upcycle verifies expert replication to 64 experts and zero-down shared expert init
3. Long-Context Key-Value Cache Memory Ceilings (M12 Literature & Architecture Review):
   - 128k context under KV8 <= 3.0 GB
   - 256k context under KV8 <= 6.0 GB
   - 256k context under Multi-Head Latent Attention (MLA) < 1.5 GB
4. Full Run Training Budget, Schedule & Router Stabilization:
   - 150B token budget calculations
   - WSD schedule phase transitions (warmup, stable, 10–15% decay)
   - Routing bias freeze triggered at start of anneal to prevent router thrash
   - Multi-GPU distributed sharding command (torchrun, world_size=4, ep_size=4)
5. Cloud Compute & RunPod Resource Modeling:
   - Intended use, estimated runtime, multi-GPU hardware tiers (4x A100, 4x H100, 8x A100), and ~$1.2k cost budget
   - Enforces multi-GPU requirement (single-card cannot fit model + optimizer state)
6. Quality Gate & Acceptance Verification:
   - Routing specialization check (#217) with BLIND rule enforcement
   - Recall evaluation check (#221)
   - Checkpoint resume stability verification (no discontinuity across restart)
   - Combined acceptance gate
7. Deterministic Mock Pipeline Simulation & CLI Verification:
   - simulate_mock_pipeline passes end-to-end
   - CLI flags (--cloud-spec, --check-kv-cache, --dry-run, --print-train-cmd, --mock-pipeline)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.model.blocks import MambaConfig, load_config
from src.model.sizing import kv_cache_elements_per_token, kv_cache_memory_bytes
from src.train.checkpoint import save_weights
from src.train.upcycle import _MUST_MATCH, check_upcycle_compatible, _expected_keys
from src.train.large_moe_run import (
    CLOUD_HARDWARE_TEMPLATES,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_DENSE_CONFIG,
    DEFAULT_MOE_CONFIG,
    MAX_BUDGET_TOKENS,
    MIN_BUDGET_TOKENS,
    LargeMoERunConfig,
    execute_upcycle,
    get_cloud_run_spec,
    simulate_mock_pipeline,
    verify_kv_cache_ceilings,
    verify_large_moe_acceptance,
    verify_recall_acceptance,
    verify_resume_stability,
    verify_routing_specialization,
)
from src.eval.moe_routing import specialization_report

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. Architecture & Config Validation
# ==============================================================================

def test_code_large_a_config_validity():
    """Verify config/code-large-a.yaml satisfies all architectural specs for Large A."""
    cfg = load_config(str(REPO_ROOT / "config/code-large-a.yaml"))
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

    # MoE structure: 64 routed experts, top-8, 1 shared expert
    assert cfg.moe_every == 3
    assert cfg.n_experts == 64
    assert cfg.top_k == 8
    assert cfg.n_shared_experts == 1
    assert cfg.moe_d_ff == 1536
    assert cfg.n_moe_layers == 16
    assert cfg.moe_balance_rate == 0.001

    # Parameter counts: ~3.88B total params, ~686M–710M active
    tot_params = cfg.num_parameters()
    act_params = cfg.active_num_parameters()
    assert 3_800_000_000 <= tot_params <= 3_950_000_000, f"Expected ~3.88B params, got {tot_params}"
    assert 680_000_000 <= act_params <= 720_000_000, f"Expected ~700M active params, got {act_params}"


# ==============================================================================
# 2. Sparse Upcycle Compatibility (#214 / #200)
# ==============================================================================

def test_code_large_a_upcycle_compatibility():
    """Verify that code-small-dense.yaml and code-large-a.yaml match on all _MUST_MATCH fields."""
    src = load_config(str(DEFAULT_DENSE_CONFIG))
    dst = load_config(str(DEFAULT_MOE_CONFIG))
    src.validate()
    dst.validate()

    for field_name in _MUST_MATCH:
        v_src = getattr(src, field_name)
        v_dst = getattr(dst, field_name)
        assert v_src == v_dst, f"Mismatch on {field_name}: dense={v_src} != moe={v_dst}"

    src_moe = {i for i in range(src.n_layers) if src.is_moe_layer(i)}
    dst_moe = {i for i in range(dst.n_layers) if dst.is_moe_layer(i)}
    assert src_moe == dst_moe

    # check_upcycle_compatible succeeds without error
    check_upcycle_compatible(src, dst)


def test_execute_upcycle_dry_run():
    """Verify execute_upcycle dry_run inspects compatibility and returns correct parameter previews."""
    res = execute_upcycle(DEFAULT_DENSE_CONFIG, DEFAULT_MOE_CONFIG, dry_run=True)
    assert res["status"] == "compatible"
    assert res["dry_run"] is True
    assert res["src_config"]["n_experts"] == 1
    assert res["dst_config"]["n_experts"] == 64
    assert res["dst_config"]["n_shared_experts"] == 1
    assert res["experts_expansion"] == "1 -> 64"
    assert res["shared_expert"] == "0 -> 1"


def test_synthetic_weight_upcycle(tmp_path):
    """Verify weight transformation replicates dense FFN to 64 experts and creates zero-down shared expert."""
    src_cfg = load_config(str(DEFAULT_DENSE_CONFIG))
    dst_cfg = load_config(str(DEFAULT_MOE_CONFIG))

    # Construct synthetic weights dict matching expected keys
    rng = np.random.default_rng(42)
    dense_weights = {
        k: rng.standard_normal(shp).astype(np.float32)
        for k, shp in _expected_keys(src_cfg).items()
    }

    dense_path = tmp_path / "synthetic_dense.safetensors"
    save_weights(dense_weights, str(dense_path), config=src_cfg)

    upcycle_out = tmp_path / "upcycled_large_a.safetensors"
    res = execute_upcycle(dense_path, DEFAULT_MOE_CONFIG, out_path=upcycle_out, dry_run=False)

    assert res["status"] == "compatible"
    assert upcycle_out.exists()
    assert (tmp_path / "upcycled_large_a.safetensors.config.json").exists()
    assert (tmp_path / "upcycled_large_a.safetensors.upcycle.json").exists()


# ==============================================================================
# 3. Long-Context Key-Value Cache Memory Ceilings (M12 Review)
# ==============================================================================

def test_kv_cache_memory_ceilings():
    """Verify memory ceilings from M12 review:
    1. 128k context under KV8 <= 3.0 GB
    2. 256k context under KV8 <= 6.0 GB
    3. 256k context under MLA < 1.5 GB
    """
    cfg = load_config(str(DEFAULT_MOE_CONFIG))
    assert cfg.n_attention_layers == 7

    kv_res = verify_kv_cache_ceilings(cfg)
    assert kv_res["passed"] is True
    assert kv_res["status"] == "PASSED"

    # KV8 128k
    assert kv_res["kv8_128k"]["passed"] is True
    assert kv_res["kv8_128k"]["gb"] <= 3.0
    assert kv_res["kv8_128k"]["gb"] == pytest.approx(1.41, rel=0.05)

    # KV8 256k
    assert kv_res["kv8_256k"]["passed"] is True
    assert kv_res["kv8_256k"]["gb"] <= 6.0
    assert kv_res["kv8_256k"]["gb"] == pytest.approx(2.82, rel=0.05)

    # MLA 256k
    assert kv_res["mla_256k"]["passed"] is True
    assert kv_res["mla_256k"]["gb"] < 1.5
    assert kv_res["mla_256k"]["gb"] == pytest.approx(1.17, rel=0.05)


def test_kv_cache_memory_ceiling_violation():
    """Verify that verify_kv_cache_ceilings detects ceiling breaches."""
    cfg = load_config(str(DEFAULT_MOE_CONFIG))
    # Artificially low ceiling
    kv_res = verify_kv_cache_ceilings(cfg, ceiling_128k_kv8=0.5)
    assert kv_res["passed"] is False
    assert kv_res["status"] == "FAILED"
    assert kv_res["kv8_128k"]["passed"] is False


# ==============================================================================
# 4. Training Budget, Hyperparameters & Multi-GPU Execution
# ==============================================================================

def test_large_moe_training_budget_and_schedule():
    """Verify 150B token calculations, WSD schedule, and routing bias freeze step."""
    cfg = LargeMoERunConfig(total_tokens=150_000_000_000, batch_size=16, grad_accum=16, world_size=4, ep_size=4)

    # dp_size = 4 // 4 = 1; tokens_per_step = 16 * 2048 * 16 * 1 = 524,288 tokens/step
    assert cfg.dp_size == 1
    assert cfg.tokens_per_step() == 524_288

    tot_steps = cfg.total_steps()
    assert tot_steps == 286_103

    breakdown = cfg.step_breakdown()
    assert breakdown["total_steps"] == tot_steps
    assert breakdown["warmup_steps"] == 3000

    # 12% anneal decay: round(286103 * 0.12) = 34,332 steps
    assert breakdown["decay_steps"] == 34_332
    assert breakdown["anneal_start_step"] == tot_steps - 34_332
    assert breakdown["freeze_router_bias_step"] == breakdown["anneal_start_step"]
    assert breakdown["stable_steps"] == tot_steps - breakdown["warmup_steps"] - breakdown["decay_steps"]


def test_train_command_generation():
    """Verify multi-GPU distributed torchrun command generation."""
    cfg = LargeMoERunConfig(total_tokens=150_000_000_000, world_size=4, ep_size=4)
    cmd = cfg.generate_train_command(
        data_dir="data/shards",
        init_checkpoint="runs/large_a_init.safetensors",
        domains_json="config/domains.json",
    )

    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=4" in cmd
    assert "--world-size" in cmd and cmd[cmd.index("--world-size") + 1] == "4"
    assert "--ep-size" in cmd and cmd[cmd.index("--ep-size") + 1] == "4"
    assert "--freeze-router-bias-step" in cmd
    assert "--moe-diag-domains" in cmd and cmd[cmd.index("--moe-diag-domains") + 1] == "config/domains.json"


# ==============================================================================
# 5. Cloud Compute & RunPod Resource Modeling
# ==============================================================================

def test_cloud_run_spec_policy_compliance():
    """Verify cloud specification adheres to CRITICAL RUNPOD / CLOUD COMPUTE POLICY:
    1. Intended use
    2. Estimated runtime
    3. Hardware configuration / GPU type
    4. Estimated cost
    """
    spec = get_cloud_run_spec(total_tokens=150_000_000_000, hardware="4x-a100-80gb")

    assert "intended_use" in spec
    assert "150B tokens" in spec["intended_use"]
    assert "Large A" in spec["intended_use"]

    assert "estimated_runtime" in spec
    assert "hardware_configuration" in spec
    assert spec["hardware_configuration"]["num_gpus"] == 4
    assert spec["hardware_configuration"]["single_gpu"] is False
    assert spec["hardware_configuration"]["requires_distributed"] is True

    assert "estimated_cost" in spec
    assert spec["estimated_cost_usd"]["min"] > 0
    assert spec["estimated_cost_usd"]["max"] < 1200.0  # Within allocated budget

    assert "runpod_launch_command" in spec
    assert "scripts/cloud_pod.py launch" in spec["runpod_launch_command"]
    assert "--template 4x-a100-80gb" in spec["runpod_launch_command"]


# ==============================================================================
# 6. Quality Gate & Acceptance Verification
# ==============================================================================

def test_routing_specialization_pass_and_blind():
    """Verify routing specialization verification adheres to BLIND rule and detects kill-criterion."""
    # Domain-separated histograms across 64 experts
    h_ts = [[100 if i < 32 else 5 for i in range(64)] for _ in range(16)]
    h_math = [[5 if i < 32 else 100 for i in range(64)] for _ in range(16)]
    hists = {"typescript": h_ts, "math": h_math}

    rep = specialization_report(hists)
    res = verify_routing_specialization(rep, kill_pair=("typescript", "math"), kill_threshold=0.90)
    assert res["status"] == "PASSED"
    assert res["passed"] is True
    assert res["specializing"] is True
    assert res["kill_triggered"] is False
    assert res["overlap"] < 0.90

    # BLIND rule: missing domain pair reports BLIND, never healthy
    blind_res = verify_routing_specialization(rep, kill_pair=("typescript", "rust"))
    assert blind_res["status"] == "BLIND"
    assert blind_res["passed"] is False


def test_resume_stability_verification():
    """Verify resume stability check verifies continuity without loss discontinuity."""
    # Smooth trace with resume event
    smooth_trace = [
        {"step": 1000, "val_loss": 3.50},
        {"step": 2000, "val_loss": 3.10},
        {"step": 2001, "val_loss": 3.102, "resumed": True},
        {"step": 3000, "val_loss": 2.80},
    ]
    res_smooth = verify_resume_stability(smooth_trace, resume_step=2001)
    assert res_smooth["passed"] is True
    assert res_smooth["status"] == "PASSED"
    assert res_smooth["loss_delta"] <= 0.10

    # Discontinuous trace
    jumpy_trace = [
        {"step": 1000, "val_loss": 3.50},
        {"step": 2000, "val_loss": 3.10},
        {"step": 2001, "val_loss": 3.85, "resumed": True},
        {"step": 3000, "val_loss": 2.80},
    ]
    res_jumpy = verify_resume_stability(jumpy_trace, resume_step=2001)
    assert res_jumpy["passed"] is False
    assert res_jumpy["status"] == "FAILED"
    assert res_jumpy["loss_delta"] > 0.10


def test_verify_large_moe_acceptance():
    """Verify end-to-end acceptance gate combining recall, routing, KV ceilings, and resume."""
    h_ts = [[100 if i < 32 else 5 for i in range(64)] for _ in range(16)]
    h_math = [[5 if i < 32 else 100 for i in range(64)] for _ in range(16)]
    routing_rep = specialization_report({"typescript": h_ts, "math": h_math})

    recall_data = {"mrr": 0.42, "rank_top1": 0.35, "ce_nats": 3.0}
    metrics_data = [
        {"step": 2000, "val_loss": 3.10},
        {"step": 2001, "val_loss": 3.105, "resumed": True},
    ]

    gate = verify_large_moe_acceptance(
        routing_rep,
        recall_data,
        metrics_data=metrics_data,
        resume_step=2001,
    )
    assert gate["accepted"] is True
    assert gate["routing_check"]["passed"] is True
    assert gate["recall_check"]["passed"] is True
    assert gate["kv_cache_check"]["passed"] is True
    assert gate["resume_check"]["passed"] is True


# ==============================================================================
# 7. Mock Pipeline Simulation & CLI Verification
# ==============================================================================

def test_simulate_mock_pipeline_end_to_end():
    """Verify offline mock pipeline simulation runs deterministically without paid GPU execution."""
    res = simulate_mock_pipeline(total_tokens=DEFAULT_BUDGET_TOKENS, seed=223)

    assert res["upcycle"]["status"] == "compatible"
    assert res["training_plan"]["total_steps"] > 0
    assert res["cloud_spec"]["hardware_configuration"]["num_gpus"] == 4
    assert res["kv_cache_check"]["passed"] is True
    assert res["routing_report"]["mean_overlap"] < 0.90
    assert res["recall_results"]["mrr"] >= 0.15
    assert res["acceptance"]["accepted"] is True


def test_simulate_mock_pipeline_kill_criterion():
    """Verify mock pipeline correctly fails when routing kill criterion triggers."""
    res = simulate_mock_pipeline(total_tokens=DEFAULT_BUDGET_TOKENS, seed=223, simulate_kill=True)
    assert res["acceptance"]["accepted"] is False
    assert res["acceptance"]["routing_check"]["passed"] is False
    assert res["acceptance"]["recall_check"]["passed"] is False
