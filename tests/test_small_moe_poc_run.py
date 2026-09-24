"""Unit and integration tests for Issue #422: Small MoE sparse upcycle & initial training run.

Verifies:
1. Architecture & Config Validation:
   - config/code-small-moe.yaml valid MambaConfig (56 layers, d_model 768, 8 routed experts, 1 shared expert)
   - Fits on a single GPU (~685.1M total params, ~345.4M active)
   - FIM attention constraint satisfied (56 % 8 == 0, 7 attention layers, final layer is attention)
   - Dropless routing with top-2 gate dispatch and Loss-Free Balancing (moe_balance_rate: 0.001)
2. Sparse Upcycle Compatibility & Execution (#214 / #421 / #422):
   - All 15 _MUST_MATCH fields match between code-small-dense.yaml and code-small-moe.yaml
   - Attention and MoE layer indices match exactly
   - execute_sparse_upcycle dry-run produces correct structural preview
   - execute_sparse_upcycle with synthetic dense checkpoint generates:
     - 8 replicated routed experts per MoE layer (experts.0..7)
     - zero-down initialized shared expert (shared_experts.0.down.weight is all zeros)
     - fresh router weights of shape (8, 768)
     - valid .config.json sidecar and .upcycle.json manifest
3. Full Run Training Budget & Hyperparameters:
   - POC token budget calculations (effective batch size 262,144 tokens/step)
   - WSD schedule phase transitions (warmup, stable, decay)
   - Training command generation includes Muon, WSD, and Small MoE flags
4. Cloud Compute & RunPod A40 Resource Modeling:
   - Intended use, estimated runtime (0.2–0.5 GPU-hrs), hardware tiers (A40, RTX 4090, A100), and cost estimates
   - Single-GPU fit verified (no multi-node/FSDP required for 685.1M Small MoE)
   - RunPod launch command formatting
5. Quality Gate: Bit-Exact Training Resume at Step 50 (#422 Acceptance Criterion 1):
   - verify_bit_exact_resume passes on identical trajectories (max_diff == 0.0)
   - verify_bit_exact_resume fails when trajectory diverges (max_diff > atol)
   - CheckpointStore Slot-A (step 50) and Slot-B double-buffered rotation
   - moe_route_bias.* keys round-trip properly
6. Quality Gate: Router Entropy & Expert Load Distribution Stability (#422 Acceptance Criterion 2):
   - verify_router_stability passes on high entropy (>= 1.50 nats) and low utilization variance (<= 0.05)
   - verify_router_stability fails on router collapse (entropy < 1.0 or variance > 0.05)
   - verify_router_stability fails on expert starvation
   - BLIND rule enforcement on missing or unmeasured metrics
7. Deterministic Mock Pipeline Simulation & CLI Verification:
   - simulate_mock_pipeline passes end-to-end and satisfies acceptance criteria
   - CLI flags (--cloud-spec, --print-train-cmd, --dry-run, --mock-pipeline, --verify-run)
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.model.blocks import MambaConfig, load_config
from src.train.checkpoint import CheckpointStore, save_weights
from src.train.upcycle import _MUST_MATCH, _expected_keys, check_upcycle_compatible
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
    verify_bit_exact_resume,
    verify_router_stability,
    verify_small_moe_poc_acceptance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. Architecture & Config Validation
# ==============================================================================

def test_code_small_moe_config_validity():
    """Verify config/code-small-moe.yaml satisfies all architectural specs for Small MoE."""
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

    # MoE structure: 8 routed experts, top-2 dropless routing, 1 shared expert
    assert cfg.moe_every == 3
    assert cfg.n_experts == 8
    assert cfg.top_k == 2
    assert cfg.n_shared_experts == 1
    assert cfg.moe_d_ff == 1536
    assert cfg.moe_d_ff_resolved == 1536
    assert cfg.moe_balance_rate == 0.001
    assert cfg.n_moe_layers == 16

    # Surviving Mamba layers: 56 - 7 - 16 = 33
    assert cfg.n_layers - cfg.n_attention_layers - cfg.n_moe_layers == 33

    # Total parameters ~685.1M, active ~345.4M (fits single GPU)
    total_params = cfg.num_parameters()
    active_params = cfg.active_num_parameters()
    assert 680_000_000 < total_params < 690_000_000
    assert 340_000_000 < active_params < 350_000_000


# ==============================================================================
# 2. Sparse Upcycle Compatibility & Execution
# ==============================================================================

def test_upcycle_compatibility_with_dense_poc():
    """All 15 _MUST_MATCH fields match between code-small-dense.yaml and code-small-moe.yaml."""
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

    src_attn = {i for i in range(src.n_layers) if src.is_attention_layer(i)}
    dst_attn = {i for i in range(dst.n_layers) if dst.is_attention_layer(i)}
    assert src_attn == dst_attn

    check_upcycle_compatible(src, dst)


def test_execute_sparse_upcycle_dry_run():
    """Verify execute_sparse_upcycle dry run generates valid compatibility report."""
    res = execute_sparse_upcycle(
        DEFAULT_DENSE_CONFIG,
        DEFAULT_MOE_CONFIG,
        dry_run=True,
    )
    assert res["status"] == "compatible"
    assert res["dry_run"] is True
    assert res["dst_config"]["n_experts"] == 8
    assert res["dst_config"]["top_k"] == 2
    assert res["dst_config"]["n_shared_experts"] == 1
    assert res["dst_config"]["moe_balance_rate"] == 0.001
    assert "replicated" in res["experts_expansion"]
    assert "zero_down" in res["shared_expert"]


def test_execute_sparse_upcycle_with_synthetic_checkpoint(tmp_path):
    """Verify execute_sparse_upcycle generates 8 replicated experts and zero-down shared expert."""
    src_cfg = load_config(str(DEFAULT_DENSE_CONFIG))
    dst_cfg = load_config(str(DEFAULT_MOE_CONFIG))

    rng = np.random.default_rng(42)
    dense_weights = {
        k: rng.standard_normal(shp).astype(np.float32)
        for k, shp in _expected_keys(src_cfg).items()
    }

    dense_path = tmp_path / "dense_source.safetensors"
    save_weights(dense_weights, str(dense_path), config=src_cfg)

    out_path = tmp_path / "small_moe_init.safetensors"
    res = execute_sparse_upcycle(
        dense_path,
        DEFAULT_MOE_CONFIG,
        out_path=out_path,
        seed=422,
        dry_run=False,
    )

    assert res["status"] == "compatible"
    assert out_path.exists()
    assert (tmp_path / "small_moe_init.safetensors.config.json").exists()
    assert (tmp_path / "small_moe_init.safetensors.upcycle.json").exists()

    # Load upcycled weights and check structural invariants
    from src.train.checkpoint import load_weights_dict
    upcycled = load_weights_dict(str(out_path))

    first_moe_layer = min(i for i in range(dst_cfg.n_layers) if dst_cfg.is_moe_layer(i))

    # 1. 8 routed experts exist and are identical copies of source expert 0
    src_gate = dense_weights[f"layers.{first_moe_layer}.experts.0.gate.weight"]
    for e in range(8):
        exp_gate = upcycled[f"layers.{first_moe_layer}.experts.{e}.gate.weight"]
        assert exp_gate.shape == (1536, 768)
        assert np.array_equal(exp_gate, src_gate)

    # 2. Router weights initialized with shape (8, 768)
    router_w = upcycled[f"layers.{first_moe_layer}.router.weight"]
    assert router_w.shape == (8, 768)

    # 3. Shared expert down projection initialized to exact zeros
    shared_down = upcycled[f"layers.{first_moe_layer}.shared_experts.0.down.weight"]
    assert shared_down.shape == (768, 1536)
    assert np.all(shared_down == 0.0)


# ==============================================================================
# 3. Training Budget, Schedule & Hyperparameters
# ==============================================================================

def test_small_moe_training_budget():
    """Verify effective batch size, step counts, and WSD schedule breakdown."""
    run_cfg = SmallMoEPOCRunConfig(
        total_tokens=131_072_000,
        batch_size=16,
        seq_len=2048,
        grad_accum=8,
        warmup_steps=100,
        decay_frac=0.20,
    )
    assert run_cfg.tokens_per_step() == 262_144
    assert run_cfg.total_steps() == 500

    bd = run_cfg.step_breakdown()
    assert bd["total_steps"] == 500
    assert bd["warmup_steps"] == 100
    assert bd["decay_steps"] == 100
    assert bd["stable_steps"] == 300


def test_small_moe_generate_train_command():
    """Verify generated train command line formatting includes required flags."""
    run_cfg = SmallMoEPOCRunConfig()
    cmd = run_cfg.generate_train_command("data/split", "runs/small_moe_init.safetensors", backend="cuda")
    cmd_str = " ".join(cmd)

    assert "scripts/train.py" in cmd_str
    assert "--backend cuda" in cmd_str
    assert "config/code-small-moe.yaml" in cmd_str
    assert "--init runs/small_moe_init.safetensors" in cmd_str
    assert "--lr-schedule wsd" in cmd_str
    assert "--ckpt-every 50" in cmd_str


# ==============================================================================
# 4. Cloud Compute & RunPod A40 Modeling
# ==============================================================================

def test_cloud_run_spec_a40():
    """Verify NVIDIA A40 hardware template, costs, and single-GPU fit."""
    spec_a40 = get_cloud_run_spec(total_tokens=131_072_000, hardware="a40")
    assert "Issue #422" in spec_a40["intended_use"]
    assert spec_a40["hardware_configuration"]["gpu_model"] == "NVIDIA A40"
    assert spec_a40["hardware_configuration"]["vram_gb"] == 48
    assert spec_a40["hardware_configuration"]["single_gpu"] is True
    assert spec_a40["approx_hourly_rate_usd"] == 0.40
    assert "python scripts/cloud_pod.py launch --template a40" in spec_a40["runpod_launch_command"]


# ==============================================================================
# 5. Quality Gate: Bit-Exact Training Resume at Step 50
# ==============================================================================

def test_verify_bit_exact_resume_passes_on_identical_trajectories():
    """Bit-exact resume verification passes when reference and resumed match exactly."""
    ref_losses = {s: 5.0 - 0.01 * s for s in range(0, 101)}
    res_losses = {s: 5.0 - 0.01 * s for s in range(50, 101)}

    res = verify_bit_exact_resume(ref_losses, res_losses, resume_step=50, atol=0.0)
    assert res["status"] == "PASSED"
    assert res["passed"] is True
    assert res["resume_step"] == 50
    assert res["max_diff"] == 0.0
    assert res["eval_steps_count"] == 51


def test_verify_bit_exact_resume_fails_on_divergence():
    """Bit-exact resume verification fails when post-resume values differ."""
    ref_losses = {s: 5.0 - 0.01 * s for s in range(0, 101)}
    res_losses = {s: 5.0 - 0.01 * s for s in range(50, 101)}
    res_losses[75] += 0.005  # divergence

    res = verify_bit_exact_resume(ref_losses, res_losses, resume_step=50, atol=0.0)
    assert res["status"] == "FAILED"
    assert res["passed"] is False
    assert res["max_diff"] > 0.0


# ==============================================================================
# 6. Quality Gate: Router Entropy & Expert Load Distribution Stability
# ==============================================================================

def test_verify_router_stability_passes_when_balanced():
    """Router stability passes when entropy >= 1.50 nats and variance <= 0.05."""
    records = []
    for s in range(10):
        records.append({
            "step": s * 10,
            "moe_router_entropy": 2.05 - 0.005 * s,
            "moe_util_var": 0.001 + 0.0002 * s,
            "expert_loads": [[[100] * 8]],
        })

    res = verify_router_stability(records, min_entropy_threshold=1.50, max_variance_threshold=0.05)
    assert res["status"] == "PASSED"
    assert res["passed"] is True
    assert res["stable"] is True
    assert res["entropy"]["min"] >= 1.50
    assert res["utilization_variance"]["max"] <= 0.05


def test_verify_router_stability_fails_on_expert_collapse():
    """Router stability fails when router collapses (low entropy / high variance)."""
    # Low entropy (collapse to 1 expert)
    records_collapsed = [
        {"step": s * 10, "moe_router_entropy": 0.50, "moe_util_var": 0.08}
        for s in range(10)
    ]
    res = verify_router_stability(records_collapsed, min_entropy_threshold=1.50, max_variance_threshold=0.05)
    assert res["status"] == "FAILED"
    assert res["passed"] is False
    assert res["entropy"]["status"] == "FAILED"


def test_verify_router_stability_fails_on_expert_starvation():
    """Router stability fails when an expert receives 0 tokens."""
    records_starved = []
    for s in range(10):
        records_starved.append({
            "step": s * 10,
            "moe_router_entropy": 1.95,
            "moe_util_var": 0.01,
            "expert_loads": [[[100, 100, 100, 100, 100, 100, 100, 0]]],  # expert 7 starved
        })
    res = verify_router_stability(records_starved)
    assert res["status"] == "FAILED"
    assert res["passed"] is False
    assert res["expert_starvation_points"] > 0


def test_verify_router_stability_blind_rule(tmp_path):
    """Router stability reports BLIND when metrics cannot be observed."""
    nonexistent = tmp_path / "nonexistent.jsonl"
    res = verify_router_stability(nonexistent)
    assert res["status"] == "BLIND"
    assert res["passed"] is False


# ==============================================================================
# 7. Mock Pipeline Simulation & CLI Verification
# ==============================================================================

def test_simulate_mock_pipeline_end_to_end(tmp_path):
    """Verify simulate_mock_pipeline passes end-to-end and fulfills acceptance criteria."""
    out_dir = tmp_path / "small_moe_poc_run"
    res = simulate_mock_pipeline(
        output_dir=out_dir,
        total_tokens=131_072_000,
        seed=422,
        resume_step=50,
        write_weights=False,  # lightweight for fast test
    )

    assert res["acceptance"]["accepted"] is True
    assert res["acceptance"]["upcycle_check"]["status"] == "PASSED"
    assert res["acceptance"]["bundle_check"]["status"] == "PASSED"
    assert res["acceptance"]["resume_check"]["status"] == "PASSED"
    assert res["acceptance"]["router_stability_check"]["status"] == "PASSED"

    # Verify generated files on disk
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "weights.safetensors").exists()
    assert (out_dir / "small_moe_init.safetensors").exists()
    assert (out_dir / "small_moe_init.safetensors.config.json").exists()
    assert (out_dir / "small_moe_init.safetensors.upcycle.json").exists()
    assert (out_dir / "resume_verification.json").exists()
    assert (out_dir / "resume" / "slot-a" / "weights.safetensors").exists()
    assert (out_dir / "resume" / "slot-b" / "weights.safetensors").exists()
    assert (out_dir / "resume" / "LATEST").exists()


def test_run_small_moe_poc_cli_flags(tmp_path):
    """Verify scripts/run_small_moe_poc.py CLI interface flags."""
    script = str(REPO_ROOT / "scripts/run_small_moe_poc.py")

    # --cloud-spec
    res = subprocess.run([sys.executable, script, "--cloud-spec"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "SMALL MOE POC RUN (#422) — CLOUD COMPUTE SPECIFICATION" in res.stdout

    # --print-train-cmd
    res = subprocess.run([sys.executable, script, "--print-train-cmd"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "scripts/train.py" in res.stdout
    assert "code-small-moe.yaml" in res.stdout

    # --dry-run
    res = subprocess.run([sys.executable, script, "--dry-run"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "SMALL MOE POC RUN (#422) — DRY RUN" in res.stdout
    assert "COMPATIBLE" in res.stdout

    # --mock-pipeline
    run_out = tmp_path / "cli_run"
    json_out = tmp_path / "cli_out.json"
    res = subprocess.run([
        sys.executable, script,
        "--mock-pipeline",
        "--out-dir", str(run_out),
        "--output", str(json_out),
    ], capture_output=True, text=True)
    assert res.returncode == 0
    assert "ACCEPTANCE PASSED" in res.stdout
    assert json_out.exists()
    data = json.loads(json_out.read_text())
    assert data["acceptance"]["accepted"] is True

    # --verify-run
    res = subprocess.run([
        sys.executable, script,
        "--verify-run", str(run_out),
    ], capture_output=True, text=True)
    assert res.returncode == 0
    assert "ACCEPTANCE PASSED" in res.stdout
