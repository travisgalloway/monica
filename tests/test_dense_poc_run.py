"""Unit and integration tests for Issue #421: dense baseline pretraining run (2B tokens).

Verifies:
1. Architecture & Config Validation:
   - config/code-small-dense.yaml valid MambaConfig (56 layers, d_model 768, d_state 16, degenerate n_experts=1)
   - Correct parameter sizing (~232.1M total parameters)
   - FIM attention constraint satisfied (56 % 8 == 0, 7 attention layers, final block is attention)
   - Degenerate MoE: moe_every=3, n_experts=1, top_k=1, moe_d_ff=1536, moe_balance_rate=None
2. Sparse Upcycle Compatibility (#214 / #200 / #421):
   - All 15 _MUST_MATCH fields match between code-small-dense.yaml and code-small-moe.yaml
   - Attention and MoE layer indices match exactly
   - check_upcycle_compatible succeeds without error
   - scripts/upcycle.py --src <ckpt> --config config/code-small-moe.yaml --dry-run exits clean
3. Full Run Training Budget & Hyperparameters:
   - 2B token budget calculations (2,000,000,000 tokens)
   - WSD schedule phase transitions (warmup, stable, decay)
   - Effective batch size and tokens/step (16 * 2048 * 8 = 262,144)
   - Training command generation includes --backend cuda, --config config/code-small-dense.yaml, WSD flags
4. Cloud Compute & RunPod Resource Modeling:
   - Intended use, estimated runtime (1.5–2.5 GPU-hrs), hardware tiers (A40, RTX 4090, A100), and cost estimates
   - Single-GPU fit verified (no multi-node/FSDP required for 232M)
   - RunPod launch command formatting
5. Quality Gate & Acceptance Verification:
   - Monotonically decreasing BPB curve check: passes on monotonic decay, fails on non-monotonicity
   - Double-buffered Slot-A and Slot-B checkpoint bundles check with sidecar configs
   - Combined acceptance gate
6. Deterministic Mock Pipeline Simulation & CLI Verification:
   - simulate_mock_pipeline passes end-to-end
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
from src.train.dense_poc_run import (
    CLOUD_HARDWARE_TEMPLATES,
    DEFAULT_BUDGET_TOKENS,
    DEFAULT_DENSE_CONFIG,
    DEFAULT_MOE_CONFIG,
    DensePOCRunConfig,
    get_cloud_run_spec,
    simulate_mock_pipeline,
    verify_bpb_monotonicity,
    verify_checkpoint_bundles,
    verify_dense_poc_acceptance,
    verify_upcycle_dry_run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 1. Architecture & Config Validation
# ==============================================================================

def test_code_small_dense_config_validity():
    """Verify config/code-small-dense.yaml satisfies all architectural specs for dense POC."""
    cfg = load_config(str(REPO_ROOT / "config/code-small-dense.yaml"))
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

    # Degenerate MoE structure (n_experts=1, top_k=1, plain SwiGLU FFN)
    assert cfg.moe_every == 3
    assert cfg.n_experts == 1
    assert cfg.top_k == 1
    assert cfg.n_shared_experts == 0
    assert cfg.moe_d_ff == 1536
    assert cfg.moe_d_ff_resolved == 1536
    assert cfg.moe_balance_rate is None
    assert cfg.n_moe_layers == 16

    # Surviving Mamba layers: 56 - 7 - 16 = 33
    assert cfg.n_layers - cfg.n_attention_layers - cfg.n_moe_layers == 33

    # Total parameters ~232.1M
    total_params = cfg.num_parameters()
    assert 230_000_000 < total_params < 235_000_000


# ==============================================================================
# 2. Sparse Upcycle Compatibility with Target MoE Configs
# ==============================================================================

def test_upcycle_compatibility_with_small_moe():
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


def test_upcycle_dry_run_with_synthetic_checkpoint(tmp_path):
    """Verify verify_upcycle_dry_run passes on checkpoint with sidecar config."""
    cfg = load_config(str(DEFAULT_DENSE_CONFIG))
    ckpt = tmp_path / "weights.safetensors"
    save_weights({}, str(ckpt), config=cfg)

    res = verify_upcycle_dry_run(ckpt, DEFAULT_MOE_CONFIG)
    assert res["status"] == "PASSED"
    assert res["passed"] is True
    assert res["must_match_satisfied"] >= 15
    assert res["attention_layers_match"] is True
    assert res["moe_layers_match"] is True


def test_scripts_upcycle_cli_dry_run(tmp_path):
    """Verify scripts/upcycle.py --src <ckpt> --config config/code-small-moe.yaml --dry-run exits 0."""
    cfg = load_config(str(DEFAULT_DENSE_CONFIG))
    ckpt = tmp_path / "weights.safetensors"
    save_weights({}, str(ckpt), config=cfg)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/upcycle.py"),
        "--src", str(ckpt),
        "--config", str(REPO_ROOT / "config/code-small-moe.yaml"),
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "15/15 MUST_MATCH fields satisfied" in res.stdout
    assert "--dry-run: exiting before reading --src or writing --out" in res.stdout


# ==============================================================================
# 3. Training Budget, Schedule & Checkpoints
# ==============================================================================

def test_training_budget_calculations():
    """Verify step counts, effective batch size, and WSD schedule breakdown for 2B tokens."""
    cfg_2b = DensePOCRunConfig(
        total_tokens=2_000_000_000,
        batch_size=16,
        seq_len=2048,
        grad_accum=8,
        warmup_steps=500,
        decay_frac=0.20,
    )
    assert cfg_2b.tokens_per_step() == 262_144
    assert cfg_2b.total_steps() == 7630

    bd = cfg_2b.step_breakdown()
    assert bd["total_steps"] == 7630
    assert bd["warmup_steps"] == 500
    assert bd["decay_steps"] == int(round(7630 * 0.20))
    assert bd["stable_steps"] == 7630 - 500 - bd["decay_steps"]


def test_generate_train_command():
    """Verify train command line formatting includes required flags."""
    cfg = DensePOCRunConfig(total_tokens=2_000_000_000)
    cmd = cfg.generate_train_command("data/split", backend="cuda")

    cmd_str = " ".join(cmd)
    assert "scripts/train.py" in cmd_str
    assert "--backend cuda" in cmd_str
    assert "config/code-small-dense.yaml" in cmd_str
    assert "--total-tokens 2000000000" in cmd_str
    assert "--lr-schedule wsd" in cmd_str
    assert "--decay-frac 0.2" in cmd_str
    assert "--eval-every 500" in cmd_str
    assert "--ckpt-every 500" in cmd_str


# ==============================================================================
# 4. Cloud Compute & RunPod Modeling
# ==============================================================================

def test_cloud_run_spec_modeling():
    """Verify RunPod / cloud hardware template, costs, and single-GPU fit for 2B tokens."""
    spec_a40 = get_cloud_run_spec(total_tokens=2_000_000_000, hardware="a40")
    assert "Issue #421" in spec_a40["intended_use"]
    assert spec_a40["hardware_configuration"]["gpu_model"] == "NVIDIA A40"
    assert spec_a40["hardware_configuration"]["single_gpu"] is True
    assert spec_a40["approx_hourly_rate_usd"] == 0.40
    assert spec_a40["estimated_cost_usd"]["min"] < 1.00
    assert "python scripts/cloud_pod.py launch --template a40" in spec_a40["runpod_launch_command"]

    spec_4090 = get_cloud_run_spec(total_tokens=2_000_000_000, hardware="rtx4090")
    assert spec_4090["hardware_configuration"]["gpu_model"] == "NVIDIA GeForce RTX 4090"
    assert spec_4090["hardware_configuration"]["single_gpu"] is True


# ==============================================================================
# 5. Quality Gate: BPB Curve Monotonicity & Checkpoint Bundles
# ==============================================================================

def test_bpb_monotonicity_verification():
    """Verify verify_bpb_monotonicity passes on decreasing series and fails on regressions."""
    # Decreasing series (passing)
    passing_records = [
        {"step": 0, "val_loss": 8.4, "val_bpb": 3.46},
        {"step": 500, "val_loss": 5.5, "val_bpb": 2.27},
        {"step": 1000, "val_loss": 3.8, "val_bpb": 1.56},
        {"step": 1500, "val_loss": 2.6, "val_bpb": 1.07},
        {"step": 2000, "val_loss": 1.8, "val_bpb": 0.74},
    ]
    res_pass = verify_bpb_monotonicity(passing_records)
    assert res_pass["status"] == "PASSED"
    assert res_pass["passed"] is True
    assert res_pass["monotonic"] is True
    assert res_pass["delta_bpb"] < 0.0

    # Non-monotonic series with regression (failing)
    failing_records = [
        {"step": 0, "val_loss": 8.4, "val_bpb": 3.46},
        {"step": 500, "val_loss": 5.5, "val_bpb": 2.27},
        {"step": 1000, "val_loss": 6.2, "val_bpb": 2.56},  # regression
        {"step": 1500, "val_loss": 2.6, "val_bpb": 1.07},
    ]
    res_fail = verify_bpb_monotonicity(failing_records)
    assert res_fail["status"] == "FAILED"
    assert res_fail["passed"] is False
    assert res_fail["monotonic"] is False
    assert len(res_fail["violations"]) == 1


def test_checkpoint_bundles_verification(tmp_path):
    """Verify verify_checkpoint_bundles validates Slot-A and Slot-B double-buffered layout."""
    cfg = load_config(str(DEFAULT_DENSE_CONFIG))
    resume_dir = tmp_path / "resume"
    store = CheckpointStore(str(resume_dir))

    # Commit slot-a
    store.save(
        step=500,
        loss_scale_state={},
        weights_serializer=lambda p: save_weights({}, p, config=cfg),
        optimizer_serializer=lambda p: Path(p).write_bytes(b"opt_500"),
    )
    # Commit slot-b
    store.save(
        step=1000,
        loss_scale_state={},
        weights_serializer=lambda p: save_weights({}, p, config=cfg),
        optimizer_serializer=lambda p: Path(p).write_bytes(b"opt_1000"),
    )

    # Save canonical weights
    save_weights({}, str(tmp_path / "weights.safetensors"), config=cfg)

    res = verify_checkpoint_bundles(tmp_path)
    assert res["status"] == "PASSED"
    assert res["passed"] is True
    assert res["slot_a"] is True
    assert res["slot_b"] is True
    assert res["latest_slot"] == "slot-b"
    assert res["canonical_weights"] is True


# ==============================================================================
# 6. Deterministic Mock Pipeline Simulation & CLI Verification
# ==============================================================================

def test_simulate_mock_pipeline_end_to_end(tmp_path):
    """Verify simulate_mock_pipeline passes end-to-end and satisfies acceptance criteria."""
    out_dir = tmp_path / "dense_poc_run"
    res = simulate_mock_pipeline(
        output_dir=out_dir,
        total_tokens=2_000_000_000,
        seed=421,
        write_weights=False,  # lightweight for fast test
    )

    assert res["acceptance"]["accepted"] is True
    assert res["acceptance"]["bpb_check"]["status"] == "PASSED"
    assert res["acceptance"]["bundle_check"]["status"] == "PASSED"
    assert res["acceptance"]["upcycle_check"]["status"] == "PASSED"

    # Verify generated files on disk
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "weights.safetensors").exists()
    assert (out_dir / "weights.safetensors.config.json").exists()
    assert (out_dir / "resume" / "slot-a" / "weights.safetensors").exists()
    assert (out_dir / "resume" / "slot-b" / "weights.safetensors").exists()
    assert (out_dir / "resume" / "LATEST").exists()


def test_run_dense_poc_cli_flags(tmp_path):
    """Verify scripts/run_dense_poc.py CLI interface flags."""
    script = str(REPO_ROOT / "scripts/run_dense_poc.py")

    # --cloud-spec
    res = subprocess.run([sys.executable, script, "--cloud-spec"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "DENSE BASELINE RUN (#421) — CLOUD COMPUTE SPECIFICATION" in res.stdout

    # --print-train-cmd
    res = subprocess.run([sys.executable, script, "--print-train-cmd"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "scripts/train.py" in res.stdout
    assert "--lr-schedule wsd" in res.stdout

    # --dry-run
    res = subprocess.run([sys.executable, script, "--dry-run"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Upcycle compatibility confirmed" in res.stdout

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
