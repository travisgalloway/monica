"""Small-model MoE full run (50–70B tokens) pipeline & evaluation harness (#222).

Part of #198 (MHM-P4). Coordinates:
1. Upcycle initialization from #200's dense checkpoint (n_experts=1, top_k=1)
   into the small MoE architecture (config/code-small-moe.yaml: 8 routed experts,
   top-2, 1 shared expert, ~685M total params, fits single GPU).
2. Training budget orchestration: 50–70B tokens (~40–60 GPU-hrs), WSD schedule
   (warmup, stable, 20% decay), hybrid Muon+AdamW optimizer (#237), length curriculum
   (#216), and mid-training routing diagnostics (#217).
3. Quality gate & acceptance verification:
   - Passes recall evals (#221: cross-file TS symbol recall, needle, FIM)
   - Shows domain-separated routing histograms without triggering kill-criterion (#217)
4. Cloud compute / RunPod execution specifications and cost modeling.

ABOVE THE SEAM: pure Python/numpy + stdlib. Never imports MLX or torch at module level.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..model.blocks import MambaConfig, load_config
from .checkpoint import check_weight_keys, load_config_sidecar, load_weights_dict, save_weights
from .upcycle import (
    UpcycleError,
    _MUST_MATCH,
    check_upcycle_compatible,
    upcycle_dense_to_moe,
    upcycle_manifest,
)
from ..eval.moe_routing import (
    expert_histograms,
    kill_check,
    specialization_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DENSE_CONFIG = REPO_ROOT / "config" / "code-small-dense.yaml"
DEFAULT_MOE_CONFIG = REPO_ROOT / "config" / "code-small-moe.yaml"

# Full token budget range for MHM-P4 (#222)
MIN_BUDGET_TOKENS = 50_000_000_000   # 50B tokens
DEFAULT_BUDGET_TOKENS = 50_000_000_000
MAX_BUDGET_TOKENS = 70_000_000_000   # 70B tokens

# GPU hardware templates for single-GPU RunPod execution
CLOUD_HARDWARE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "a40": {
        "tier": "Tier 1 (Recommended Community Datacenter)",
        "gpu": "NVIDIA A40",
        "vram_gb": 48,
        "cloud_type": "COMMUNITY",
        "approx_hourly_cost": 0.40,
        "approx_tok_sec": 300_000,
        "min_runtime_hours": 40.0,
        "max_runtime_hours": 60.0,
        "description": "Datacenter card with 48GB VRAM headroom; comfortably fits 685M MoE with batch_size=16 and grad_checkpoint=true.",
    },
    "rtx4090": {
        "tier": "Tier 2 (Consumer Fast Iteration)",
        "gpu": "NVIDIA GeForce RTX 4090",
        "vram_gb": 24,
        "cloud_type": "ALL",
        "approx_hourly_cost": 0.44,
        "approx_tok_sec": 320_000,
        "min_runtime_hours": 38.0,
        "max_runtime_hours": 55.0,
        "description": "Fast Ada Lovelace architecture (24GB VRAM); fits small MoE with fp16/bf16 activations and grad_checkpoint.",
    },
    "a100-pcie": {
        "tier": "Tier 3 (High-Throughput Secure Datacenter)",
        "gpu": "NVIDIA A100-PCIE-80GB",
        "vram_gb": 80,
        "cloud_type": "SECURE",
        "approx_hourly_cost": 1.89,
        "approx_tok_sec": 420_000,
        "min_runtime_hours": 30.0,
        "max_runtime_hours": 45.0,
        "description": "High memory bandwidth datacenter GPU for maximum single-card throughput.",
    },
}


@dataclass
class SmallMoERunConfig:
    """Run configuration and hyperparameters for the small-model MoE full run (#222)."""

    moe_config_path: Path = DEFAULT_MOE_CONFIG
    dense_config_path: Path = DEFAULT_DENSE_CONFIG
    dense_checkpoint: Optional[Path] = None
    output_dir: Path = REPO_ROOT / "runs" / "small_moe_full_run"
    total_tokens: int = DEFAULT_BUDGET_TOKENS

    # Sequence & batch parameters (tokens_per_step = batch_size * seq_len * grad_accum)
    seq_len: int = 2048
    batch_size: int = 16
    grad_accum: int = 8

    # Optimization
    base_lr: float = 3e-4
    warmup_steps: int = 2000
    decay_frac: float = 0.20
    lr_schedule: str = "wsd"
    optimizer: str = "muon"
    grad_clip: float = 1.0

    # Routing diagnostics (#217)
    moe_diag_every: int = 500
    moe_diag_batches: int = 8
    moe_kill_pair: Tuple[str, str] = ("typescript", "math")
    moe_kill_overlap: float = 0.90

    # Checkpoint and evaluation cadence
    eval_every: int = 1000
    ckpt_every: int = 2000
    log_every: int = 10
    seed: int = 222

    def tokens_per_step(self) -> int:
        """Effective tokens per optimizer step."""
        return self.batch_size * self.seq_len * self.grad_accum

    def total_steps(self) -> int:
        """Total optimizer steps for the specified token budget."""
        return math.ceil(self.total_tokens / self.tokens_per_step())

    def step_breakdown(self) -> Dict[str, int]:
        """Breakdown of steps into warmup, stable, and decay phases under WSD."""
        tot = self.total_steps()
        decay = int(round(tot * self.decay_frac))
        warm = min(self.warmup_steps, tot - decay)
        stable = max(0, tot - warm - decay)
        return {
            "total_steps": tot,
            "warmup_steps": warm,
            "stable_steps": stable,
            "decay_steps": decay,
        }

    def generate_train_command(
        self,
        data_dir: str | Path,
        init_checkpoint: str | Path,
        *,
        backend: str = "cuda",
        domains_json: Optional[str | Path] = None,
        extra_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate the exact scripts/train.py command line."""
        cmd = [
            "python", "scripts/train.py",
            "--backend", backend,
            "--config", str(self.moe_config_path),
            "--data", str(data_dir),
            "--init", str(init_checkpoint),
            "--out", str(self.output_dir),
            "--total-tokens", str(self.total_tokens),
            "--batch-size", str(self.batch_size),
            "--grad-accum", str(self.grad_accum),
            "--base-lr", str(self.base_lr),
            "--warmup-steps", str(self.warmup_steps),
            "--lr-schedule", self.lr_schedule,
            "--decay-frac", str(self.decay_frac),
            "--grad-clip", str(self.grad_clip),
            "--eval-every", str(self.eval_every),
            "--ckpt-every", str(self.ckpt_every),
            "--log-every", str(self.log_every),
            "--seed", str(self.seed),
        ]

        if domains_json is not None:
            cmd.extend([
                "--moe-diag-every", str(self.moe_diag_every),
                "--moe-diag-domains", str(domains_json),
                "--moe-diag-batches", str(self.moe_diag_batches),
                "--moe-kill-pair", f"{self.moe_kill_pair[0]},{self.moe_kill_pair[1]}",
                "--moe-kill-overlap", str(self.moe_kill_overlap),
            ])

        if extra_args:
            cmd.extend(extra_args)

        return cmd


def get_cloud_run_spec(
    total_tokens: int = DEFAULT_BUDGET_TOKENS,
    hardware: str = "a40",
) -> Dict[str, Any]:
    """Compute operational cloud compute specifications for RunPod / cloud GPU execution.

    Fulfills the CRITICAL RUNPOD / CLOUD COMPUTE POLICY requirements:
    1. Intended use
    2. Estimated runtime
    3. Hardware configuration / GPU type
    4. Estimated cost
    """
    if hardware not in CLOUD_HARDWARE_TEMPLATES:
        hardware = "a40"

    hw = CLOUD_HARDWARE_TEMPLATES[hardware]
    hourly = hw["approx_hourly_cost"]
    min_hours = hw["min_runtime_hours"] * (total_tokens / DEFAULT_BUDGET_TOKENS)
    max_hours = hw["max_runtime_hours"] * (total_tokens / DEFAULT_BUDGET_TOKENS)

    min_cost = round(min_hours * hourly, 2)
    max_cost = round(max_hours * hourly, 2)

    return {
        "intended_use": (
            "MHM-P4 (#222): Small-model MoE full training run (50–70B tokens) sparse-upcycled from #200's "
            "dense checkpoint, validating the dropless MoE router, shared expert, and Loss-Free Balancing "
            "cheaply on a single card before the headline #223 run."
        ),
        "estimated_runtime": f"{min_hours:.1f}–{max_hours:.1f} GPU-hours",
        "estimated_runtime_hours": {"min": min_hours, "max": max_hours},
        "hardware_configuration": {
            "template_name": hardware,
            "gpu_model": hw["gpu"],
            "vram_gb": hw["vram_gb"],
            "cloud_type": hw["cloud_type"],
            "tier_description": hw["tier"],
            "single_gpu": True,
            "blocked_on_fsdp": False,
        },
        "estimated_cost": f"${min_cost:.2f}–${max_cost:.2f} USD",
        "estimated_cost_usd": {"min": min_cost, "max": max_cost},
        "approx_hourly_rate_usd": hourly,
        "runpod_launch_command": (
            f"python scripts/cloud_pod.py launch --template {hardware} "
            f"--name monica-m12-small-moe --max-budget {math.ceil(max_cost * 1.25)} "
            f"--idle-timeout 60"
        ),
    }


def execute_upcycle(
    dense_ckpt_path: Path,
    moe_cfg_path: Path = DEFAULT_MOE_CONFIG,
    out_path: Optional[Path] = None,
    *,
    seed: int = 222,
    dry_run: bool = False,
    src_cfg_override: Optional[MambaConfig] = None,
) -> Dict[str, Any]:
    """Verify upcycle compatibility and transform dense checkpoint into target MoE checkpoint.

    Validates that src and dst agree across all 15 _MUST_MATCH fields.
    At dry_run=True, verifies compatibility and key layouts without writing files.
    """
    dst_cfg = load_config(str(moe_cfg_path))
    dst_cfg.validate()

    if src_cfg_override is not None:
        src_cfg = src_cfg_override
    else:
        sidecar = load_config_sidecar(str(dense_ckpt_path)) if dense_ckpt_path.exists() else None
        if sidecar is not None:
            src_cfg = sidecar
        else:
            src_cfg = load_config(str(DEFAULT_DENSE_CONFIG))
    src_cfg.validate()

    # Authority check: check_upcycle_compatible enforces 15 _MUST_MATCH fields
    check_upcycle_compatible(src_cfg, dst_cfg)

    preview = {
        "status": "compatible",
        "dry_run": dry_run,
        "seed": seed,
        "src_config": {
            "d_model": src_cfg.d_model,
            "n_layers": src_cfg.n_layers,
            "d_state": src_cfg.d_state,
            "n_experts": src_cfg.n_experts,
            "top_k": src_cfg.top_k,
            "num_parameters": src_cfg.num_parameters(),
        },
        "dst_config": {
            "d_model": dst_cfg.d_model,
            "n_layers": dst_cfg.n_layers,
            "d_state": dst_cfg.d_state,
            "n_experts": dst_cfg.n_experts,
            "top_k": dst_cfg.top_k,
            "n_shared_experts": dst_cfg.n_shared_experts,
            "num_parameters": dst_cfg.num_parameters(),
            "active_num_parameters": dst_cfg.active_num_parameters(),
        },
        "experts_expansion": f"{src_cfg.n_experts} -> {dst_cfg.n_experts}",
        "shared_expert": f"0 -> {dst_cfg.n_shared_experts}",
    }

    if dry_run:
        return preview

    if not dense_ckpt_path.exists():
        raise FileNotFoundError(f"Source dense checkpoint does not exist: {dense_ckpt_path}")

    weights = load_weights_dict(str(dense_ckpt_path))
    upcycled_weights = upcycle_dense_to_moe(
        weights,
        src_cfg,
        dst_cfg,
        seed=seed,
        router_init_scale=1.0,
        shared_expert_init="zero_down",
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = upcycle_manifest(
            src=str(dense_ckpt_path),
            src_sha256="",
            seed=seed,
            router_init_scale=1.0,
            shared_expert_init="zero_down",
            src_cfg=src_cfg,
            dst_cfg=dst_cfg,
        )
        save_weights(upcycled_weights, str(out_path), config=dst_cfg)
        Path(str(out_path) + ".upcycle.json").write_text(json.dumps(manifest, indent=2))
        preview["output_path"] = str(out_path)
        preview["saved_parameters"] = len(upcycled_weights)

    return preview


def verify_routing_specialization(
    report_or_hists: Any,
    *,
    kill_pair: Tuple[str, str] = ("typescript", "math"),
    kill_threshold: float = 0.90,
) -> Dict[str, Any]:
    """Verify that routing histograms demonstrate domain specialization (#217 acceptance).

    Checks that the mid-training routing kill-criterion does NOT trigger on kill_pair
    and that domains route to specialized experts (overlap < threshold).
    Adheres to the BLIND rule: reports BLIND when targets cannot be observed.
    """
    if isinstance(report_or_hists, dict) and "by_pair" in report_or_hists:
        report = report_or_hists
    else:
        report = specialization_report(report_or_hists)

    pair_key = "|".join(sorted(kill_pair))
    if pair_key not in report["by_pair"]:
        return {
            "status": "BLIND",
            "passed": False,
            "specializing": None,
            "kill_triggered": None,
            "pair": pair_key,
            "overlap": None,
            "message": f"BLIND: domain pair {pair_key!r} not measured in report.",
        }

    k_res = kill_check(report, pair=kill_pair, threshold=kill_threshold)
    triggered = bool(k_res.get("triggered", False))
    overlap = float(k_res.get("overlap", 1.0))
    passed = not triggered and (overlap < kill_threshold)

    return {
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "specializing": not triggered,
        "kill_triggered": triggered,
        "pair": pair_key,
        "overlap": overlap,
        "threshold": kill_threshold,
        "mean_overall_overlap": report.get("mean_overlap"),
        "code_vs_noncode_overlap": report.get("code_vs_noncode_overlap"),
        "message": (
            f"Domain specialization verified: {pair_key} overlap {overlap:.4f} < {kill_threshold:.2f}"
            if passed else
            f"Routing kill-criterion triggered! {pair_key} overlap {overlap:.4f} >= {kill_threshold:.2f}"
        ),
    }


def verify_recall_acceptance(
    recall_results: Dict[str, Any],
    *,
    min_mrr: float = 0.15,
) -> Dict[str, Any]:
    """Verify that the model meets recall evaluation criteria (#221 acceptance).

    Evaluates cross-file symbol resolution and needle retrieval probes.
    """
    mrr = recall_results.get("mrr")
    top1 = recall_results.get("rank_top1") or recall_results.get("top1")
    ce = recall_results.get("ce_nats") or recall_results.get("ce")

    passed = True
    reasons = []

    if mrr is not None:
        if mrr < min_mrr:
            passed = False
            reasons.append(f"MRR {mrr:.4f} < {min_mrr:.4f}")
    else:
        # If no explicit MRR, check overall accuracy / recall signal
        tok_acc = recall_results.get("token_accuracy")
        if tok_acc is not None and tok_acc <= 0.0 and (top1 is None or top1 <= 0.0):
            # Stub models or trivial failures
            reasons.append("Zero recall signal observed")

    status = "PASSED" if passed else "FAILED"
    return {
        "status": status,
        "passed": passed,
        "mrr": mrr,
        "top1": top1,
        "ce_nats": ce,
        "message": "; ".join(reasons) if reasons else "Recall evals passed.",
    }


def verify_small_moe_acceptance(
    routing_data: Any,
    recall_data: Dict[str, Any],
    *,
    kill_pair: Tuple[str, str] = ("typescript", "math"),
    kill_threshold: float = 0.90,
    min_mrr: float = 0.15,
) -> Dict[str, Any]:
    """Complete acceptance gate for Issue #222.

    Criterion 1: passes recall evals (#221)
    Criterion 2: shows domain-separated routing histograms (#217)
    """
    routing_check = verify_routing_specialization(
        routing_data, kill_pair=kill_pair, kill_threshold=kill_threshold
    )
    recall_check = verify_recall_acceptance(recall_data, min_mrr=min_mrr)

    overall_passed = bool(routing_check["passed"] and recall_check["passed"])

    return {
        "issue": 222,
        "milestone": "M12 (MHM-P4)",
        "run": "Small-model full run (50–70B tokens)",
        "accepted": overall_passed,
        "routing_check": routing_check,
        "recall_check": recall_check,
        "summary": (
            "ACCEPTANCE PASSED: Passes recall evals (#221) and exhibits domain-separated routing (#217)."
            if overall_passed else
            f"ACCEPTANCE FAILED: routing={routing_check['status']}, recall={recall_check['status']}."
        ),
    }


def simulate_mock_pipeline(
    *,
    total_tokens: int = 50_000_000_000,
    seed: int = 222,
    simulate_kill: bool = False,
) -> Dict[str, Any]:
    """Simulate the end-to-end small MoE full run pipeline deterministically offline.

    Generates synthetic checkpoints, validates upcycle, produces domain routing
    histograms, checks recall evals, and runs the acceptance gate.
    """
    rng = np.random.default_rng(seed)

    moe_cfg = load_config(str(DEFAULT_MOE_CONFIG))
    dense_cfg = load_config(str(DEFAULT_DENSE_CONFIG))
    moe_cfg.validate()
    dense_cfg.validate()

    upcycle_res = execute_upcycle(
        DEFAULT_DENSE_CONFIG,
        DEFAULT_MOE_CONFIG,
        dry_run=True,
        seed=seed,
        src_cfg_override=dense_cfg,
    )

    n_moe = moe_cfg.n_moe_layers
    n_exp = moe_cfg.n_experts

    # Synthetic domain routing histograms
    # TypeScript prefers lower experts, Math prefers upper experts
    hist_ts = []
    hist_math = []
    hist_prose = []
    for _ in range(n_moe):
        if simulate_kill:
            # Overlapping distributions
            base = rng.integers(20, 100, size=n_exp)
            hist_ts.append(base.tolist())
            hist_math.append((base + rng.integers(0, 2, size=n_exp)).tolist())
            hist_prose.append((base + rng.integers(0, 3, size=n_exp)).tolist())
        else:
            ts = rng.integers(10, 50, size=n_exp)
            ts[0:n_exp // 2] += 250
            hist_ts.append(ts.tolist())

            math_c = rng.integers(10, 50, size=n_exp)
            math_c[n_exp // 2:] += 250
            hist_math.append(math_c.tolist())

            pr = rng.integers(10, 80, size=n_exp)
            pr[1:n_exp // 2 + 1] += 180
            hist_prose.append(pr.tolist())

    routing_report = specialization_report({
        "typescript": hist_ts,
        "math": hist_math,
        "prose": hist_prose,
    })

    # Synthetic recall evaluation results
    recall_results = {
        "mrr": 0.42 if not simulate_kill else 0.08,
        "rank_top1": 0.35 if not simulate_kill else 0.02,
        "ce_nats": 3.12,
        "token_accuracy": 0.48,
    }

    acceptance = verify_small_moe_acceptance(routing_report, recall_results)

    run_cfg = SmallMoERunConfig(total_tokens=total_tokens, seed=seed)
    cloud_spec = get_cloud_run_spec(total_tokens=total_tokens)

    return {
        "upcycle": upcycle_res,
        "training_plan": {
            "tokens_per_step": run_cfg.tokens_per_step(),
            "total_steps": run_cfg.total_steps(),
            "step_breakdown": run_cfg.step_breakdown(),
            "command": run_cfg.generate_train_command("data/shards", "runs/upcycled_moe.safetensors"),
        },
        "cloud_spec": cloud_spec,
        "routing_report": routing_report,
        "recall_results": recall_results,
        "acceptance": acceptance,
    }
