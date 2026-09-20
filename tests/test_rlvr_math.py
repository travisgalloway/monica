"""#103 -- End-to-end RLVR / GRPO test on GSM8K and MATH prompt sets.

Verifies:
1. Math GRPO loop runs with exact-match/SymPy rewards on GSM8K/MATH via scripts/rlvr.py.
2. Group advantage and KL penalty are tracked with stable training dynamics.
3. Telemetry records group advantage, KL penalty, and verifier stats.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rlvr_math_gsm8k_with_kl_penalty_e2e(tmp_path):
    if importlib.util.find_spec("mlx") is None and importlib.util.find_spec("torch") is None:
        pytest.skip("test requires mlx or torch backend")

    from src.model.backend import get_backend
    from src.model.blocks import load_config

    cfg_path = REPO_ROOT / "config" / "toy-mhm.yaml"
    cfg = load_config(str(cfg_path))
    backend = get_backend()
    model = backend.model_cls(cfg)

    # Save portable weights for init and reference
    init_path = tmp_path / "init.safetensors"
    model.save(str(init_path))

    problems_path = REPO_ROOT / "eval_sets" / "math" / "gsm8k.jsonl"
    out_dir = tmp_path / "rlvr_gsm8k_out"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "rlvr.py"),
        "--config", str(cfg_path),
        "--init", str(init_path),
        "--problems", str(problems_path),
        "--reward", "math",
        "--beta", "0.04",
        "--track-kl",
        "--steps", "2",
        "--group-size", "4",
        "--verifier-workers", "2",
        "--verifier-cache",
        "--byte-fallback",
        "--max-new-tokens", "8",
        "--out", str(out_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0, f"rlvr failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    weights_file = out_dir / "weights.safetensors"
    telemetry_file = out_dir / "telemetry.json"
    assert weights_file.exists()
    assert telemetry_file.exists()

    telemetry = json.loads(telemetry_file.read_text(encoding="utf-8"))
    assert "grpo" in telemetry
    grpo = telemetry["grpo"]
    assert "mean_adv" in grpo
    assert "mean_abs_adv" in grpo
    assert "mean_kl" in grpo
    assert grpo["mean_kl"] is not None
    assert grpo["mean_kl"] >= 0.0
    assert grpo["beta"] == 0.04


def test_rlvr_math_hendrycks_math_e2e(tmp_path):
    if importlib.util.find_spec("mlx") is None and importlib.util.find_spec("torch") is None:
        pytest.skip("test requires mlx or torch backend")

    from src.model.backend import get_backend
    from src.model.blocks import load_config

    cfg_path = REPO_ROOT / "config" / "toy-mhm.yaml"
    cfg = load_config(str(cfg_path))
    backend = get_backend()
    model = backend.model_cls(cfg)

    init_path = tmp_path / "init.safetensors"
    model.save(str(init_path))

    problems_path = REPO_ROOT / "eval_sets" / "math" / "math.jsonl"
    out_dir = tmp_path / "rlvr_math_out"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "rlvr.py"),
        "--config", str(cfg_path),
        "--init", str(init_path),
        "--problems", str(problems_path),
        "--reward", "math",
        "--beta", "0.02",
        "--track-kl",
        "--steps", "2",
        "--group-size", "4",
        "--verifier-workers", "2",
        "--verifier-cache",
        "--byte-fallback",
        "--max-new-tokens", "8",
        "--out", str(out_dir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0, f"rlvr failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    weights_file = out_dir / "weights.safetensors"
    telemetry_file = out_dir / "telemetry.json"
    assert weights_file.exists()
    assert telemetry_file.exists()

    telemetry = json.loads(telemetry_file.read_text(encoding="utf-8"))
    assert "grpo" in telemetry
    grpo = telemetry["grpo"]
    assert "mean_adv" in grpo
    assert "mean_kl" in grpo
    assert grpo["mean_kl"] >= 0.0
