"""End-to-end distill driver gate (#81). Runs scripts/distill_smoke.py — build toy corpus +
synthetic teacher top-k, then scripts/distill.py through all three stages on MLX, asserting each
stage's loss decreases, the portable weights + per-stage resume bundles are written, and a
--resume invocation reloads cleanly. MLX-only (the smoke runs on the dev backend).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

REPO = Path(__file__).resolve().parents[1]


def test_distill_smoke_end_to_end(tmp_path):
    out = tmp_path / "distill-smoke"
    res = subprocess.run(
        [sys.executable, "scripts/distill_smoke.py", "--out", str(out), "--steps-per-stage", "12"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "PATH": os.environ.get("PATH", "")},
    )
    assert res.returncode == 0, f"distill_smoke failed:\n{res.stdout}\n{res.stderr}"
    assert "DISTILL SMOKE PASSED" in res.stdout
    run = out / "run"
    assert (run / "weights.safetensors").exists()
    for stage in ("mixing-match", "hidden-align", "logit-distill"):
        assert (run / stage / "metrics.jsonl").exists()
        assert (run / stage / "resume").exists()
