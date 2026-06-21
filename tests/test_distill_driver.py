"""End-to-end distill driver gate (#81). Runs scripts/distill_smoke.py — build toy corpus +
synthetic teacher top-k, then scripts/distill.py through all three stages, asserting each stage's
loss drops below its start, the portable weights + per-stage resume bundles are written, and a
--resume invocation reloads cleanly. Parametrized over the available backends (mlx on Apple
Silicon, cuda where torch is installed — incl. torch-CPU), so a CUDA-only box still gets the gate.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _available_backends():
    backends = []
    try:
        import mlx.core  # noqa: F401
        backends.append("mlx")
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        backends.append("cuda")
    except ImportError:
        pass
    return backends


_BACKENDS = _available_backends()


@pytest.mark.skipif(not _BACKENDS, reason="no backend (mlx/torch) available")
@pytest.mark.parametrize("backend", _BACKENDS)
def test_distill_smoke_end_to_end(tmp_path, backend):
    out = tmp_path / f"distill-smoke-{backend}"
    res = subprocess.run(
        [sys.executable, "scripts/distill_smoke.py", "--backend", backend, "--out", str(out),
         "--steps-per-stage", "12"],
        cwd=REPO, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "PATH": os.environ.get("PATH", "")},
    )
    assert res.returncode == 0, f"distill_smoke ({backend}) failed:\n{res.stdout}\n{res.stderr}"
    assert "DISTILL SMOKE PASSED" in res.stdout
    run = out / "run"
    assert (run / "weights.safetensors").exists()
    for stage in ("mixing-match", "hidden-align", "logit-distill"):
        assert (run / stage / "metrics.jsonl").exists()
        assert (run / stage / "resume").exists()
