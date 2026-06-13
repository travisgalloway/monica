"""Seam guard: nothing portable may import a hardware backend.

If importing the interface or any above-the-seam package pulls in `mlx` or torch's
CUDA stack, the migration plan is broken. The check runs in a FRESH subprocess,
not in-process, for two reasons:

  * pytest imports every test module at collection, and the test files already
    import nearly all the portable modules — an in-process re-import is a
    sys.modules cache hit that re-triggers nothing, so a real module-level
    backend import would be invisible (a false pass).
  * the old design deleted mlx/torch from sys.modules to compensate, which made
    any later fresh import of a native backend inside a test body abort the
    interpreter.

A fresh interpreter makes every import genuinely fresh and leaves this process's
sys.modules untouched.
"""

import subprocess
import sys
from pathlib import Path

try:
    import mlx.core  # noqa: F401 — presence probe for the mechanism self-test
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

import pytest


PORTABLE_MODULES = [
    "src.model.interface",
    "src.model.blocks",
    "src.model.backend",
    "src.data.loader",
    "src.data.pack",
    "src.data.split",
    "src.data.download",
    "src.data.instruct_format",
    "src.data.sft_data",
    "src.data.sft_loader",
    "src.train.schedule",
    "src.train.checkpoint",
    "src.train.loss_scale",
    "src.train.loop",
    "src.eval.val_loss",
    "src.eval.olmes_adapter",
    "src.conformance.forward_step_parity",
    "src.conformance.backend_parity",
    "src.serve.sessions",
    "src.serve.rewind",
    "src.serve.sampling",
    "src.serve.generate",
]

FORBIDDEN_ROOTS = ("mlx", "torch")


def _run_guard(modules):
    """Import `modules` in a fresh interpreter; fail if a backend lands in sys.modules."""
    code = (
        "import sys, importlib\n"
        f"for mod in {modules!r}:\n"
        "    importlib.import_module(mod)\n"
        # Match by top-level package so submodules (mlx.core, torch._C, ...) also fail.
        f"leaked = sorted({{m for m in sys.modules if m.split('.')[0] in {FORBIDDEN_ROOTS!r}}})\n"
        "assert not leaked, f'backend leaked above the seam: {leaked}'\n"
    )
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, "-c", code],
                          cwd=repo_root, capture_output=True, text=True)


def test_portable_modules_do_not_import_backends():
    res = _run_guard(PORTABLE_MODULES)
    assert res.returncode == 0, f"seam guard failed:\n{res.stderr}"


@pytest.mark.skipif(not HAVE_MLX, reason="mlx unavailable")
def test_guard_mechanism_detects_leaks():
    # The MLX backend legitimately imports mlx; the guard must flag it, proving
    # the check is not vacuously green.
    res = _run_guard(["src.model.mlx_backend"])
    assert res.returncode != 0, "guard failed to detect a known backend import"
    assert "backend leaked above the seam" in res.stderr
