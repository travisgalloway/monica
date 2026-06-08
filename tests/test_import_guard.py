"""Seam guard: nothing portable may import a hardware backend.

If importing the interface or any above-the-seam package pulls in `mlx` or torch's
CUDA stack, the migration plan is broken. We import the portable modules and assert
no backend module landed in sys.modules as a side effect.
"""

import sys
import importlib


PORTABLE_MODULES = [
    "src.model.interface",
    "src.model.blocks",
    "src.data.loader",
    "src.data.pack",
    "src.data.split",
    "src.train.schedule",
    "src.train.checkpoint",
    "src.train.loss_scale",
    "src.train.loop",
    "src.eval.val_loss",
    "src.conformance.forward_step_parity",
]

FORBIDDEN_ROOTS = ("mlx", "torch")


def test_portable_modules_do_not_import_backends():
    # Drop any backend modules a prior test may have loaded.
    for name in list(sys.modules):
        if name.split(".")[0] in ("mlx", "torch"):
            del sys.modules[name]

    for mod in PORTABLE_MODULES:
        importlib.import_module(mod)

    # Match by top-level package so submodules (mlx.core, torch._C, ...) also fail.
    leaked = sorted({m for m in sys.modules if m.split(".")[0] in FORBIDDEN_ROOTS})
    assert not leaked, f"backend leaked above the seam: {leaked}"
