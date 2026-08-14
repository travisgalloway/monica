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
    "src.model.sizing",
    "src.model.train_time",
    "src.data.loader",
    "src.data.corpus",
    "src.data.storage",
    "src.data.r2_sync",
    "src.data.datatrove_pipeline",
    "src.data.filters",
    "src.data.dedup",
    "src.data.shard",
    "src.data.pack",
    "src.data.stack_v2",
    "src.data.vocab_sample",
    "src.data.ts_clean",
    "src.data.split",
    "src.data.download",
    "src.data.tokenize",
    "src.data.instruct_format",
    "src.data.chat_template",
    "src.data.instruct_sft",
    "src.data.reasoning_traces",
    "src.data.reasoning_sft",
    "src.data.tool_sources",
    "src.data.tool_sft",
    "src.data.sft_data",
    "src.data.sft_loader",
    "src.data.sft_sources",
    "src.data.dpo_data",
    "src.data.dpo_loader",
    "src.data.dpo_sources",
    "src.train.dpo_math",
    "src.train.grpo",
    "src.train.verifiers",
    "src.train.schedule",
    "src.train.checkpoint",
    "src.train.loss_scale",
    "src.train.moe_balance",
    "src.train.upcycle",
    "src.train.loop",
    "src.train.curriculum",
    "src.train.stream",
    "src.train.logging",
    "src.eval.val_loss",
    "src.eval.quantize",
    "src.eval.long_context",
    "src.eval.fim_eval",
    "src.eval.code_suite",
    "src.eval.code_recall",
    "src.eval.code_needle",
    "src.eval.domain_bpb",
    "src.eval.moe_routing",
    "src.eval.external_sets",
    "src.eval.olmes_adapter",
    "src.eval.bfcl_adapter",
    "src.eval.retrieval_probe",
    "src.eval.probes",
    "src.eval.ts_error_eval",
    "src.eval.lsp_eval",
    "src.eval.ssi_contract",
    "src.lsp.tsc",
    "src.lsp.prettier",
    "src.lsp.diagnostics",
    "src.lsp.lm",
    "src.lsp.harness",
    "src.lsp.chat",
    "src.lsp.execute",
    "src.lsp.jsonrpc",
    "src.lsp.ts_lsp",
    "src.lsp.ts_service",
    "src.lsp.opengrep",
    "src.lsp.oracle",
    "src.lsp.ts_boundaries",
    "src.conformance.forward_step_parity",
    "src.conformance.backend_parity",
    "src.conformance.doc_boundary_parity",
    "src.conformance.prefill_decode_parity",
    "src.serve.sessions",
    "src.serve.rewind",
    "src.serve.sampling",
    "src.serve.generate",
    "src.serve.spec_decode",
]

FORBIDDEN_ROOTS = ("mlx", "torch", "bitsandbytes")


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
