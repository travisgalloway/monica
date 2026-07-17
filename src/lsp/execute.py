"""Functional execution of generated TypeScript (#199 F1) — the correctness guard.

`diagnostic_clean_rate` measures whether generated code *type-checks*. It does not measure
whether the code is *correct*, and the two can move in opposite directions: hard-ban can
make a body compile by banning a token, producing compiling nonsense (the `/*age*/age`
comment-insertion pathology observed in Phase 0). So a clean-rate win is untrustworthy on
its own. This module runs the code against its benchmark tests, so `pass@1` can be reported
next to clean-rate and a hollow gain (clean-rate up, pass@1 down) is caught, not hidden.

**Type-checking and running are deliberately decoupled.** `tsc` is invoked with
`--noEmitOnError false`, so JS is emitted *even when types don't check*, and the program
runs regardless of the clean-rate verdict. This is the whole point: `pass@1` must be an
independent axis, or it cannot serve as a guard on the clean-rate metric.

`run_tests` returns a small structured verdict rather than a bare bool, because *how* a
program fails is informative (a compile failure that still ran, a runtime assertion, a
timeout/infinite loop) and the aggregate table reports the breakdown.

ABOVE THE SEAM — stdlib + subprocess only. No `mlx`/`torch` import (guarded by
`tests/test_import_guard.py`). Shells out to the same `node`/`tsc` toolchain `tsc.py`
resolves; runs benign, fixed benchmark code under a hard per-call timeout.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .tsc import SET_DIR, resolve_tsc

# Emit JS regardless of type errors (`--noEmitOnError false`) and skip lib-dts checking for
# speed; target/module chosen so plain `node` can run the output. Deliberately NOT the pinned
# strict eval tsconfig — this step measures behaviour, not types.
_EMIT_FLAGS = ["--target", "ES2020", "--module", "commonjs",
               "--noEmitOnError", "false", "--skipLibCheck", "--esModuleInterop"]

_DEFAULT_TIMEOUT_S = 15.0


@dataclass
class ExecResult:
    passed: bool
    outcome: str            # "pass" | "runtime_fail" | "compile_fail" | "timeout" | "no_toolchain"
    detail: str = ""        # trailing stderr / message, for the transcript


class Executor:
    """Persistent-scratch-dir compile+run, mirroring `TscRunner`'s per-run lifetime.

    Reuses one directory across every record rather than an `mkdtemp` per call. Not
    concurrency-safe (one executor == one sequential stream), matching the single-threaded
    harness.
    """

    def __init__(self, tsc_argv: Optional[List[str]] = None,
                 node: str = "node", timeout_s: float = _DEFAULT_TIMEOUT_S):
        self.tsc_argv = tsc_argv if tsc_argv is not None else resolve_tsc()
        self.node = node
        self.timeout_s = timeout_s
        self.n_runs = 0
        self._tmpdir_obj = tempfile.TemporaryDirectory(dir=SET_DIR, prefix="lsp_exec_")
        self.scratch_dir = Path(self._tmpdir_obj.name)

    def run_tests(self, prompt: str, completion: str, tests: str) -> ExecResult:
        """Compile `prompt + completion + tests` and execute it.

        `passed` is true iff the program compiles-to-JS and exits 0 (MultiPL-E tests
        `process.exit`/throw on failure). A program that fails to emit JS at all is a
        `compile_fail`; one that emits but exits non-zero is a `runtime_fail`; one that
        exceeds the timeout is a `timeout` (an infinite loop is a real failure mode of
        generated code, not an error to swallow).
        """
        if self.tsc_argv is None:
            return ExecResult(False, "no_toolchain")

        self.n_runs += 1
        program = prompt + completion + "\n" + tests
        src = self.scratch_dir / "program.ts"
        js = self.scratch_dir / "program.js"
        if js.exists():
            js.unlink()
        src.write_text(program, encoding="utf-8")

        comp = subprocess.run(
            self.tsc_argv + ["program.ts", "--outDir", str(self.scratch_dir)] + _EMIT_FLAGS,
            cwd=self.scratch_dir, capture_output=True, text=True)
        if not js.exists():
            return ExecResult(False, "compile_fail", comp.stdout.strip()[-400:])

        try:
            run = subprocess.run([self.node, str(js)], cwd=self.scratch_dir,
                                 capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            return ExecResult(False, "timeout", f">{self.timeout_s}s")

        if run.returncode == 0:
            return ExecResult(True, "pass")
        return ExecResult(False, "runtime_fail", run.stderr.strip()[-400:])

    def close(self) -> None:
        self._tmpdir_obj.cleanup()

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
