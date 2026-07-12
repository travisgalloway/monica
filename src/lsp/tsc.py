"""Shells out to the pinned `tsc` compiler (#194's toolchain) for the LSP harness.

`resolve_tsc()` is the original from `scripts/validate_ts_error_set.py`, moved here
so both that script and the #199 harness share one source of truth; the script now
re-exports it as a thin shim (see its module docstring).

`TscRunner` differs from the old per-call `tempfile.TemporaryDirectory`: the harness
makes thousands of `tsc` calls (every generated token can trigger a re-check), so it
holds **one persistent scratch dir per run** instead of paying `mkdtemp`/`rmtree` on
every call. The directory is still nested under `SET_DIR` (not system temp) — that
nesting is load-bearing, since TS's default `typeRoots` walk needs to find
`SET_DIR/node_modules/@types/node` (ambient `console`, etc.) from wherever the
scratch dir sits.

ABOVE THE SEAM — stdlib only. No `mlx`/`torch` import anywhere in this module
(guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from .diagnostics import Diagnostic, parse_tsc_output

SET_DIR = Path(__file__).resolve().parent.parent.parent / "eval_sets" / "ts_error_injection"
DEFAULT_TSCONFIG_PATH = SET_DIR / "tsconfig.json"
LOCAL_TSC = SET_DIR / "node_modules" / ".bin" / "tsc"

_DIAGNOSTIC_CODE_RE = re.compile(r"error (TS\d+):")


def resolve_tsc() -> Optional[List[str]]:
    """Return the argv prefix to invoke `tsc`, or None if no usable toolchain exists.

    Deliberately does not fall back to `npx -p typescript tsc`: npx would fetch a bare
    `typescript` package with no access to `SET_DIR/node_modules/@types/node`, so
    ambient globals like `console` would spuriously fail to resolve under this
    project's `lib: ["ES2020"]` (no-DOM) tsconfig — producing diagnostics unrelated to
    the labeled error and making validation flaky on hosts with `node`/`npx` but no
    `npm install` run in `SET_DIR`.

    Also requires `node` itself on PATH: the local `tsc` shim's `#!/usr/bin/env node`
    shebang needs it, and without this check a node-less host would hit a raw
    `FileNotFoundError` from `subprocess.run` instead of a clean "unavailable" signal.
    """
    if LOCAL_TSC.exists() and shutil.which("node") is not None:
        return [str(LOCAL_TSC)]
    return None


def tsc_diagnostics(source: str, tsconfig: Path, tsc_argv: List[str]) -> List[str]:
    """Compile `source` under `tsconfig` in a fresh scratch dir and return the
    `TSxxxx` codes reported. Legacy per-call shape (one `TemporaryDirectory` per
    invocation) kept for `scripts/validate_ts_error_set.py` / its one-shot
    whole-set validation run, where thousands of `tsc` calls in a tight loop never
    happen. The harness's hot path is `TscRunner.diagnostics`, below.
    """
    with tempfile.TemporaryDirectory(dir=SET_DIR) as td:
        tmpdir = Path(td)
        (tmpdir / "tsconfig.json").write_text(tsconfig.read_text(encoding="utf-8"), encoding="utf-8")
        (tmpdir / "snippet.ts").write_text(source, encoding="utf-8")
        proc = subprocess.run(tsc_argv + ["-p", str(tmpdir), "--pretty", "false"],
                               capture_output=True, text=True)
        return _DIAGNOSTIC_CODE_RE.findall(proc.stdout + proc.stderr)


class TscRunner:
    """A persistent-scratch-dir `tsc` invoker for the repair loop's hot path.

    One `snippet.ts` is overwritten per call rather than creating a fresh directory
    each time. Not safe for concurrent use (one runner == one sequential compile
    stream) — the harness is single-threaded per record, matching `TscRunner`'s
    per-run lifetime.
    """

    def __init__(self, tsc_argv: Optional[List[str]] = None,
                 tsconfig: Path = DEFAULT_TSCONFIG_PATH,
                 scratch_parent: Path = SET_DIR):
        self.tsc_argv = tsc_argv if tsc_argv is not None else resolve_tsc()
        if self.tsc_argv is None:
            raise RuntimeError("no tsc toolchain resolvable (run `npm install` in "
                                f"{scratch_parent})")
        self.n_calls = 0
        self.wall_s = 0.0

        self._tmpdir_obj = tempfile.TemporaryDirectory(dir=scratch_parent, prefix="lsp_scratch_")
        self.scratch_dir = Path(self._tmpdir_obj.name)
        (self.scratch_dir / "tsconfig.json").write_text(
            tsconfig.read_text(encoding="utf-8"), encoding="utf-8")
        self.snippet_path = self.scratch_dir / "snippet.ts"

    def diagnostics(self, source: str) -> List[Diagnostic]:
        """Compile `source` (overwriting the run's `snippet.ts`) and return every
        `Diagnostic` tsc reports, unfiltered — the caller applies
        `diagnostics.filter_diagnostics` for frontier/TS1xxx gating."""
        self.snippet_path.write_text(source, encoding="utf-8")
        t0 = time.monotonic()
        proc = subprocess.run(self.tsc_argv + ["-p", str(self.scratch_dir), "--pretty", "false"],
                               capture_output=True, text=True)
        self.wall_s += time.monotonic() - t0
        self.n_calls += 1
        return parse_tsc_output(proc.stdout + proc.stderr, source)

    def codes(self, source: str) -> List[str]:
        """`[d.code for d in self.diagnostics(source)]` — the common case when the
        caller only needs the diagnostic-code set (e.g. `error_avoidance_rate`)."""
        return [d.code for d in self.diagnostics(source)]

    def close(self) -> None:
        self._tmpdir_obj.cleanup()

    def __enter__(self) -> "TscRunner":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
