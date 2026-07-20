"""Shells out to the pinned `prettier` formatter (Stage 3 of #193's TS clean pipeline).

Mirrors `src/lsp/tsc.py`'s resolve/run shape: `resolve_prettier()` is the same
graceful-skip idiom as `resolve_tsc()` (missing toolchain -> `None`, never an exception),
and `format_source` never corrupts or drops a record over a formatting failure — any
non-zero exit, timeout, or unexpected error returns the ORIGINAL source unchanged.
Formatting is a normalization pass, not a correctness gate — Stage 4 (`ts_clean.tsc_clean`)
is what actually filters on compiler diagnostics, so a prettier hiccup here must never
silently drop a record.

ABOVE THE SEAM — stdlib only. No `mlx`/`torch` import anywhere in this module (guarded by
`tests/test_import_guard.py`).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import List, Optional

from .tsc import SET_DIR

LOCAL_PRETTIER = SET_DIR / "node_modules" / ".bin" / "prettier"


def resolve_prettier() -> Optional[List[str]]:
    """Return the argv prefix to invoke `prettier`, or None if no usable toolchain exists.
    Same shape as `tsc.resolve_tsc`: a local install under `SET_DIR` (`npm install`, see
    `eval_sets/ts_error_injection/package.json`) plus `node` on PATH."""
    if LOCAL_PRETTIER.exists() and shutil.which("node") is not None:
        return [str(LOCAL_PRETTIER)]
    return None


def format_source(source: str, prettier_argv: List[str], *, parser: str = "typescript",
                   timeout: float = 30.0) -> str:
    """Run `prettier` over `source` via stdin (`--parser <parser>`) and return the
    formatted text. On ANY non-zero exit, timeout, or unexpected error (missing binary,
    OS error, ...), returns `source` UNCHANGED — a formatting failure must never drop or
    corrupt a corpus record."""
    try:
        proc = subprocess.run(prettier_argv + ["--parser", parser], input=source,
                              capture_output=True, text=True, timeout=timeout)
    except Exception:
        return source
    if proc.returncode != 0:
        return source
    return proc.stdout


class PrettierRunner:
    """Thin stateful wrapper around `format_source` for orchestrator call sites that want
    a fixed `prettier_argv`/`parser` bound once (mirrors `TscRunner`'s shape, though
    prettier has no persistent-process/scratch-dir cost to amortize — each call is a
    self-contained subprocess over stdin)."""

    def __init__(self, prettier_argv: Optional[List[str]] = None, *, parser: str = "typescript"):
        self.prettier_argv = prettier_argv if prettier_argv is not None else resolve_prettier()
        if self.prettier_argv is None:
            raise RuntimeError("no prettier toolchain resolvable (run `npm install` in "
                                f"{SET_DIR})")
        self.parser = parser

    def format(self, source: str) -> str:
        """Format `source`, falling back to it unchanged on any prettier failure."""
        return format_source(source, self.prettier_argv, parser=self.parser)
