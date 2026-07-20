"""Stage 4: the LSP-clean filter — keep only TS files `tsc --noEmit` accepts with zero
diagnostics (#193, locked composability decision #3: train on LSP-clean code so
inference-time LSP feedback in M12 corrects distribution shift instead of fighting the
training prior).

Generalizes the filter loop `scripts/build_clean_prefix_set.py` proved out for the
over-repair probe set: drop `MODULE_RESOLUTION_CODES` (import-bearing files aren't all
marked dirty just because the isolated eval tsconfig can't resolve their imports — see
that script's module docstring for the full rationale) before deciding "clean". This is
the **acceptance-critical** rate for #193: `CleanRateStats` records exactly what fraction
of the sampled corpus survives, and the orchestrator (`scripts/build_ts_clean_corpus.py`)
writes it into `manifest.json`.

ABOVE THE SEAM — stdlib only (the real `tsc_runner` is injected by the caller; this module
never imports `src.lsp.tsc` at module level, only via type-checking-only usage in
docstrings). No `mlx`/`torch` (guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from src.lsp.diagnostics import MODULE_RESOLUTION_CODES

from .corpus import Record


@dataclass
class CleanRateStats:
    """Stage-4 tallies: how many records went in, how many were LSP-clean, how many were
    dirty (had a surviving diagnostic), and how many raised while being checked."""

    n_seen: int = 0
    n_clean: int = 0
    n_dirty: int = 0
    n_error: int = 0

    def as_dict(self) -> dict:
        return {"n_seen": self.n_seen, "n_clean": self.n_clean,
                "n_dirty": self.n_dirty, "n_error": self.n_error}


def tsc_clean(records: Iterable[Record], *, tsc_runner, ignore_module_resolution: bool = True,
              stats: Optional[CleanRateStats] = None) -> Iterator[Record]:
    """Yield only records whose text compiles with zero (surviving) `tsc` diagnostics.

    `tsc_runner` is injected — the real `src.lsp.tsc.TscRunner` for a live run, or a stub
    exposing `.codes(source) -> List[str]` in tests. When `ignore_module_resolution` is
    True (the default — matches `build_clean_prefix_set.py`'s load-bearing lesson),
    `src.lsp.diagnostics.MODULE_RESOLUTION_CODES` are dropped from the code list before the
    clean/dirty decision, so import-bearing real-world files aren't all marked dirty just
    because the sample's tsconfig can't resolve their imports.

    A record whose `tsc_runner.codes()` call raises is counted as `n_error` and skipped
    (never yielded, never counted as clean or dirty) — a toolchain hiccup on one file must
    not corrupt the filter rate for the rest of the corpus."""
    for record in records:
        if stats is not None:
            stats.n_seen += 1
        try:
            codes = tsc_runner.codes(record.text)
        except Exception:
            if stats is not None:
                stats.n_error += 1
            continue
        if ignore_module_resolution:
            codes = [c for c in codes if c not in MODULE_RESOLUTION_CODES]
        if codes:
            if stats is not None:
                stats.n_dirty += 1
            continue
        if stats is not None:
            stats.n_clean += 1
        yield record
