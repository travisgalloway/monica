"""The drop-in oracle -- `CompositeOracle` is the seam-facing replacement for
`TscRunner`, wiring together the TS-LSP arm (`ts_lsp.py`, always reliable) and the
opengrep arm (`opengrep.py`, correct but experimental -- see its module docstring's
"Known residual limitation") behind one object with `TscRunner`'s exact public
contract (`.diagnostics`, `.codes`, `.close`, `__enter__`/`__exit__`, `.n_calls`,
`.wall_s`), so both drivers (`scripts/eval_lsp_humaneval.py`,
`scripts/eval_lsp_harness.py`) construct it and read its cost counters unchanged.

`kind` selects which arm(s) run:

  - `"ts"` (the default, and Stage A's shipped default) -- TS-LSP only.
  - `"opengrep"` -- opengrep only, no type-checking arm at all.
  - `"both"` -- both arms, findings merged and deduplicated by `(offset, code)`.
    If opengrep can't be constructed (binary missing, or its startup warmup never
    settles -- see `opengrep.py`), `"both"` **degrades to TS-LSP alone** rather
    than taking the whole harness down over the experimental arm; the degradation
    is recorded in `sources_active` so a results JSON is self-describing about
    what actually ran. `"opengrep"` alone has nothing to degrade TO, so it raises
    if opengrep isn't constructible.

`n_calls`/`wall_s` are **summed across the active arms** (properties, not
independently tracked) -- they reflect each arm's own real cost rather than
double-counting a wrapper-level stopwatch.

ABOVE THE SEAM -- stdlib only. No `mlx`/`torch` import anywhere in this module
(guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .diagnostics import Diagnostic
from .opengrep import OpengrepOracle, resolve_opengrep
from .ts_lsp import TsLspOracle, resolve_ts_lsp

VALID_KINDS = ("ts", "opengrep", "both")

_DEFAULT_TIMEOUT_S = 10.0


def resolve_oracle(kind: str) -> bool:
    """True if `kind`'s REQUIRED toolchain is resolvable. `"ts"` and `"both"`
    both need TS-LSP (opengrep is optional for `"both"` -- it degrades, it
    doesn't block); `"opengrep"` needs the opengrep binary, since that mode has
    no TS-LSP to fall back to."""
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown oracle kind {kind!r} (want one of {VALID_KINDS})")
    if kind == "opengrep":
        return resolve_opengrep() is not None
    return resolve_ts_lsp() is not None


class CompositeOracle:
    """Reproduces `TscRunner`'s public surface over one or both LSP oracle arms.
    Not safe for concurrent use, matching both underlying oracles."""

    def __init__(self, kind: str = "ts", *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown oracle kind {kind!r} (want one of {VALID_KINDS})")
        self.kind = kind
        self.sources_active: List[str] = []

        self._ts: Optional[TsLspOracle] = None
        self._opengrep: Optional[OpengrepOracle] = None

        if kind in ("ts", "both"):
            self._ts = TsLspOracle(timeout_s=timeout_s)
            self.sources_active.append("ts")

        if kind in ("opengrep", "both"):
            if resolve_opengrep() is None:
                if kind == "opengrep":
                    raise RuntimeError(
                        "no opengrep toolchain resolvable -- see "
                        "eval_sets/opengrep_rules/README.md")
                # kind == "both": no TS-LSP to lose here since it's already up;
                # just proceed without the opengrep arm.
            else:
                try:
                    self._opengrep = OpengrepOracle(timeout_s=timeout_s)
                    self.sources_active.append("opengrep")
                except RuntimeError:
                    if kind == "opengrep":
                        raise
                    # kind == "both": opengrep failed to become ready (e.g. its
                    # startup warmup never settled) -- degrade to TS-LSP alone
                    # rather than losing the whole harness over the
                    # experimental arm. self._opengrep stays None.
                    pass

    def _active_oracles(self) -> Tuple:
        return tuple(o for o in (self._ts, self._opengrep) if o is not None)

    @property
    def n_calls(self) -> int:
        return sum(o.n_calls for o in self._active_oracles())

    @property
    def wall_s(self) -> float:
        return sum(o.wall_s for o in self._active_oracles())

    def diagnostics(self, source: str) -> List[Diagnostic]:
        """Every arm's findings, UNFILTERED (the caller applies
        `diagnostics.filter_diagnostics`, same as `TscRunner`/`TsLspOracle`),
        merged and deduplicated by `(offset, code)` -- the two arms can agree
        on the same defect (e.g. a syntax error both a parser and a pattern
        matcher would flag) and should count once, not twice."""
        merged: List[Diagnostic] = []
        seen = set()
        for oracle in self._active_oracles():
            for d in oracle.diagnostics(source):
                key = (d.offset, d.code)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(d)
        return merged

    def codes(self, source: str) -> List[str]:
        """`[d.code for d in self.diagnostics(source)]` -- mirrors `TscRunner.codes`."""
        return [d.code for d in self.diagnostics(source)]

    def close(self) -> None:
        for oracle in self._active_oracles():
            oracle.close()

    def __enter__(self) -> "CompositeOracle":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
