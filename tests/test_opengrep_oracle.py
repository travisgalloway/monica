"""Deterministic mechanism tests for `OpengrepOracle`'s reliability handling --
the proactive-recycle + reactive-restart-on-stall logic added to `diagnostics()`
to survive the measured ~10% full-timeout stall (see `src/lsp/opengrep.py`'s
module docstring).

These need **no** `opengrep` binary: they bypass the real `__init__`/`_start_server`
(which would spawn a subprocess) and script `_rescan`'s return values directly, so
they run in CI on every host. The live-binary validation lives in
`scripts/opengrep_soak.py`; the ruleset-fixture tests live in `tests/test_opengrep.py`.
"""

from __future__ import annotations

from typing import List, Tuple

from src.lsp.opengrep import OpengrepOracle

# A minimal well-formed raw LSP finding, enough for `_map_finding` to build a
# Diagnostic (line/character 0-indexed; maps to offset 0 for any source).
_RAW_FINDING = {
    "range": {"start": {"line": 0, "character": 0}},
    "code": "loop-bound-off-by-one",
    "message": "off-by-one",
    "severity": 1,
}


class _FakeOracle(OpengrepOracle):
    """An `OpengrepOracle` whose subprocess-touching pieces are replaced by
    counters, and whose `_rescan` plays back a scripted list of
    `(got, payload)` tuples. The real `diagnostics()` / `_restart()` run
    unchanged -- that is exactly what's under test."""

    def __init__(self, rescan_script: List[Tuple[bool, list]], *,
                 recycle_every: int, restart_on_stall: bool) -> None:
        # Deliberately bypass OpengrepOracle.__init__ (no binary, no spawn).
        self.timeout_s = 5.0
        self.recycle_every = recycle_every
        self.restart_on_stall = restart_on_stall
        self.n_calls = 0
        self.wall_s = 0.0
        self.n_timeouts = 0
        self.n_restarts = 0
        self.n_recycles = 0
        self.n_stall_recoveries = 0
        self._script = list(rescan_script)
        self.rescan_calls = 0
        self.starts = 0
        self.teardowns = 0

    # --- subprocess seam, stubbed to counters -------------------------- #
    def _ensure_alive(self) -> None:
        pass  # no real process in the fake

    def _teardown_process(self) -> None:
        self.teardowns += 1

    def _start_server(self) -> None:
        self.starts += 1

    def _rescan(self, source: str, timeout_s: float):
        self.rescan_calls += 1
        if self._script:
            return self._script.pop(0)
        return (True, [])  # default: responded, no findings


def test_proactive_recycle_fires_on_the_call_count_boundary():
    # recycle_every=3: over 7 calls, the boundary (n_calls % 3 == 0, n_calls > 0)
    # is crossed before call 4 (n_calls==3) and call 7 (n_calls==6) -> 2 recycles.
    o = _FakeOracle([(True, [])] * 7, recycle_every=3, restart_on_stall=False)
    for _ in range(7):
        assert o.diagnostics("x") == []
    assert o.n_calls == 7
    assert o.n_recycles == 2
    assert o.n_restarts == 2          # every recycle funnels through _restart
    assert o.starts == 2 and o.teardowns == 2
    assert o.n_stall_recoveries == 0
    assert o.n_timeouts == 0


def test_reactive_restart_recovers_a_single_stall():
    # First scan stalls (no response), retry against a fresh process succeeds.
    o = _FakeOracle([(False, []), (True, [_RAW_FINDING])],
                    recycle_every=0, restart_on_stall=True)
    diags = o.diagnostics("x")
    assert len(diags) == 1 and diags[0].code == "loop-bound-off-by-one"
    assert o.rescan_calls == 2         # original + one retry
    assert o.n_stall_recoveries == 1
    assert o.n_restarts == 1
    assert o.n_timeouts == 0
    assert o.n_calls == 1              # one *logical* call


def test_persistent_stall_is_counted_after_one_retry():
    # Both the original and the retry stall -> counted as a timeout, returns [].
    o = _FakeOracle([(False, []), (False, [])],
                    recycle_every=0, restart_on_stall=True)
    assert o.diagnostics("x") == []
    assert o.rescan_calls == 2         # original + one retry, then give up
    assert o.n_stall_recoveries == 1   # the retry was attempted
    assert o.n_restarts == 1
    assert o.n_timeouts == 1
    assert o.n_calls == 1


def test_mitigation_off_reproduces_pre_fix_behavior():
    # recycle_every=0 + restart_on_stall=False: a stall is a silent [] with no respawn.
    o = _FakeOracle([(False, [])], recycle_every=0, restart_on_stall=False)
    assert o.diagnostics("x") == []
    assert o.rescan_calls == 1         # no retry
    assert o.n_restarts == 0
    assert o.n_recycles == 0
    assert o.n_stall_recoveries == 0
    assert o.n_timeouts == 1
