"""#230 -- CI-safe unit tests for the LSP/opengrep verifier reward: the pure
reward-shaping core (`severity_weight`, `is_degenerate_output`,
`diagnostics_to_reward`) on synthetic `Diagnostic`s, and `LspVerifier` against a
fake oracle -- no toolchain, no subprocess, no real `typescript-language-server`.

One toolchain-gated integration test lives at the bottom, behind the same
`resolve_ts_lsp() is None` skip idiom `tests/test_ts_lsp.py` uses.
"""

from __future__ import annotations

import pytest

from src.lsp.diagnostics import Diagnostic
from src.train.verifiers import (LspVerifier, diagnostics_to_reward, is_degenerate_output,
                                 severity_weight)


def _diag(code: str = "TS2339", severity: int = 1) -> Diagnostic:
    return Diagnostic(code=code, line=1, col=1, message="synthetic", offset=0, severity=severity)


# --------------------------------------------------------------------------- #
# severity_weight / diagnostics_to_reward -- pure, no oracle
# --------------------------------------------------------------------------- #

def test_severity_weight_default_table():
    assert severity_weight(_diag(severity=1)) == pytest.approx(0.30)
    assert severity_weight(_diag(severity=2)) == pytest.approx(0.15)
    assert severity_weight(_diag(severity=3)) == pytest.approx(0.05)
    assert severity_weight(_diag(severity=4)) == pytest.approx(0.05)


def test_severity_weight_code_override():
    d = _diag(code="TS9999", severity=1)
    assert severity_weight(d, code_overrides={"TS9999": 0.9}) == pytest.approx(0.9)
    # a code not in the override table falls back to the severity table.
    assert severity_weight(_diag(code="TS1", severity=1),
                           code_overrides={"TS9999": 0.9}) == pytest.approx(0.30)


def test_diagnostics_to_reward_clean_is_perfect():
    assert diagnostics_to_reward([]) == 1.0


def test_diagnostics_to_reward_one_error():
    assert diagnostics_to_reward([_diag(severity=1)]) == pytest.approx(0.7)


def test_diagnostics_to_reward_three_errors():
    assert diagnostics_to_reward([_diag(severity=1) for _ in range(3)]) == pytest.approx(0.1)


def test_diagnostics_to_reward_clamps_at_diag_floor():
    assert diagnostics_to_reward([_diag(severity=1) for _ in range(10)]) == pytest.approx(-0.5)


def test_diagnostics_to_reward_severity_2_and_3_weights():
    assert diagnostics_to_reward([_diag(severity=2)]) == pytest.approx(0.85)
    assert diagnostics_to_reward([_diag(severity=3)]) == pytest.approx(0.95)


def test_diagnostics_to_reward_hacked_dominates_even_when_otherwise_clean():
    assert diagnostics_to_reward([], hacked=True) == -1.0


def test_diagnostics_to_reward_hacked_dominates_over_diagnostics():
    # a hacked artifact scores -1.0 no matter how many (or few) diagnostics
    # survive -- the hack floor must sit BELOW diag_floor, never above it.
    assert diagnostics_to_reward([_diag(severity=1)], hacked=True) == -1.0


def test_diagnostics_to_reward_degenerate_is_minus_one():
    assert diagnostics_to_reward([], degenerate=True) == -1.0


def test_diagnostics_to_reward_both_flags_still_minus_one():
    assert diagnostics_to_reward([_diag(severity=1) for _ in range(10)],
                                 hacked=True, degenerate=True) == -1.0


# --------------------------------------------------------------------------- #
# is_degenerate_output -- pure, no oracle
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", ["", "   \n", "// only a comment\n", "/* block */", ";"])
def test_is_degenerate_output_flags_the_known_traps(text):
    reason = is_degenerate_output(text)
    assert reason is not None, f"{text!r} should be flagged degenerate"
    assert isinstance(reason, str)


def test_is_degenerate_output_empty_and_whitespace_reasons():
    assert is_degenerate_output("") == "empty"
    assert is_degenerate_output("   \n\t") == "whitespace_only"


def test_is_degenerate_output_real_code_is_none():
    assert is_degenerate_output("const x: number = 1 + 2;\nconsole.log(x);\n") is None
    assert is_degenerate_output("function add(a: number, b: number) { return a + b; }") is None


# --------------------------------------------------------------------------- #
# LspVerifier against a fake oracle -- no real typescript-language-server
# --------------------------------------------------------------------------- #

class FakeOracle:
    """Stands in for `CompositeOracle`'s public surface
    (`.diagnostics()/.close()/.n_calls/.wall_s/.ts_stats/.opengrep_stats`).
    `diags` is either a fixed list returned on every call, or a callable
    `source -> List[Diagnostic]` for tests that need call-dependent output."""

    def __init__(self, diags=None, raises: bool = False):
        self._diags = [] if diags is None else diags
        self._raises = raises
        self.calls = []
        self.n_calls = 0
        self.wall_s = 0.0
        self.ts_stats = {"n_calls": 0, "wall_s": 0.0, "n_timeouts": 0, "n_restarts": 0}
        self.opengrep_stats = None
        self.close_count = 0

    def diagnostics(self, source):
        self.calls.append(source)
        self.n_calls += 1
        self.wall_s += 0.01
        if self._raises:
            raise RuntimeError("simulated oracle failure")
        return list(self._diags(source)) if callable(self._diags) else list(self._diags)

    def close(self):
        self.close_count += 1


def test_lsp_verifier_clean_sample_scores_one():
    v = LspVerifier(oracle=FakeOracle(diags=[]))
    assert v.reward("const x = 1;\n", prompt="") == 1.0


def test_lsp_verifier_suppression_trap_dominates_clean_diagnostics():
    # the fake oracle reports NO diagnostics -- but `as any` in the completion
    # must still zero the reward via the hack floor, not the (empty) diagnostic set.
    v = LspVerifier(oracle=FakeOracle(diags=[]))
    assert v.reward("const x = y as any;\n") == -1.0


def test_lsp_verifier_empty_completion_trap():
    v = LspVerifier(oracle=FakeOracle(diags=[]))
    assert v.reward("") == -1.0


def test_lsp_verifier_module_resolution_dropped_by_default():
    v = LspVerifier(oracle=FakeOracle(diags=[_diag(code="TS2307")]))
    assert v.reward("import { x } from 'y';\n") == 1.0


def test_lsp_verifier_module_resolution_counted_when_disabled():
    v = LspVerifier(oracle=FakeOracle(diags=[_diag(code="TS2307")]),
                    ignore_module_resolution=False)
    assert v.reward("import { x } from 'y';\n") == pytest.approx(0.7)


def test_lsp_verifier_hatches_directives_ignores_superset_only_members():
    # empty_body is a superset-only hatch (a known-noisy false positive); under
    # "directives" it must not trip the hack floor, only the diagnostic penalty applies.
    v = LspVerifier(oracle=FakeOracle(diags=[]), hatches="directives")
    assert v.reward("function f() {}\n") == 1.0


def test_lsp_verifier_hatches_none_disables_hack_floor():
    v = LspVerifier(oracle=FakeOracle(diags=[]), hatches="none")
    assert v.reward("const x = y as any;\n") == 1.0


def test_lsp_verifier_telemetry_keys_and_values():
    v = LspVerifier(oracle=FakeOracle(diags=[]))
    v.reward("const x = y as any;\n")
    v.reward("const y = 1;\n")
    t = v.telemetry()
    for key in ("n_samples", "n_clean", "n_hacked", "n_degenerate", "hatch_counts",
               "degenerate_reasons", "mean_diagnostics", "mean_reward", "n_calls",
               "wall_s", "ts_stats", "opengrep_stats"):
        assert key in t
    assert t["n_samples"] == 2
    assert t["n_hacked"] == 1
    assert t["n_clean"] == 1
    assert t["hatch_counts"].get("as_any") == 1
    assert t["n_calls"] == 2


def test_lsp_verifier_close_is_idempotent():
    oracle = FakeOracle(diags=[])
    v = LspVerifier(oracle=oracle)
    v.reward("const x = 1;\n")
    v.close()
    v.close()
    assert oracle.close_count == 1


def test_lsp_verifier_context_manager_closes_once():
    oracle = FakeOracle(diags=[])
    with LspVerifier(oracle=oracle) as v:
        v.reward("const x = 1;\n")
    v.close()  # a second close after __exit__ must still be a no-op
    assert oracle.close_count == 1


def test_lsp_verifier_on_error_raise_propagates():
    v = LspVerifier(oracle=FakeOracle(raises=True))
    with pytest.raises(RuntimeError):
        v.reward("const x = 1;\n")
    assert v.telemetry()["n_oracle_errors"] == 1


def test_lsp_verifier_on_error_skip_returns_none():
    v = LspVerifier(oracle=FakeOracle(raises=True), on_error="skip")
    assert v.reward("const x = 1;\n") is None
    assert v.telemetry()["n_oracle_errors"] == 1


def test_lsp_verifier_rejects_unknown_hatches_mode():
    with pytest.raises(ValueError):
        LspVerifier(hatches="bogus")


def test_lsp_verifier_rejects_unknown_on_error():
    with pytest.raises(ValueError):
        LspVerifier(on_error="bogus")


# --------------------------------------------------------------------------- #
# Toolchain-gated integration test (skipped without a real TS-LSP toolchain)
# --------------------------------------------------------------------------- #

from src.lsp.ts_lsp import resolve_ts_lsp  # noqa: E402 -- grouped with its own section

_ts_lsp_available = resolve_ts_lsp() is not None


@pytest.mark.skipif(not _ts_lsp_available,
                    reason="no typescript-language-server toolchain on this host")
def test_lsp_verifier_real_oracle_ranks_clean_above_broken():
    with LspVerifier(kind="ts") as v:
        clean = v.reward("function add(a: number, b: number): number {\n  return a + b;\n}\n")
        broken = v.reward(
            "function add(a: number, b: number): number {\n  return a + gorblak;\n}\n")
        assert clean > broken
        assert v.telemetry()["n_calls"] == 2
