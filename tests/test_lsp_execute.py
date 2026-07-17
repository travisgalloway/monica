"""Functional execution guard (#199 F1) — pass/fail/timeout classification.

The `pass@1` guard is only trustworthy if it distinguishes "the code is correct" from
"the code compiles". These pin that: a correct body passes, a wrong body fails at
runtime (not at compile), a syntax-broken body is a compile_fail, and an infinite loop
is a timeout rather than a hang. Skips gracefully without a node toolchain.
"""

from __future__ import annotations

import pytest

from src.lsp.execute import Executor
from src.lsp.tsc import resolve_tsc

pytestmark = pytest.mark.skipif(resolve_tsc() is None,
                                 reason="no node/tsc toolchain (run npm install in eval_sets/ts_error_injection)")

# A tiny self-contained problem in MultiPL-E shape: open-body prompt + tests that exit nonzero on failure.
PROMPT = "function add(a: number, b: number): number {\n"
TESTS = """
declare var process: any;
function assert(c: boolean) { if (!c) { process.exit(1); } }
assert(add(2, 3) === 5);
assert(add(-1, 1) === 0);
"""


@pytest.fixture(scope="module")
def ex():
    e = Executor()
    yield e
    e.close()


def test_correct_body_passes(ex):
    r = ex.run_tests(PROMPT, "  return a + b;\n}\n", TESTS)
    assert r.passed and r.outcome == "pass"


def test_wrong_body_fails_at_runtime_not_compile(ex):
    """A type-correct but behaviourally-wrong body compiles and RUNS, then fails its
    tests — the case that makes pass@1 independent of clean-rate."""
    r = ex.run_tests(PROMPT, "  return a - b;\n}\n", TESTS)
    assert not r.passed and r.outcome == "runtime_fail"


def test_type_error_still_runs_so_passat1_is_independent_of_cleanrate(ex):
    """`--noEmitOnError false`: a body with a type error still emits JS and runs. If the
    behaviour is correct it PASSES pass@1 even though it is not tsc-clean — the whole
    reason pass@1 can guard the clean-rate metric."""
    # `a + b` is correct behaviour; the stray `as any` cast is a type smell, not a runtime bug.
    r = ex.run_tests(PROMPT, "  return (a as any) + (b as any);\n}\n", TESTS)
    assert r.passed


def test_syntax_error_does_not_pass(ex):
    """A broken body must not pass. Whether it lands as compile_fail or runtime_fail is
    not load-bearing (tsc with --noEmitOnError false emits degenerate JS for some parse
    errors, which then throws at runtime); the guarantee is `not passed`."""
    r = ex.run_tests(PROMPT, "  return a + ;\n}\n", TESTS)
    assert not r.passed and r.outcome in ("compile_fail", "runtime_fail")


def test_infinite_loop_is_timeout_not_hang(ex):
    slow = Executor(timeout_s=2.0)
    try:
        r = slow.run_tests(PROMPT, "  while (true) {}\n  return a + b;\n}\n", TESTS)
        assert not r.passed and r.outcome == "timeout"
    finally:
        slow.close()
