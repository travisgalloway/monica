"""Verifiable rewards for RLVR/GRPO (#78). Pure stdlib; the code path is gated."""

import os

import pytest

from src.train.verifiers import (CodeVerifier, exact_match_reward, extract_final_number,
                                 math_reward, normalize_text)


def test_exact_match_normalizes():
    assert exact_match_reward("Yes", " yes ") == 1.0
    assert exact_match_reward("a   b", "a b") == 1.0          # whitespace collapse
    assert exact_match_reward("foo", "bar") == 0.0
    assert normalize_text("  Hello   World ") == "hello world"


def test_extract_final_number():
    assert extract_final_number("The answer is 42.") == 42.0
    assert extract_final_number("blah #### 1,234") == 1234.0   # GSM8K marker + separators
    assert extract_final_number("step 3, then -7.5") == -7.5   # last number
    assert extract_final_number("no digits here") is None


def test_math_reward():
    assert math_reward("so there are 18 apples", "#### 18") == 1.0
    assert math_reward("the total is 19", "18") == 0.0
    assert math_reward("no number at all", "5") == 0.0
    assert math_reward("answer: 3.0", "3") == 1.0             # tolerance


def test_code_verifier_disabled_raises():
    # Executing untrusted model output must be an explicit opt-in.
    with pytest.raises(RuntimeError):
        CodeVerifier().reward("x = 1", ["assert x == 1"])
    assert CodeVerifier(enabled=True).reward("x = 1", []) == 0.0   # no tests -> 0


@pytest.mark.skipif(not os.environ.get("RUN_CODE_VERIFIER"),
                    reason="CodeVerifier runs code in a subprocess; opt-in via RUN_CODE_VERIFIER")
def test_code_verifier_partial_credit():
    cv = CodeVerifier(enabled=True)
    assert cv.reward("def f():\n    return 2\n", ["assert f() == 2", "assert f() == 3"]) == 0.5


def test_memoized_verifier_caches_rewards():
    from src.train.verifiers import MemoizedVerifier

    call_count = 0

    def mock_reward(c, ref=None, *, prompt=""):
        nonlocal call_count
        call_count += 1
        return 1.0 if c == "good" else 0.0

    mv = MemoizedVerifier(mock_reward)
    assert mv.reward("good", prompt="p1") == 1.0
    assert mv.reward("good", prompt="p1") == 1.0
    assert mv.reward("good", prompt="p1") == 1.0
    assert mv.reward("bad", prompt="p1") == 0.0

    assert call_count == 2
    t = mv.telemetry()
    assert t["cache_hits"] == 2
    assert t["cache_misses"] == 2
    assert t["cache_hit_rate"] == 0.5
    assert t["cache_size"] == 2


def test_memoized_verifier_context_manager_and_telemetry():
    from src.train.verifiers import MemoizedVerifier

    class MockTarget:
        def __init__(self):
            self.closed = False

        def reward(self, c, ref=None, *, prompt=""):
            return 0.5

        def telemetry(self):
            return {"base_metric": 42}

        def close(self):
            self.closed = True

    target = MockTarget()
    with MemoizedVerifier(target) as mv:
        assert mv.reward("test") == 0.5
        t = mv.telemetry()
        assert t["base_metric"] == 42
        assert t["cache_misses"] == 1
    assert target.closed


def test_lsp_verifier_fail_fast_skips_oracle():
    from src.train.verifiers import LspVerifier

    class FakeOracle:
        def __init__(self):
            self.n_calls = 0
            self.wall_s = 0.0
            self.ts_stats = None
            self.opengrep_stats = None

        def diagnostics(self, s):
            self.n_calls += 1
            return []

        def close(self):
            pass

    # fail_fast=True skips oracle when hacked or degenerate
    oracle_fast = FakeOracle()
    v_fast = LspVerifier(oracle=oracle_fast, fail_fast=True)
    assert v_fast.reward("const x = y as any;\n") == -1.0
    assert v_fast.reward("") == -1.0
    assert oracle_fast.n_calls == 0
    assert v_fast.telemetry()["n_hacked"] == 1
    assert v_fast.telemetry()["n_degenerate"] == 1

    # Clean code still reaches the oracle
    assert v_fast.reward("const x = 1;\n") == 1.0
    assert oracle_fast.n_calls == 1

    # fail_fast=False (default backward compatible) still queries oracle
    oracle_slow = FakeOracle()
    v_slow = LspVerifier(oracle=oracle_slow, fail_fast=False)
    assert v_slow.reward("const x = y as any;\n") == -1.0
    assert oracle_slow.n_calls == 1


def test_score_rollouts_sequential_and_concurrent_parity():
    from concurrent.futures import ThreadPoolExecutor
    from src.train.verifiers import score_rollouts

    def dummy_reward(c, ref=None):
        return float(len(c))

    completions = [f"code_sample_{i}" for i in range(16)]

    # Sequential
    seq_scores = score_rollouts(dummy_reward, completions, max_workers=1)
    # Concurrent internal pool
    par_scores = score_rollouts(dummy_reward, completions, max_workers=4)
    # Concurrent external pool
    with ThreadPoolExecutor(max_workers=4) as ex:
        pool_scores = score_rollouts(dummy_reward, completions, executor=ex)

    assert seq_scores == par_scores
    assert seq_scores == pool_scores
    assert len(seq_scores) == 16
    assert seq_scores[0] == float(len("code_sample_0"))

