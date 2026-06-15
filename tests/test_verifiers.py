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
