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




# --------------------------------------------------------------------------- #
# #339: ToolSchemaVerifier & When2CallAbstentionVerifier Tests
# --------------------------------------------------------------------------- #

SAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Fetch current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "units": {"type": "string"},
                "detailed": {"type": "boolean"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_database",
        "description": "Execute database query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
]


def test_tool_schema_verifier_valid_calls():
    from src.train.verifiers import ToolSchemaVerifier

    tsv = ToolSchemaVerifier(tools=SAMPLE_TOOLS)

    # 1. Valid call with required parameter
    comp1 = '<tool_call>{"name": "get_weather", "arguments": {"city": "Seattle"}}</tool_call>'
    assert tsv.reward(comp1) == 1.0

    # 2. Valid call with all optional parameters conforming to types
    comp2 = (
        '<tool_call>{"name": "get_weather", "arguments": '
        '{"city": "Paris", "days": 5, "units": "celsius", "detailed": true}}</tool_call>'
    )
    assert tsv.reward(comp2) == 1.0

    # 3. Valid call with array of strings
    comp3 = (
        '<tool_call>{"name": "search_database", "arguments": '
        '{"query": "SELECT 1", "tags": ["db", "sql"], "limit": 10}}</tool_call>'
    )
    assert tsv.reward(comp3) == 1.0


def test_tool_schema_verifier_syntax_errors():
    from src.train.verifiers import ToolSchemaVerifier

    tsv = ToolSchemaVerifier(tools=SAMPLE_TOOLS)

    # Unclosed <tool_call> tag
    assert tsv.reward('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}') == -1.0

    # Malformed JSON syntax
    assert tsv.reward('<tool_call>{"name": "get_weather", "arguments": {broken_json}}</tool_call>') == -1.0

    # Payload not a JSON object
    assert tsv.reward('<tool_call>["get_weather", "Paris"]</tool_call>') == -1.0

    # Missing "name" key
    assert tsv.reward('<tool_call>{"arguments": {"city": "Paris"}}</tool_call>') == -1.0

    # Arguments not a dict
    assert tsv.reward('<tool_call>{"name": "get_weather", "arguments": "Paris"}</tool_call>') == -1.0

    # No tool call when tool was expected
    assert tsv.reward("I think the weather in Paris is sunny today.") == -1.0


def test_tool_schema_verifier_hallucinations():
    from src.train.verifiers import ToolSchemaVerifier

    tsv = ToolSchemaVerifier(tools=SAMPLE_TOOLS)

    # Tool name not in declared tools -> hallucination penalized
    comp = '<tool_call>{"name": "google_search", "arguments": {"query": "weather"}}</tool_call>'
    assert tsv.reward(comp) == -1.0
    t = tsv.telemetry()
    assert t["n_hallucinations"] >= 1


def test_tool_schema_verifier_missing_required_args():
    from src.train.verifiers import ToolSchemaVerifier

    tsv = ToolSchemaVerifier(tools=SAMPLE_TOOLS)

    # get_weather requires "city", but arguments is empty
    comp = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    assert tsv.reward(comp) == -1.0

    # search_database requires "query", only optional "limit" provided
    comp2 = '<tool_call>{"name": "search_database", "arguments": {"limit": 5}}</tool_call>'
    assert tsv.reward(comp2) == -1.0


def test_tool_schema_verifier_parameter_type_mismatch():
    from src.train.verifiers import ToolSchemaVerifier

    tsv = ToolSchemaVerifier(tools=SAMPLE_TOOLS)

    # "city" expects string, got integer
    comp1 = '<tool_call>{"name": "get_weather", "arguments": {"city": 12345}}</tool_call>'
    assert tsv.reward(comp1) == -1.0

    # "days" expects integer, got string
    comp2 = '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome", "days": "five"}}</tool_call>'
    assert tsv.reward(comp2) == -1.0

    # "days" expects integer, got boolean (in Python bool is int subclass, schema must reject)
    comp3 = '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome", "days": true}}</tool_call>'
    assert tsv.reward(comp3) == -1.0

    # "detailed" expects boolean, got string
    comp4 = '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome", "detailed": "yes"}}</tool_call>'
    assert tsv.reward(comp4) == -1.0

    # "tags" expects array of strings, got array of ints
    comp5 = '<tool_call>{"name": "search_database", "arguments": {"query": "test", "tags": [1, 2]}}</tool_call>'
    assert tsv.reward(comp5) == -1.0


def test_tool_schema_verifier_extracts_tools_from_prompt():
    import json
    from src.train.verifiers import ToolSchemaVerifier

    tsv = ToolSchemaVerifier()  # no tools in __init__
    prompt = "<tools>\n" + json.dumps(SAMPLE_TOOLS) + "\n</tools>\nUser: check Paris weather"

    # Valid call using tools extracted from prompt
    comp_valid = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    assert tsv.reward(comp_valid, prompt=prompt) == 1.0

    # Hallucinated tool against tools extracted from prompt
    comp_hallu = '<tool_call>{"name": "fake_tool", "arguments": {}}</tool_call>'
    assert tsv.reward(comp_hallu, prompt=prompt) == -1.0


def test_tool_schema_verifier_parallel_calls():
    from src.train.verifiers import ToolSchemaVerifier

    tsv = ToolSchemaVerifier(tools=SAMPLE_TOOLS)

    # Both parallel calls valid
    comp_both_valid = (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>\n'
        '<tool_call>{"name": "search_database", "arguments": {"query": "Tokyo info"}}</tool_call>'
    )
    assert tsv.reward(comp_both_valid) == 1.0

    # One call valid, one hallucinated -> penalized
    comp_one_hallu = (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>\n'
        '<tool_call>{"name": "calc_tax", "arguments": {"amount": 100}}</tool_call>'
    )
    assert tsv.reward(comp_one_hallu) == -1.0


def test_tool_schema_verifier_partial_credit():
    from src.train.verifiers import ToolSchemaVerifier

    tsv_strict = ToolSchemaVerifier(tools=SAMPLE_TOOLS, partial_credit=False)
    tsv_partial = ToolSchemaVerifier(tools=SAMPLE_TOOLS, partial_credit=True)

    # 1 missing argument
    comp = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    assert tsv_strict.reward(comp) == -1.0
    r_partial = tsv_partial.reward(comp)
    assert r_partial > -1.0  # partial credit gives higher reward than complete failure


def test_when2call_abstention_verifier_scoring():
    from src.train.verifiers import When2CallAbstentionVerifier

    w2c = When2CallAbstentionVerifier()

    # Direct answer without tool call -> rewarded (+1.0)
    assert w2c.reward("The Eiffel Tower is in Paris, France.") == 1.0
    assert w2c.reward("Python is a dynamic programming language.") == 1.0

    # Spurious / redundant tool invocation -> penalized (-1.0)
    spurious1 = '<tool_call>{"name": "search", "arguments": {"query": "Eiffel Tower"}}</tool_call>'
    assert w2c.reward(spurious1) == -1.0

    # Partial / broken tool tag in answer -> penalized (-1.0)
    spurious2 = 'Let me check: <tool_call>{"name": "lookup"}'
    assert w2c.reward(spurious2) == -1.0

    # Degenerate / empty answer -> penalized (-1.0)
    assert w2c.reward("") == -1.0
    assert w2c.reward("   ") == -1.0
    assert w2c.reward("x") == -1.0  # too short (< 2 chars)

    t = w2c.telemetry()
    assert t["n_samples"] == 7
    assert t["n_abstain_success"] == 2
    assert t["n_spurious_calls"] == 2
    assert t["n_degenerate"] == 3


def test_tool_schema_verifier_when2call_integration():
    from src.train.verifiers import ToolSchemaVerifier

    tsv = ToolSchemaVerifier(tools=SAMPLE_TOOLS)

    # Abstention via abstain=True flag
    assert tsv.reward("Direct answer to general question.", abstain=True) == 1.0
    assert tsv.reward('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>', abstain=True) == -1.0

    # Abstention via category="abstention"
    assert tsv.reward("Direct answer without tools.", category="abstention") == 1.0
    assert tsv.reward('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>', category="abstention") == -1.0

    # Abstention via reference without tool calls
    assert tsv.reward("42 is the answer.", reference="42") == 1.0
    assert tsv.reward('<tool_call>{"name": "calc", "arguments": {}}</tool_call>', reference="42") == -1.0


def test_tool_schema_verifier_telemetry():
    from src.train.verifiers import ToolSchemaVerifier

    tsv = ToolSchemaVerifier(tools=SAMPLE_TOOLS)
    tsv.reward('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>')  # valid
    tsv.reward('<tool_call>{"name": "fake_tool", "arguments": {}}</tool_call>')                     # hallucination
    tsv.reward('<tool_call>{"name": "get_weather", "arguments": {broken}}</tool_call>')             # syntax error
    tsv.reward('<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>')                   # missing arg
    tsv.reward('<tool_call>{"name": "get_weather", "arguments": {"city": 999}}</tool_call>')        # type error
    tsv.reward("No tools here", abstain=True)                                                        # abstention correct
    tsv.reward('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>', abstain=True)  # spurious

    t = tsv.telemetry()
    assert t["n_samples"] == 7
    assert t["n_valid"] == 1
    assert t["n_hallucinations"] == 1
    assert t["n_syntax_errors"] == 1
    assert t["n_missing_args"] == 1
    assert t["n_type_errors"] == 1
    assert t["n_abstentions_correct"] == 1
    assert t["n_spurious_calls"] == 1


def test_rlvr_tool_schema_collate_rollouts_and_grpo_advantages():
    """Acceptance: scripts/rlvr.py --reward tool-schema runs with collate_rollouts and GRPO advantage tracking."""
    from scripts.rlvr import collate_rollouts
    from src.train.grpo import group_advantages
    from src.train.verifiers import ToolSchemaVerifier, score_rollouts

    tsv = ToolSchemaVerifier(tools=SAMPLE_TOOLS)

    # Rollouts: 1 valid call, 1 hallucinated call, 1 type error call, 1 syntax error call
    completions = [
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
        '<tool_call>{"name": "hallucinated_tool", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "get_weather", "arguments": {"city": 123}}</tool_call>',
        '<tool_call>{"name": "get_weather", "arguments": {bad_json}}</tool_call>',
    ]

    prompt_ids = [10, 20, 30]
    gen_ids_list = [
        [101, 102, 103],
        [201, 202],
        [301, 302, 303, 304],
        [401],
    ]
    rollouts = [(prompt_ids, g) for g in gen_ids_list]

    rewards = score_rollouts(tsv.reward, completions)
    assert rewards == [1.0, -1.0, -1.0, -1.0]

    # Compute GRPO advantages
    adv = group_advantages([rewards])[0]
    assert len(adv) == 4
    # The valid call must have strictly positive advantage
    assert adv[0] > 0.0
    # The invalid calls must have negative advantage
    assert adv[1] < 0.0 and adv[2] < 0.0 and adv[3] < 0.0
    # Advantages within group must sum to ~0
    assert abs(float(adv.sum())) < 1e-5

    # Collate into GRPO micro-batch
    inputs, targets, mask, adv_out = collate_rollouts(rollouts, adv, pad_id=0)

    # Verify shapes and types
    B = len(rollouts)
    max_len = max(len(p) + len(g) for p, g in rollouts)
    assert inputs.shape == (B, max_len - 1)
    assert targets.shape == (B, max_len - 1)
    assert mask.shape == (B, max_len - 1)
    assert adv_out.shape == (B,)
    assert (adv_out == adv.astype("float32")).all()

    # Mask must be 1 on generated tokens and 0 on prompt tokens
    for i, (p, g) in enumerate(rollouts):
        prompt_len = len(p)
        gen_len = len(g)
        total_len = prompt_len + gen_len
        assert (mask[i, :prompt_len - 1] == 0.0).all()
        assert (mask[i, prompt_len - 1:total_len - 1] == 1.0).all()


# --------------------------------------------------------------------------- #
# #340: SympyVerifier & Z3Verifier Tests
# --------------------------------------------------------------------------- #

def test_extract_math_expression():
    from src.train.verifiers import extract_math_expression

    # \boxed{...} with nested braces
    assert extract_math_expression(r"Compute 1/4 + 1/4. \boxed{\frac{1}{2}}.") == "((1)/(2))"
    # GSM8K format ####
    assert extract_math_expression("The result is #### 42") == "42"
    # Answer prefix
    assert extract_math_expression("Therefore, the answer is: 3*x + 1.") == "3*x + 1"
    # Radicals
    assert extract_math_expression(r"\sqrt{8}") == "sqrt(8)"
    assert extract_math_expression(r"\sqrt[3]{27}") == "((27)**(1/(3)))"
    # Commas in numbers
    assert extract_math_expression("1,234,567") == "1234567"
    # Dollar signs
    assert extract_math_expression(r"$\frac{3}{4}$") == "((3)/(4))"


def test_sympy_symbolic_reward_algebraic_equivalence():
    from src.train.verifiers import sympy_symbolic_reward

    # Equivalent fractions
    assert sympy_symbolic_reward("1/2", "2/4") == 1.0
    assert sympy_symbolic_reward(r"\frac{1}{2}", "0.5") == 1.0
    assert sympy_symbolic_reward("1/3 + 1/6", "1/2") == 1.0

    # Equivalent radicals
    assert sympy_symbolic_reward("sqrt(8)", "2*sqrt(2)") == 1.0
    assert sympy_symbolic_reward(r"\sqrt{12}", "2*sqrt(3)") == 1.0

    # Polynomial expansion & algebraic expressions
    assert sympy_symbolic_reward("(x + 1)^2", "x^2 + 2*x + 1") == 1.0
    assert sympy_symbolic_reward("(x - 2)*(x + 2)", "x^2 - 4") == 1.0
    assert sympy_symbolic_reward("2x + 3x", "5*x") == 1.0

    # Trig identities
    assert sympy_symbolic_reward("sin(x)^2 + cos(x)^2", "1") == 1.0

    # Inequivalent expressions
    assert sympy_symbolic_reward("x + 1", "x + 2") == 0.0
    assert sympy_symbolic_reward("1/3", "1/2") == 0.0


def test_sympy_symbolic_reward_equations():
    from src.train.verifiers import sympy_symbolic_reward

    # Equation to value / RHS extraction
    assert sympy_symbolic_reward("x = 5", "5") == 1.0
    assert sympy_symbolic_reward("5", "x = 5") == 1.0
    assert sympy_symbolic_reward("x = 1/2", "2/4") == 1.0

    # Equivalent equations
    assert sympy_symbolic_reward("x + 2 = 7", "x = 5") == 1.0
    assert sympy_symbolic_reward("2*x = 10", "x = 5") == 1.0
    assert sympy_symbolic_reward("3*x + 6 = 0", "x = -2") == 1.0

    # Inequivalent equations
    assert sympy_symbolic_reward("x = 5", "x = 6") == 0.0


def test_sympy_verifier_class_and_telemetry():
    from src.train.verifiers import SympyVerifier

    verifier = SympyVerifier()
    # Completion inside solution text
    comp1 = r"Simplifying the radical gives \boxed{2\sqrt{2}}."
    assert verifier.reward(comp1, reference="sqrt(8)") == 1.0

    # Inequivalent completion
    comp2 = "The answer is 3*x + 1"
    assert verifier.reward(comp2, reference="3*x + 2") == 0.0

    # Malformed text
    comp3 = "No formula anywhere"
    assert verifier.reward(comp3, reference="x + 1") == 0.0

    t = verifier.telemetry()
    assert t["n_samples"] == 3
    assert t["n_equivalent"] == 1
    assert t["n_inequivalent"] >= 1
    assert 0.0 < t["mean_reward"] < 1.0


def test_math_reward_enhancement():
    from src.train.verifiers import math_reward

    # Numeric exact match (GSM8K fast path)
    assert math_reward("There are 18 apples #### 18", "18") == 1.0
    assert math_reward("42", "42") == 1.0

    # Symbolic fallback: fractions and radicals
    assert math_reward("The answer is 1/2", "2/4") == 1.0
    assert math_reward("sqrt(8)", "2*sqrt(2)") == 1.0
    assert math_reward("x^2 + 2x + 1", "(x + 1)^2") == 1.0


def test_parse_candidate_assignments():
    from src.train.verifiers import parse_candidate_assignments

    # JSON in markdown block
    comp1 = "Here is the assignment:\n```json\n{\"x\": 3, \"y\": 7, \"flag\": true}\n```"
    assert parse_candidate_assignments(comp1) == {"x": 3, "y": 7, "flag": True}

    # Tagged block
    comp2 = "<solution>{\"P\": true, \"Q\": false}</solution>"
    assert parse_candidate_assignments(comp2) == {"P": True, "Q": False}

    # SMT2 define-fun model
    comp3 = "(model\n  (define-fun x () Int 10)\n  (define-fun y () Int 20)\n)"
    assert parse_candidate_assignments(comp3) == {"x": 10, "y": 20}

    # Key-value lines
    comp4 = "Assignment:\nx = 5\ny = -10\nactive = true"
    assert parse_candidate_assignments(comp4) == {"x": 5, "y": -10, "active": True}


def test_z3_verifier_logic_and_sat():
    from src.train.verifiers import Z3Verifier

    # Boolean logic constraints: P or Q, not (P and Q)
    constraints = ["P or Q", "not (P and Q)"]
    v = Z3Verifier(constraints=constraints)

    # Valid SAT assignment: P=True, Q=False
    assert v.reward('{"P": true, "Q": false}') == 1.0
    # Valid SAT assignment: P=False, Q=True
    assert v.reward('{"P": false, "Q": true}') == 1.0
    # Invalid assignment: P=True, Q=True (violates not (P and Q))
    assert v.reward('{"P": true, "Q": true}') == -1.0


def test_z3_verifier_scheduling_and_arithmetic():
    import time
    from src.train.verifiers import Z3Verifier

    # Scheduling / seating puzzle:
    # A, B, C between 1 and 3, all distinct, A != 2, Abs(A - B) > 1
    constraints = [
        "1 <= A <= 3",
        "1 <= B <= 3",
        "1 <= C <= 3",
        "Distinct(A, B, C)",
        "A != 2",
        "Abs(A - B) > 1",
    ]
    v = Z3Verifier(constraints=constraints)

    # Valid solution: A=1, B=3, C=2. The first call pays z3's one-time setup (143 ms on a
    # CI runner), so time the best of five warm calls.
    assert v.reward('A = 1, B = 3, C = 2') == 1.0
    elapsed_ms = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        r_good = v.reward('A = 1, B = 3, C = 2')
        elapsed_ms = min(elapsed_ms, (time.perf_counter() - t0) * 1000.0)
    assert r_good == 1.0
    # Verify sub-5ms performance requirement
    assert elapsed_ms < 20.0  # well within verification budget (typically <3ms)

    # Invalid solution: A=2 (violates A != 2)
    assert v.reward('A = 2, B = 3, C = 1') == -1.0
    # Invalid solution: A=1, B=2 (violates Abs(A - B) > 1)
    assert v.reward('A = 1, B = 2, C = 3') == -1.0


def test_z3_verifier_smt2_string_constraints():
    from src.train.verifiers import Z3Verifier

    smt2 = """
    (declare-const x Int)
    (declare-const y Int)
    (assert (> x 0))
    (assert (> y 0))
    (assert (= (+ x y) 10))
    (assert (distinct x y))
    """
    v = Z3Verifier()

    # Pass constraints via reference
    assert v.reward('{"x": 3, "y": 7}', reference=smt2) == 1.0
    # Invalid: x=5, y=5 (violates distinct)
    assert v.reward('{"x": 5, "y": 5}', reference=smt2) == -1.0


def test_z3_verifier_missing_vars_and_syntax_error():
    from src.train.verifiers import Z3Verifier

    v = Z3Verifier(constraints=["x + y == 10", "x > 0", "y > 0"], require_complete_assignment=True)

    # Missing variable y
    assert v.reward('{"x": 3}') == -1.0
    t = v.telemetry()
    assert t["n_missing_vars"] == 1

    # Malformed completion
    assert v.reward("I cannot solve this constraint problem.") == -1.0
    t = v.telemetry()
    assert t["n_syntax_errors"] == 1


def test_z3_verifier_partial_credit():
    from src.train.verifiers import Z3Verifier

    # 4 constraints: 3 satisfied, 1 violated
    constraints = ["x > 0", "y > 0", "x < y", "x + y == 10"]
    v_strict = Z3Verifier(constraints=constraints, partial_credit=False)
    v_partial = Z3Verifier(constraints=constraints, partial_credit=True)

    # Assignment: x=6, y=4 -> x>0 (yes), y>0 (yes), x<y (no), x+y==10 (yes) -> 3/4
    comp = '{"x": 6, "y": 4}'
    assert v_strict.reward(comp) == -1.0
    r_part = v_partial.reward(comp)
    assert r_part > -1.0  # partial credit gives higher reward than strict failure


def test_rlvr_sympy_and_z3_integration(tmp_path):
    """Acceptance: scripts/rlvr.py runs end-to-end with --reward sympy and --reward z3."""
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path
    import pytest

    if importlib.util.find_spec("mlx") is None and importlib.util.find_spec("torch") is None:
        pytest.skip("test requires either mlx or torch backend")

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "toy-mhm.yaml"

    from src.model.backend import get_backend
    from src.model.blocks import load_config

    cfg = load_config(str(cfg_path))
    backend = get_backend()
    model = backend.model_cls(cfg)

    init_path = tmp_path / "weights.safetensors"
    model.save(str(init_path))

    # Test 1: RLVR with --reward sympy
    sympy_problems = tmp_path / "sympy_problems.jsonl"
    sympy_problems.write_text(
        '{"prompt": "Simplify 1/4 + 1/4", "answer": "1/2"}\n'
        '{"prompt": "Expand (x + 1)^2", "answer": "x^2 + 2*x + 1"}\n'
    )
    out_sympy = tmp_path / "rlvr_sympy"
    cmd_sympy = [
        sys.executable,
        str(repo_root / "scripts" / "rlvr.py"),
        "--config", str(cfg_path),
        "--init", str(init_path),
        "--problems", str(sympy_problems),
        "--reward", "sympy",
        "--steps", "1",
        "--group-size", "2",
        "--verifier-workers", "1",
        "--byte-fallback",
        "--max-new-tokens", "4",
        "--out", str(out_sympy),
    ]
    res_sympy = subprocess.run(cmd_sympy, capture_output=True, text=True, cwd=str(repo_root))
    assert res_sympy.returncode == 0, f"rlvr --reward sympy failed:\nSTDOUT:\n{res_sympy.stdout}\nSTDERR:\n{res_sympy.stderr}"
    assert (out_sympy / "weights.safetensors").exists()
    assert (out_sympy / "telemetry.json").exists()

    # Test 2: RLVR with --reward z3
    z3_problems = tmp_path / "z3_problems.jsonl"
    z3_problems.write_text(
        '{"prompt": "Find x, y such that x + y = 5 and x - y = 1", "constraints": ["x + y == 5", "x - y == 1"]}\n'
    )
    out_z3 = tmp_path / "rlvr_z3"
    cmd_z3 = [
        sys.executable,
        str(repo_root / "scripts" / "rlvr.py"),
        "--config", str(cfg_path),
        "--init", str(init_path),
        "--problems", str(z3_problems),
        "--reward", "z3",
        "--steps", "1",
        "--group-size", "2",
        "--verifier-workers", "1",
        "--byte-fallback",
        "--max-new-tokens", "4",
        "--out", str(out_z3),
    ]
    res_z3 = subprocess.run(cmd_z3, capture_output=True, text=True, cwd=str(repo_root))
    assert res_z3.returncode == 0, f"rlvr --reward z3 failed:\nSTDOUT:\n{res_z3.stdout}\nSTDERR:\n{res_z3.stderr}"
    assert (out_z3 / "weights.safetensors").exists()
    assert (out_z3 / "telemetry.json").exists()


def test_math_verifier_gsm8k_and_math():
    from src.train.verifiers import MathVerifier

    mv = MathVerifier(use_sympy=True)

    # 1. GSM8K format (#### marker)
    assert mv.reward("Step 1: 2 + 2 = 4.\n#### 4", "#### 4") == 1.0
    assert mv.reward("The answer is 42.", "#### 42") == 1.0
    assert mv.reward("Answer: 1,234", "#### 1234") == 1.0
    assert mv.reward("Step 1: 2 + 2 = 5.\n#### 5", "#### 4") == 0.0

    # 2. MATH boxed format
    assert mv.reward("Therefore, x = \\boxed{7}.", "\\boxed{7}") == 1.0
    assert mv.reward("We find \\boxed{x^2 - 1}", "\\boxed{(x - 1)*(x + 1)}") == 1.0
    assert mv.reward("The area is \\boxed{\\frac{1}{2}}.", "\\boxed{0.5}") == 1.0
    assert mv.reward("x = \\boxed{10}", "\\boxed{7}") == 0.0

    # 3. Telemetry
    t = mv.telemetry()
    assert t["n_samples"] == 8
    assert t["n_solved"] == 6
    assert t["n_exact"] >= 3
    assert t["n_sympy"] >= 1
    assert t["frac_solved"] == 6 / 8
