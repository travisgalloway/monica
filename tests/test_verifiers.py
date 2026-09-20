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
