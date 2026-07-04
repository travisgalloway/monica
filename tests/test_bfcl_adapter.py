"""BFCL-style eval harness tests (#102): parse / match / score / aggregate.

Offline via a fake `generate_fn` returning canned decoded strings (no network, no
backend — this module never imports mlx/torch, guarded by test_import_guard.py)."""

import json

from src.data.tool_sources import TOOL_CALL_CLOSE, TOOL_CALL_OPEN
from src.eval.bfcl_adapter import (
    BFCL_FIXTURE,
    evaluate_bfcl,
    match_call,
    parse_tool_calls,
    score_example,
)


def _block(call: dict) -> str:
    return f"{TOOL_CALL_OPEN}\n{json.dumps(call)}\n{TOOL_CALL_CLOSE}"


# --------------------------------------------------------------------------- #
# parse_tool_calls: stacked/parallel blocks + malformed JSON
# --------------------------------------------------------------------------- #

def test_parse_tool_calls_single_block():
    text = _block({"name": "get_weather", "arguments": {"city": "Paris"}})
    calls = parse_tool_calls(text)
    assert calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


def test_parse_tool_calls_stacked_parallel_blocks():
    c1 = {"name": "get_weather", "arguments": {"city": "Tokyo"}}
    c2 = {"name": "set_timer", "arguments": {"seconds": 300}}
    text = _block(c1) + "\n" + _block(c2)
    calls = parse_tool_calls(text)
    assert calls == [c1, c2]


def test_parse_tool_calls_no_blocks_returns_empty():
    assert parse_tool_calls("Sure, here's the answer with no calls.") == []


def test_parse_tool_calls_malformed_json_is_skipped():
    good = {"name": "get_weather", "arguments": {"city": "Paris"}}
    text = f"{TOOL_CALL_OPEN}\nnot json at all\n{TOOL_CALL_CLOSE}\n" + _block(good)
    calls = parse_tool_calls(text)
    assert calls == [good]


def test_parse_tool_calls_missing_name_key_is_skipped():
    text = f"{TOOL_CALL_OPEN}\n{json.dumps({'arguments': {'city': 'Paris'}})}\n{TOOL_CALL_CLOSE}"
    assert parse_tool_calls(text) == []


def test_parse_tool_calls_unclosed_block_is_ignored():
    text = f"{TOOL_CALL_OPEN}\n{json.dumps({'name': 'x', 'arguments': {}})}"  # no closing tag
    assert parse_tool_calls(text) == []


# --------------------------------------------------------------------------- #
# match_call: scalar and list-valued (BFCL "possible answers") gold values
# --------------------------------------------------------------------------- #

def test_match_call_scalar_exact_match():
    pred = {"name": "get_weather", "arguments": {"city": "Paris"}}
    gold = {"name": "get_weather", "arguments": {"city": "Paris"}}
    assert match_call(pred, gold) is True


def test_match_call_scalar_mismatch():
    pred = {"name": "get_weather", "arguments": {"city": "London"}}
    gold = {"name": "get_weather", "arguments": {"city": "Paris"}}
    assert match_call(pred, gold) is False


def test_match_call_name_mismatch():
    pred = {"name": "get_time", "arguments": {"city": "Paris"}}
    gold = {"name": "get_weather", "arguments": {"city": "Paris"}}
    assert match_call(pred, gold) is False


def test_match_call_list_valued_gold_accepts_any_member():
    gold = {"name": "get_weather", "arguments": {"city": ["Boston", "Boston, MA"]}}
    assert match_call({"name": "get_weather", "arguments": {"city": "Boston"}}, gold) is True
    assert match_call({"name": "get_weather", "arguments": {"city": "Boston, MA"}}, gold) is True
    assert match_call({"name": "get_weather", "arguments": {"city": "NYC"}}, gold) is False


def test_match_call_extra_pred_args_ignored():
    pred = {"name": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}
    gold = {"name": "get_weather", "arguments": {"city": "Paris"}}
    assert match_call(pred, gold) is True


def test_match_call_missing_pred_arg_fails():
    pred = {"name": "get_weather", "arguments": {}}
    gold = {"name": "get_weather", "arguments": {"city": "Paris"}}
    assert match_call(pred, gold) is False


# --------------------------------------------------------------------------- #
# score_example: simple / parallel / abstention, correct + incorrect
# --------------------------------------------------------------------------- #

def test_score_example_simple_correct():
    gold_ex = {"category": "simple", "tools": [],
               "gold": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}
    pred_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    result = score_example(pred_calls, gold_ex)
    assert result["correct"] is True
    assert result["category"] == "simple"


def test_score_example_simple_wrong_call():
    gold_ex = {"category": "simple", "tools": [],
               "gold": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}
    pred_calls = [{"name": "get_weather", "arguments": {"city": "London"}}]
    assert score_example(pred_calls, gold_ex)["correct"] is False


def test_score_example_simple_extra_call_is_wrong():
    gold_ex = {"category": "simple", "tools": [],
               "gold": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}
    pred_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}},
                  {"name": "set_timer", "arguments": {"seconds": 10}}]
    assert score_example(pred_calls, gold_ex)["correct"] is False


def test_score_example_parallel_order_insensitive():
    gold_ex = {"category": "parallel", "tools": [], "gold": [
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "set_timer", "arguments": {"seconds": 300}},
    ]}
    # predicted in the opposite order -> still correct (order-insensitive)
    pred_calls = [
        {"name": "set_timer", "arguments": {"seconds": 300}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
    ]
    assert score_example(pred_calls, gold_ex)["correct"] is True


def test_score_example_parallel_count_sensitive_duplicate():
    gold_ex = {"category": "parallel", "tools": [],
               "gold": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]}
    # duplicate predicted call when gold only has one -> wrong (count-sensitive)
    pred_calls = [{"name": "get_weather", "arguments": {"city": "Tokyo"}},
                  {"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    assert score_example(pred_calls, gold_ex)["correct"] is False


def test_score_example_abstention_correct_when_no_call():
    gold_ex = {"category": "abstention", "tools": [], "gold": []}
    assert score_example([], gold_ex)["correct"] is True


def test_score_example_abstention_wrong_when_spurious_call():
    gold_ex = {"category": "abstention", "tools": [], "gold": []}
    pred_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    assert score_example(pred_calls, gold_ex)["correct"] is False


def test_score_example_relevance_alias_behaves_like_abstention():
    gold_ex = {"category": "relevance", "tools": [], "gold": []}
    assert score_example([], gold_ex)["correct"] is True
    assert score_example([{"name": "x", "arguments": {}}], gold_ex)["correct"] is False


# --------------------------------------------------------------------------- #
# schema_valid reporting
# --------------------------------------------------------------------------- #

def test_score_example_schema_valid_true_for_conforming_call():
    tools = [{"name": "get_weather", "parameters": {"type": "object",
                                                     "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}]
    gold_ex = {"category": "simple", "tools": tools,
               "gold": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}
    pred_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    result = score_example(pred_calls, gold_ex)
    assert result["schema_valid"] is True
    assert result["correct"] is True


def test_score_example_schema_valid_false_for_missing_required_arg():
    tools = [{"name": "get_weather", "parameters": {"type": "object",
                                                     "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}]
    gold_ex = {"category": "simple", "tools": tools,
               "gold": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}
    # A call that happens to match gold's own city loosely but omits "city" outright
    # can't score correct anyway; use a schema-invalid-but-name-matching call instead.
    pred_calls = [{"name": "get_weather", "arguments": {}}]
    result = score_example(pred_calls, gold_ex)
    assert result["schema_valid"] is False


def test_score_example_schema_valid_vacuously_true_for_no_calls():
    gold_ex = {"category": "abstention", "tools": [{"name": "x", "parameters": {}}], "gold": []}
    assert score_example([], gold_ex)["schema_valid"] is True


# --------------------------------------------------------------------------- #
# evaluate_bfcl: end-to-end aggregation with a fake generate_fn + the inline fixture
# --------------------------------------------------------------------------- #

def _perfect_generate_fn(prompt_or_messages) -> str:
    """Fake generator that always emits the gold answer for whichever fixture example
    is being asked about (keyed by the user turn's content, robust to fixture order)."""
    text = json.dumps(prompt_or_messages)
    if "Boston" in text:
        return _block({"name": "get_weather", "arguments": {"city": "Boston"}})
    if "5 minute timer" in text:
        return (_block({"name": "get_weather", "arguments": {"city": "Tokyo"}}) + "\n"
                + _block({"name": "set_timer", "arguments": {"seconds": 300}}))
    if "Translate" in text:
        return "I don't have a translation tool, but 'hello' in French is 'bonjour'."
    return "I don't know."


def test_evaluate_bfcl_perfect_generation_scores_100_percent():
    summary = evaluate_bfcl(BFCL_FIXTURE, _perfect_generate_fn)
    assert summary["n_examples"] == len(BFCL_FIXTURE)
    assert summary["accuracy"] == 1.0
    assert summary["schema_valid_rate"] == 1.0
    assert set(summary["per_category_accuracy"]) == {"simple", "parallel", "abstention"}
    for acc in summary["per_category_accuracy"].values():
        assert acc == 1.0


def _always_wrong_generate_fn(prompt_or_messages) -> str:
    """Fake generator that always calls the wrong tool with the wrong args — including
    on the abstention example, where any call at all is wrong."""
    return _block({"name": "nonexistent_tool", "arguments": {"foo": "bar"}})


def test_evaluate_bfcl_all_wrong_generation_scores_0_percent():
    summary = evaluate_bfcl(BFCL_FIXTURE, _always_wrong_generate_fn)
    assert summary["accuracy"] == 0.0
    for acc in summary["per_category_accuracy"].values():
        assert acc == 0.0
    # nonexistent_tool never appears in any fixture example's declared tools.
    assert summary["schema_valid_rate"] == 0.0


def test_evaluate_bfcl_uses_prompt_when_messages_absent():
    examples = [{"category": "abstention", "tools": [], "gold": [], "prompt": "hi"}]
    seen = []

    def gen(prompt_or_messages):
        seen.append(prompt_or_messages)
        return "no calls here"

    summary = evaluate_bfcl(examples, gen)
    assert seen == ["hi"]
    assert summary["accuracy"] == 1.0


def test_evaluate_bfcl_empty_examples_reports_nan_not_crash():
    summary = evaluate_bfcl([], lambda p: "")
    assert summary["n_examples"] == 0
    assert summary["accuracy"] != summary["accuracy"]  # NaN != NaN
    assert summary["per_category_accuracy"] == {}
