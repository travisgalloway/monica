"""Reasoning-trace sources + <think>/<answer> formatting (#96). Pure stdlib (no backend/network)."""

from src.data.reasoning_traces import (ANSWER_CLOSE, ANSWER_OPEN, THINK_CLOSE, THINK_OPEN,
                                       format_trace, handauthored_trace_records,
                                       iter_reasoning_traces, mot_row_to_messages,
                                       trace_to_messages)


def test_format_trace_wraps_think_then_answer():
    out = format_trace("step one\nstep two", "42")
    assert out == (f"{THINK_OPEN}\nstep one\nstep two\n{THINK_CLOSE}\n"
                   f"{ANSWER_OPEN}\n42\n{ANSWER_CLOSE}")
    # think precedes answer; both tag pairs present.
    assert out.index(THINK_OPEN) < out.index(ANSWER_OPEN)


def test_trace_to_messages_assistant_carries_tags():
    rec = trace_to_messages("What is 2+2?", "add them", "4", system="be terse")
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    content = rec["messages"][-1]["content"]
    assert THINK_OPEN in content and ANSWER_OPEN in content and "4" in content
    assert rec["source"] == "handauthored" and rec["license"] == "cc0"


def test_mot_row_messages_passthrough():
    row = {"messages": [{"role": "user", "content": "Q"},
                        {"role": "assistant", "content": f"{THINK_OPEN}\nthink\n{THINK_CLOSE}\nA"}]}
    rec = mot_row_to_messages(row)
    assert rec is not None and rec["source"] == "mot"
    assert rec["messages"][-1]["role"] == "assistant"


def test_mot_row_problem_solution_fallback():
    rec = mot_row_to_messages({"problem": "2+2?", "reasoning": "add", "solution": "4"})
    assert rec is not None
    content = rec["messages"][-1]["content"]
    assert THINK_OPEN in content and "4" in content


def test_mot_row_skips_incomplete():
    assert mot_row_to_messages({"problem": "only a question"}) is None
    assert mot_row_to_messages({"messages": [{"role": "user", "content": "no answer"}]}) is None


def test_iter_reasoning_traces_handauthored():
    rows = list(iter_reasoning_traces(["handauthored"]))
    assert rows == list(handauthored_trace_records())
    assert len(rows) >= 4 and all(r["messages"][-1]["role"] == "assistant" for r in rows)
