"""Unit tests for staged context compaction (T4) with soft elision and hard summarization (#350)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.agent.compaction import (
    ContextCompactor,
    elide_observation_content,
    estimate_tokens,
    partition_conversation,
)
from src.agent.runtime import (
    AgentRuntime,
)
from src.data.tool_sources import (
    CODING_AGENT_TOOLS,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
)

# --------------------------------------------------------------------------- #
# Helper Fixtures & Builders
# --------------------------------------------------------------------------- #

def _build_react_turn(turn_idx: int, thought: str, action: str, args: dict, obs: str) -> tuple[dict, dict]:
    """Helper to generate an assistant tool call message and user observation message."""
    call_json = json.dumps({"name": action, "arguments": args})
    assistant_msg = {
        "role": "assistant",
        "content": f"<think>{thought}</think>\n{TOOL_CALL_OPEN}\n{call_json}\n{TOOL_CALL_CLOSE}",
    }
    user_msg = {
        "role": "user",
        "content": f"{TOOL_RESPONSE_OPEN}\n{obs}\n{TOOL_RESPONSE_CLOSE}",
    }
    return assistant_msg, user_msg


# --------------------------------------------------------------------------- #
# Unit Tests: Token Estimation & Conversation Partitioning
# --------------------------------------------------------------------------- #

def test_estimate_tokens_string_and_messages():
    """Verify estimate_tokens returns reasonable approximations and respects custom counter."""
    assert estimate_tokens("") == 0
    text = "hello world from monica agent"
    tokens = estimate_tokens(text)
    assert tokens > 0

    msgs = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Fix bug in tests."},
    ]
    t_msgs = estimate_tokens(msgs)
    assert t_msgs > len(msgs) * 4

    custom = lambda x: 42
    assert estimate_tokens(msgs, token_counter=custom) == 42


def test_partition_conversation_boundaries():
    """Verify partition_conversation accurately segments preamble, middle turns, and recent window."""
    sys_msg = {"role": "system", "content": "System instruction"}
    task_msg = {"role": "user", "content": "Task description"}

    t1_a, t1_u = _build_react_turn(1, "Think 1", "view_file", {"path": "a.py"}, "output 1")
    t2_a, t2_u = _build_react_turn(2, "Think 2", "view_file", {"path": "b.py"}, "output 2")
    t3_a, t3_u = _build_react_turn(3, "Think 3", "view_file", {"path": "c.py"}, "output 3")
    t4_a, t4_u = _build_react_turn(4, "Think 4", "view_file", {"path": "d.py"}, "output 4")

    messages = [sys_msg, task_msg, t1_a, t1_u, t2_a, t2_u, t3_a, t3_u, t4_a, t4_u]

    preamble, middle_turns, recent_turns = partition_conversation(messages, protect_recent_turns=2)

    assert len(preamble) == 2
    assert preamble[0]["content"] == "System instruction"
    assert preamble[1]["content"] == "Task description"

    assert len(middle_turns) == 2
    assert middle_turns[0] == [t1_a, t1_u]
    assert middle_turns[1] == [t2_a, t2_u]

    assert len(recent_turns) == 2
    assert recent_turns[0] == [t3_a, t3_u]
    assert recent_turns[1] == [t4_a, t4_u]


def test_partition_conversation_fewer_turns_protects_all():
    """Verify that when turns <= protect_recent_turns, all turns are recent and middle is empty."""
    sys_msg = {"role": "system", "content": "System"}
    task_msg = {"role": "user", "content": "Task"}
    t1_a, t1_u = _build_react_turn(1, "Think 1", "view_file", {"path": "a.py"}, "output 1")
    t2_a, t2_u = _build_react_turn(2, "Think 2", "view_file", {"path": "b.py"}, "output 2")

    messages = [sys_msg, task_msg, t1_a, t1_u, t2_a, t2_u]
    preamble, middle_turns, recent_turns = partition_conversation(messages, protect_recent_turns=2)

    assert len(preamble) == 2
    assert len(middle_turns) == 0
    assert len(recent_turns) == 2


def test_partition_conversation_with_prior_summary_message():
    """Verify partitioning handles a pre-existing [Context Summary: ...] middle message."""
    sys_msg = {"role": "system", "content": "System"}
    task_msg = {"role": "user", "content": "Task"}
    summary_msg = {"role": "user", "content": "[Context Summary: Prior steps inspected a.py]"}
    t2_a, t2_u = _build_react_turn(2, "Think 2", "edit_file", {"path": "a.py"}, "edited")
    t3_a, t3_u = _build_react_turn(3, "Think 3", "view_file", {"path": "b.py"}, "content b")
    t4_a, t4_u = _build_react_turn(4, "Think 4", "execute_bash", {"command": "pytest"}, "passed")

    messages = [sys_msg, task_msg, summary_msg, t2_a, t2_u, t3_a, t3_u, t4_a, t4_u]
    preamble, middle_turns, recent_turns = partition_conversation(messages, protect_recent_turns=2)

    assert len(preamble) == 2
    assert len(middle_turns) == 2
    assert middle_turns[0] == [summary_msg]
    assert middle_turns[1] == [t2_a, t2_u]
    assert len(recent_turns) == 2
    assert recent_turns[0] == [t3_a, t3_u]
    assert recent_turns[1] == [t4_a, t4_u]


# --------------------------------------------------------------------------- #
# Unit Tests: Observation Elision
# --------------------------------------------------------------------------- #

def test_elide_observation_content_tool_response_tags():
    """Verify bulky outputs inside <tool_response> are replaced with [Output elided: N chars]."""
    bulky_body = "A" * 500
    content = f"{TOOL_RESPONSE_OPEN}\n{bulky_body}\n{TOOL_RESPONSE_CLOSE}"

    elided, n_blocks, n_chars = elide_observation_content(content, elision_char_threshold=100)
    assert n_blocks == 1
    assert n_chars == 500
    assert "[Output elided: 500 chars]" in elided
    assert TOOL_RESPONSE_OPEN in elided
    assert TOOL_RESPONSE_CLOSE in elided
    assert bulky_body not in elided


def test_elide_observation_content_short_not_elided():
    """Verify observations below threshold are left unchanged."""
    short_body = "File successfully updated."
    content = f"{TOOL_RESPONSE_OPEN}\n{short_body}\n{TOOL_RESPONSE_CLOSE}"

    elided, n_blocks, n_chars = elide_observation_content(content, elision_char_threshold=100)
    assert n_blocks == 0
    assert n_chars == 0
    assert elided == content


def test_elide_observation_content_no_double_elision():
    """Verify that already-elided content is not elided again."""
    content = f"{TOOL_RESPONSE_OPEN}\n[Output elided: 1000 chars]\n{TOOL_RESPONSE_CLOSE}"
    elided, n_blocks, n_chars = elide_observation_content(content, elision_char_threshold=50)
    assert n_blocks == 0
    assert n_chars == 0
    assert elided == content


# --------------------------------------------------------------------------- #
# Unit Tests: Staged Two-Tier Compactor Pipeline (T4)
# --------------------------------------------------------------------------- #

def test_soft_threshold_elision_without_llm_calls():
    """Verify soft threshold elision strips large observation bodies without triggering LLM calls."""
    sys_msg = {"role": "system", "content": "You are Monica."}
    task_msg = {"role": "user", "content": "Compile project."}

    huge_compiler_log = "Error on line 12\nWarning: unused import\n" * 100
    t1_a, t1_u = _build_react_turn(1, "Running make", "execute_bash", {"command": "make"}, huge_compiler_log)
    t2_a, t2_u = _build_react_turn(2, "Reading fix", "view_file", {"path": "fix.py"}, "short output")
    t3_a, t3_u = _build_react_turn(3, "Testing", "execute_bash", {"command": "test"}, "passed")

    messages = [sys_msg, task_msg, t1_a, t1_u, t2_a, t2_u, t3_a, t3_u]

    init_tok = estimate_tokens(messages)
    compactor = ContextCompactor(
        max_context_tokens=init_tok + 200,
        soft_threshold=0.50,
        hard_threshold=0.85,
        protect_recent_turns=2,
        elision_char_threshold=200,
    )

    mock_summarizer = MagicMock(return_value="Mock summary narrative")
    compactor.summarizer = mock_summarizer

    compacted_messages, report = compactor.compact_with_report(messages)

    assert report.compacted is True
    assert report.elision_applied is True
    assert report.summarization_applied is False
    assert report.elided_observations >= 1
    assert report.final_tokens < report.initial_tokens
    assert mock_summarizer.call_count == 0

    assert "[Output elided:" in compacted_messages[3]["content"]
    assert "chars]" in compacted_messages[3]["content"]
    assert huge_compiler_log not in compacted_messages[3]["content"]


def test_protected_window_invariant_strictly_preserved_verbatim():
    """Verify system preamble and recent window (>= 2 turns) are preserved verbatim across compaction."""
    sys_content = "SYSTEM PREAMBLE INSTRUCTIONS: DO NOT CHANGE ANY PART OF THIS"
    task_content = "TASK PROMPT: REPRODUCE BUG #350 VERBATIM"
    sys_msg = {"role": "system", "content": sys_content}
    task_msg = {"role": "user", "content": task_content}

    t1_a, t1_u = _build_react_turn(1, "Explore", "execute_bash", {"command": "ls"}, "BULKY " * 500)

    recent_bulky_obs = "RECENT BULKY OBSERVATION " * 300
    t2_a, t2_u = _build_react_turn(2, "View recent", "view_file", {"path": "recent.py"}, recent_bulky_obs)
    t3_a, t3_u = _build_react_turn(3, "Finalize", "view_file", {"path": "done.py"}, "done output")

    messages = [sys_msg, task_msg, t1_a, t1_u, t2_a, t2_u, t3_a, t3_u]

    compactor = ContextCompactor(
        max_context_tokens=4000,
        soft_threshold=0.30,
        hard_threshold=0.85,
        protect_recent_turns=2,
        elision_char_threshold=100,
    )

    compacted, report = compactor.compact_with_report(messages)

    assert report.elision_applied is True
    assert compacted[0] == sys_msg
    assert compacted[1] == task_msg
    assert compacted[0]["content"] == sys_content
    assert compacted[1]["content"] == task_content

    assert "[Output elided:" in compacted[3]["content"]

    assert compacted[4] == t2_a
    assert compacted[5] == t2_u
    assert compacted[5]["content"] == t2_u["content"]
    assert recent_bulky_obs in compacted[5]["content"]
    assert "[Output elided:" not in compacted[5]["content"]

    assert compacted[6] == t3_a
    assert compacted[7] == t3_u


def test_hard_threshold_summarization_triggers_llm():
    """Verify hard threshold summarization invokes LLM when context still exceeds threshold after elision."""
    sys_msg = {"role": "system", "content": "You are Monica."}
    task_msg = {"role": "user", "content": "Solve task."}

    t1_a, t1_u = _build_react_turn(1, "Detailed thought 1 " * 80, "execute_bash", {"command": "build"}, "output 1 " * 50)
    t2_a, t2_u = _build_react_turn(2, "Detailed thought 2 " * 80, "view_file", {"path": "b.py"}, "output 2 " * 50)
    t3_a, t3_u = _build_react_turn(3, "Recent thought 3", "view_file", {"path": "c.py"}, "short 3")
    t4_a, t4_u = _build_react_turn(4, "Recent thought 4", "view_file", {"path": "d.py"}, "short 4")

    messages = [sys_msg, task_msg, t1_a, t1_u, t2_a, t2_u, t3_a, t3_u, t4_a, t4_u]

    mock_summary_narrative = "Narrative: Built codebase and inspected b.py."
    mock_summarizer = MagicMock(return_value=mock_summary_narrative)

    compactor = ContextCompactor(
        max_context_tokens=600,
        soft_threshold=0.20,
        hard_threshold=0.40,
        protect_recent_turns=2,
        elision_char_threshold=50,
        summarizer=mock_summarizer,
    )

    compacted, report = compactor.compact_with_report(messages)

    assert report.compacted is True
    assert report.summarization_applied is True
    assert mock_summarizer.call_count >= 1

    summary_messages = [m for m in compacted if "[Context Summary:" in m.get("content", "")]
    assert len(summary_messages) == 1
    assert mock_summary_narrative in summary_messages[0]["content"]

    assert compacted[0] == sys_msg
    assert compacted[1] == task_msg
    assert compacted[-2] == t3_a or compacted[-2] == t4_a
    assert compacted[-1] == t4_u


def test_sub_threshold_is_noop():
    """Verify that conversation history below soft threshold is returned untouched."""
    sys_msg = {"role": "system", "content": "You are Monica."}
    task_msg = {"role": "user", "content": "Task"}
    t1_a, t1_u = _build_react_turn(1, "Think", "view_file", {"path": "a.py"}, "short")
    t2_a, t2_u = _build_react_turn(2, "Think", "view_file", {"path": "b.py"}, "short")

    messages = [sys_msg, task_msg, t1_a, t1_u, t2_a, t2_u]

    compactor = ContextCompactor(
        max_context_tokens=10000,
        soft_threshold=0.60,
        hard_threshold=0.85,
    )

    compacted, report = compactor.compact_with_report(messages)
    assert report.compacted is False
    assert report.elision_applied is False
    assert report.summarization_applied is False
    assert compacted == messages


def test_fallback_deterministic_narrative_when_no_summarizer():
    """Verify deterministic narrative summary operates cleanly without external LLM."""
    sys_msg = {"role": "system", "content": "You are Monica."}
    task_msg = {"role": "user", "content": "Task"}
    t1_a, t1_u = _build_react_turn(1, "Think 1", "view_file", {"path": "main.py"}, "code")
    t2_a, t2_u = _build_react_turn(2, "Think 2", "edit_file", {"path": "main.py"}, "edited")
    t3_a, t3_u = _build_react_turn(3, "Think 3", "execute_bash", {"command": "pytest"}, "ok")
    t4_a, t4_u = _build_react_turn(4, "Think 4", "view_file", {"path": "tests.py"}, "tests")

    messages = [sys_msg, task_msg, t1_a, t1_u, t2_a, t2_u, t3_a, t3_u, t4_a, t4_u]

    compactor = ContextCompactor(
        max_context_tokens=300,
        soft_threshold=0.10,
        hard_threshold=0.20,
        protect_recent_turns=2,
        summarizer=None,
    )

    compacted, report = compactor.compact_with_report(messages)
    assert report.summarization_applied is True
    summary_messages = [m for m in compacted if "[Context Summary:" in m.get("content", "")]
    assert len(summary_messages) == 1
    assert "inspected main.py" in summary_messages[0]["content"] or "edited main.py" in summary_messages[0]["content"]


# --------------------------------------------------------------------------- #
# Unit Tests: AgentRuntime Integration
# --------------------------------------------------------------------------- #

def test_agent_runtime_with_compactor_integration():
    """Verify AgentRuntime applies compactor during multi-turn loop and logs telemetry event."""
    compactor = ContextCompactor(
        max_context_tokens=500,
        soft_threshold=0.30,
        hard_threshold=0.85,
        protect_recent_turns=2,
        elision_char_threshold=50,
    )

    turn_counter = 0
    def mock_lm(messages):
        nonlocal turn_counter
        turn_counter += 1
        if turn_counter == 1:
            call = json.dumps({"name": "view_file", "arguments": {"path": "heavy.log"}})
            return f"<think>Examine heavy log</think>\n{TOOL_CALL_OPEN}\n{call}\n{TOOL_CALL_CLOSE}"
        if turn_counter == 2:
            call = json.dumps({"name": "view_file", "arguments": {"path": "step2.txt"}})
            return f"<think>Examine step 2</think>\n{TOOL_CALL_OPEN}\n{call}\n{TOOL_CALL_CLOSE}"
        if turn_counter == 3:
            call = json.dumps({"name": "view_file", "arguments": {"path": "step3.txt"}})
            return f"<think>Examine step 3</think>\n{TOOL_CALL_OPEN}\n{call}\n{TOOL_CALL_CLOSE}"
        return "Final Answer: Done"

    def mock_executor(name, args):
        if args.get("path") == "heavy.log":
            return "LOG LINE OUTPUT DATA " * 50
        return "Short output"

    runtime = AgentRuntime(
        lm=mock_lm,
        tool_executor=mock_executor,
        compactor=compactor,
        max_turns=5,
    )

    result = runtime.run("Debug heavy log")
    assert result.success is True
    assert result.total_turns >= 3

    compaction_events = [e for e in result.events if e.get("event") == "context_compacted"]
    assert len(compaction_events) >= 1
    assert compaction_events[0]["elision_applied"] is True


def test_no_lossless_recall_machinery_invariant():
    """Verify recall_event is explicitly absent from tools and compactor (arXiv:2609.20804)."""
    tool_names = [t["name"] for t in CODING_AGENT_TOOLS]
    assert "recall_event" not in tool_names
    assert not hasattr(ContextCompactor, "recall_event")
    assert not hasattr(ContextCompactor, "vector_cache")
    assert not hasattr(ContextCompactor, "disk_cache")
