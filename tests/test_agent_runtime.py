"""Unit tests for autonomous multi-turn coding agent execution loop and anti-spin circuit breakers (#349)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.agent.runtime import (
    AgentRuntime,
    AntiSpinCircuitBreaker,
    InMemoryTrajectoryLogger,
    JsonlTrajectoryLogger,
    WorkspaceToolExecutor,
    canonical_call_bytes,
    is_tool_failure,
    run_agent_loop,
)

# --------------------------------------------------------------------------- #
# Canonical Serialization & Failure Detection Tests
# --------------------------------------------------------------------------- #

def test_canonical_call_bytes_consistency():
    """Verify dictionary key ordering and whitespace invariance in canonical serialization."""
    call1 = canonical_call_bytes("view_file", {"path": "src/main.py", "start_line": 1, "end_line": 20})
    call2 = canonical_call_bytes("view_file", {"end_line": 20, "path": "src/main.py", "start_line": 1})
    call3 = canonical_call_bytes("view_file", '{"start_line": 1, "end_line": 20, "path": "src/main.py"}')
    assert call1 == call2
    assert call1 == call3

    diff_args = canonical_call_bytes("view_file", {"path": "src/other.py", "start_line": 1, "end_line": 20})
    assert call1 != diff_args

    diff_tool = canonical_call_bytes("edit_file", {"path": "src/main.py", "start_line": 1, "end_line": 20})
    assert call1 != diff_tool


def test_is_tool_failure_detection():
    """Verify classification of tool results into successes vs failures."""
    assert is_tool_failure(RuntimeError("disk error"))
    assert is_tool_failure({"error": "file not found"})
    assert is_tool_failure({"is_error": True, "output": ""})
    assert is_tool_failure({"exit_code": 1, "stdout": "", "stderr": "command failed"})
    assert is_tool_failure("Error: file not found")
    assert is_tool_failure("[ERROR] invalid syntax")
    assert is_tool_failure("Traceback (most recent call last):\n  File 'a.py', line 1")

    assert not is_tool_failure({"status": "ok", "message": "done"})
    assert not is_tool_failure({"exit_code": 0, "stdout": "hello\n"})
    assert not is_tool_failure("File content successfully displayed.")
    assert not is_tool_failure(["file1.py", "file2.py"])


# --------------------------------------------------------------------------- #
# Anti-Spin Circuit Breaker Unit Tests
# --------------------------------------------------------------------------- #

def test_circuit_breaker_identical_call_redirection_and_termination():
    """Verify circuit breaker triggers redirection at 5 identical calls and terminates at 8."""
    cb = AntiSpinCircuitBreaker(redirection_threshold=5, termination_threshold=8)

    # Turns 1 to 4: consecutive calls below threshold
    for i in range(1, 5):
        status = cb.record_call("view_file", {"path": "test.py"})
        assert status.consecutive_calls == i
        assert not status.should_redirect
        assert not status.should_terminate

    # Turn 5: 5th identical call triggers redirection reminder
    status5 = cb.record_call("view_file", {"path": "test.py"})
    assert status5.consecutive_calls == 5
    assert status5.should_redirect
    assert "5 consecutive identical tool calls" in (status5.redirection_message or "")
    assert not status5.should_terminate

    # Turns 6 and 7: continues redirecting but not yet terminated
    for i in (6, 7):
        status = cb.record_call("view_file", {"path": "test.py"})
        assert status.consecutive_calls == i
        assert status.should_redirect
        assert not status.should_terminate

    # Turn 8: 8th identical call triggers termination in record_result
    cb.record_call("view_file", {"path": "test.py"})
    status8 = cb.record_result("view_file", {"path": "test.py"}, is_error=False)
    assert status8.consecutive_calls == 8
    assert status8.should_terminate
    assert cb.is_terminated
    assert "anti_spin_circuit_breaker" in (status8.termination_reason or "")


def test_circuit_breaker_identical_failure_redirection_and_termination():
    """Verify circuit breaker triggers redirection at 5 identical failures and terminates at 8."""
    cb = AntiSpinCircuitBreaker(redirection_threshold=5, termination_threshold=8)
    args = {"path": "nonexistent.py"}

    # Turns 1 to 4: failures below threshold
    for i in range(1, 5):
        cb.record_call("view_file", args)
        res_status = cb.record_result("view_file", args, is_error=True)
        assert res_status.consecutive_failures == i
        assert not res_status.should_redirect
        assert not res_status.should_terminate

    # Turn 5: 5th identical failure triggers corrective redirection reminder
    cb.record_call("view_file", args)
    res_status5 = cb.record_result("view_file", args, is_error=True)
    assert res_status5.consecutive_failures == 5
    assert res_status5.should_redirect
    assert "failed 5 times consecutively" in (res_status5.redirection_message or "")
    assert not res_status5.should_terminate

    # Turns 6 and 7:
    for i in (6, 7):
        cb.record_call("view_file", args)
        res_status = cb.record_result("view_file", args, is_error=True)
        assert res_status.consecutive_failures == i
        assert res_status.should_redirect
        assert not res_status.should_terminate

    # Turn 8: 8th identical failure terminates execution
    cb.record_call("view_file", args)
    res_status8 = cb.record_result("view_file", args, is_error=True)
    assert res_status8.consecutive_failures == 8
    assert res_status8.should_terminate
    assert cb.is_terminated
    assert "anti_spin_circuit_breaker: 8 consecutive identical failures" in (res_status8.termination_reason or "")


def test_circuit_breaker_streak_resets_on_different_action():
    """Verify streak counters reset when tool name or arguments change."""
    cb = AntiSpinCircuitBreaker(redirection_threshold=5, termination_threshold=8)

    # 4 identical calls
    for _ in range(4):
        cb.record_call("view_file", {"path": "test.py"})
        cb.record_result("view_file", {"path": "test.py"}, is_error=True)

    assert cb.consecutive_identical_calls == 4
    assert cb.consecutive_identical_failures == 4

    # Call with different arguments resets failure count and sets calls count to 1
    call_status = cb.record_call("view_file", {"path": "other.py"})
    assert call_status.consecutive_calls == 1
    assert call_status.consecutive_failures == 0
    assert not call_status.should_redirect

    # Success resets consecutive failures
    res_status = cb.record_result("view_file", {"path": "other.py"}, is_error=False)
    assert res_status.consecutive_failures == 0


# --------------------------------------------------------------------------- #
# Workspace Tool Execution Unit Tests
# --------------------------------------------------------------------------- #

def test_workspace_tool_executor(tmp_path: Path):
    """Verify WorkspaceToolExecutor executes CODING_AGENT_TOOLS within workspace."""
    workspace = tmp_path / "repo"
    workspace.mkdir()

    executor = WorkspaceToolExecutor(workspace_dir=workspace)

    # 1. execute_bash
    bash_res = executor.execute_bash("echo 'hello from bash'")
    assert bash_res["exit_code"] == 0
    assert "hello from bash" in bash_res["stdout"]
    assert not bash_res["is_error"]

    # 2. edit_file & view_file
    test_file = workspace / "module.py"
    test_file.write_text("def original_func():\n    return 42\n", encoding="utf-8")

    view_res = executor.view_file("module.py", start_line=1, end_line=2)
    assert not view_res.get("is_error")
    assert "original_func" in view_res["content"]
    assert view_res["total_lines"] == 2

    # Edit file
    edit_res = executor.edit_file("module.py", old_str="42", new_str="100")
    assert not edit_res.get("is_error")
    assert edit_res["status"] == "ok"
    assert "return 100" in test_file.read_text(encoding="utf-8")

    # Ambiguous edit error
    test_file.write_text("a = 1\na = 1\n", encoding="utf-8")
    ambig_res = executor.edit_file("module.py", old_str="a = 1", new_str="a = 2")
    assert ambig_res.get("is_error")
    assert "occurs 2 times" in ambig_res["error"]

    # 3. find_files & grep_search
    sub_dir = workspace / "pkg"
    sub_dir.mkdir()
    (sub_dir / "target.py").write_text("def find_me():\n    pass\n", encoding="utf-8")

    find_res = executor.find_files("*.py")
    assert not find_res.get("is_error")
    assert find_res["count"] >= 2
    assert "pkg/target.py" in find_res["files"]

    grep_res = executor.grep_search("find_me")
    assert not grep_res.get("is_error")
    assert grep_res["total_matches"] >= 1
    assert any("find_me" in m["content"] for m in grep_res["matches"])

    # 4. Sandboxing security
    escape_res = executor.view_file("../../etc/passwd")
    assert escape_res.get("is_error")
    assert "Security error" in escape_res["error"]


# --------------------------------------------------------------------------- #
# Multi-Turn ReAct Loop Tests with Mock & FakeLM Backends
# --------------------------------------------------------------------------- #

def test_agent_runtime_immediate_completion():
    """Verify runtime completes cleanly when agent answers on turn 1 without tool calls."""
    def mock_lm(messages: list[dict[str, Any]]) -> str:
        return "<think>Task is trivial.</think>The solution is 42."

    runtime = AgentRuntime(lm=mock_lm, max_turns=5)
    result = runtime.run("What is the meaning of life?")

    assert result.success
    assert result.status == "completed"
    assert result.total_turns == 1
    assert result.final_answer == "The solution is 42."
    assert result.turns[0].thought == "Task is trivial."
    assert result.turns[0].tool_calls == []
    assert not result.circuit_breaker_triggered


def test_agent_runtime_multiturn_tool_execution(tmp_path: Path):
    """Verify multi-turn Thought -> Action -> Observation -> Final Answer loop."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data.txt").write_text("monica agent runtime data", encoding="utf-8")

    executor = WorkspaceToolExecutor(workspace_dir=workspace)

    call_count = 0

    def mock_lm(messages: list[dict[str, Any]]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                "I will view the file data.txt to inspect its contents.\n"
                "<tool_call>\n"
                '{"name": "view_file", "arguments": {"path": "data.txt"}}\n'
                "</tool_call>"
            )
        else:
            assert any("<tool_response>" in m.get("content", "") for m in messages)
            return "<think>Inspected file.</think>The file contains 'monica agent runtime data'."

    runtime = AgentRuntime(lm=mock_lm, tool_executor=executor, max_turns=5)
    result = runtime.run("Read data.txt and summarize it.")

    assert result.success
    assert result.status == "completed"
    assert result.total_turns == 2
    assert "monica agent runtime data" in (result.final_answer or "")

    turn1 = result.turns[0]
    assert len(turn1.tool_calls) == 1
    assert turn1.tool_calls[0].name == "view_file"
    assert len(turn1.tool_observations) == 1
    assert not turn1.tool_observations[0].is_error
    assert "monica agent runtime data" in turn1.tool_observations[0].output["content"]
    assert turn1.phase_transitions == ["thought", "action", "observation"]

    turn2 = result.turns[1]
    assert turn2.tool_calls == []
    assert turn2.phase_transitions == ["thought", "completion"]


def test_agent_runtime_anti_spin_redirection_and_termination(tmp_path: Path):
    """Verify anti-spin circuit breaker terminates early at 8 identical failures."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = WorkspaceToolExecutor(workspace_dir=workspace)

    redirection_seen = False

    def stubborn_spinning_lm(messages: list[dict[str, Any]]) -> str:
        nonlocal redirection_seen
        for m in messages:
            content = m.get("content", "")
            if "[CIRCUIT BREAKER REDIRECTION]" in content:
                redirection_seen = True

        return (
            "Trying view_file again.\n"
            "<tool_call>\n"
            '{"name": "view_file", "arguments": {"path": "missing_file.txt"}}\n'
            "</tool_call>"
        )

    runtime = AgentRuntime(
        lm=stubborn_spinning_lm,
        tool_executor=executor,
        max_turns=20,
        redirection_threshold=5,
        termination_threshold=8,
    )
    result = runtime.run("Inspect missing_file.txt")

    assert not result.success
    assert result.status == "circuit_breaker_tripped"
    assert result.circuit_breaker_triggered
    assert result.total_turns == 8  # Terminated early, DID NOT exhaust 20 turns
    assert redirection_seen  # Redirection reminder was observed by turn 6
    assert "8 consecutive identical failures" in (result.failure_reason or "")


def test_agent_runtime_anti_spin_identical_calls_termination():
    """Verify anti-spin circuit breaker terminates at 8 identical calls even on success."""
    def mock_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "ok", "value": 123}

    def infinite_loop_lm(messages: list[dict[str, Any]]) -> str:
        return (
            "Calling status tool.\n"
            "<tool_call>\n"
            '{"name": "get_status", "arguments": {"check": "all"}}\n'
            "</tool_call>"
        )

    runtime = AgentRuntime(
        lm=infinite_loop_lm,
        tool_executor=mock_executor,
        max_turns=25,
        redirection_threshold=5,
        termination_threshold=8,
    )
    result = runtime.run("Keep polling status")

    assert not result.success
    assert result.status == "circuit_breaker_tripped"
    assert result.circuit_breaker_triggered
    assert result.total_turns == 8
    assert "8 consecutive identical calls" in (result.failure_reason or "")


def test_agent_runtime_budget_exhaustion():
    """Verify runtime exits gracefully with budget_exhausted when max_turns is reached."""
    def wandering_lm(messages: list[dict[str, Any]]) -> str:
        turn_count = len(messages)
        return (
            f"Checking step {turn_count}\n"
            "<tool_call>\n"
            f'{{"name": "step_tool", "arguments": {{"step": {turn_count}}}}}\n'
            "</tool_call>"
        )

    def step_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"step": args.get("step"), "status": "ok"}

    runtime = AgentRuntime(
        lm=wandering_lm,
        tool_executor=step_executor,
        max_turns=4,
    )
    result = runtime.run("Run multiple steps")

    assert not result.success
    assert result.status == "budget_exhausted"
    assert result.total_turns == 4
    assert not result.circuit_breaker_triggered
    assert "budget exhausted" in (result.failure_reason or "")


# --------------------------------------------------------------------------- #
# Trajectory Logging & Telemetry Tests
# --------------------------------------------------------------------------- #

def test_trajectory_logging_in_memory_and_jsonl(tmp_path: Path):
    """Verify telemetry recording and trajectory output to memory and JSONL."""
    log_file = tmp_path / "trajectories.jsonl"
    mem_logger = InMemoryTrajectoryLogger()
    jsonl_logger = JsonlTrajectoryLogger(path=log_file)

    class MultiSinkLogger:
        def log_turn(self, turn: Any) -> None:
            mem_logger.log_turn(turn)
            jsonl_logger.log_turn(turn)

        def log_trajectory(self, trajectory: Any) -> None:
            mem_logger.log_trajectory(trajectory)
            jsonl_logger.log_trajectory(trajectory)

    call_num = 0

    def mock_lm(messages: list[dict[str, Any]]) -> str:
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            return '<tool_call>\n{"name": "test_tool", "arguments": {"x": 1}}\n</tool_call>'
        return "Done."

    runtime = AgentRuntime(
        lm=mock_lm,
        tool_executor=lambda name, args: {"res": 1},
        max_turns=3,
        trajectory_logger=MultiSinkLogger(),
    )
    result = runtime.run("Perform task")

    assert result.success
    assert result.total_turns == 2

    # Memory logger assertions
    assert len(mem_logger.turns) == 2
    assert len(mem_logger.trajectories) == 1

    # JSONL file assertions
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3  # 2 turn records + 1 trajectory record
    turn_rec = json.loads(lines[0])
    assert turn_rec["type"] == "turn"
    traj_rec = json.loads(lines[-1])
    assert traj_rec["type"] == "trajectory"
    assert traj_rec["data"]["status"] == "completed"

    # JSON serialization of AgentRunResult
    json_str = result.to_json()
    parsed = json.loads(json_str)
    assert parsed["task"] == "Perform task"
    assert parsed["status"] == "completed"
    assert parsed["total_turns"] == 2


# --------------------------------------------------------------------------- #
# LMAdapter Compatibility Test
# --------------------------------------------------------------------------- #

class FakeTokenizer:
    eos_token_id = 0


class ScriptedFakeLM:
    """Mock LMAdapter implementation verifying LMAdapter step/reset execution."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_idx = 0
        self.n_forward_tokens = 0
        self.n_forward_tokens_nocache = 0
        self.tokenizer = FakeTokenizer()
        self._tokens: list[int] = []

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(i) for i in token_ids if i != 0)

    def reset(self, context: str) -> np.ndarray:
        resp = self.responses[min(self.call_idx, len(self.responses) - 1)]
        self.call_idx += 1
        self._tokens = [ord(c) for c in resp] + [0]
        logits = np.zeros(256, dtype=np.float32)
        if self._tokens:
            logits[self._tokens[0]] = 100.0
        return logits

    def step(self, token_id: int) -> np.ndarray:
        if self._tokens:
            self._tokens.pop(0)
        logits = np.zeros(256, dtype=np.float32)
        if self._tokens:
            logits[self._tokens[0]] = 100.0
        return logits


def test_agent_runtime_with_lmadapter_interface():
    """Verify AgentRuntime drives LMAdapter backends with reset/step/decode."""
    lm = ScriptedFakeLM([
        '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "echo test"}}\n</tool_call>',
        "Final bash execution complete.",
    ])

    result = run_agent_loop(
        task="Test LMAdapter integration",
        lm=lm,
        tool_executor=lambda name, args: {"stdout": "test\n", "exit_code": 0},
        max_turns=3,
    )

    assert result.success
    assert result.total_turns == 2
    assert "Final bash execution complete." in (result.final_answer or "")
