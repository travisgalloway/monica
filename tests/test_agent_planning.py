"""Unit and integration tests for capability-adaptive planning scaffolding and out-of-history injection (#352)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agent.planning import (
    PlanItem,
    PlanManager,
    PlanningPolicy,
    parse_plan_markdown,
    resolve_planning_policy,
)
from src.agent.runtime import (
    AgentRuntime,
    WorkspaceToolExecutor,
    run_agent_loop,
)

# --------------------------------------------------------------------------- #
# 1. PlanItem and Markdown Parsing Tests
# --------------------------------------------------------------------------- #

def test_plan_item_rendering_and_serialization():
    """Verify PlanItem renders markdown checklist items and serializes to dict."""
    item_unchecked = PlanItem(index=1, description="Inspect codebase", completed=False)
    assert item_unchecked.render() == "- [ ] 1. Inspect codebase"
    assert item_unchecked.to_dict() == {"index": 1, "description": "Inspect codebase", "completed": False}

    item_checked = PlanItem(index=2, description="Run unit tests", completed=True)
    assert item_checked.render() == "- [x] 2. Run unit tests"
    assert item_checked.to_dict() == {"index": 2, "description": "Run unit tests", "completed": True}

    restored = PlanItem.from_dict(item_checked.to_dict())
    assert restored.index == 2
    assert restored.description == "Run unit tests"
    assert restored.completed is True


def test_parse_plan_markdown_variations():
    """Verify markdown plan parser handles standard checkboxes, numbers, and bullets."""
    md_text = """
    # Active Plan
    - [ ] 1. Locate root cause in src/agent/
    - [x] 2. Implement PlanManager
    * [ ] 3. Run pytest verification
    - [X] 4. Update documentation
    """
    items = parse_plan_markdown(md_text)
    assert len(items) == 4
    assert items[0].index == 1
    assert items[0].description == "Locate root cause in src/agent/"
    assert items[0].completed is False

    assert items[1].index == 2
    assert items[1].description == "Implement PlanManager"
    assert items[1].completed is True

    assert items[2].index == 3
    assert items[2].description == "Run pytest verification"
    assert items[2].completed is False

    assert items[3].index == 4
    assert items[3].description == "Update documentation"
    assert items[3].completed is True


def test_parse_plan_markdown_plain_lists():
    """Verify markdown plan parser converts plain numbered or bulleted steps into uncompleted items."""
    plain_text = """
    1. Read documentation
    2. Edit source files
    3. Verify test suite
    """
    items = parse_plan_markdown(plain_text)
    assert len(items) == 3
    assert [it.description for it in items] == [
        "Read documentation",
        "Edit source files",
        "Verify test suite",
    ]
    assert all(not it.completed for it in items)


# --------------------------------------------------------------------------- #
# 2. PlanManager State and Mutation Tests
# --------------------------------------------------------------------------- #

def test_plan_manager_lifecycle():
    """Verify PlanManager tracks completion states and step updates."""
    pm = PlanManager(
        policy=PlanningPolicy.FRONTIER,
        initial_plan=["Find bug", "Fix bug", "Test fix"],
    )
    assert pm.has_plan
    assert not pm.is_complete
    assert pm.total_count == 3
    assert pm.completed_count == 0
    assert pm.remaining_count == 3

    # Update step 1
    updated = pm.update_step(1, completed=True)
    assert updated.completed is True
    assert pm.completed_count == 1
    assert pm.remaining_count == 2
    assert not pm.is_complete

    # Update steps 2 and 3
    pm.update_step(2, completed=True)
    pm.update_step(3, completed=True)
    assert pm.completed_count == 3
    assert pm.remaining_count == 0
    assert pm.is_complete

    # Invalid step update
    with pytest.raises(ValueError, match="Step 99 not found"):
        pm.update_step(99, completed=True)

    # Reset
    pm.reset()
    assert pm.completed_count == 0
    assert pm.remaining_count == 3


def test_plan_manager_execute_update_plan():
    """Verify execute_update_plan tool invocation handler."""
    pm = PlanManager(policy=PlanningPolicy.SUB_FRONTIER)
    assert not pm.has_plan

    # 1. Initialize plan via 'steps'
    res_init = pm.execute_update_plan({"steps": ["Step A", "Step B"]})
    assert res_init["status"] == "ok"
    assert pm.total_count == 2
    assert not res_init["all_completed"]

    # 2. Update single step via 'step'
    res_step = pm.execute_update_plan({"step": 1, "completed": True})
    assert res_step["status"] == "ok"
    assert pm.completed_count == 1
    assert not res_step["all_completed"]

    # 3. Batch updates
    res_batch = pm.execute_update_plan({
        "updates": [
            {"step": 2, "completed": True},
        ]
    })
    assert res_batch["status"] == "ok"
    assert pm.is_complete
    assert res_batch["all_completed"]

    # 4. Out of range error
    res_err = pm.execute_update_plan({"step": 10})
    assert res_err["status"] == "error"
    assert res_err["is_error"] is True


def test_plan_manager_serialization():
    """Verify PlanManager serializes to dict and JSON."""
    pm = PlanManager(policy=PlanningPolicy.SUB_FRONTIER, initial_plan=["Do X"])
    d = pm.to_dict()
    assert d["policy"] == "sub_frontier"
    assert d["has_plan"] is True
    assert len(d["items"]) == 1

    j = pm.to_json()
    parsed = json.loads(j)
    assert parsed["items"][0]["description"] == "Do X"


# --------------------------------------------------------------------------- #
# 3. Out-of-History Plan Injection Tests
# --------------------------------------------------------------------------- #

def test_out_of_history_plan_injection_does_not_mutate_messages():
    """Verify PlanManager.inject_plan injects clean {PLAN} without mutating persistent messages."""
    pm = PlanManager(
        policy=PlanningPolicy.FRONTIER,
        initial_plan=["Task A", "Task B"],
    )

    base_messages = [
        {"role": "system", "content": "You are Monica."},
        {"role": "user", "content": "Please complete Task A and B."},
    ]

    # Persistent messages length before injection
    orig_count = len(base_messages)
    orig_sys = base_messages[0]["content"]

    conditioned = pm.inject_plan(base_messages)

    # Injected copy contains {PLAN} block
    assert "{PLAN}" in conditioned[0]["content"]
    assert "- [ ] 1. Task A" in conditioned[0]["content"]
    assert "- [ ] 2. Task B" in conditioned[0]["content"]

    # Base messages list remains completely unmutated and clean
    assert len(base_messages) == orig_count
    assert base_messages[0]["content"] == orig_sys
    assert "{PLAN}" not in base_messages[0]["content"]


def test_out_of_history_plan_injection_placeholder_replacement():
    """Verify {PLAN} placeholder in prompt template is replaced with plan block."""
    pm = PlanManager(
        policy=PlanningPolicy.SUB_FRONTIER,
        initial_plan=["Investigate bug", "Fix code"],
    )

    template_messages = [
        {"role": "system", "content": "You are Monica.\nActive Scaffolding:\n{PLAN}\nEnd plan."},
        {"role": "user", "content": "Start work."},
    ]

    conditioned = pm.inject_plan(template_messages)
    content = conditioned[0]["content"]

    assert "Active Scaffolding:\n{PLAN}" in content
    assert "- [ ] 1. Investigate bug" in content
    assert "- [ ] 2. Fix code" in content
    assert "End plan." in content


# --------------------------------------------------------------------------- #
# 4. Capability-Adaptive Policy Resolution Tests
# --------------------------------------------------------------------------- #

def test_resolve_planning_policy_adaptive():
    """Verify policy resolution: <100B -> SUB_FRONTIER, >=100B -> FRONTIER."""
    # Model size parameter
    assert resolve_planning_policy("adaptive", model_size_b=7.0) == PlanningPolicy.SUB_FRONTIER
    assert resolve_planning_policy("adaptive", model_size_b=70.0) == PlanningPolicy.SUB_FRONTIER
    assert resolve_planning_policy("adaptive", model_size_b=120.0) == PlanningPolicy.FRONTIER
    assert resolve_planning_policy("adaptive", model_size_b=405.0) == PlanningPolicy.FRONTIER

    # Model name patterns
    assert resolve_planning_policy("adaptive", model_name="monica-7b-instruct") == PlanningPolicy.SUB_FRONTIER
    assert resolve_planning_policy("adaptive", model_name="llama-3-8b-instruct") == PlanningPolicy.SUB_FRONTIER
    assert resolve_planning_policy("adaptive", model_name="claude-3-5-sonnet") == PlanningPolicy.FRONTIER
    assert resolve_planning_policy("adaptive", model_name="gpt-4o") == PlanningPolicy.FRONTIER

    # Explicit policies
    assert resolve_planning_policy("frontier") == PlanningPolicy.FRONTIER
    assert resolve_planning_policy("sub_frontier") == PlanningPolicy.SUB_FRONTIER


# --------------------------------------------------------------------------- #
# 5. Frontier Exit-Gate Policy Runtime Tests
# --------------------------------------------------------------------------- #

def test_frontier_exit_gate_terminates_upon_checklist_completion():
    """Verify frontier exit-gate policy cleanly terminates the run immediately upon checklist completion."""
    pm = PlanManager(
        policy=PlanningPolicy.FRONTIER,
        initial_plan=["Update config", "Verify build"],
    )

    turn_counter = 0

    def mock_lm(messages: list[dict[str, Any]]) -> str:
        nonlocal turn_counter
        turn_counter += 1
        # Model sees {PLAN} block in prompt
        assert any("{PLAN}" in m.get("content", "") for m in messages)

        if turn_counter == 1:
            return (
                "<think>Completing step 1.</think>\n"
                "<tool_call>\n"
                '{"name": "update_plan", "arguments": {"step": 1, "completed": true}}\n'
                "</tool_call>"
            )
        elif turn_counter == 2:
            return (
                "<think>Completing step 2, which finishes all verification.</think>\n"
                "<tool_call>\n"
                '{"name": "update_plan", "arguments": {"step": 2, "completed": true}}\n'
                "</tool_call>"
            )
        else:
            # Should NOT be reached because exit gate terminates immediately on turn 2
            return "<think>Wandering post-edit loop...</think>"

    runtime = AgentRuntime(
        lm=mock_lm,
        plan_manager=pm,
        max_turns=10,
    )
    result = runtime.run("Execute deployment checklist")

    assert result.success
    assert result.status == "completed"
    assert result.exit_gate_triggered
    assert result.total_turns == 2  # Clean exit at turn 2, NO wandering post-edit turns
    assert pm.is_complete
    assert any(e["event"] == "exit_gate_terminated" for e in result.events)


# --------------------------------------------------------------------------- #
# 6. Sub-Frontier Scaffolding Policy Runtime Tests
# --------------------------------------------------------------------------- #

def test_sub_frontier_turn1_mandatory_plan_and_premature_abort_prevention():
    """Verify sub-frontier policy rejects turn 1 completion without plan and blocks premature abort."""
    pm = PlanManager(policy=PlanningPolicy.SUB_FRONTIER)

    call_count = 0

    def reluctant_lm(messages: list[dict[str, Any]]) -> str:
        nonlocal call_count
        call_count += 1
        last_user = messages[-1].get("content", "")

        if call_count == 1:
            # Turn 1: Model tries to complete immediately without creating a plan (premature abort)
            return "<think>Trivial task.</think>I am finished."
        elif call_count == 2:
            # Turn 2: Model received mandatory plan enforcement notice
            assert "[PLAN ENFORCEMENT]" in last_user
            assert "Mandatory plan generation is required on turn 1" in last_user
            return (
                "<think>Creating plan now.</think>\n"
                "<tool_call>\n"
                '{"name": "update_plan", "arguments": {"plan": "- [ ] 1. Write code\\n- [ ] 2. Run tests"}}\n'
                "</tool_call>"
            )
        elif call_count == 3:
            # Turn 3: Model tries to finish, but items 1 and 2 are still pending!
            return "<think>Code written in my head.</think>All done!"
        elif call_count == 4:
            # Turn 4: Model received premature abort notice listing pending items
            assert "[PLAN ENFORCEMENT]" in last_user
            assert "checklist item(s) are still pending" in last_user
            return (
                "<think>Checking off both items.</think>\n"
                "<tool_call>\n"
                '{"name": "update_plan", "arguments": {"updates": [{"step": 1, "completed": true}, {"step": 2, "completed": true}]}}\n'
                "</tool_call>"
            )
        else:
            # Turn 5: Final completion with all items checked off
            return "<think>Plan verified.</think>Task completed successfully."

    runtime = AgentRuntime(
        lm=reluctant_lm,
        plan_manager=pm,
        max_turns=8,
    )
    result = runtime.run("Implement and test module")

    assert result.success
    assert result.status == "completed"
    assert result.total_turns == 5
    assert pm.is_complete
    assert any(e["event"] == "premature_abort_prevented" for e in result.events)


def test_sub_frontier_turn1_scaffolding_reminder():
    """Verify sub-frontier policy injects a plan reminder when tools run on turn 1 without a plan."""
    pm = PlanManager(policy=PlanningPolicy.SUB_FRONTIER)
    call_idx = 0

    def mock_lm(messages: list[dict[str, Any]]) -> str:
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return (
                "<tool_call>\n"
                '{"name": "view_file", "arguments": {"path": "main.py"}}\n'
                "</tool_call>"
            )
        else:
            # Check that turn 1 tool observation received the scaffolding reminder
            last_msg = messages[-1].get("content", "")
            assert "[PLAN REMINDER]" in last_msg
            return "Done."

    runtime = AgentRuntime(
        lm=mock_lm,
        plan_manager=pm,
        tool_executor=lambda name, args: {"content": "print('hello')", "total_lines": 1},
        max_turns=3,
    )
    result = runtime.run("Inspect and report")

    assert any(e["event"] == "plan_scaffolding_reminder" for e in result.events)


# --------------------------------------------------------------------------- #
# 7. WorkspaceToolExecutor Integration
# --------------------------------------------------------------------------- #

def test_workspace_tool_executor_update_plan(tmp_path: Path):
    """Verify WorkspaceToolExecutor executes update_plan via PlanManager."""
    pm = PlanManager(policy=PlanningPolicy.FRONTIER, initial_plan=["Step 1", "Step 2"])
    executor = WorkspaceToolExecutor(workspace_dir=tmp_path, plan_manager=pm)

    res = executor.execute("update_plan", {"step": 1, "completed": True})
    assert not res.get("is_error")
    assert res["status"] == "ok"
    assert res["completed_items"] == 1
    assert pm.completed_count == 1


# --------------------------------------------------------------------------- #
# 8. Out-of-History Plan Injection Verification Against Conversation Turns
# --------------------------------------------------------------------------- #

def test_plan_state_does_not_grow_conversation_turns():
    """Verify out-of-history injection conditions prompt without adding extra planning turns to messages."""
    pm = PlanManager(
        policy=PlanningPolicy.FRONTIER,
        initial_plan=["Step A", "Step B"],
    )

    call_num = 0

    def mock_lm(messages: list[dict[str, Any]]) -> str:
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            return '<tool_call>\n{"name": "update_plan", "arguments": {"step": 1, "completed": true}}\n</tool_call>'
        return '<tool_call>\n{"name": "update_plan", "arguments": {"step": 2, "completed": true}}\n</tool_call>'

    result = run_agent_loop(
        task="Complete steps out-of-history",
        lm=mock_lm,
        plan_manager=pm,
        max_turns=5,
    )

    assert result.success
    assert result.exit_gate_triggered
    # Exactly 2 turns executed
    assert result.total_turns == 2
    # Messages list only contains:
    # 0: system, 1: initial user, 2: assistant turn 1, 3: user tool resp turn 1, 4: assistant turn 2, 5: user tool resp turn 2
    assert len(result.messages) == 6
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
