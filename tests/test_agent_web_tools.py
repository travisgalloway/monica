"""End-to-end integration tests for web search and fetch tools in multi-turn ReAct runtime (#368).

Validates:
1. Runtime Dispatch Integration:
   - Registration of web_search and fetch_web_page in default tool registry (WorkspaceToolExecutor).
   - Execution timings, token counts, and network events flowing into TrajectoryTelemetry.
2. Compaction Awareness:
   - ContextCompactor soft-elision of bulky fetch_web_page observations ([Web page content elided: N chars])
     at the 0.60 usable context threshold.
3. Tool SFT & Evaluation Distractors:
   - Presence of web_search and fetch_web_page in CODING_AGENT_TOOLS as available tools / distractor candidates.
4. End-to-end multi-turn discovery -> fetch -> task completion with mock agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from src.agent.compaction import (
    ContextCompactor,
    elide_observation_content,
    estimate_tokens,
)
from src.agent.runtime import (
    AgentRuntime,
    TrajectoryTelemetry,
    WorkspaceToolExecutor,
    run_agent_loop,
)
from src.data.tool_sources import (
    CODING_AGENT_TOOLS,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    sample_distractors,
    validate_call_against_tools,
)


def _build_react_turn(thought: str, action: str, args: dict, obs: str) -> tuple[dict, dict]:
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
# 1. Tool Sources & SFT Distractor Integration Tests
# --------------------------------------------------------------------------- #

def test_coding_agent_tools_includes_web_search_and_fetch():
    """Verify web_search and fetch_web_page are in CODING_AGENT_TOOLS with valid schemas."""
    tool_names = [t["name"] for t in CODING_AGENT_TOOLS]
    assert "web_search" in tool_names
    assert "fetch_web_page" in tool_names
    assert "execute_bash" in tool_names
    assert "view_file" in tool_names
    assert "edit_file" in tool_names
    assert "write_file" in tool_names
    assert "grep_search" in tool_names
    assert "find_files" in tool_names

    # Check schemas
    assert validate_call_against_tools({"name": "web_search", "arguments": {"query": "python"}}, CODING_AGENT_TOOLS)
    assert not validate_call_against_tools({"name": "web_search", "arguments": {}}, CODING_AGENT_TOOLS)
    assert validate_call_against_tools({"name": "fetch_web_page", "arguments": {"url": "https://example.com"}}, CODING_AGENT_TOOLS)
    assert not validate_call_against_tools({"name": "fetch_web_page", "arguments": {}}, CODING_AGENT_TOOLS)


def test_sample_distractors_with_coding_agent_tools():
    """Verify distractor sampling works correctly alongside CODING_AGENT_TOOLS."""
    import random
    rng = random.Random(42)
    names = {t["name"] for t in CODING_AGENT_TOOLS}
    distractors = sample_distractors(names, 2, rng=rng)
    assert len(distractors) == 2
    for d in distractors:
        assert d["name"] not in names


# --------------------------------------------------------------------------- #
# 2. Runtime Dispatch & Tool Registry Tests
# --------------------------------------------------------------------------- #

def test_workspace_tool_executor_registry_registration(tmp_path: Path):
    """Verify WorkspaceToolExecutor registers web_search and fetch_web_page in its default registry."""
    executor = WorkspaceToolExecutor(workspace_dir=tmp_path)
    registry = executor.registry

    assert "web_search" in registry
    assert "fetch_web_page" in registry
    assert "execute_bash" in registry
    assert "view_file" in registry

    # Test custom tool registration
    executor.register_tool("custom_tool", lambda args: {"status": "custom", "value": args.get("val")})
    assert "custom_tool" in executor.registry
    res = executor.execute("custom_tool", {"val": 123})
    assert res == {"status": "custom", "value": 123}


def test_workspace_tool_executor_web_dispatch(tmp_path: Path):
    """Verify WorkspaceToolExecutor dispatches web_search and fetch_web_page cleanly."""
    executor = WorkspaceToolExecutor(workspace_dir=tmp_path)

    # Mock web_search
    with patch("src.agent.runtime.web_search", return_value=json.dumps([{"title": "Test Title", "url": "https://example.com", "snippet": "Test snippet"}])):
        search_res = executor.execute("web_search", {"query": "test query", "count": 1})
        assert "Test Title" in search_res
        assert "https://example.com" in search_res

    # Mock fetch_web_page
    with patch("src.agent.runtime.fetch_web_page", return_value="# Page Title\n\nPage body content markdown."):
        fetch_res = executor.execute("fetch_web_page", {"url": "https://example.com"})
        assert "Page Title" in fetch_res
        assert "Page body content markdown" in fetch_res


def test_workspace_tool_executor_disabled_web_tools(tmp_path: Path):
    """Verify executor returns clean error message when web tools are disabled."""
    executor = WorkspaceToolExecutor(workspace_dir=tmp_path, enable_web_tools=False)
    search_res = executor.execute("web_search", {"query": "test"})
    assert "disabled" in search_res.lower()
    fetch_res = executor.execute("fetch_web_page", {"url": "https://example.com"})
    assert "disabled" in fetch_res.lower()


# --------------------------------------------------------------------------- #
# 3. Compaction Awareness Tests
# --------------------------------------------------------------------------- #

def test_compactor_soft_elides_bulky_fetch_web_page_at_60_threshold():
    """Verify ContextCompactor soft-elides bulky fetch_web_page outputs as [Web page content elided: N chars]."""
    sys_msg = {"role": "system", "content": "You are Monica."}
    task_msg = {"role": "user", "content": "Fetch documentation."}

    # Turn 1: web search (middle turn)
    search_body = json.dumps([{"title": "Doc", "url": "https://example.com", "snippet": "Snippet"}])
    t1_a, t1_u = _build_react_turn("Search docs", "web_search", {"query": "docs"}, search_body)

    # Turn 2: bulky fetch_web_page output (middle turn)
    bulky_markdown = "# API Reference\n" + ("This is detailed API documentation with many parameters.\n" * 80)
    assert len(bulky_markdown) > 1000
    t2_a, t2_u = _build_react_turn("Fetch doc page", "fetch_web_page", {"url": "https://example.com"}, bulky_markdown)

    # Turn 3: bulky bash output (middle turn to verify distinction between output elided and web page elided)
    bulky_bash = "compiler build log line...\n" * 80
    t3_a, t3_u = _build_react_turn("Compile", "execute_bash", {"command": "make"}, bulky_bash)

    # Turn 4 & 5: recent turns (protected verbatim)
    t4_a, t4_u = _build_react_turn("View fix", "view_file", {"path": "fix.py"}, "short fix line")
    t5_a, t5_u = _build_react_turn("Test", "execute_bash", {"command": "pytest"}, "1 passed")

    messages = [sys_msg, task_msg, t1_a, t1_u, t2_a, t2_u, t3_a, t3_u, t4_a, t4_u, t5_a, t5_u]

    init_tokens = estimate_tokens(messages)
    # Set max_context_tokens so that init_tokens exceeds soft_threshold (0.60)
    compactor = ContextCompactor(
        max_context_tokens=int(init_tokens / 0.70),  # init is ~70% of max, above 0.60 soft threshold
        soft_threshold=0.60,
        hard_threshold=0.90,
        protect_recent_turns=2,
        elision_char_threshold=200,
    )

    compacted_messages, report = compactor.compact_with_report(messages)

    assert report.compacted is True
    assert report.elision_applied is True
    assert report.summarization_applied is False
    assert report.elided_observations >= 2
    assert report.final_tokens < report.initial_tokens

    # Verify Turn 2 (fetch_web_page) was elided with [Web page content elided: N chars]
    t2_user_compacted = compacted_messages[5]["content"]
    assert "[Web page content elided:" in t2_user_compacted
    assert f"{len(bulky_markdown.strip())} chars]" in t2_user_compacted
    assert bulky_markdown not in t2_user_compacted

    # Verify Turn 3 (execute_bash) was elided with [Output elided: N chars]
    t3_user_compacted = compacted_messages[7]["content"]
    assert "[Output elided:" in t3_user_compacted
    assert "[Web page content elided:" not in t3_user_compacted

    # Verify Protected Window Invariant: preamble and recent turns (turns 4 & 5) intact
    assert compacted_messages[0] == sys_msg
    assert compacted_messages[1] == task_msg
    assert compacted_messages[8] == t4_a
    assert compacted_messages[9] == t4_u
    assert compacted_messages[10] == t5_a
    assert compacted_messages[11] == t5_u


def test_elide_observation_content_direct_web_page_tool():
    """Verify elide_observation_content handles tool_name='fetch_web_page' directly."""
    bulky_body = "# Markdown Header\n" + "Page content details\n" * 50
    content = f"{TOOL_RESPONSE_OPEN}\n{bulky_body}\n{TOOL_RESPONSE_CLOSE}"

    elided, n_obs, n_chars = elide_observation_content(content, elision_char_threshold=100, tool_name="fetch_web_page")
    assert n_obs == 1
    assert n_chars == len(bulky_body.strip())
    assert f"[Web page content elided: {n_chars} chars]" in elided

    # Non-web tool uses [Output elided: N chars]
    elided_other, _, _ = elide_observation_content(content, elision_char_threshold=100, tool_name="view_file")
    assert f"[Output elided: {n_chars} chars]" in elided_other


# --------------------------------------------------------------------------- #
# 4. End-to-End Multi-Turn Mock Agent Test
# --------------------------------------------------------------------------- #

def test_end_to_end_multiturn_web_tools_mock_agent(tmp_path: Path):
    """Test full multi-turn ReAct workflow: discovery (web_search) -> extraction (fetch_web_page) -> task completion."""
    search_json = json.dumps([
        {
            "title": "Monica Documentation: Fast KV Caching",
            "url": "https://monica.ai/docs/kv-cache",
            "snippet": "Configure kv_cache_mode to 'fused' for 2x decode throughput.",
        }
    ])
    page_markdown = "# Fast KV Caching\n\nTo enable fused KV caching, set `fused_cache=True` in ModelConfig."

    call_index = 0

    def mock_model_backend(messages: list[dict]) -> str:
        nonlocal call_index
        call_index += 1

        if call_index == 1:
            # Turn 1: Model issues web_search discovery query
            return (
                "<think>The user wants to configure fast KV caching. Let me search the docs.</think>\n"
                f"{TOOL_CALL_OPEN}\n"
                '{"name": "web_search", "arguments": {"query": "Monica fast KV caching docs"}}\n'
                f"{TOOL_CALL_CLOSE}"
            )
        elif call_index == 2:
            # Turn 2: Model sees search result and fetches the specific page URL
            last_msg = messages[-1]["content"]
            assert "https://monica.ai/docs/kv-cache" in last_msg
            return (
                "<think>Search returned the KV cache doc URL. Now I need to fetch the page content.</think>\n"
                f"{TOOL_CALL_OPEN}\n"
                '{"name": "fetch_web_page", "arguments": {"url": "https://monica.ai/docs/kv-cache"}}\n'
                f"{TOOL_CALL_CLOSE}"
            )
        elif call_index == 3:
            # Turn 3: Model extracts knowledge and writes configuration to workspace
            last_msg = messages[-1]["content"]
            assert "fused_cache=True" in last_msg
            return (
                "<think>The doc says set fused_cache=True. Let me write this to config.py.</think>\n"
                f"{TOOL_CALL_OPEN}\n"
                '{"name": "write_file", "arguments": {"path": "config.py", "content": "fused_cache = True\\n"}}\n'
                f"{TOOL_CALL_CLOSE}"
            )
        else:
            # Turn 4: Model delivers final response
            return (
                "<think>Configuration has been written successfully.</think>\n"
                "I searched the documentation, fetched the KV cache guide, and configured fused_cache = True in config.py."
            )

    executor = WorkspaceToolExecutor(workspace_dir=tmp_path)

    with (
        patch("src.agent.runtime.web_search", return_value=search_json),
        patch("src.agent.runtime.fetch_web_page", return_value=page_markdown),
    ):
        runtime = AgentRuntime(
            lm=mock_model_backend,
            workspace_dir=tmp_path,
            tool_executor=executor,
            max_turns=5,
        )
        result = runtime.run("Find documentation on fast KV caching and configure it in config.py")

    # 1. Verify Task Completion
    assert result.status == "completed"
    assert result.success is True
    assert result.total_turns == 4
    assert (tmp_path / "config.py").read_text() == "fused_cache = True\n"
    assert "configured fused_cache = True" in (result.final_answer or "")

    # 2. Verify TrajectoryTelemetry
    telemetry = result.telemetry
    assert isinstance(telemetry, TrajectoryTelemetry)
    assert telemetry.total_turns == 4
    assert telemetry.web_search_count == 1
    assert telemetry.web_fetch_count == 1
    assert telemetry.total_tool_wall_s > 0
    assert telemetry.total_tokens > 0

    # 3. Verify Network Events
    assert len(telemetry.network_events) == 2
    search_event = telemetry.network_events[0]
    assert search_event["event"] == "network_event"
    assert search_event["tool"] == "web_search"
    assert search_event["action"] == "search"
    assert search_event["query"] == "Monica fast KV caching docs"
    assert search_event["token_count"] > 0
    assert search_event["is_error"] is False

    fetch_event = telemetry.network_events[1]
    assert fetch_event["event"] == "network_event"
    assert fetch_event["tool"] == "fetch_web_page"
    assert fetch_event["action"] == "fetch"
    assert fetch_event["url"] == "https://monica.ai/docs/kv-cache"
    assert fetch_event["token_count"] > 0
    assert fetch_event["char_count"] == len(page_markdown)
    assert fetch_event["is_error"] is False

    # 4. Verify Turn Tool Observations Telemetry
    turn1_obs = result.turns[0].tool_observations[0]
    assert turn1_obs.name == "web_search"
    assert turn1_obs.token_count > 0
    assert not turn1_obs.is_error

    turn2_obs = result.turns[1].tool_observations[0]
    assert turn2_obs.name == "fetch_web_page"
    assert turn2_obs.token_count > 0
    assert not turn2_obs.is_error

    # 5. Verify to_dict serialization
    res_dict = result.to_dict()
    assert "telemetry" in res_dict
    assert res_dict["telemetry"]["web_search_count"] == 1
    assert res_dict["telemetry"]["web_fetch_count"] == 1
    assert len(res_dict["telemetry"]["network_events"]) == 2


def test_run_agent_loop_convenience_wrapper_with_web_tools(tmp_path: Path):
    """Verify run_agent_loop wrapper passes tools and returns AgentRunResult with TrajectoryTelemetry."""
    executor = WorkspaceToolExecutor(workspace_dir=tmp_path)
    res = run_agent_loop(
        task="Immediate finish",
        lm=lambda msgs: "All done immediately.",
        workspace_dir=tmp_path,
        tool_executor=executor,
        max_turns=2,
    )
    assert res.status == "completed"
    assert res.success is True
    assert isinstance(res.telemetry, TrajectoryTelemetry)
