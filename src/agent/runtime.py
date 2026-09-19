"""Autonomous multi-turn coding agent execution loop with anti-spin circuit breakers (#349).

Implements the M12 agent runtime driving ReAct loops (Thought -> Action -> Tool Observation)
over repository workspaces using CODING_AGENT_TOOLS and native web tools.

Anti-Spin Circuit Breaker:
  - Monitors the tool call stream for consecutive identical calls (identical tool name and
    byte-identical canonical arguments).
  - Injects a corrective redirection reminder at 5 consecutive identical calls or failures.
  - Terminates execution early at 8 consecutive identical failures (or 8 identical repeats)
    with a dedicated failure reason rather than exhausting the max-turn budget.

Telemetry & Telemetry Logging:
  - Records per-turn event metrics, tool execution timings, token counts, and phase transitions.
  - Supports pluggable LMAdapter backends, custom tool executors, and trajectory loggers.

ABOVE THE SEAM — pure Python standard library + NumPy only. No hardware backends (mlx/torch)
are imported anywhere in this module (enforced by tests/test_import_guard.py).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..data.chat_template import CHAT_EOS, IM_END, render
from ..data.tool_sources import (
    CODING_AGENT_TOOLS,
    FETCH_WEB_PAGE_TOOL,
    TOOL_CALL_OPEN,
    UPDATE_PLAN_TOOL,
    WEB_SEARCH_TOOL,
    format_tool_response,
    render_tool_system,
)
from ..eval.bfcl_adapter import parse_tool_calls
from ..serve.sampling import sample
from .fetcher import fetch_web_page
from .planning import PlanManager, PlanningPolicy
from .safety import (
    FileReadRegistry,
    format_diagnostic_summary,
    resolve_safe_workspace_path,
    run_file_diagnostics,
)
from .search import web_search

DEFAULT_SYSTEM_PROMPT = (
    "You are Monica, an autonomous coding agent. Solve the user's task by exploring the codebase, "
    "editing files, and running commands using your tools. Think step-by-step before taking actions. "
    "When your task is complete or no further actions are needed, output your final response."
)

DEFAULT_MAX_TURNS = 20
DEFAULT_MAX_GEN_TOKENS = 512
DEFAULT_REDIRECTION_THRESHOLD = 5
DEFAULT_TERMINATION_THRESHOLD = 8


# --------------------------------------------------------------------------- #
# Canonical Serialization & Helper Utilities
# --------------------------------------------------------------------------- #

def canonical_call_bytes(name: str, arguments: dict[str, Any] | str | None) -> bytes:
    """Generate canonical byte representation for a tool call (name and arguments).

    Ensures identical tool calls with differing dictionary key order or spacing
    are identified as byte-identical.
    """
    clean_name = name.strip() if isinstance(name, str) else str(name)
    if arguments is None:
        args_bytes = b"{}"
    elif isinstance(arguments, dict):
        args_bytes = json.dumps(arguments, sort_keys=True, separators=(",", ":")).encode("utf-8")
    elif isinstance(arguments, str):
        raw = arguments.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                args_bytes = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
            else:
                args_bytes = raw.encode("utf-8")
        except (json.JSONDecodeError, ValueError):
            args_bytes = raw.encode("utf-8")
    else:
        args_bytes = str(arguments).encode("utf-8")

    return f"{clean_name}:".encode() + args_bytes


def is_tool_failure(result: Any) -> bool:
    """Determine whether a tool execution result represents a failure."""
    if isinstance(result, Exception):
        return True
    if isinstance(result, dict):
        if result.get("is_error") is True:
            return True
        if "error" in result and result["error"] is not None:
            return True
        return bool("exit_code" in result and result["exit_code"] != 0)
    if hasattr(result, "is_success") and not result.is_success:
        return True
    if hasattr(result, "error") and result.error is not None:
        return True
    if hasattr(result, "exit_code") and result.exit_code != 0:
        return True
    if isinstance(result, str):
        stripped = result.strip()
        if stripped.startswith(("Error:", "error:", "[ERROR]", "Traceback (most recent call last):")):
            return True
    return False


# --------------------------------------------------------------------------- #
# Telemetry and Turn Structures
# --------------------------------------------------------------------------- #

@dataclass
class ToolCall:
    """Structured representation of a single tool invocation."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_call_bytes(self.name, self.arguments)

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {"name": self.name, "arguments": self.arguments}
        if self.call_id is not None:
            res["call_id"] = self.call_id
        return res


@dataclass
class ToolObservation:
    """Observation resulting from executing a tool call."""

    name: str
    output: Any
    is_error: bool = False
    wall_s: float = 0.0
    call_id: str | None = None
    redirection_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out_repr = self.output
        if hasattr(out_repr, "to_json"):
            try:
                out_repr = json.loads(out_repr.to_json())
            except (json.JSONDecodeError, ValueError, TypeError):
                out_repr = str(out_repr)
        elif not isinstance(out_repr, (dict, list, str, int, float, bool, type(None))):
            out_repr = str(out_repr)

        res: dict[str, Any] = {
            "name": self.name,
            "output": out_repr,
            "is_error": self.is_error,
            "wall_s": self.wall_s,
        }
        if self.call_id is not None:
            res["call_id"] = self.call_id
        if self.redirection_warning is not None:
            res["redirection_warning"] = self.redirection_warning
        return res


@dataclass
class AgentTurn:
    """Telemetry and content for one turn in the ReAct loop."""

    turn_index: int
    thought: str = ""
    action: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_wall_s: float = 0.0
    tool_wall_s: float = 0.0
    turn_wall_s: float = 0.0
    phase_transitions: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "thought": self.thought,
            "action": self.action,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "tool_observations": [o.to_dict() for o in self.tool_observations],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "generation_wall_s": self.generation_wall_s,
            "tool_wall_s": self.tool_wall_s,
            "turn_wall_s": self.turn_wall_s,
            "phase_transitions": list(self.phase_transitions),
            "events": list(self.events),
        }


@dataclass
class AgentRunResult:
    """Full trajectory result returned by an agent run."""

    task: str
    turns: list[AgentTurn] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    status: str = "completed"  # "completed", "budget_exhausted", "circuit_breaker_tripped", "failed"
    success: bool = True
    failure_reason: str | None = None
    circuit_breaker_triggered: bool = False
    exit_gate_triggered: bool = False
    plan_state: dict[str, Any] | None = None
    total_wall_s: float = 0.0
    total_tool_wall_s: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "exit_gate_triggered": self.exit_gate_triggered,
            "plan_state": self.plan_state,
            "total_turns": self.total_turns,
            "total_wall_s": self.total_wall_s,
            "total_tool_wall_s": self.total_tool_wall_s,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "final_answer": self.final_answer,
            "turns": [t.to_dict() for t in self.turns],
            "messages": self.messages,
            "events": list(self.events),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------- #
# Anti-Spin Circuit Breaker
# --------------------------------------------------------------------------- #

@dataclass
class CircuitBreakerStatus:
    """State reported by the circuit breaker for a call and its outcome."""

    is_identical: bool
    consecutive_calls: int
    consecutive_failures: int
    should_redirect: bool = False
    should_terminate: bool = False
    redirection_message: str | None = None
    termination_reason: str | None = None


class AntiSpinCircuitBreaker:
    """Monitors the tool call stream for consecutive identical calls and failures.

    Triggers corrective redirection reminders at 5 consecutive repeats (identical calls
    or identical failures). Terminates execution early at 8 consecutive identical repeats/failures
    with a dedicated failure reason rather than exhausting the max-turn budget.
    """

    def __init__(
        self,
        redirection_threshold: int = DEFAULT_REDIRECTION_THRESHOLD,
        termination_threshold: int = DEFAULT_TERMINATION_THRESHOLD,
    ) -> None:
        self.redirection_threshold = int(redirection_threshold)
        self.termination_threshold = int(termination_threshold)
        self.last_call_bytes: bytes | None = None
        self.last_tool_name: str | None = None
        self.consecutive_identical_calls: int = 0
        self.consecutive_identical_failures: int = 0
        self.total_redirections_injected: int = 0
        self.is_terminated: bool = False
        self.termination_reason: str | None = None

    def reset(self) -> None:
        """Reset internal circuit breaker counters."""
        self.last_call_bytes = None
        self.last_tool_name = None
        self.consecutive_identical_calls = 0
        self.consecutive_identical_failures = 0
        self.total_redirections_injected = 0
        self.is_terminated = False
        self.termination_reason = None

    def record_call(self, name: str, arguments: dict[str, Any] | str | None) -> CircuitBreakerStatus:
        """Inspect an incoming tool call before execution."""
        call_bytes = canonical_call_bytes(name, arguments)
        is_identical = self.last_call_bytes is not None and call_bytes == self.last_call_bytes

        if is_identical:
            self.consecutive_identical_calls += 1
        else:
            self.last_call_bytes = call_bytes
            self.last_tool_name = name
            self.consecutive_identical_calls = 1
            self.consecutive_identical_failures = 0

        should_redirect = False
        redirection_message = None

        if self.consecutive_identical_calls >= self.redirection_threshold:
            should_redirect = True
            redirection_message = (
                f"[CIRCUIT BREAKER REDIRECTION] Detected {self.consecutive_identical_calls} consecutive "
                f"identical tool calls for '{name}'. Stop repeating identical calls. Adjust arguments, "
                f"use a different tool, or provide your final answer."
            )

        return CircuitBreakerStatus(
            is_identical=is_identical,
            consecutive_calls=self.consecutive_identical_calls,
            consecutive_failures=self.consecutive_identical_failures,
            should_redirect=should_redirect,
            should_terminate=False,
            redirection_message=redirection_message,
            termination_reason=None,
        )

    def record_result(
        self,
        name: str,
        arguments: dict[str, Any] | str | None,
        *,
        is_error: bool,
    ) -> CircuitBreakerStatus:
        """Inspect the outcome of a tool execution."""
        call_bytes = canonical_call_bytes(name, arguments)
        is_identical = self.last_call_bytes is not None and call_bytes == self.last_call_bytes

        if is_error:
            if is_identical:
                self.consecutive_identical_failures += 1
            else:
                self.consecutive_identical_failures = 1
        else:
            self.consecutive_identical_failures = 0

        should_redirect = False
        redirection_message = None
        should_terminate = False
        termination_reason = None

        # Redirection reminder at threshold
        if (
            self.consecutive_identical_failures >= self.redirection_threshold
            or self.consecutive_identical_calls >= self.redirection_threshold
        ):
            should_redirect = True
            self.total_redirections_injected += 1
            if self.consecutive_identical_failures >= self.redirection_threshold:
                redirection_message = (
                    f"[CIRCUIT BREAKER REDIRECTION] Tool '{name}' has failed "
                    f"{self.consecutive_identical_failures} times consecutively with identical arguments. "
                    f"Stop repeating this failing action and try a different approach."
                )
            else:
                redirection_message = (
                    f"[CIRCUIT BREAKER REDIRECTION] Detected {self.consecutive_identical_calls} consecutive "
                    f"identical tool calls for '{name}'. Stop repeating identical calls. Adjust arguments, "
                    f"use a different tool, or provide your final answer."
                )

        # Early termination at threshold: prioritize failure count over identical call count
        if self.consecutive_identical_failures >= self.termination_threshold:
            should_terminate = True
            termination_reason = (
                f"anti_spin_circuit_breaker: {self.consecutive_identical_failures} consecutive identical "
                f"failures for tool '{name}'"
            )
            self.is_terminated = True
            self.termination_reason = termination_reason
        elif self.consecutive_identical_calls >= self.termination_threshold:
            should_terminate = True
            termination_reason = (
                f"anti_spin_circuit_breaker: {self.consecutive_identical_calls} consecutive identical "
                f"calls for tool '{name}'"
            )
            self.is_terminated = True
            self.termination_reason = termination_reason

        return CircuitBreakerStatus(
            is_identical=is_identical,
            consecutive_calls=self.consecutive_identical_calls,
            consecutive_failures=self.consecutive_identical_failures,
            should_redirect=should_redirect,
            should_terminate=should_terminate,
            redirection_message=redirection_message,
            termination_reason=termination_reason,
        )


# --------------------------------------------------------------------------- #
# Workspace Tool Execution
# --------------------------------------------------------------------------- #

class WorkspaceToolExecutor:
    """Sandboxed execution provider for CODING_AGENT_TOOLS on a local repository (#349, #351).

    Enforces deterministic safety gates (#351):
      - Workspace Containment: Path jailbreak checks preventing symlinks or relative
        traversals outside repository root.
      - Read-Before-Write Gate: Reject edits and overwrites to files not read in the session.
      - Immediate Post-Mutation Diagnostics: Fast static analysis and linter feedback
        attached directly to tool observations.
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        timeout_s: float = 30.0,
        *,
        enable_web_tools: bool = True,
        enforce_read_before_write: bool = True,
        enable_diagnostic_feedback: bool = True,
        read_registry: FileReadRegistry | None = None,
        diagnostic_provider: Any | None = None,
        ts_lsp_service: Any | None = None,
        plan_manager: PlanManager | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_dir or os.getcwd()).resolve()
        self.timeout_s = float(timeout_s)
        self.enable_web_tools = bool(enable_web_tools)
        self.enforce_read_before_write = bool(enforce_read_before_write)
        self.enable_diagnostic_feedback = bool(enable_diagnostic_feedback)
        self.read_registry = read_registry if read_registry is not None else FileReadRegistry()
        self.diagnostic_provider = diagnostic_provider
        self.ts_lsp_service = ts_lsp_service
        self.plan_manager = plan_manager

    def reset(self) -> None:
        """Reset session-level safety state (read registry and content hashes)."""
        self.read_registry.clear()
        if self.plan_manager is not None:
            self.plan_manager.reset()

    def _resolve_safe_path(self, path_str: str) -> tuple[Path | None, str | None]:
        """Resolve a workspace path, preventing directory traversal and symlink jailbreaks outside root."""
        return resolve_safe_workspace_path(path_str, self.workspace_root)

    def _run_diagnostics(self, target: Path, content: str) -> list[dict[str, Any]]:
        """Run fast post-mutation static analysis and linter diagnostics (#351)."""
        if not self.enable_diagnostic_feedback:
            return []
        try:
            return run_file_diagnostics(
                target,
                content=content,
                custom_provider=self.diagnostic_provider,
                ts_lsp_service=self.ts_lsp_service,
            )
        except Exception:  # noqa: BLE001
            return []

    def execute_bash(self, command: str) -> dict[str, Any]:
        """Execute a shell command within the repository workspace."""
        try:
            proc = subprocess.run(
                command,
                shell=True,
                check=False,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "is_error": proc.returncode != 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "error": f"Command timed out after {self.timeout_s} seconds",
                "is_error": True,
            }
        except (OSError, RuntimeError) as e:
            return {"exit_code": -1, "error": str(e), "is_error": True}

    def view_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """View content lines of a file in the workspace."""
        target, err = self._resolve_safe_path(path)
        if err or target is None:
            return {"error": err, "is_error": True, "safety_violation": "path_jailbreak"}

        if not target.is_file():
            return {"error": f"File not found: {path}", "is_error": True}

        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            rel_path = str(target.relative_to(self.workspace_root))
            # Register in session read registry with SHA-256 content hash (#351)
            self.read_registry.register_read(rel_path, content)

            lines = content.splitlines(keepends=True)
            total_lines = len(lines)
            s_idx = max(1, int(start_line)) if start_line is not None else 1
            e_idx = min(total_lines, int(end_line)) if end_line is not None else total_lines

            if s_idx > total_lines or s_idx > e_idx:
                selected_lines = []
            else:
                selected_lines = lines[s_idx - 1 : e_idx]

            numbered = [f"{s_idx + i}: {line}" for i, line in enumerate(selected_lines)]
            return {
                "path": rel_path,
                "total_lines": total_lines,
                "start_line": s_idx,
                "end_line": e_idx,
                "content": "".join(selected_lines),
                "numbered": "".join(numbered),
            }
        except (OSError, UnicodeDecodeError) as e:
            return {"error": f"Failed to read file {path}: {e}", "is_error": True}

    def edit_file(self, path: str, old_str: str, new_str: str) -> dict[str, Any]:
        """Replace a unique occurrence of old_str with new_str in a workspace file (#351)."""
        target, err = self._resolve_safe_path(path)
        if err or target is None:
            return {"error": err, "is_error": True, "safety_violation": "path_jailbreak"}

        if not target.is_file():
            return {"error": f"File not found: {path}", "is_error": True}

        rel_path = str(target.relative_to(self.workspace_root))

        # Read-before-write safety gate (#351)
        if self.enforce_read_before_write and not self.read_registry.is_read(rel_path):
            return {
                "error": (
                    f"Read-before-write safety violation: file '{path}' has not been read in this session. "
                    f"Use view_file to inspect the file before modifying it."
                ),
                "is_error": True,
                "safety_violation": "unread_file",
            }

        try:
            with open(target, "r", encoding="utf-8") as f:
                content = f.read()

            count = content.count(old_str)
            if count == 0:
                return {
                    "error": f"old_str not found in {path}. Make sure target string matches exactly.",
                    "is_error": True,
                }
            if count > 1:
                return {
                    "error": f"old_str occurs {count} times in {path}; must match exactly one occurrence.",
                    "is_error": True,
                }

            updated = content.replace(old_str, new_str, 1)
            with open(target, "w", encoding="utf-8") as f:
                f.write(updated)

            # Update session read registry with new content hash (#351)
            self.read_registry.register_write(rel_path, updated)

            # Post-mutation diagnostic feedback (#351)
            diagnostics = self._run_diagnostics(target, updated)
            msg = f"Successfully edited {path}"
            if diagnostics:
                summary = format_diagnostic_summary(diagnostics)
                msg = f"{msg}. {summary}"

            return {
                "status": "ok",
                "path": rel_path,
                "message": msg,
                "diagnostics": diagnostics,
            }
        except (OSError, UnicodeDecodeError) as e:
            return {"error": f"Failed to edit file {path}: {e}", "is_error": True}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write or overwrite content to a workspace file (#351)."""
        target, err = self._resolve_safe_path(path)
        if err or target is None:
            return {"error": err, "is_error": True, "safety_violation": "path_jailbreak"}

        rel_path = str(target.relative_to(self.workspace_root))

        # Read-before-write safety check (#351): existing files must be read before overwrite
        if (
            self.enforce_read_before_write
            and target.is_file()
            and not self.read_registry.is_read(rel_path)
        ):
            return {
                    "error": (
                        f"Read-before-write safety violation: file '{path}' already exists but has not been read in this session. "
                        f"Use view_file to inspect the file before overwriting it."
                    ),
                    "is_error": True,
                    "safety_violation": "unread_file",
                }

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)

            # Update session read registry with new content hash (#351)
            self.read_registry.register_write(rel_path, content)

            # Post-mutation diagnostic feedback (#351)
            diagnostics = self._run_diagnostics(target, content)
            msg = f"Successfully wrote {path}"
            if diagnostics:
                summary = format_diagnostic_summary(diagnostics)
                msg = f"{msg}. {summary}"

            return {
                "status": "ok",
                "path": rel_path,
                "message": msg,
                "diagnostics": diagnostics,
            }
        except (OSError, UnicodeEncodeError) as e:
            return {"error": f"Failed to write file {path}: {e}", "is_error": True}

    def grep_search(self, query: str, path: str | None = None) -> dict[str, Any]:
        """Search for a regex or string pattern across files in the workspace."""
        search_dir = self.workspace_root
        if path:
            target, err = self._resolve_safe_path(path)
            if err or target is None:
                return {"error": err, "is_error": True, "safety_violation": "path_jailbreak"}
            search_dir = target

        results: list[dict[str, Any]] = []
        try:
            pattern = re.compile(query)
        except re.error:
            pattern = re.compile(re.escape(query))

        try:
            if search_dir.is_file():
                files = [search_dir]
            else:
                files = [p for p in search_dir.rglob("*") if p.is_file()]

            for fpath in files:
                if any(part.startswith(".") for part in fpath.parts):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        for line_idx, line in enumerate(f, start=1):
                            if pattern.search(line):
                                rel_path = str(fpath.relative_to(self.workspace_root))
                                results.append({
                                    "file": rel_path,
                                    "line": line_idx,
                                    "content": line.rstrip("\r\n"),
                                })
                                if len(results) >= 100:
                                    break
                except (OSError, UnicodeDecodeError):
                    continue
                if len(results) >= 100:
                    break

            return {"query": query, "total_matches": len(results), "matches": results}
        except (OSError, re.error) as e:
            return {"error": f"grep_search failed: {e}", "is_error": True}

    def find_files(self, pattern: str, dir: str | None = None) -> dict[str, Any]:
        """Find files matching a glob pattern."""
        search_dir = self.workspace_root
        if dir:
            target, err = self._resolve_safe_path(dir)
            if err or target is None:
                return {"error": err, "is_error": True, "safety_violation": "path_jailbreak"}
            search_dir = target

        results: list[str] = []
        try:
            for root, _, filenames in os.walk(search_dir):
                rel_root = Path(root)
                if any(part.startswith(".") for part in rel_root.relative_to(self.workspace_root).parts):
                    continue
                for fname in filenames:
                    if fnmatch.fnmatch(fname, pattern):
                        full_path = rel_root / fname
                        rel_path = str(full_path.relative_to(self.workspace_root))
                        results.append(rel_path)
                        if len(results) >= 100:
                            break
                if len(results) >= 100:
                    break

            return {"pattern": pattern, "count": len(results), "files": sorted(results)}
        except OSError as e:
            return {"error": f"find_files failed: {e}", "is_error": True}

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch tool execution by name."""
        args = arguments or {}
        if name == "execute_bash":
            return self.execute_bash(str(args.get("command", "")))
        elif name == "view_file":
            return self.view_file(
                path=str(args.get("path", "")),
                start_line=args.get("start_line"),
                end_line=args.get("end_line"),
            )
        elif name == "edit_file":
            return self.edit_file(
                path=str(args.get("path", "")),
                old_str=str(args.get("old_str", "")),
                new_str=str(args.get("new_str", "")),
            )
        elif name == "write_file":
            return self.write_file(
                path=str(args.get("path", "")),
                content=str(args.get("content", "")),
            )
        elif name == "grep_search":
            return self.grep_search(query=str(args.get("query", "")), path=args.get("path"))
        elif name == "find_files":
            return self.find_files(pattern=str(args.get("pattern", "*")), dir=args.get("dir"))
        elif name == "web_search" and self.enable_web_tools:
            return web_search(query=str(args.get("query", "")), count=int(args.get("count", 5)))
        elif name == "fetch_web_page" and self.enable_web_tools:
            return fetch_web_page(url=str(args.get("url", "")))
        elif name == "update_plan":
            if self.plan_manager is not None:
                return self.plan_manager.execute_update_plan(args)
            return {"error": "No PlanManager configured for update_plan", "is_error": True}
        else:
            return {"error": f"Unknown or unsupported tool: {name}", "is_error": True}


# --------------------------------------------------------------------------- #
# Trajectory Logging Interfaces
# --------------------------------------------------------------------------- #

@runtime_checkable
class TrajectoryLogger(Protocol):
    """Protocol for trajectory log sinks."""

    def log_turn(self, turn: AgentTurn) -> None:
        ...

    def log_trajectory(self, trajectory: AgentRunResult) -> None:
        ...


class InMemoryTrajectoryLogger:
    """In-memory telemetry store for testing and analysis."""

    def __init__(self) -> None:
        self.turns: list[AgentTurn] = []
        self.trajectories: list[AgentRunResult] = []

    def log_turn(self, turn: AgentTurn) -> None:
        self.turns.append(turn)

    def log_trajectory(self, trajectory: AgentRunResult) -> None:
        self.trajectories.append(trajectory)

    def clear(self) -> None:
        self.turns.clear()
        self.trajectories.clear()


class JsonlTrajectoryLogger:
    """Appends turns and trajectories to a JSONL file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_turn(self, turn: AgentTurn) -> None:
        record = {"type": "turn", "data": turn.to_dict(), "timestamp": time.time()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log_trajectory(self, trajectory: AgentRunResult) -> None:
        record = {"type": "trajectory", "data": trajectory.to_dict(), "timestamp": time.time()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
# ReAct Response Parsing
# --------------------------------------------------------------------------- #

_ACTION_REGEX = re.compile(
    r"(?:^|\n)(?:Action|Tool):\s*([a-zA-Z0-9_-]+)\s*(?:(?:\(|\s*)\s*(\{.*?\})\s*\)?)",
    re.DOTALL,
)


def parse_react_response(text: str) -> tuple[str, list[ToolCall], str | None]:
    """Parse an assistant ReAct response into (thought, tool_calls, final_answer).

    Extracts <tool_call> JSON blocks, <think>...</think> traces, and regular completions.
    """
    calls_raw = parse_tool_calls(text)
    tool_calls: list[ToolCall] = [
        ToolCall(name=c["name"], arguments=c.get("arguments", {})) for c in calls_raw if "name" in c
    ]

    # Fallback to Action: tool_name(args) if no <tool_call> tags present
    if not tool_calls:
        for m in _ACTION_REGEX.finditer(text):
            tool_name = m.group(1).strip()
            args_str = m.group(2).strip()
            try:
                args = json.loads(args_str)
                tool_calls.append(ToolCall(name=tool_name, arguments=args))
            except (json.JSONDecodeError, ValueError):
                continue

    # Extract thought / reasoning trace
    thought = ""
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    if think_start != -1 and think_end != -1 and think_end > think_start:
        thought = text[think_start + len("<think>") : think_end].strip()
    elif tool_calls:
        # Text preceding the first tool call
        first_call_pos = text.find(TOOL_CALL_OPEN)
        if first_call_pos != -1:
            thought = text[:first_call_pos].strip()
        else:
            action_pos = text.find("Action:")
            if action_pos != -1:
                thought = text[:action_pos].strip()

    # Clean leading "Thought:" label if present
    if thought.startswith("Thought:"):
        thought = thought[len("Thought:") :].strip()

    # If no tool calls, response is the final answer
    final_answer: str | None = None
    if not tool_calls:
        if think_end != -1:
            final_answer = text[think_end + len("</think>") :].strip()
        else:
            final_answer = text.strip()

    return thought, tool_calls, final_answer


def _eos_token_ids(lm: Any) -> set[int]:
    """Inspect LM adapter tokenizer for EOS and turn-ending token IDs."""
    ids: set[int] = set()
    tok = getattr(lm, "tokenizer", None)
    if tok is not None:
        v = getattr(tok, "eos_token_id", None)
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(int(x) for x in v)
        inner = getattr(tok, "_tokenizer", tok)
        for name in ("<|im_end|>", "<|endoftext|>", "</s>"):
            try:
                tid = inner.convert_tokens_to_ids(name)
                if isinstance(tid, int) and tid >= 0:
                    ids.add(tid)
            except (AttributeError, KeyError, ValueError):
                pass
    return ids


# --------------------------------------------------------------------------- #
# Multi-Turn Agent Runtime
# --------------------------------------------------------------------------- #

class AgentRuntime:
    """Autonomous ReAct coding agent execution loop with anti-spin circuit breakers.

    Drives multi-turn Thought -> Action -> Tool Observation interactions against
    repositories and LMAdapter backends, collecting comprehensive trajectory telemetry.
    """

    def __init__(
        self,
        lm: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | WorkspaceToolExecutor | None = None,
        workspace_dir: str | Path | None = None,
        system_prompt: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_gen_tokens: int = DEFAULT_MAX_GEN_TOKENS,
        temperature: float = 0.0,
        redirection_threshold: int = DEFAULT_REDIRECTION_THRESHOLD,
        termination_threshold: int = DEFAULT_TERMINATION_THRESHOLD,
        trajectory_logger: TrajectoryLogger | Callable[[AgentRunResult], None] | str | Path | None = None,
        compactor: Any | None = None,
        plan_manager: PlanManager | None = None,
        planning_policy: str | PlanningPolicy | None = None,
    ) -> None:
        self.lm = lm
        self.tools = list(tools) if tools is not None else list(CODING_AGENT_TOOLS) + [WEB_SEARCH_TOOL, FETCH_WEB_PAGE_TOOL]
        self.workspace_dir = workspace_dir
        if tool_executor is not None:
            self.tool_executor = tool_executor
        else:
            self.tool_executor = WorkspaceToolExecutor(workspace_dir=workspace_dir)

        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_turns = int(max_turns)
        self.max_gen_tokens = int(max_gen_tokens)
        self.temperature = float(temperature)

        self.circuit_breaker = AntiSpinCircuitBreaker(
            redirection_threshold=redirection_threshold,
            termination_threshold=termination_threshold,
        )

        # Configure trajectory logger
        if isinstance(trajectory_logger, (str, Path)):
            self.trajectory_logger: TrajectoryLogger | None = JsonlTrajectoryLogger(trajectory_logger)
        elif isinstance(trajectory_logger, TrajectoryLogger):
            self.trajectory_logger = trajectory_logger
        elif callable(trajectory_logger):
            self.trajectory_logger = trajectory_logger  # type: ignore
        else:
            self.trajectory_logger = None

        self.compactor = compactor

        if plan_manager is not None:
            self.plan_manager = plan_manager
        elif planning_policy is not None:
            self.plan_manager = PlanManager(policy=planning_policy)
        else:
            self.plan_manager = None

        if self.plan_manager is not None:
            if not any(t.get("name") == "update_plan" for t in self.tools):
                self.tools.append(UPDATE_PLAN_TOOL)
            if hasattr(self.tool_executor, "plan_manager") and self.tool_executor.plan_manager is None:
                self.tool_executor.plan_manager = self.plan_manager

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> tuple[Any, bool, float]:
        """Execute a tool call using configured executor, handling exceptions safely."""
        t0 = time.monotonic()
        try:
            if name == "update_plan" and self.plan_manager is not None:
                res = self.plan_manager.execute_update_plan(arguments)
            elif hasattr(self.tool_executor, "execute") and callable(self.tool_executor.execute):
                res = self.tool_executor.execute(name, arguments)
            elif callable(self.tool_executor):
                res = self.tool_executor(name, arguments)
            else:
                res = {"error": f"Tool executor is not callable: {self.tool_executor}", "is_error": True}
        except Exception as e:  # noqa: BLE001
            res = {"error": f"Exception during {name} execution: {e}", "is_error": True}

        duration_s = time.monotonic() - t0
        err = is_tool_failure(res)
        return res, err, duration_s

    def _generate_turn(
        self,
        messages: list[dict[str, Any]],
        rng: np.random.Generator | None = None,
    ) -> tuple[str, int, int, float]:
        """Generate assistant response from LMAdapter or callable backend.

        Returns (completion_text, prompt_tokens, completion_tokens, duration_s).
        """
        t0 = time.monotonic()
        lm = self.lm
        if lm is None:
            raise ValueError("No LM backend provided to AgentRuntime")

        # 1. Callable interface (mock or high-level function)
        if callable(lm) and not hasattr(lm, "reset"):
            output = lm(messages)
            if isinstance(output, dict):
                text = str(output.get("content", ""))
            else:
                text = str(output)
            prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
            comp_tokens = len(text.split())
            return text, prompt_tokens, comp_tokens, time.monotonic() - t0

        # 2. Object with generate() or chat() method
        if hasattr(lm, "generate") and callable(lm.generate):
            text = lm.generate(messages)
            prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
            comp_tokens = len(str(text).split())
            return str(text), prompt_tokens, comp_tokens, time.monotonic() - t0

        # 3. LMAdapter interface (reset / step / decode)
        if hasattr(lm, "reset") and hasattr(lm, "step"):
            if hasattr(lm, "render_chat") and callable(lm.render_chat):
                context = lm.render_chat(messages)
            else:
                context = render(messages, add_generation_prompt=True)

            prompt_tokens = len(lm.encode(context)) if hasattr(lm, "encode") else len(context.split())
            logits = lm.reset(context)
            gen_ids: list[int] = []
            eos = _eos_token_ids(lm)

            for _ in range(self.max_gen_tokens):
                tok = sample(logits, temperature=self.temperature, rng=rng)
                if tok in eos:
                    break
                logits = lm.step(tok)
                gen_ids.append(tok)

            if hasattr(lm, "decode"):
                text = lm.decode(gen_ids)
            else:
                text = "".join(chr(i) for i in gen_ids)

            # Strip IM_END delimiter if present
            if text.endswith(IM_END):
                text = text[: -len(IM_END)].rstrip()
            elif text.endswith(CHAT_EOS):
                text = text[: -len(CHAT_EOS)].rstrip()

            return text, prompt_tokens, len(gen_ids), time.monotonic() - t0

        raise TypeError(f"Unsupported LM backend type: {type(lm)}")

    def run(
        self,
        task: str,
        *,
        max_turns: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> AgentRunResult:
        """Run the autonomous multi-turn ReAct coding agent execution loop."""
        turn_budget = int(max_turns) if max_turns is not None else self.max_turns
        start_time = time.monotonic()

        # Reset anti-spin circuit breaker and session safety state for this run (#351)
        self.circuit_breaker.reset()
        if hasattr(self.tool_executor, "reset") and callable(self.tool_executor.reset):
            self.tool_executor.reset()
        elif hasattr(self.tool_executor, "read_registry") and hasattr(self.tool_executor.read_registry, "clear"):
            self.tool_executor.read_registry.clear()
        if self.plan_manager is not None:
            self.plan_manager.reset()

        # Build initial messages
        system_content = render_tool_system(self.tools, preamble=self.system_prompt)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": task},
        ]

        turns: list[AgentTurn] = []
        all_events: list[dict[str, Any]] = [
            {"event": "run_start", "task": task, "max_turns": turn_budget, "timestamp": time.time()}
        ]

        final_answer: str | None = None
        status = "completed"
        success = True
        failure_reason: str | None = None
        circuit_breaker_triggered = False
        exit_gate_triggered = False

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tool_wall_s = 0.0

        for turn_idx in range(1, turn_budget + 1):
            t_turn_start = time.monotonic()
            turn_events: list[dict[str, Any]] = [
                {"event": "turn_start", "turn": turn_idx, "timestamp": time.time()}
            ]
            phase_transitions: list[str] = ["thought"]

            # Staged context compaction (#350)
            if self.compactor is not None:
                if hasattr(self.compactor, "compact_with_report"):
                    messages, comp_report = self.compactor.compact_with_report(messages)
                    if comp_report.compacted:
                        phase_transitions.append("compaction")
                        turn_events.append({
                            "event": "context_compacted",
                            "elision_applied": comp_report.elision_applied,
                            "summarization_applied": comp_report.summarization_applied,
                            "initial_tokens": comp_report.initial_tokens,
                            "final_tokens": comp_report.final_tokens,
                            "elided_observations": comp_report.elided_observations,
                            "summarized_turns": comp_report.summarized_turns,
                        })
                elif hasattr(self.compactor, "compact"):
                    messages = self.compactor.compact(messages)
                elif callable(self.compactor):
                    messages = self.compactor(messages)

            # Out-of-history plan injection into prompt conditioning (#352)
            if self.plan_manager is not None:
                conditioned_messages = self.plan_manager.inject_plan(messages)
            else:
                conditioned_messages = messages

            # Generate model turn
            try:
                response_text, p_tokens, c_tokens, gen_wall_s = self._generate_turn(conditioned_messages, rng=rng)
            except Exception as e:  # noqa: BLE001
                status = "failed"
                success = False
                failure_reason = f"Generation failure at turn {turn_idx}: {e}"
                turn_events.append({"event": "generation_error", "error": str(e)})
                break

            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens

            # Parse thought, tool calls, and final answer
            thought, tool_calls, answer = parse_react_response(response_text)
            if thought:
                turn_events.append({"event": "thought_generated", "length": len(thought)})

            # Case A: Agent finishes without further tool calls
            if not tool_calls:
                # Capability-adaptive scaffolding: detect and prevent premature task abort (#352)
                premature_abort = False
                abort_reason = None
                if self.plan_manager is not None:
                    premature_abort, abort_reason = self.plan_manager.check_premature_abort(turn_idx)

                if premature_abort and abort_reason:
                    phase_transitions.append("scaffolding_redirection")
                    turn_events.append({
                        "event": "premature_abort_prevented",
                        "turn": turn_idx,
                        "reason": abort_reason,
                    })
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": abort_reason})
                    turn = AgentTurn(
                        turn_index=turn_idx,
                        thought=thought,
                        action=None,
                        tool_calls=[],
                        tool_observations=[],
                        prompt_tokens=p_tokens,
                        completion_tokens=c_tokens,
                        generation_wall_s=gen_wall_s,
                        tool_wall_s=0.0,
                        turn_wall_s=time.monotonic() - t_turn_start,
                        phase_transitions=phase_transitions,
                        events=turn_events,
                    )
                    turns.append(turn)
                    all_events.extend(turn_events)
                    if self.trajectory_logger and hasattr(self.trajectory_logger, "log_turn"):
                        self.trajectory_logger.log_turn(turn)
                    continue

                final_answer = answer or response_text
                messages.append({"role": "assistant", "content": response_text})
                phase_transitions.append("completion")
                turn_events.append({"event": "task_completed", "turn": turn_idx})

                turn = AgentTurn(
                    turn_index=turn_idx,
                    thought=thought,
                    action=None,
                    tool_calls=[],
                    tool_observations=[],
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    generation_wall_s=gen_wall_s,
                    tool_wall_s=0.0,
                    turn_wall_s=time.monotonic() - t_turn_start,
                    phase_transitions=phase_transitions,
                    events=turn_events,
                )
                turns.append(turn)
                all_events.extend(turn_events)
                if self.trajectory_logger and hasattr(self.trajectory_logger, "log_turn"):
                    self.trajectory_logger.log_turn(turn)
                break

            # Case B: Agent makes one or more tool calls
            phase_transitions.append("action")
            messages.append({"role": "assistant", "content": response_text})

            phase_transitions.append("observation")
            tool_obs_list: list[ToolObservation] = []
            turn_tool_wall_s = 0.0
            trip_breaker_this_turn = False
            redirection_reminders: list[str] = []

            for call in tool_calls:
                turn_events.append({
                    "event": "tool_call",
                    "name": call.name,
                    "arguments": call.arguments,
                })

                # Check circuit breaker before execution for redirection
                cb_call_status = self.circuit_breaker.record_call(call.name, call.arguments)
                if cb_call_status.should_redirect and cb_call_status.redirection_message:
                    redirection_reminders.append(cb_call_status.redirection_message)
                    turn_events.append({
                        "event": "circuit_breaker_redirection",
                        "tool": call.name,
                        "repeats": cb_call_status.consecutive_calls,
                        "message": cb_call_status.redirection_message,
                    })

                # Execute tool
                result, is_err, dur_s = self._execute_tool(call.name, call.arguments)
                turn_tool_wall_s += dur_s
                total_tool_wall_s += dur_s

                # Check circuit breaker after execution outcome
                cb_res_status = self.circuit_breaker.record_result(call.name, call.arguments, is_error=is_err)

                redir_warn = None
                if cb_res_status.should_redirect and cb_res_status.redirection_message:
                    redir_warn = cb_res_status.redirection_message
                    if redir_warn not in redirection_reminders:
                        redirection_reminders.append(redir_warn)
                    turn_events.append({
                        "event": "circuit_breaker_redirection",
                        "tool": call.name,
                        "repeats": cb_res_status.consecutive_failures or cb_res_status.consecutive_calls,
                        "message": redir_warn,
                    })

                if cb_res_status.should_terminate:
                    trip_breaker_this_turn = True
                    circuit_breaker_triggered = True
                    status = "circuit_breaker_tripped"
                    failure_reason = cb_res_status.termination_reason
                    turn_events.append({
                        "event": "circuit_breaker_terminated",
                        "tool": call.name,
                        "repeats": cb_res_status.consecutive_failures or cb_res_status.consecutive_calls,
                        "reason": cb_res_status.termination_reason,
                    })

                tool_obs_list.append(
                    ToolObservation(
                        name=call.name,
                        output=result,
                        is_error=is_err,
                        wall_s=dur_s,
                        redirection_warning=redir_warn,
                    )
                )

                turn_events.append({
                    "event": "tool_executed",
                    "name": call.name,
                    "is_error": is_err,
                    "duration_s": dur_s,
                })

                if call.name == "update_plan":
                    turn_events.append({
                        "event": "plan_updated",
                        "completed_items": self.plan_manager.completed_count if self.plan_manager else 0,
                        "total_items": self.plan_manager.total_count if self.plan_manager else 0,
                        "is_complete": self.plan_manager.is_complete if self.plan_manager else False,
                    })

                # Check completion exit-gate policy (#352)
                if self.plan_manager is not None and self.plan_manager.should_exit_gate_terminate():
                    exit_gate_triggered = True
                    status = "completed"
                    success = True
                    final_answer = (
                        answer or f"Task completed: all {self.plan_manager.total_count} plan checklist items verified."
                    )
                    phase_transitions.append("exit_gate")
                    turn_events.append({
                        "event": "exit_gate_terminated",
                        "turn": turn_idx,
                        "completed_items": self.plan_manager.completed_count,
                        "total_items": self.plan_manager.total_count,
                    })
                    break

                if trip_breaker_this_turn:
                    break

            # Format tool observations into user message
            obs_payloads = [obs.output for obs in tool_obs_list]
            tool_resp_str = format_tool_response(obs_payloads)

            # If redirection reminder was triggered, append to user turn context
            if redirection_reminders:
                reminder_block = "\n\n" + "\n".join(redirection_reminders)
                tool_resp_str += reminder_block

            if turn_idx == 1 and not exit_gate_triggered and self.plan_manager is not None:
                plan_reminder = self.plan_manager.get_turn1_reminder()
                if plan_reminder:
                    tool_resp_str += f"\n\n{plan_reminder}"
                    turn_events.append({"event": "plan_scaffolding_reminder", "message": plan_reminder})

            messages.append({"role": "user", "content": tool_resp_str})

            turn = AgentTurn(
                turn_index=turn_idx,
                thought=thought,
                action=tool_calls[0].name if tool_calls else None,
                tool_calls=tool_calls,
                tool_observations=tool_obs_list,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                generation_wall_s=gen_wall_s,
                tool_wall_s=turn_tool_wall_s,
                turn_wall_s=time.monotonic() - t_turn_start,
                phase_transitions=phase_transitions,
                events=turn_events,
            )
            turns.append(turn)
            all_events.extend(turn_events)

            if self.trajectory_logger and hasattr(self.trajectory_logger, "log_turn"):
                self.trajectory_logger.log_turn(turn)

            if trip_breaker_this_turn or exit_gate_triggered:
                if trip_breaker_this_turn:
                    success = False
                break
        else:
            # Reached max turns without natural termination or breaker trip
            status = "budget_exhausted"
            success = False
            failure_reason = f"max_turns ({turn_budget}) budget exhausted"
            all_events.append({"event": "budget_exhausted", "max_turns": turn_budget})

        total_wall_s = time.monotonic() - start_time
        all_events.append({
            "event": "run_complete",
            "status": status,
            "success": success,
            "total_turns": len(turns),
            "total_wall_s": total_wall_s,
        })

        run_result = AgentRunResult(
            task=task,
            turns=turns,
            messages=messages,
            final_answer=final_answer,
            status=status,
            success=success,
            failure_reason=failure_reason,
            circuit_breaker_triggered=circuit_breaker_triggered,
            exit_gate_triggered=exit_gate_triggered,
            plan_state=self.plan_manager.to_dict() if self.plan_manager is not None else None,
            total_wall_s=total_wall_s,
            total_tool_wall_s=total_tool_wall_s,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            events=all_events,
        )

        if self.trajectory_logger:
            if hasattr(self.trajectory_logger, "log_trajectory"):
                self.trajectory_logger.log_trajectory(run_result)
            elif callable(self.trajectory_logger):
                self.trajectory_logger(run_result)

        return run_result


def run_agent_loop(
    task: str,
    lm: Any,
    *,
    workspace_dir: str | Path | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | WorkspaceToolExecutor | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_gen_tokens: int = DEFAULT_MAX_GEN_TOKENS,
    temperature: float = 0.0,
    redirection_threshold: int = DEFAULT_REDIRECTION_THRESHOLD,
    termination_threshold: int = DEFAULT_TERMINATION_THRESHOLD,
    trajectory_logger: TrajectoryLogger | Callable[[AgentRunResult], None] | str | Path | None = None,
    system_prompt: str | None = None,
    compactor: Any | None = None,
    plan_manager: PlanManager | None = None,
    planning_policy: str | PlanningPolicy | None = None,
) -> AgentRunResult:
    """Convenience functional wrapper around AgentRuntime."""
    runtime = AgentRuntime(
        lm=lm,
        tools=tools,
        tool_executor=tool_executor,
        workspace_dir=workspace_dir,
        system_prompt=system_prompt,
        max_turns=max_turns,
        max_gen_tokens=max_gen_tokens,
        temperature=temperature,
        redirection_threshold=redirection_threshold,
        termination_threshold=termination_threshold,
        trajectory_logger=trajectory_logger,
        compactor=compactor,
        plan_manager=plan_manager,
        planning_policy=planning_policy,
    )
    return runtime.run(task)
