"""Unit tests for harness safety gates, path containment, and post-edit diagnostics (#351)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.runtime import AgentRuntime, WorkspaceToolExecutor
from src.agent.safety import (
    FileReadRecord,
    FileReadRegistry,
    format_diagnostic_summary,
    resolve_safe_workspace_path,
    run_file_diagnostics,
)

# --------------------------------------------------------------------------- #
# FileReadRegistry & SHA-256 Content Hash Tests
# --------------------------------------------------------------------------- #

def test_file_read_registry_registration_and_hash():
    """Verify registry records reads, writes, and computes accurate SHA-256 hashes."""
    registry = FileReadRegistry()
    assert len(registry) == 0
    assert not registry.is_read("src/main.py")

    content = "print('hello world')\n"
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    rec = registry.register_read("src/main.py", content)
    assert rec.path == "src/main.py"
    assert rec.sha256 == expected_hash
    assert rec.size_bytes == len(content.encode("utf-8"))
    assert rec.to_dict()["sha256"] == expected_hash
    assert registry.is_read("src/main.py")
    assert registry.get_hash("src/main.py") == expected_hash
    assert registry.get_record("src/main.py") is not None
    assert "src/main.py" in registry
    assert len(registry) == 1
    assert "src/main.py" in registry.all_records()

    # Update on write
    new_content = "print('updated')\n"
    new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
    rec2 = registry.register_write("src/main.py", new_content)
    assert rec2.sha256 == new_hash
    assert registry.get_hash("src/main.py") == new_hash

    # Clear registry
    registry.clear()
    assert len(registry) == 0
    assert not registry.is_read("src/main.py")


def test_file_read_record_dataclass():
    """Verify FileReadRecord dataclass properties."""
    rec = FileReadRecord(path="a.py", sha256="abc", read_at=100.0, size_bytes=10)
    d = rec.to_dict()
    assert d == {"path": "a.py", "sha256": "abc", "read_at": 100.0, "size_bytes": 10}


# --------------------------------------------------------------------------- #
# Workspace Containment & Jailbreak Prevention Tests
# --------------------------------------------------------------------------- #

def test_resolve_safe_workspace_path_valid_paths(tmp_path: Path):
    """Verify valid relative and absolute paths within workspace root are allowed."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sub = workspace / "src"
    sub.mkdir()
    file1 = sub / "app.py"
    file1.write_text("print(1)", encoding="utf-8")

    # Relative path
    p, err = resolve_safe_workspace_path("src/app.py", workspace)
    assert err is None
    assert p == file1.resolve()

    # Relative path with internal dot-dots
    p, err = resolve_safe_workspace_path("src/../src/app.py", workspace)
    assert err is None
    assert p == file1.resolve()

    # Absolute path within workspace
    p, err = resolve_safe_workspace_path(str(file1), workspace)
    assert err is None
    assert p == file1.resolve()


def test_resolve_safe_workspace_path_traversal_jailbreak(tmp_path: Path):
    """Verify directory traversal attempts escaping workspace boundary are rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Direct parent traversal
    p, err = resolve_safe_workspace_path("../outside.txt", workspace)
    assert p is None
    assert "escapes workspace boundary" in (err or "")

    # Multi-level parent traversal
    p, err = resolve_safe_workspace_path("foo/../../../../etc/passwd", workspace)
    assert p is None
    assert "escapes workspace boundary" in (err or "")

    # Absolute path outside workspace
    p, err = resolve_safe_workspace_path("/etc/passwd", workspace)
    assert p is None
    assert "escapes workspace boundary" in (err or "")

    # Invalid paths: empty and null bytes
    p, err = resolve_safe_workspace_path("", workspace)
    assert p is None
    assert "invalid path" in (err or "")

    p, err = resolve_safe_workspace_path("file" + chr(0) + ".py", workspace)
    assert p is None
    assert "invalid path" in (err or "")


def test_resolve_safe_workspace_path_symlink_jailbreak(tmp_path: Path):
    """Verify symlinks pointing outside the repository root are safely caught and rejected."""
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("sensitive data", encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # 1. Symlink file pointing outside workspace
    evil_file_link = workspace / "evil_file.txt"
    os.symlink(outside_file, evil_file_link)

    p, err = resolve_safe_workspace_path("evil_file.txt", workspace)
    assert p is None
    assert "escapes workspace boundary" in (err or "")

    # 2. Symlink directory pointing outside workspace
    evil_dir_link = workspace / "evil_dir"
    os.symlink(outside_dir, evil_dir_link)

    p, err = resolve_safe_workspace_path("evil_dir/secret.txt", workspace)
    assert p is None
    assert "escapes workspace boundary" in (err or "")

    # 3. Non-existent file under symlinked outside directory
    p, err = resolve_safe_workspace_path("evil_dir/new_file.txt", workspace)
    assert p is None
    assert "escapes workspace boundary" in (err or "")

    # 4. Internal symlink pointing inside workspace should be allowed
    inside_dir = workspace / "internal"
    inside_dir.mkdir()
    target_inside = inside_dir / "safe.txt"
    target_inside.write_text("safe content", encoding="utf-8")

    safe_link = workspace / "safe_link.txt"
    os.symlink(target_inside, safe_link)

    p, err = resolve_safe_workspace_path("safe_link.txt", workspace)
    assert err is None
    assert p == target_inside.resolve()


# --------------------------------------------------------------------------- #
# Read-Before-Write Safety Gate Tests
# --------------------------------------------------------------------------- #

def test_read_before_write_gate_rejects_unread_edit(tmp_path: Path):
    """Verify edit_file on an existing file that has not been read in the session is rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "target.py"
    target.write_text("val = 1\n", encoding="utf-8")

    executor = WorkspaceToolExecutor(workspace_dir=workspace)

    # Edit without viewing first
    edit_res = executor.edit_file("target.py", old_str="val = 1", new_str="val = 2")
    assert edit_res.get("is_error") is True
    assert edit_res.get("safety_violation") == "unread_file"
    assert "Read-before-write safety violation" in edit_res["error"]
    assert "has not been read in this session" in edit_res["error"]
    assert target.read_text(encoding="utf-8") == "val = 1\n"

    # Now view the file first
    view_res = executor.view_file("target.py")
    assert not view_res.get("is_error")
    assert executor.read_registry.is_read("target.py")

    # Second edit now succeeds
    edit_res2 = executor.edit_file("target.py", old_str="val = 1", new_str="val = 2")
    assert not edit_res2.get("is_error")
    assert edit_res2["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "val = 2\n"


def test_read_before_write_gate_rejects_unread_overwrite(tmp_path: Path):
    """Verify write_file overwriting an existing file not read in the session is rejected."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "existing.py"
    target.write_text("original = True\n", encoding="utf-8")

    executor = WorkspaceToolExecutor(workspace_dir=workspace)

    # Overwrite without viewing first
    write_res = executor.write_file("existing.py", "new = True\n")
    assert write_res.get("is_error") is True
    assert write_res.get("safety_violation") == "unread_file"
    assert "already exists but has not been read in this session" in write_res["error"]
    assert target.read_text(encoding="utf-8") == "original = True\n"

    # View file
    executor.view_file("existing.py")
    assert executor.read_registry.is_read("existing.py")

    # Overwrite now succeeds
    write_res2 = executor.write_file("existing.py", "new = True\n")
    assert not write_res2.get("is_error")
    assert write_res2["status"] == "ok"
    assert target.read_text(encoding="utf-8") == "new = True\n"


def test_write_new_file_succeeds_and_registers_in_read_registry(tmp_path: Path):
    """Verify write_file creating a new non-existent file succeeds and registers for future edits."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    executor = WorkspaceToolExecutor(workspace_dir=workspace)

    # Write brand-new file
    write_res = executor.write_file("new_module.py", "x = 42\n")
    assert not write_res.get("is_error")
    assert write_res["status"] == "ok"
    assert executor.read_registry.is_read("new_module.py")

    # Subsequent edit succeeds without explicit view_file because write registered it
    edit_res = executor.edit_file("new_module.py", old_str="42", new_str="100")
    assert not edit_res.get("is_error")
    assert edit_res["status"] == "ok"
    assert (workspace / "new_module.py").read_text(encoding="utf-8") == "x = 100\n"


def test_session_reset_clears_read_registry_requiring_re_read(tmp_path: Path):
    """Verify executor.reset() clears session read records, re-enabling safety protection."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "file.py"
    target.write_text("a = 1\n", encoding="utf-8")

    executor = WorkspaceToolExecutor(workspace_dir=workspace)
    executor.view_file("file.py")
    assert executor.read_registry.is_read("file.py")

    # Reset session state
    executor.reset()
    assert not executor.read_registry.is_read("file.py")

    # Edit is rejected until viewed again in the new session
    edit_res = executor.edit_file("file.py", old_str="a = 1", new_str="a = 2")
    assert edit_res.get("is_error") is True
    assert edit_res.get("safety_violation") == "unread_file"


# --------------------------------------------------------------------------- #
# Post-Edit Diagnostic Feedback Tests
# --------------------------------------------------------------------------- #

def test_run_file_diagnostics_direct(tmp_path: Path):
    """Verify run_file_diagnostics helper directly."""
    f = tmp_path / "test.py"
    f.write_text("def good():\n    return 1\n", encoding="utf-8")
    diags = run_file_diagnostics(f)
    assert diags == []

    # Custom provider
    custom_diags = run_file_diagnostics(
        f,
        custom_provider=lambda p, c: [{"line": 1, "column": 1, "message": "custom", "code": "C01", "severity": "error", "source": "test"}]
    )
    assert len(custom_diags) == 1
    assert custom_diags[0]["code"] == "C01"


def test_post_edit_diagnostics_python_clean(tmp_path: Path):
    """Verify clean Python edits attach diagnostics: [] to tool observation."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "calc.py"
    f.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    executor = WorkspaceToolExecutor(workspace_dir=workspace)
    executor.view_file("calc.py")

    res = executor.edit_file("calc.py", old_str="a + b", new_str="a + b + 0")
    assert not res.get("is_error")
    assert res["status"] == "ok"
    assert "diagnostics" in res
    assert res["diagnostics"] == []
    assert res["message"] == "Successfully edited calc.py"


def test_post_edit_diagnostics_python_syntax_error(tmp_path: Path):
    """Verify Python syntax errors introduced by edit_file are immediately reported in diagnostics."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "syntax_test.py"
    f.write_text("def good():\n    return 42\n", encoding="utf-8")

    executor = WorkspaceToolExecutor(workspace_dir=workspace)
    executor.view_file("syntax_test.py")

    # Introduce syntax error
    res = executor.edit_file("syntax_test.py", old_str="def good():", new_str="def broken(:")
    assert not res.get("is_error")  # File mutation succeeded
    assert res["status"] == "ok"
    assert len(res["diagnostics"]) > 0

    diag = res["diagnostics"][0]
    assert diag["line"] == 1
    assert "column" in diag
    assert diag["severity"] == "error"
    assert "Static analysis:" in res["message"]


def test_post_write_diagnostics_python_syntax_error(tmp_path: Path):
    """Verify Python syntax errors in write_file are immediately captured in diagnostics."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    executor = WorkspaceToolExecutor(workspace_dir=workspace)
    res = executor.write_file("broken.py", "if True\n    pass\n")
    assert not res.get("is_error")
    assert len(res["diagnostics"]) > 0

    diag = res["diagnostics"][0]
    assert diag["line"] == 1
    assert "Syntax" in diag["message"] or "invalid-syntax" in diag["code"]
    assert "Static analysis:" in res["message"]


def test_post_edit_diagnostics_typescript_lsp(tmp_path: Path):
    """Verify TypeScript LSP service (TsLspService) diagnostics are captured and attached."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "index.ts"
    f.write_text("const x: number = 10;\n", encoding="utf-8")

    @dataclass
    class FakeDiagnostic:
        code: str
        line: int
        col: int
        message: str
        severity: int = 1
        source: str = "ts"

    class FakeTsLspService:
        def __init__(self):
            self.last_path = None
            self.last_text = None

        def update(self, path: str, text: str):
            self.last_path = path
            self.last_text = text

        def diagnostics(self, path: str):
            if "bad_type" in (self.last_text or ""):
                return [
                    FakeDiagnostic(
                        code="TS2322",
                        line=2,
                        col=5,
                        message="Type 'string' is not assignable to type 'number'.",
                        severity=1,
                    )
                ]
            return []

    fake_lsp = FakeTsLspService()
    executor = WorkspaceToolExecutor(workspace_dir=workspace, ts_lsp_service=fake_lsp)
    executor.view_file("index.ts")

    # Clean edit
    clean_res = executor.edit_file("index.ts", old_str="10", new_str="20")
    assert not clean_res.get("is_error")
    assert clean_res["diagnostics"] == []

    # Edit introducing type error
    err_res = executor.edit_file("index.ts", old_str="20", new_str="'bad_type'")
    assert not err_res.get("is_error")
    assert len(err_res["diagnostics"]) == 1
    d = err_res["diagnostics"][0]
    assert d["code"] == "TS2322"
    assert d["line"] == 2
    assert d["column"] == 5
    assert "Type 'string' is not assignable" in d["message"]
    assert "TS2322" in err_res["message"]


def test_post_edit_diagnostics_json_syntax(tmp_path: Path):
    """Verify JSON formatting errors are caught and attached to diagnostics."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "config.json"
    f.write_text('{"key": "value"}\n', encoding="utf-8")

    executor = WorkspaceToolExecutor(workspace_dir=workspace)
    executor.view_file("config.json")

    res = executor.edit_file("config.json", old_str='"value"', new_str="not_valid_json")
    assert not res.get("is_error")
    assert len(res["diagnostics"]) == 1
    assert res["diagnostics"][0]["code"] == "JSONDecodeError"
    assert res["diagnostics"][0]["source"] == "json_parser"


def test_diagnostic_summary_formatting():
    """Verify format_diagnostic_summary formatting."""
    assert format_diagnostic_summary([]) == ""

    diags = [
        {"line": 4, "code": "E999", "message": "invalid syntax", "severity": "error"}
    ]
    summary = format_diagnostic_summary(diags)
    assert "1 issue(s) detected" in summary
    assert "Line 4: [E999] invalid syntax" in summary


# --------------------------------------------------------------------------- #
# End-to-End AgentRuntime Integration Tests
# --------------------------------------------------------------------------- #

def test_agent_runtime_react_loop_with_safety_and_diagnostics(tmp_path: Path):
    """Verify multi-turn agent ReAct loop with safety rejection recovery and diagnostics."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    code_file = workspace / "app.py"
    code_file.write_text("def calc():\n    return 42\n", encoding="utf-8")

    executor = WorkspaceToolExecutor(workspace_dir=workspace)
    call_turn = 0

    def mock_agent_lm(messages: list[dict[str, Any]]) -> str:
        nonlocal call_turn
        call_turn += 1

        if call_turn == 1:
            # Turn 1: Blind edit attempt without reading first -> Should trigger safety gate!
            return (
                "Let me update calc() directly.\n"
                "<tool_call>\n"
                '{"name": "edit_file", "arguments": {"path": "app.py", "old_str": "42", "new_str": "100"}}\n'
                "</tool_call>"
            )
        elif call_turn == 2:
            # Turn 2: Verify agent received safety rejection and self-corrects to view_file
            last_msg = messages[-1]["content"]
            assert "Read-before-write safety violation" in last_msg
            return (
                "Safety gate prompted me to read app.py first. Reading file.\n"
                "<tool_call>\n"
                '{"name": "view_file", "arguments": {"path": "app.py"}}\n'
                "</tool_call>"
            )
        elif call_turn == 3:
            # Turn 3: Now edit after viewing
            last_msg = messages[-1]["content"]
            assert "def calc():" in last_msg
            return (
                "Now editing app.py safely.\n"
                "<tool_call>\n"
                '{"name": "edit_file", "arguments": {"path": "app.py", "old_str": "42", "new_str": "100"}}\n'
                "</tool_call>"
            )
        else:
            # Turn 4: Final completion
            last_msg = messages[-1]["content"]
            assert "Successfully edited app.py" in last_msg
            return "<think>Task complete.</think>Successfully updated calc to return 100."

    runtime = AgentRuntime(lm=mock_agent_lm, tool_executor=executor, max_turns=6)
    result = runtime.run("Update calc in app.py to return 100")

    assert result.success
    assert result.status == "completed"
    assert result.total_turns == 4
    assert (workspace / "app.py").read_text(encoding="utf-8") == "def calc():\n    return 100\n"

    # Inspect turn 1 observation: safety rejection
    obs1 = result.turns[0].tool_observations[0]
    assert obs1.is_error is True
    assert "Read-before-write safety violation" in obs1.output["error"]

    # Inspect turn 3 observation: edit success with diagnostics attached
    obs3 = result.turns[2].tool_observations[0]
    assert obs3.is_error is False
    assert obs3.output["status"] == "ok"
    assert "diagnostics" in obs3.output
    assert obs3.output["diagnostics"] == []
