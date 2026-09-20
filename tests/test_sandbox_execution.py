"""Unit tests for containerized execution sandboxes and verifier integration (#353)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.runtime import AgentRuntime, WorkspaceToolExecutor
from src.runtime.sandbox import (
    ContainerSandbox,
    ExecutionResult,
    LocalSandboxProvider,
    SandboxConfig,
    SandboxProvider,
    create_sandbox,
)
from src.train.verifiers import CodeVerifier

# --------------------------------------------------------------------------- #
# ExecutionResult and SandboxConfig Tests
# --------------------------------------------------------------------------- #


def test_execution_result_properties():
    """Verify ExecutionResult fields and is_error logic."""
    success = ExecutionResult(exit_code=0, stdout="ok", duration_s=0.1)
    assert not success.is_error
    assert not success.timed_out
    assert success.stdout == "ok"

    failure = ExecutionResult(exit_code=1, stderr="err", duration_s=0.1)
    assert failure.is_error
    assert not failure.timed_out

    timeout = ExecutionResult(
        exit_code=-1, timed_out=True, error="timed out", duration_s=5.0
    )
    assert timeout.is_error
    assert timeout.timed_out
    assert timeout.error == "timed out"


def test_sandbox_config_defaults():
    """Verify default resource limits and sandbox configuration."""
    config = SandboxConfig()
    assert config.image == "python:3.12-slim"
    assert config.engine == "docker"
    assert config.container_workdir == "/workspace"
    assert config.timeout_s == 30.0
    assert config.cpu_limit == 2.0
    assert config.memory_limit == "2g"
    assert config.pids_limit == 256
    assert config.network == "none"
    assert not config.read_only
    assert config.cap_drop == ["ALL"]
    assert not config.persistent


# --------------------------------------------------------------------------- #
# LocalSandboxProvider Tests
# --------------------------------------------------------------------------- #


def test_local_sandbox_execution(tmp_path: Path):
    """Verify LocalSandboxProvider executes commands, bash, and python."""
    provider = LocalSandboxProvider(workspace_dir=tmp_path, timeout_s=5.0)
    assert provider.is_local
    assert not provider.is_container

    # 1. run_command
    res = provider.run_command(["echo", "hello local"])
    assert res.exit_code == 0
    assert "hello local" in res.stdout
    assert not res.is_error

    # 2. run_bash
    test_file = tmp_path / "hello.txt"
    bash_res = provider.run_bash(f"echo 'sample' > {test_file.name}", cwd=tmp_path)
    assert bash_res.exit_code == 0
    assert test_file.read_text(encoding="utf-8").strip() == "sample"

    # 3. run_python
    py_res = provider.run_python("print(6 * 7)")
    assert py_res.exit_code == 0
    assert py_res.stdout.strip() == "42"


def test_local_sandbox_timeout():
    """Verify LocalSandboxProvider timeout guard."""
    provider = LocalSandboxProvider(timeout_s=0.1)
    res = provider.run_command(["python3", "-c", "import time; time.sleep(1.0)"], timeout=0.1)
    assert res.exit_code == -1
    assert res.timed_out
    assert res.is_error
    assert "timed out" in (res.error or "").lower()


# --------------------------------------------------------------------------- #
# ContainerSandbox CLI Generation Tests
# --------------------------------------------------------------------------- #


def test_container_sandbox_build_run_args(tmp_path: Path):
    """Verify container run CLI argument generation with resource limits."""
    sandbox = ContainerSandbox(
        image="python:3.12-alpine",
        engine="docker",
        workspace_dir=tmp_path,
        container_workdir="/app",
        cpu_limit=1.5,
        memory_limit="1g",
        pids_limit=128,
        network="none",
        read_only=True,
        cap_drop=["ALL"],
        env={"TEST_VAR": "val1"},
    )

    args = sandbox.build_run_args(
        ["python3", "-c", "print(1)"],
        container_name="sb_test_1",
        cwd=tmp_path / "subdir",
        env={"EXTRA_VAR": "val2"},
    )

    # Validate engine and flags
    assert args[0] == "docker"
    assert args[1] == "run"
    assert "--rm" in args
    assert "--name" in args and args[args.index("--name") + 1] == "sb_test_1"
    assert "--cpus" in args and args[args.index("--cpus") + 1] == "1.5"
    assert "--memory" in args and args[args.index("--memory") + 1] == "1g"
    assert "--pids-limit" in args and args[args.index("--pids-limit") + 1] == "128"
    assert "--network" in args and args[args.index("--network") + 1] == "none"
    assert "--read-only" in args
    assert "--cap-drop" in args and args[args.index("--cap-drop") + 1] == "ALL"

    # Volume mount check
    expected_mount = f"{tmp_path.resolve()}:/app:ro"
    assert "-v" in args and args[args.index("-v") + 1] == expected_mount

    # Workdir check
    assert "-w" in args and args[args.index("-w") + 1] == "/app/subdir"

    # Environment variables
    assert "-e" in args
    assert "TEST_VAR=val1" in args
    assert "EXTRA_VAR=val2" in args

    # Image and command
    assert args[-4] == "python:3.12-alpine"
    assert args[-3:] == ["python3", "-c", "print(1)"]


def test_container_sandbox_podman_engine(tmp_path: Path):
    """Verify ContainerSandbox supports podman CLI engine."""
    sandbox = ContainerSandbox(
        engine="podman",
        workspace_dir=tmp_path,
    )
    args = sandbox.build_run_args(["ls"])
    assert args[0] == "podman"


# --------------------------------------------------------------------------- #
# ContainerSandbox Lifecycle & Mocking Tests
# --------------------------------------------------------------------------- #


def test_container_sandbox_mock_ephemeral_execution(tmp_path: Path):
    """Verify ephemeral container execution lifecycle with mocked subprocess."""
    sandbox = ContainerSandbox(workspace_dir=tmp_path, engine="docker")

    mock_proc = MagicMock(returncode=0, stdout="hello container\n", stderr="")
    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        res = sandbox.run_command(["echo", "hello container"])
        assert res.exit_code == 0
        assert res.stdout == "hello container\n"
        assert not res.is_error

        # Ensure subprocess.run was called with docker run --rm
        called_args = mock_run.call_args[0][0]
        assert called_args[0] == "docker"
        assert called_args[1] == "run"
        assert "--rm" in called_args


def test_container_sandbox_mock_ephemeral_timeout(tmp_path: Path):
    """Verify timeout triggers immediate container cleanup."""
    sandbox = ContainerSandbox(workspace_dir=tmp_path, engine="docker")

    # Simulate timeout on the first subprocess call, success on cleanup
    with patch(
        "subprocess.run",
        side_effect=[
            subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=5.0),
            MagicMock(returncode=0),
        ],
    ) as mock_run:
        res = sandbox.run_command(["sleep", "10"], timeout=5.0)
        assert res.timed_out
        assert res.is_error
        assert res.exit_code == -1

        # Second call should be docker rm -f <container_name>
        assert mock_run.call_count == 2
        cleanup_args = mock_run.call_args_list[1][0][0]
        assert cleanup_args[0] == "docker"
        assert cleanup_args[1] == "rm"
        assert "-f" in cleanup_args


def test_container_sandbox_mock_persistent_lifecycle(tmp_path: Path):
    """Verify persistent container start, exec, and stop lifecycle."""
    sandbox = ContainerSandbox(workspace_dir=tmp_path, persistent=True, engine="docker")

    with patch("subprocess.run") as mock_run:
        # Mock container startup
        mock_run.return_value = MagicMock(returncode=0, stdout="mock_container_id\n", stderr="")

        with sandbox:
            assert sandbox.container_id is not None
            assert sandbox._is_running

            # Exec command inside persistent container
            mock_run.return_value = MagicMock(returncode=0, stdout="exec ok", stderr="")
            res = sandbox.run_command(["pytest"])
            assert res.exit_code == 0
            assert res.stdout == "exec ok"

            # Check that exec arguments were used
            last_call = mock_run.call_args[0][0]
            assert last_call[0] == "docker"
            assert last_call[1] == "exec"

        # After exiting context, container should be stopped and cleaned up
        assert sandbox.container_id is None
        assert not sandbox._is_running


# --------------------------------------------------------------------------- #
# Factory `create_sandbox` Tests
# --------------------------------------------------------------------------- #


def test_create_sandbox_factory():
    """Verify create_sandbox correctly instantiates provider types."""
    # Local provider
    sb_local = create_sandbox("local")
    assert isinstance(sb_local, LocalSandboxProvider)
    assert sb_local.is_local

    sb_subproc = create_sandbox("subprocess")
    assert isinstance(sb_subproc, LocalSandboxProvider)

    # Container provider
    sb_docker = create_sandbox("docker")
    assert isinstance(sb_docker, ContainerSandbox)
    assert sb_docker.engine == "docker"
    assert sb_docker.is_container

    sb_podman = create_sandbox("podman")
    assert isinstance(sb_podman, ContainerSandbox)
    assert sb_podman.engine == "podman"

    # Environment variable override
    with patch.dict(os.environ, {"MONICA_SANDBOX_BACKEND": "docker"}):
        sb_auto = create_sandbox("auto")
        assert isinstance(sb_auto, ContainerSandbox)

    with pytest.raises(ValueError, match="Unknown sandbox backend"):
        create_sandbox("invalid_backend")


# --------------------------------------------------------------------------- #
# CodeVerifier Integration Tests
# --------------------------------------------------------------------------- #


def test_code_verifier_with_container_sandbox():
    """Verify CodeVerifier routes evaluation through container sandbox."""
    mock_sandbox = MagicMock(spec=ContainerSandbox)
    mock_sandbox.is_local = False
    mock_sandbox.is_container = True

    # 1. CodeVerifier enabled without explicit enabled=True when container sandbox provided
    cv = CodeVerifier(sandbox=mock_sandbox, timeout=2.0)

    # Mock all tests passing
    mock_sandbox.run_python.return_value = ExecutionResult(exit_code=0, stdout="pass")
    reward_full = cv.reward("def f(): return 1", ["assert f() == 1", "assert f() > 0"])
    assert reward_full == 1.0
    assert mock_sandbox.run_python.call_count == 2

    # Mock 1 of 2 tests failing
    mock_sandbox.run_python.reset_mock()
    mock_sandbox.run_python.side_effect = [
        ExecutionResult(exit_code=0, stdout="pass"),
        ExecutionResult(exit_code=1, stderr="AssertionError"),
    ]
    reward_partial = cv.reward("def f(): return 1", ["assert f() == 1", "assert f() == 2"])
    assert reward_partial == 0.5


def test_code_verifier_local_requires_opt_in():
    """Verify local execution (no sandbox or local sandbox) requires enabled=True."""
    # No sandbox, enabled=False -> raises
    with pytest.raises(RuntimeError, match="CodeVerifier is disabled"):
        CodeVerifier().reward("x = 1", ["assert x == 1"])

    # LocalSandboxProvider, enabled=False -> raises
    local_sb = LocalSandboxProvider()
    with pytest.raises(RuntimeError, match="CodeVerifier is disabled"):
        CodeVerifier(sandbox=local_sb, enabled=False).reward("x = 1", ["assert x == 1"])

    # LocalSandboxProvider, enabled=True -> executes
    cv_local = CodeVerifier(sandbox=local_sb, enabled=True)
    assert cv_local.reward("x = 1", ["assert x == 1"]) == 1.0


# --------------------------------------------------------------------------- #
# WorkspaceToolExecutor & AgentRuntime Integration Tests
# --------------------------------------------------------------------------- #


def test_workspace_tool_executor_with_sandbox(tmp_path: Path):
    """Verify execute_bash routes through sandbox when configured."""
    mock_sandbox = MagicMock(spec=SandboxProvider)
    mock_sandbox.run_bash.return_value = ExecutionResult(
        exit_code=0, stdout="sandboxed output", stderr=""
    )

    executor = WorkspaceToolExecutor(workspace_dir=tmp_path, sandbox=mock_sandbox)
    res = executor.execute_bash("ls -la")

    assert res["exit_code"] == 0
    assert res["stdout"] == "sandboxed output"
    assert not res["is_error"]
    mock_sandbox.run_bash.assert_called_once_with(
        "ls -la", cwd=tmp_path.resolve(), timeout=30.0
    )


def test_workspace_tool_executor_sandbox_toggle(tmp_path: Path):
    """Verify toggling between local subprocess and container sandbox."""
    # 1. Local execution (sandbox=None)
    local_exec = WorkspaceToolExecutor(workspace_dir=tmp_path, sandbox=None)
    local_res = local_exec.execute_bash("echo 'local mode'")
    assert local_res["exit_code"] == 0
    assert "local mode" in local_res["stdout"]

    # 2. Sandboxed execution (string identifier 'docker')
    with patch("src.runtime.sandbox.ContainerSandbox.run_command") as mock_cmd:
        mock_cmd.return_value = ExecutionResult(
            exit_code=0, stdout="docker mode", stderr=""
        )
        sb_exec = WorkspaceToolExecutor(workspace_dir=tmp_path, sandbox="docker")
        assert isinstance(sb_exec.sandbox, ContainerSandbox)
        sb_res = sb_exec.execute_bash("echo 'docker mode'")
        assert sb_res["exit_code"] == 0
        assert sb_res["stdout"] == "docker mode"


def test_agent_runtime_sandbox_propagation(tmp_path: Path):
    """Verify sandbox parameter propagates from AgentRuntime to WorkspaceToolExecutor."""
    mock_sandbox = MagicMock(spec=ContainerSandbox)
    runtime = AgentRuntime(
        lm=MagicMock(),
        workspace_dir=tmp_path,
        sandbox=mock_sandbox,
    )
    assert isinstance(runtime.tool_executor, WorkspaceToolExecutor)
    assert runtime.tool_executor.sandbox is mock_sandbox


# --------------------------------------------------------------------------- #
# Live Docker Smoke Test (Optional integration test when Docker is active)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not shutil.which("docker"),
    reason="Docker engine not available in environment",
)
def test_live_docker_smoke(tmp_path: Path):
    """Smoke test running an ephemeral container if Docker daemon is running."""
    # Check if Docker daemon is responsive
    try:
        check = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=3.0,
            check=False,
        )
        if check.returncode != 0:
            pytest.skip("Docker daemon is not responsive")
    except (subprocess.SubprocessError, OSError):
        pytest.skip("Docker command failed")

    # Run real alpine container
    sandbox = ContainerSandbox(
        image="alpine:latest",
        workspace_dir=tmp_path,
        network="none",
        timeout_s=10.0,
    )
    res = sandbox.run_command(["echo", "live docker test"])
    assert res.exit_code == 0
    assert "live docker test" in res.stdout
    assert not res.is_error
