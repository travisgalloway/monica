"""Containerized execution sandbox for agent tools and RLVR verifiers (#353).

Provides environmental isolation for untrusted model output execution and coding agent
tool commands (arXiv:2609.20804, Harbor container orchestration).

Features:
- Ephemeral container lifecycle management (Docker and Podman CLI backends).
- Persistent container session mode via context manager.
- Workspace filesystem volume mounting with host-path translation.
- Configurable CPU, memory, PIDs, and network isolation limits.
- Process timeout guards with automatic container cleanup.
- Abstract SandboxProvider interface with LocalSandboxProvider and ContainerSandbox implementations.
- Seamless toggling between local subprocess execution and isolated container execution.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import types
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of command execution within a sandbox."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    error: str | None = None

    @property
    def is_error(self) -> bool:
        """True if the command returned a non-zero exit code or timed out."""
        return self.exit_code != 0 or self.timed_out


@dataclass
class SandboxConfig:
    """Configuration for containerized sandboxing."""

    image: str = "python:3.12-slim"
    engine: str = "docker"  # "docker", "podman", etc.
    workspace_dir: str | Path | None = None
    container_workdir: str = "/workspace"
    timeout_s: float = 30.0
    cpu_limit: float | None = 2.0  # CPU count / quota
    memory_limit: str | None = "2g"  # Memory constraint (e.g. "512m", "2g")
    pids_limit: int | None = 256  # Guard against fork-bombs
    network: str = "none"  # "none" for isolated RLVR/evals, "bridge", "host"
    read_only: bool = False  # Mount workspace read-only
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    env: dict[str, str] = field(default_factory=dict)
    persistent: bool = False  # Maintain long-running container session


class SandboxProvider(ABC):
    """Abstract base class for execution providers."""

    is_local: bool = False
    is_container: bool = False

    @abstractmethod
    def run_command(
        self,
        cmd: list[str] | str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a command inside the sandbox."""
        ...

    def run_python(
        self,
        code: str,
        *,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a Python code string inside the sandbox."""
        return self.run_command(["python3", "-c", code], timeout=timeout, env=env)

    def run_bash(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a bash command string inside the sandbox."""
        return self.run_command(["/bin/bash", "-c", command], cwd=cwd, timeout=timeout, env=env)

    def close(self) -> None:
        """Clean up any persistent resources."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()


class LocalSandboxProvider(SandboxProvider):
    """Local host subprocess execution provider (for trusted fast smoke runs)."""

    is_local: bool = True
    is_container: bool = False

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        timeout_s: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else Path.cwd()
        self.timeout_s = float(timeout_s)
        self.base_env = dict(env or {})

    def run_command(
        self,
        cmd: list[str] | str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        effective_timeout = timeout if timeout is not None else self.timeout_s
        effective_cwd = cwd if cwd is not None else self.workspace_dir
        merged_env = os.environ.copy()
        merged_env.update(self.base_env)
        if env:
            merged_env.update(env)

        t0 = time.monotonic()
        try:
            shell = isinstance(cmd, str)
            proc = subprocess.run(
                cmd,
                shell=shell,
                cwd=effective_cwd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=merged_env,
                check=False,
            )
            duration = time.monotonic() - t0
            return ExecutionResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_s=duration,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - t0
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="",
                duration_s=duration,
                timed_out=True,
                error=f"Command timed out after {effective_timeout} seconds",
            )
        except (subprocess.SubprocessError, OSError) as e:
            duration = time.monotonic() - t0
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration_s=duration,
                timed_out=False,
                error=str(e),
            )


class ContainerSandbox(SandboxProvider):
    """Containerized sandbox provider utilizing Docker or Podman CLI (#353)."""

    is_local: bool = False
    is_container: bool = True

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        image: str = "python:3.12-slim",
        engine: str = "docker",
        workspace_dir: str | Path | None = None,
        container_workdir: str = "/workspace",
        timeout_s: float = 30.0,
        cpu_limit: float | None = 2.0,
        memory_limit: str | None = "2g",
        pids_limit: int | None = 256,
        network: str = "none",
        read_only: bool = False,
        cap_drop: list[str] | None = None,
        env: dict[str, str] | None = None,
        persistent: bool = False,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = SandboxConfig(
                image=image,
                engine=engine,
                workspace_dir=workspace_dir,
                container_workdir=container_workdir,
                timeout_s=timeout_s,
                cpu_limit=cpu_limit,
                memory_limit=memory_limit,
                pids_limit=pids_limit,
                network=network,
                read_only=read_only,
                cap_drop=cap_drop if cap_drop is not None else ["ALL"],
                env=dict(env or {}),
                persistent=persistent,
            )
        self.engine = self.config.engine
        self.workspace_dir = (
            Path(self.config.workspace_dir).resolve() if self.config.workspace_dir else None
        )
        self.container_id: str | None = None
        self._is_running: bool = False

    def _resolve_workdir(self, cwd: str | Path | None) -> str:
        """Resolve host working directory to container working directory path."""
        if cwd is None:
            return self.config.container_workdir
        if self.workspace_dir is not None:
            try:
                target_path = Path(cwd)
                if not target_path.is_absolute():
                    target_path = (self.workspace_dir / target_path).resolve()
                else:
                    target_path = target_path.resolve()
                rel = target_path.relative_to(self.workspace_dir)
                if str(rel) == ".":
                    return self.config.container_workdir
                return f"{self.config.container_workdir}/{rel}".replace("\\", "/")
            except ValueError:
                return str(cwd)
        return str(cwd)

    def build_run_args(
        self,
        cmd: list[str],
        *,
        container_name: str | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """Construct the CLI argument list for an ephemeral container run."""
        args = [self.engine, "run", "--rm"]
        if container_name:
            args.extend(["--name", container_name])
        if self.config.cpu_limit is not None:
            args.extend(["--cpus", str(self.config.cpu_limit)])
        if self.config.memory_limit is not None:
            args.extend(["--memory", str(self.config.memory_limit)])
        if self.config.pids_limit is not None:
            args.extend(["--pids-limit", str(self.config.pids_limit)])
        if self.config.network:
            args.extend(["--network", str(self.config.network)])
        if self.config.read_only:
            args.append("--read-only")
        for cap in self.config.cap_drop:
            args.extend(["--cap-drop", cap])

        if self.workspace_dir is not None:
            mount_mode = "ro" if self.config.read_only else "rw"
            args.extend(["-v", f"{self.workspace_dir}:{self.config.container_workdir}:{mount_mode}"])

        workdir = self._resolve_workdir(cwd)
        args.extend(["-w", workdir])

        merged_env = {**self.config.env, **(env or {})}
        for k, v in merged_env.items():
            args.extend(["-e", f"{k}={v}"])

        args.append(self.config.image)
        args.extend(cmd)
        return args

    def build_exec_args(
        self,
        cmd: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """Construct the CLI argument list for exec inside a running container."""
        if not self.container_id:
            raise RuntimeError("No running persistent container")
        args = [self.engine, "exec"]
        workdir = self._resolve_workdir(cwd)
        args.extend(["-w", workdir])
        merged_env = {**self.config.env, **(env or {})}
        for k, v in merged_env.items():
            args.extend(["-e", f"{k}={v}"])
        args.append(self.container_id)
        args.extend(cmd)
        return args

    def start(self) -> str:
        """Start a persistent background container session."""
        if self._is_running and self.container_id:
            return self.container_id

        container_name = f"monica_sb_{uuid.uuid4().hex[:12]}"
        args = [self.engine, "run", "-d", "--name", container_name]
        if self.config.cpu_limit is not None:
            args.extend(["--cpus", str(self.config.cpu_limit)])
        if self.config.memory_limit is not None:
            args.extend(["--memory", str(self.config.memory_limit)])
        if self.config.pids_limit is not None:
            args.extend(["--pids-limit", str(self.config.pids_limit)])
        if self.config.network:
            args.extend(["--network", str(self.config.network)])
        if self.config.read_only:
            args.append("--read-only")
        for cap in self.config.cap_drop:
            args.extend(["--cap-drop", cap])

        if self.workspace_dir is not None:
            mount_mode = "ro" if self.config.read_only else "rw"
            args.extend(["-v", f"{self.workspace_dir}:{self.config.container_workdir}:{mount_mode}"])

        workdir = self._resolve_workdir(None)
        args.extend(["-w", workdir])

        for k, v in self.config.env.items():
            args.extend(["-e", f"{k}={v}"])

        args.extend([self.config.image, "sleep", "infinity"])

        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to start container sandbox: {proc.stderr.strip()}")

        self.container_id = container_name
        self._is_running = True
        return self.container_id

    def stop(self) -> None:
        """Stop and remove persistent container session."""
        if self.container_id:
            try:
                subprocess.run(
                    [self.engine, "rm", "-f", self.container_id],
                    capture_output=True,
                    check=False,
                    timeout=10.0,
                )
            except (subprocess.SubprocessError, OSError) as e:
                logger.debug("Failed to remove container %s: %s", self.container_id, e)
        self.container_id = None
        self._is_running = False

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> Self:
        if self.config.persistent:
            self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.stop()

    def __del__(self) -> None:
        if getattr(self, "_is_running", False):
            self.stop()

    def run_command(
        self,
        cmd: list[str] | str,
        *,
        cwd: str | Path | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        effective_timeout = timeout if timeout is not None else self.config.timeout_s
        cmd_list = [cmd] if isinstance(cmd, str) else list(cmd)
        if isinstance(cmd, str):
            cmd_list = ["/bin/sh", "-c", cmd]

        t0 = time.monotonic()
        if self.config.persistent and self._is_running:
            exec_args = self.build_exec_args(cmd_list, cwd=cwd, env=env)
            try:
                proc = subprocess.run(
                    exec_args,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                    check=False,
                )
                duration = time.monotonic() - t0
                return ExecutionResult(
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    duration_s=duration,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                duration = time.monotonic() - t0
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    duration_s=duration,
                    timed_out=True,
                    error=f"Command timed out after {effective_timeout} seconds",
                )
            except (subprocess.SubprocessError, OSError) as e:
                duration = time.monotonic() - t0
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    duration_s=duration,
                    timed_out=False,
                    error=str(e),
                )
        else:
            container_name = f"monica_sb_{uuid.uuid4().hex[:12]}"
            run_args = self.build_run_args(
                cmd_list,
                container_name=container_name,
                cwd=cwd,
                env=env,
            )
            try:
                proc = subprocess.run(
                    run_args,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                    check=False,
                )
                duration = time.monotonic() - t0
                return ExecutionResult(
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    duration_s=duration,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                duration = time.monotonic() - t0
                try:
                    subprocess.run(
                        [self.engine, "rm", "-f", container_name],
                        capture_output=True,
                        check=False,
                        timeout=5.0,
                    )
                except (subprocess.SubprocessError, OSError) as e:
                    logger.debug("Failed to clean up timed out container %s: %s", container_name, e)
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    duration_s=duration,
                    timed_out=True,
                    error=f"Command timed out after {effective_timeout} seconds",
                )
            except (subprocess.SubprocessError, OSError) as e:
                duration = time.monotonic() - t0
                try:
                    subprocess.run(
                        [self.engine, "rm", "-f", container_name],
                        capture_output=True,
                        check=False,
                        timeout=5.0,
                    )
                except (subprocess.SubprocessError, OSError) as cleanup_err:
                    logger.debug("Failed to clean up failed container %s: %s", container_name, cleanup_err)
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=str(e),
                    duration_s=duration,
                    timed_out=False,
                    error=str(e),
                )


def create_sandbox(
    backend: str = "auto",
    *,
    workspace_dir: str | Path | None = None,
    timeout_s: float = 30.0,
    engine: str | None = None,
    **kwargs: Any,
) -> SandboxProvider:
    """Factory creating an execution sandbox based on backend identifier.

    Backend options:
      - "local" or "subprocess": Local host execution via subprocess.
      - "docker": Container execution using Docker CLI.
      - "podman": Container execution using Podman CLI.
      - "container": Container execution, auto-detecting docker/podman.
      - "auto": Checks MONICA_SANDBOX_BACKEND env var; if unset, auto-detects
        container engine, falling back to local.
    """
    mode = backend.lower().strip()
    if mode == "auto":
        env_backend = os.environ.get("MONICA_SANDBOX_BACKEND", "").lower().strip()
        if env_backend:
            mode = env_backend
        elif shutil.which("docker") or shutil.which("podman"):
            mode = "container"
        else:
            mode = "local"

    if mode in ("local", "subprocess"):
        return LocalSandboxProvider(
            workspace_dir=workspace_dir,
            timeout_s=timeout_s,
            **kwargs,
        )

    if mode in ("container", "docker", "podman"):
        resolved_engine = engine or ("podman" if mode == "podman" else "docker")
        if mode == "container" and not engine:
            if shutil.which("docker"):
                resolved_engine = "docker"
            elif shutil.which("podman"):
                resolved_engine = "podman"

        return ContainerSandbox(
            engine=resolved_engine,
            workspace_dir=workspace_dir,
            timeout_s=timeout_s,
            **kwargs,
        )

    raise ValueError(f"Unknown sandbox backend: {backend}")
