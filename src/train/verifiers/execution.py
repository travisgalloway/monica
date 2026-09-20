"""#361 -- RLVR: Isolated Execution Runner and Sandboxed Verifier.

Executes candidate solutions in an isolated sub-process sandbox with resource limits:
- 256 MB memory limit (enforced via OS process RSS monitoring and V8 heap limits)
- 2-second timeout guard with process tree termination
- Restricted system calls (blocking network sockets and unauthorized subprocesses)

Supports:
- Python via pytest test execution runner
- TypeScript via node:test execution runner

Integrates mutation scoring to compute post-training RLVR rewards:
    R = R_pass * (1 + gamma * R_mutation)
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.train.verifiers import is_degenerate_output
from src.train.verifiers.mutations import MutationEngine, Mutant, score_mutation


def _resolve_python_runner() -> List[str]:
    """Find the best pytest command (active venv, PATH, or sys.executable)."""
    pytest_path = shutil.which("pytest")
    if pytest_path:
        return [pytest_path, "-q", "--tb=no"]
    for candidate in (
        Path.cwd() / ".venv" / "bin" / "pytest",
        Path.cwd() / ".venv" / "bin" / "python3",
        Path(sys.executable).parent / "pytest",
    ):
        if candidate.exists() and "pytest" in candidate.name:
            return [str(candidate), "-q", "--tb=no"]
        elif candidate.exists() and "python" in candidate.name:
            return [str(candidate), "-m", "pytest", "-q", "--tb=no"]
    return [sys.executable, "-m", "pytest", "-q", "--tb=no"]


def resolve_execution_toolchain() -> bool:
    """Verify toolchains for sandboxed code execution (Python/pytest and Node)."""
    has_python = sys.executable is not None and os.path.exists(sys.executable)
    has_node = shutil.which("node") is not None
    return has_python or has_node


def _get_process_rss_bytes(pid: int) -> Optional[int]:
    """Retrieve process Resident Set Size (RSS) in bytes across Linux and macOS."""
    statm = Path(f"/proc/{pid}/statm")
    if statm.exists():
        try:
            pages = int(statm.read_text().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
            return pages * page_size
        except Exception:
            pass

    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode().strip()) * 1024
    except Exception:
        return None


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Terminate an entire process group cleanly and forcefully."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=0.5)
    except Exception:
        pass


@dataclass
class TestExecutionResult:
    """Result of running tests against a candidate solution."""

    pass_fraction: float
    passed_count: int = 0
    failed_count: int = 0
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    memory_exceeded: bool = False
    restricted_blocked: bool = False
    error: Optional[str] = None


@dataclass
class ExecutionResultRaw:
    """Internal raw subprocess execution outcome."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    memory_exceeded: bool = False


class SandboxedCodeVerifier:
    """Isolated execution verifier with resource limits and mutation testing (#361)."""

    def __init__(
        self,
        *,
        timeout_s: float = 2.0,
        memory_limit_mb: int = 256,
        restricted_system_calls: bool = True,
        gamma: float = 0.5,
        max_mutants: int = 10,
        fail_fast: bool = True,
        default_runner: str = "auto",
        mutation_engine: Optional[MutationEngine] = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.memory_limit_bytes = int(memory_limit_mb * 1024 * 1024)
        self.memory_limit_mb = int(memory_limit_mb)
        self.restricted_system_calls = bool(restricted_system_calls)
        self.gamma = float(gamma)
        self.max_mutants = max_mutants
        self.fail_fast = bool(fail_fast)
        self.default_runner = default_runner
        self.mutation_engine = mutation_engine or MutationEngine(max_mutants=max_mutants)

        self._lock = threading.Lock()
        self._n_samples = 0
        self._n_pass = 0
        self._pass_rate_sum = 0.0
        self._mutation_score_sum = 0.0
        self._n_mutants_tested = 0
        self._n_mutants_killed = 0
        self._n_timeouts = 0
        self._n_memory_exceeded = 0
        self._n_restricted_blocked = 0
        self._n_degenerate = 0
        self._wall_s = 0.0

    def __enter__(self) -> SandboxedCodeVerifier:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        pass

    def telemetry(self) -> Dict[str, Any]:
        """Return aggregated verifier telemetry."""
        with self._lock:
            n = self._n_samples
            pass_rate = (self._pass_rate_sum / n) if n > 0 else 0.0
            mut_score = (self._mutation_score_sum / n) if n > 0 else 0.0
            mut_kill_rate = (
                (self._n_mutants_killed / self._n_mutants_tested)
                if self._n_mutants_tested > 0
                else 0.0
            )
            return {
                "n_samples": n,
                "n_pass": self._n_pass,
                "assertion_pass_rate": pass_rate,
                "mean_pass_rate": pass_rate,
                "mutation_score": mut_score,
                "mean_mutation_score": mut_score,
                "mutant_kill_rate": mut_kill_rate,
                "n_mutants_tested": self._n_mutants_tested,
                "n_mutants_killed": self._n_mutants_killed,
                "n_timeouts": self._n_timeouts,
                "n_memory_exceeded": self._n_memory_exceeded,
                "n_restricted_blocked": self._n_restricted_blocked,
                "n_degenerate": self._n_degenerate,
                "wall_s": self._wall_s,
            }

    @staticmethod
    def _strip_markdown_fences(code: str) -> str:
        """Strip markdown ``` code blocks if present."""
        text = code.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    def _infer_language_and_runner(
        self,
        code: str,
        language: Optional[str] = None,
        runner: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Infer target programming language and test runner."""
        if language:
            lang = language.lower().strip()
        else:
            ts_indicators = (
                "export ", "function ", ": number", ": string", ": boolean",
                "const ", "let ", "node:test", "node:assert", "import ", "interface "
            )
            if any(ind in code for ind in ts_indicators) and ("def " not in code):
                lang = "typescript"
            else:
                lang = "python"

        if runner and runner != "auto":
            rn = runner
        elif lang in ("typescript", "ts", "javascript", "js"):
            rn = "node:test"
        else:
            rn = "pytest"

        return lang, rn

    def _execute_subproc(
        self,
        cmd: List[str],
        cwd: Path,
        timeout: float,
    ) -> ExecutionResultRaw:
        """Execute command in subprocess with process group isolation and resource bounds."""
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "NODE_NO_WARNINGS": "1",
            "HOME": str(cwd),
            "TMPDIR": str(cwd),
        }
        if "VIRTUAL_ENV" in os.environ:
            env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]

        def _preexec() -> None:
            os.setsid()
            try:
                import resource
                cpu_limit = max(1, int(timeout + 2))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
            except Exception:
                pass
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit_bytes, self.memory_limit_bytes))
            except Exception:
                pass

        t0 = time.monotonic()
        timed_out = False
        mem_exceeded = False

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_preexec,
        )

        while True:
            ret = proc.poll()
            if ret is not None:
                break
            elapsed = time.monotonic() - t0
            if elapsed > timeout:
                timed_out = True
                _kill_process_group(proc)
                break
            rss = _get_process_rss_bytes(proc.pid)
            if rss is not None and rss > self.memory_limit_bytes:
                mem_exceeded = True
                _kill_process_group(proc)
                break
            time.sleep(0.01)

        try:
            stdout, stderr = proc.communicate(timeout=1.0)
        except Exception:
            _kill_process_group(proc)
            stdout, stderr = "", ""

        duration = time.monotonic() - t0

        if "out of memory" in stderr.lower() or "allocation failed" in stderr.lower():
            mem_exceeded = True

        return ExecutionResultRaw(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            timed_out=timed_out,
            memory_exceeded=mem_exceeded,
        )

    def run_python_pytest(
        self,
        code: str,
        tests: Union[Sequence[str], str],
        sandbox_dir: Path,
        timeout: float,
    ) -> TestExecutionResult:
        """Run candidate Python code against pytest test suite in isolated sandbox."""
        test_file = sandbox_dir / "test_solution.py"

        parts = [code, "\n\n"]
        if isinstance(tests, str):
            parts.append(tests)
        else:
            for idx, item in enumerate(tests):
                item_str = item.strip()
                if item_str.startswith("assert ") or not item_str.startswith("def "):
                    parts.append(f"def test_case_{idx}():\n")
                    for line in item_str.splitlines():
                        parts.append(f"    {line}\n")
                    parts.append("\n")
                else:
                    parts.append(f"{item_str}\n\n")

        test_file.write_text("".join(parts), encoding="utf-8")

        if self.restricted_system_calls:
            conftest = sandbox_dir / "conftest.py"
            conftest.write_text(
                "import sys\n"
                "def _audit_hook(event, args):\n"
                "    if event in ('socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system', 'posix.system'):\n"
                "        raise PermissionError(f'Restricted system call in sandbox: {event}')\n"
                "sys.addaudithook(_audit_hook)\n",
                encoding="utf-8",
            )

        cmd = list(_resolve_python_runner()) + [str(test_file.name)]
        raw = self._execute_subproc(cmd, sandbox_dir, timeout)

        passed = 0
        failed = 0
        errors = 0

        m_pass = re.search(r"(\d+) passed", raw.stdout)
        if m_pass:
            passed = int(m_pass.group(1))
        m_fail = re.search(r"(\d+) failed", raw.stdout)
        if m_fail:
            failed = int(m_fail.group(1))
        m_err = re.search(r"(\d+) error", raw.stdout)
        if m_err:
            errors = int(m_err.group(1))

        total = passed + failed + errors
        if total > 0:
            pass_fraction = passed / total
        elif raw.exit_code == 0 and not raw.timed_out and not raw.memory_exceeded:
            pass_fraction = 1.0
        else:
            pass_fraction = 0.0

        restricted_blocked = "Restricted system call in sandbox" in (raw.stdout + raw.stderr)

        return TestExecutionResult(
            pass_fraction=pass_fraction,
            passed_count=passed,
            failed_count=failed + errors,
            exit_code=raw.exit_code,
            stdout=raw.stdout,
            stderr=raw.stderr,
            duration_s=raw.duration_s,
            timed_out=raw.timed_out,
            memory_exceeded=raw.memory_exceeded,
            restricted_blocked=restricted_blocked,
        )

    def run_ts_node_test(
        self,
        code: str,
        tests: Union[Sequence[str], str],
        sandbox_dir: Path,
        timeout: float,
    ) -> TestExecutionResult:
        """Run candidate TypeScript code against node:test runner in isolated sandbox."""
        test_file = sandbox_dir / "test_solution.ts"

        parts = [
            "import test from 'node:test';\n",
            "import assert from 'node:assert';\n\n",
            code,
            "\n\n",
        ]
        if isinstance(tests, str):
            parts.append(tests)
        else:
            for idx, item in enumerate(tests):
                item_str = item.strip()
                if item_str.startswith("test(") or item_str.startswith("test.it("):
                    parts.append(f"{item_str}\n\n")
                else:
                    parts.append(f"test('test_case_{idx}', () => {{\n")
                    for line in item_str.splitlines():
                        parts.append(f"    {line}\n")
                    parts.append("});\n\n")

        test_file.write_text("".join(parts), encoding="utf-8")

        node_bin = shutil.which("node") or "node"
        cmd = [node_bin, f"--max-old-space-size={self.memory_limit_mb}"]

        if self.restricted_system_calls:
            guard_file = sandbox_dir / "_sandbox_guard.cjs"
            guard_file.write_text(
                "const net = require('net');\n"
                "net.Socket.prototype.connect = function() {\n"
                "  throw new Error('Restricted system call in sandbox: socket.connect');\n"
                "};\n"
                "const cp = require('child_process');\n"
                "cp.spawn = cp.exec = cp.execFile = cp.fork = function() {\n"
                "  throw new Error('Restricted system call in sandbox: child_process');\n"
                "};\n",
                encoding="utf-8",
            )
            cmd.extend(["--require", "./_sandbox_guard.cjs"])

        cmd.extend(["--experimental-strip-types", "--test", str(test_file.name)])
        raw = self._execute_subproc(cmd, sandbox_dir, timeout)

        passed = 0
        failed = 0
        m_pass = re.search(r"ℹ pass (\d+)", raw.stdout)
        if m_pass:
            passed = int(m_pass.group(1))
        m_fail = re.search(r"ℹ fail (\d+)", raw.stdout)
        if m_fail:
            failed = int(m_fail.group(1))

        total = passed + failed
        if total > 0:
            pass_fraction = passed / total
        elif raw.exit_code == 0 and not raw.timed_out and not raw.memory_exceeded:
            pass_fraction = 1.0
        else:
            pass_fraction = 0.0

        restricted_blocked = "Restricted system call in sandbox" in (raw.stdout + raw.stderr)

        return TestExecutionResult(
            pass_fraction=pass_fraction,
            passed_count=passed,
            failed_count=failed,
            exit_code=raw.exit_code,
            stdout=raw.stdout,
            stderr=raw.stderr,
            duration_s=raw.duration_s,
            timed_out=raw.timed_out,
            memory_exceeded=raw.memory_exceeded,
            restricted_blocked=restricted_blocked,
        )

    def execute_with_tests(
        self,
        code: str,
        tests: Union[Sequence[str], str],
        *,
        language: str = "python",
        runner: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> TestExecutionResult:
        """Execute code with tests in an ephemeral isolated sandbox directory."""
        to = timeout if timeout is not None else self.timeout_s
        lang, rn = self._infer_language_and_runner(code, language, runner)

        with tempfile.TemporaryDirectory() as td:
            s_dir = Path(td)
            if rn == "node:test":
                return self.run_ts_node_test(code, tests, s_dir, to)
            return self.run_python_pytest(code, tests, s_dir, to)

    def reward(
        self,
        completion: str,
        reference: Optional[str] = None,
        *,
        prompt: str = "",
        tests: Optional[Sequence[str]] = None,
        test_code: Optional[str] = None,
        language: Optional[str] = None,
        runner: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        """Score one completion based on assertion pass rate and mutant kill rate:

            R = R_pass * (1 + gamma * R_mutation)

        Matches scripts/rlvr.py reward_fn(decoded, ref, *, prompt=...) contract.
        """
        raw_completion = "" if completion is None else str(completion)
        t_start = time.monotonic()

        with self._lock:
            self._n_samples += 1

        reason = is_degenerate_output(raw_completion)
        if reason is not None:
            with self._lock:
                self._n_degenerate += 1
                self._wall_s += time.monotonic() - t_start
            return -1.0

        code = self._strip_markdown_fences(raw_completion)

        resolved_tests: Union[Sequence[str], str] = []
        if tests is not None:
            resolved_tests = tests
        elif test_code is not None:
            resolved_tests = test_code
        elif "tests" in kwargs:
            resolved_tests = kwargs["tests"]
        elif "test_code" in kwargs:
            resolved_tests = kwargs["test_code"]
        elif reference and ("assert " in reference or "test(" in reference or "def test_" in reference):
            resolved_tests = reference
        elif "assert " in prompt:
            lines = [ln.strip() for ln in prompt.splitlines() if ln.strip().startswith("assert ")]
            if lines:
                resolved_tests = lines

        if not resolved_tests:
            resolved_tests = ["assert True"]

        lang, rn = self._infer_language_and_runner(
            code,
            language=language or kwargs.get("language"),
            runner=runner or kwargs.get("runner"),
        )

        exec_res = self.execute_with_tests(code, resolved_tests, language=lang, runner=rn)

        with self._lock:
            if exec_res.timed_out:
                self._n_timeouts += 1
            if exec_res.memory_exceeded:
                self._n_memory_exceeded += 1
            if exec_res.restricted_blocked:
                self._n_restricted_blocked += 1

        r_pass = exec_res.pass_fraction

        if r_pass >= 1.0:
            with self._lock:
                self._n_pass += 1

        if r_pass <= 0.0 or (self.fail_fast and r_pass < 1.0):
            with self._lock:
                self._pass_rate_sum += r_pass
                self._wall_s += time.monotonic() - t_start
            return float(r_pass)

        mutants = self.mutation_engine.generate_mutants(code, language=lang)
        mutants_tested = len(mutants)
        mutants_killed = 0

        for mutant in mutants:
            m_res = self.execute_with_tests(mutant.mutated_code, resolved_tests, language=lang, runner=rn)
            if m_res.pass_fraction < 1.0 or m_res.timed_out or m_res.memory_exceeded or m_res.exit_code != 0:
                mutants_killed += 1

        r_mutation = (mutants_killed / mutants_tested) if mutants_tested > 0 else 0.0
        r_total = score_mutation(r_pass, mutants_tested, mutants_killed, gamma=self.gamma)

        with self._lock:
            self._pass_rate_sum += r_pass
            self._mutation_score_sum += r_mutation
            self._n_mutants_tested += mutants_tested
            self._n_mutants_killed += mutants_killed
            self._wall_s += time.monotonic() - t_start

        return r_total
