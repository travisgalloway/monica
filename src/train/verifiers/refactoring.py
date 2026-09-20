"""#347 -- RLVR/Eval: Behavioral-invariance refactoring & interface decoupling verifiers.

Part of the M12 software design and refactoring verification track (#198, #221, #347).
Evaluate the model's ability to transform monolithic and tightly-coupled code into
decoupled patterns while strictly preserving behavioral correctness:

1. Behavioral Invariance Under Structural Refactoring:
   - Multi-file refactoring tasks (e.g. monolithic procedural code -> Strategy / Factory /
     Repository design patterns).
   - Invariant: run existing comprehensive unit and property-based test suites against
     the model's refactored architecture.
   - Reward: 1.0 if and only if 100% of existing functional tests pass without behavioral regressions.

2. Interface Decoupling & Mockability Verifier:
   - Task: extract interfaces and refactor classes to use Dependency Injection (DIP / Inversion of Control).
   - Verifier checks that the refactored class can be instantiated with mocked interfaces.
   - Assert unit tests execute against mocks with zero real network sockets or database connections opened.

3. Sandboxed Test Execution Gate:
   - Isolated temporary workspace execution.
   - Timeout guards preventing runaway infinite loops.
   - Memory and CPU resource limits.
   - Socket and database interceptor guards flagging leaky real I/O.
   - Anti-Goodhart escape-hatch guards (refactor-ignore, @unittest.skip, eval bypasses) -> -1.0.
   - Degenerate output guards (empty, whitespace, comment-only) -> -1.0.
   - Thread-safe telemetry and injectable oracle seam for testing.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

from src.lsp.diagnostics import Diagnostic, mask_strings_and_comments
from src.train.verifiers import (
    DEFAULT_SEVERITY_WEIGHTS,
    diagnostics_to_reward,
    is_degenerate_output,
    severity_weight,
)

# --------------------------------------------------------------------------- #
# Constants & Enums
# --------------------------------------------------------------------------- #

class RefactoringPattern(str, Enum):
    STRATEGY = "strategy"
    FACTORY = "factory"
    REPOSITORY = "repository"
    DEPENDENCY_INJECTION = "dependency_injection"
    MOCKABILITY = "mockability"
    INTERFACE_EXTRACTION = "interface_extraction"
    CUSTOM = "custom"


# Fine-grained penalty overrides for RLVR reward shaping
DEFAULT_REFACTORING_CODE_OVERRIDES: Mapping[str, float] = {
    "REFACTOR_SYNTAX_ERROR": 0.50,
    "REFACTOR_TEST_FAILURE": 0.50,
    "REFACTOR_TEST_TIMEOUT": 0.50,
    "REFACTOR_MOCKABILITY_FAILED": 0.40,
    "REFACTOR_LEAKY_IO": 0.50,
    "REFACTOR_MISSING_INTERFACE": 0.25,
    "REFACTOR_HARDCODED_DEPENDENCY": 0.25,
}

# Anti-Goodhart escape-hatch patterns
REFACTORING_ESCAPE_HATCH_PATTERNS: Mapping[str, re.Pattern] = {
    "refactor_suppress_directive": re.compile(
        r"(?:#|//)\s*(?:refactor-ignore|noqa:\s*REFACTOR|type:\s*ignore\s*\[refactor\])\b",
        re.IGNORECASE,
    ),
    "test_skip_directive": re.compile(
        r"@(?:unittest\.skip|pytest\.mark\.skip)(?:\(.*\))?",
    ),
    "dynamic_import_eval": re.compile(
        r"\b(?:eval|exec)\s*\(",
    ),
    "monkeypatch_assert_bypass": re.compile(
        r"(?:unittest\.TestCase\.assertTrue|assert_true|expect)\s*=\s*(?:lambda|True)",
    ),
}


# --------------------------------------------------------------------------- #
# Path Normalization & Multi-File Parsing
# --------------------------------------------------------------------------- #

def normalize_file_path(path: str) -> str:
    """Normalize file path into a clean relative POSIX path."""
    p = str(path).replace("\\", "/").strip()
    p = p.removeprefix("./")
    parts: List[str] = []
    for part in PurePosixPath(p).parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def parse_multi_file_source(
    source: str | Mapping[str, str] | Sequence[dict],
    default_filename: str = "refactored.py",
) -> dict[str, str]:
    """Parse multi-file model completion or repository representation into {path: content}."""
    if isinstance(source, Mapping):
        return {normalize_file_path(k): str(v) for k, v in source.items()}

    if isinstance(source, Sequence) and not isinstance(source, str):
        files: dict[str, str] = {}
        for item in source:
            if isinstance(item, dict) and "path" in item:
                content = item.get("content") or item.get("text") or ""
                files[normalize_file_path(item["path"])] = str(content)
        return files

    text = str(source).strip()
    files: dict[str, str] = {}

    # 1. JSON-encoded repository manifest
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if "files" in parsed and isinstance(parsed["files"], (dict, list)):
                    return parse_multi_file_source(parsed["files"], default_filename=default_filename)
                if all(isinstance(v, str) for v in parsed.values()):
                    return {normalize_file_path(k): v for k, v in parsed.items()}
            elif isinstance(parsed, list):
                return parse_multi_file_source(parsed, default_filename=default_filename)
        except Exception:
            pass

    # 2. XML file tags: <file path="...">...</file>
    xml_re = re.compile(
        r'<file\s+(?:path|name)=["\']([^"\']+)["\']\s*>(.*?)</file>',
        re.DOTALL | re.IGNORECASE,
    )
    xml_matches = list(xml_re.finditer(text))
    if xml_matches:
        for m in xml_matches:
            path = normalize_file_path(m.group(1))
            files[path] = m.group(2).strip("\n")
        return files

    # 3. Markdown code blocks with header preceding the fence:
    #    ### path/to/file.py
    #    ```python
    #    ...
    #    ```
    header_fence_re = re.compile(
        r"""(?:^|\n)(?:#+|//|\*\*|File:?|Filename:?)\s*([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)\*?\*?\s*\n+```(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)```""",
        re.MULTILINE,
    )
    for m in header_fence_re.finditer(text):
        p = normalize_file_path(m.group(1))
        files[p] = m.group(2)

    if files:
        return files

    # 4. Markdown code fence with embedded path:
    #    ```python:path/to/file.py
    fence_path_re = re.compile(
        r"""```(?:[a-zA-Z0-9_\-]+)?[:\s]+(?:filename=|title=)?["'`]?([a-zA-Z0-9_./\-]+\.[a-zA-Z0-9]+)["'`]?\s*\n([\s\S]*?)```""",
        re.MULTILINE,
    )
    for m in fence_path_re.finditer(text):
        p = normalize_file_path(m.group(1))
        files[p] = m.group(2)

    if files:
        return files

    # 5. Markdown bare fence:
    bare_fence_re = re.compile(r"""^```(?:[a-zA-Z0-9_\-]+)?\n([\s\S]*?)```$""", re.MULTILINE)
    m = bare_fence_re.search(text)
    if m:
        files[default_filename] = m.group(1)
        return files

    # Fallback: raw code snippet
    if text:
        files[default_filename] = text

    return files


def find_refactoring_escape_hatches(text: str) -> List[str]:
    """Identify anti-Goodhart escape hatches or test execution bypasses."""
    found: List[str] = []
    for name, pattern in REFACTORING_ESCAPE_HATCH_PATTERNS.items():
        if pattern.search(text):
            found.append(name)
    return found


def is_refactor_degenerate_output(text: str, *, min_code_chars: int = 2) -> Optional[str]:
    """Check for empty, whitespace-only, comment-only (Python and TS), or too-short code."""
    base_res = is_degenerate_output(text, min_code_chars=min_code_chars)
    if base_res is not None:
        return base_res

    # Strip Python '#' comments and check remaining executable code
    lines: List[str] = []
    for line in text.splitlines():
        code_part = re.sub(r"#.*$", "", line).strip()
        if code_part:
            lines.append(code_part)

    residual = "".join(lines).strip()
    if not residual:
        return "comment_only" if text.strip() else "whitespace_only"
    if len(residual) < min_code_chars:
        return "too_short"

    return None


# --------------------------------------------------------------------------- #
# Static & AST Refactoring Pattern Analysis
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RefactoringAstSummary:
    """Static AST characteristics of the refactored code bundle."""
    classes: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    injected_classes: List[str] = field(default_factory=list)
    hardcoded_io_calls: List[str] = field(default_factory=list)
    patterns_detected: List[str] = field(default_factory=list)
    syntax_errors: List[str] = field(default_factory=list)


def analyze_refactoring_ast(files: Mapping[str, str]) -> RefactoringAstSummary:
    """Analyze multi-file code bundle statically using Python AST."""
    classes: List[str] = []
    interfaces: List[str] = []
    methods: List[str] = []
    injected_classes: List[str] = []
    hardcoded_io: List[str] = []
    patterns: Set[str] = set()
    syntax_errors: List[str] = []

    for path, content in files.items():
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(content, filename=path)
        except SyntaxError as e:
            syntax_errors.append(f"{path}:{e.lineno}:{e.offset}: {e.msg}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
                # Check for ABC or Protocol or abstractmethod
                is_iface = False
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in ("ABC", "Protocol"):
                        is_iface = True
                    elif isinstance(base, ast.Attribute) and base.attr in ("ABC", "Protocol"):
                        is_iface = True

                has_abstract = any(
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and any(
                        (isinstance(d, ast.Name) and d.id == "abstractmethod")
                        or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
                        for d in item.decorator_list
                    )
                    for item in node.body
                )
                if is_iface or has_abstract:
                    interfaces.append(node.name)
                    patterns.add(RefactoringPattern.INTERFACE_EXTRACTION.value)

                # Check __init__ constructor parameters for Dependency Injection
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        args = [a.arg for a in item.args.args if a.arg != "self"]
                        if args:
                            injected_classes.append(node.name)
                            patterns.add(RefactoringPattern.DEPENDENCY_INJECTION.value)

                # Detect Repository pattern methods
                repo_method_names = {"get", "save", "find", "delete", "add", "list", "get_by_id"}
                class_methods = {
                    item.name for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                if len(repo_method_names.intersection(class_methods)) >= 2:
                    patterns.add(RefactoringPattern.REPOSITORY.value)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(node.name)
                # Factory pattern: creator function/method
                if node.name.startswith("create_") or node.name.startswith("make_") or node.name.startswith("get_"):
                    patterns.add(RefactoringPattern.FACTORY.value)

            elif isinstance(node, ast.Call):
                # Inspect for raw hardcoded socket or sqlite connections in AST
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("socket", "create_connection"):
                        hardcoded_io.append(f"{path}:{node.lineno}: socket call")
                    elif node.func.attr == "connect" and isinstance(node.func.value, ast.Name) and node.func.value.id == "sqlite3":
                        hardcoded_io.append(f"{path}:{node.lineno}: sqlite3.connect")

    # Strategy pattern detection: interface/protocol exists and multiple classes exist
    if interfaces and len(classes) >= 2:
        patterns.add(RefactoringPattern.STRATEGY.value)

    return RefactoringAstSummary(
        classes=classes,
        interfaces=interfaces,
        methods=methods,
        injected_classes=injected_classes,
        hardcoded_io_calls=hardcoded_io,
        patterns_detected=sorted(patterns),
        syntax_errors=syntax_errors,
    )


# --------------------------------------------------------------------------- #
# Sandboxed Test Execution Gate
# --------------------------------------------------------------------------- #

@dataclass
class TestExecutionResult:
    """Execution outcome of running unit and mockability tests."""
    __test__ = False  # Avoid pytest attempting to collect this dataclass as a test

    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    functional_passed: bool = False
    mockability_passed: bool = False
    timed_out: bool = False
    real_socket_detected: bool = False
    real_db_detected: bool = False
    socket_details: List[str] = field(default_factory=list)
    db_details: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0


_HARNESS_TEMPLATE = """# Refactoring Verification Test Runner
import sys
import os
import json
import unittest
import importlib
import traceback

# 1. IO Interceptor Guards
_real_socket_called = False
_socket_details = []
_real_db_called = False
_db_details = []

BLOCK_NETWORK = {block_network}
BLOCK_DATABASE = {block_database}
ALLOW_SQLITE_MEMORY = {allow_sqlite_memory}

if BLOCK_NETWORK:
    import socket
    _orig_socket = socket.socket
    _orig_create_conn = socket.create_connection

    class _GuardedSocket(_orig_socket):
        def __init__(self, *args, **kwargs):
            global _real_socket_called
            _real_socket_called = True
            _socket_details.append(f"socket.socket(args={{args}}, kwargs={{kwargs}})")
            raise RuntimeError("REAL_SOCKET_OPENED: Network socket access attempted during unit/mock tests")

    def _guarded_create_conn(*args, **kwargs):
        global _real_socket_called
        _real_socket_called = True
        _socket_details.append(f"socket.create_connection({{args}}, {{kwargs}})")
        raise RuntimeError("REAL_SOCKET_OPENED: Network connection attempted during unit/mock tests")

    socket.socket = _GuardedSocket
    socket.create_connection = _guarded_create_conn

if BLOCK_DATABASE:
    import sqlite3
    _orig_sqlite3_connect = sqlite3.connect

    def _guarded_sqlite3_connect(*args, **kwargs):
        global _real_db_called
        db_name = str(args[0]) if args else str(kwargs.get("database", ""))
        if ALLOW_SQLITE_MEMORY and (db_name == ":memory:" or "mode=memory" in db_name):
            return _orig_sqlite3_connect(*args, **kwargs)
        _real_db_called = True
        _db_details.append(f"sqlite3.connect(database={{db_name}})")
        raise RuntimeError("REAL_DB_OPENED: Real database connection attempted during unit/mock tests")

    sqlite3.connect = _guarded_sqlite3_connect

# 2. Execution Setup
sys.path.insert(0, os.getcwd())
loader = unittest.TestLoader()

def run_suite(module_names):
    suite = unittest.TestSuite()
    load_errors = []
    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
            suite.addTests(loader.loadTestsFromModule(mod))
        except Exception as e:
            load_errors.append(f"Failed to load {{mod_name}}: {{e}}\\n{{traceback.format_exc()}}")
    
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    
    total = result.testsRun + len(load_errors)
    failures = len(result.failures)
    errors = len(result.errors) + len(load_errors)
    passed = total - failures - errors
    success = (total > 0) and (failures == 0) and (errors == 0)
    
    fail_msgs = [f"{{test}}: {{err}}" for test, err in result.failures]
    err_msgs = [f"{{test}}: {{err}}" for test, err in result.errors] + load_errors
    
    return {{
        "success": success,
        "total": total,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "fail_msgs": fail_msgs,
        "err_msgs": err_msgs,
    }}

# Execute functional and mockability suites
functional_mods = {functional_mods}
mock_mods = {mock_mods}

func_res = run_suite(functional_mods) if functional_mods else {{"success": True, "total": 0, "passed": 0, "failures": 0, "errors": 0, "fail_msgs": [], "err_msgs": []}}
mock_res = run_suite(mock_mods) if mock_mods else {{"success": True, "total": 0, "passed": 0, "failures": 0, "errors": 0, "fail_msgs": [], "err_msgs": []}}

overall_total = func_res["total"] + mock_res["total"]
overall_passed = func_res["passed"] + mock_res["passed"]
overall_failures = func_res["failures"] + mock_res["failures"]
overall_errors = func_res["errors"] + mock_res["errors"]

overall_success = (
    func_res["success"]
    and mock_res["success"]
    and not _real_socket_called
    and not _real_db_called
    and (overall_total > 0)
)

out = {{
    "success": overall_success,
    "functional_passed": func_res["success"],
    "mockability_passed": mock_res["success"],
    "total_tests": overall_total,
    "passed_tests": overall_passed,
    "failed_tests": overall_failures,
    "error_tests": overall_errors,
    "real_socket_called": _real_socket_called,
    "real_db_called": _real_db_called,
    "socket_details": _socket_details,
    "db_details": _db_details,
    "failure_messages": func_res["fail_msgs"] + mock_res["fail_msgs"],
    "error_messages": func_res["err_msgs"] + mock_res["err_msgs"],
}}

with open("_result.json", "w") as f:
    json.dump(out, f)

print("__REFACTOR_RESULT_JSON__ = " + json.dumps(out))
"""


class SandboxedExecutionGate:
    """Sandboxed test execution gate with timeout guards and resource limits."""

    def __init__(
        self,
        *,
        timeout_s: float = 5.0,
        max_memory_mb: int = 512,
        block_network: bool = True,
        block_database: bool = True,
        allow_sqlite_memory: bool = False,
        sandbox_provider: Optional[Any] = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_memory_mb = max_memory_mb
        self.block_network = block_network
        self.block_database = block_database
        self.allow_sqlite_memory = allow_sqlite_memory
        self.sandbox_provider = sandbox_provider

    def _set_resource_limits(self) -> None:
        """Apply OS-level resource limits in child subprocess on Unix platforms."""
        try:
            import resource

            if self.max_memory_mb > 0:
                mem_bytes = int(self.max_memory_mb * 1024 * 1024)
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            if self.timeout_s > 0:
                cpu_secs = int(max(self.timeout_s * 2, 5))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_secs, cpu_secs))
        except Exception:
            pass

    def run_tests(
        self,
        files: Mapping[str, str],
        *,
        tests: Optional[str | Mapping[str, str] | Sequence[str]] = None,
        mock_tests: Optional[str | Mapping[str, str] | Sequence[str]] = None,
    ) -> TestExecutionResult:
        """Run functional and mockability test suites in isolated sandbox."""
        t0 = time.monotonic()
        temp_dir = tempfile.mkdtemp(prefix="refactor_gate_")

        try:
            # 1. Write target refactored files
            for rel_path, content in files.items():
                target_path = Path(temp_dir) / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Ensure __init__.py files in parents
                cur = target_path.parent
                while cur != Path(temp_dir):
                    init_py = cur / "__init__.py"
                    if not init_py.exists():
                        init_py.write_text("")
                    cur = cur.parent
                target_path.write_text(content)

            # 2. Write functional test files
            functional_mods: List[str] = []
            if tests:
                if isinstance(tests, str):
                    test_file = Path(temp_dir) / "test_suite.py"
                    test_file.write_text(tests)
                    functional_mods.append("test_suite")
                elif isinstance(tests, Mapping):
                    for k, v in tests.items():
                        norm_k = normalize_file_path(k)
                        (Path(temp_dir) / norm_k).parent.mkdir(parents=True, exist_ok=True)
                        (Path(temp_dir) / norm_k).write_text(str(v))
                        mod = norm_k.removesuffix(".py").replace("/", ".")
                        functional_mods.append(mod)
                elif isinstance(tests, Sequence):
                    for i, t in enumerate(tests):
                        fname = f"test_suite_{i}.py"
                        (Path(temp_dir) / fname).write_text(str(t))
                        functional_mods.append(f"test_suite_{i}")

            # 3. Write mockability test files
            mock_mods: List[str] = []
            if mock_tests:
                if isinstance(mock_tests, str):
                    mock_file = Path(temp_dir) / "test_mockability.py"
                    mock_file.write_text(mock_tests)
                    mock_mods.append("test_mockability")
                elif isinstance(mock_tests, Mapping):
                    for k, v in mock_tests.items():
                        norm_k = normalize_file_path(k)
                        (Path(temp_dir) / norm_k).parent.mkdir(parents=True, exist_ok=True)
                        (Path(temp_dir) / norm_k).write_text(str(v))
                        mod = norm_k.removesuffix(".py").replace("/", ".")
                        mock_mods.append(mod)
                elif isinstance(mock_tests, Sequence):
                    for i, t in enumerate(mock_tests):
                        fname = f"test_mockability_{i}.py"
                        (Path(temp_dir) / fname).write_text(str(t))
                        mock_mods.append(f"test_mockability_{i}")

            # If no explicit tests provided, look for test_*.py in files
            if not functional_mods and not mock_mods:
                for rel_path in files:
                    if (
                        rel_path.startswith("test_")
                        or rel_path.endswith("_test.py")
                        or "/test_" in rel_path
                    ):
                        mod = rel_path.removesuffix(".py").replace("/", ".")
                        functional_mods.append(mod)

            # 4. Generate and write harness runner
            harness_content = _HARNESS_TEMPLATE.format(
                block_network=self.block_network,
                block_database=self.block_database,
                allow_sqlite_memory=self.allow_sqlite_memory,
                functional_mods=repr(functional_mods),
                mock_mods=repr(mock_mods),
            )
            harness_file = Path(temp_dir) / "_harness.py"
            harness_file.write_text(harness_content)

            # 5. Execute runner in sandbox
            if self.sandbox_provider is not None:
                res = self.sandbox_provider.run_command(
                    [sys.executable, "_harness.py"],
                    cwd=temp_dir,
                    timeout=self.timeout_s,
                )
                stdout = res.stdout
                stderr = res.stderr
                timed_out = res.timed_out
            else:
                preexec = self._set_resource_limits if sys.platform != "win32" else None
                try:
                    proc = subprocess.run(
                        [sys.executable, "_harness.py"],
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_s,
                        preexec_fn=preexec,
                    )
                    stdout = proc.stdout
                    stderr = proc.stderr
                    timed_out = False
                except subprocess.TimeoutExpired:
                    stdout = ""
                    stderr = f"Execution timed out after {self.timeout_s}s"
                    timed_out = True

            duration = time.monotonic() - t0

            # 6. Parse result JSON
            result_file = Path(temp_dir) / "_result.json"
            if result_file.exists() and not timed_out:
                try:
                    data = json.loads(result_file.read_text())
                    return TestExecutionResult(
                        passed=bool(data.get("success", False)),
                        total_tests=int(data.get("total_tests", 0)),
                        passed_tests=int(data.get("passed_tests", 0)),
                        failed_tests=int(data.get("failed_tests", 0)),
                        error_tests=int(data.get("error_tests", 0)),
                        functional_passed=bool(data.get("functional_passed", False)),
                        mockability_passed=bool(data.get("mockability_passed", False)),
                        timed_out=False,
                        real_socket_detected=bool(data.get("real_socket_called", False)),
                        real_db_detected=bool(data.get("real_db_called", False)),
                        socket_details=data.get("socket_details", []),
                        db_details=data.get("db_details", []),
                        stdout=stdout,
                        stderr=stderr,
                        error=None if data.get("success") else ("\n".join(data.get("failure_messages", []) + data.get("error_messages", []))),
                        duration_s=duration,
                    )
                except Exception as e:
                    stderr += f"\nError parsing _result.json: {e}"

            # Fallback if no result JSON written (e.g. timeout or fatal crash)
            return TestExecutionResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                error_tests=1 if not timed_out else 0,
                functional_passed=False,
                mockability_passed=False,
                timed_out=timed_out,
                real_socket_detected=False,
                real_db_detected=False,
                stdout=stdout,
                stderr=stderr,
                error="Timeout" if timed_out else (stderr or "Execution failed before emitting results"),
                duration_s=duration,
            )

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Refactoring Oracle & Invariant Verifier
# --------------------------------------------------------------------------- #

@dataclass
class RefactoringTask:
    """Specification of a refactoring evaluation case."""
    id: str
    pattern: str = RefactoringPattern.STRATEGY.value
    prompt: str = ""
    initial_code: str | Mapping[str, str] = ""
    tests: str | Mapping[str, str] = ""
    mock_tests: Optional[str | Mapping[str, str]] = None
    target_interfaces: Optional[Sequence[str]] = None
    entrypoint: Optional[str] = None


@dataclass
class RefactoringAnalysisResult:
    """Outcome of verifying a refactoring completion."""
    reward: float
    is_clean: bool
    all_tests_passed: bool
    functional_passed: bool
    mockability_passed: bool
    no_socket_opened: bool
    no_db_opened: bool
    is_degenerate: bool
    degenerate_reason: Optional[str]
    is_hacked: bool
    hatches: List[str]
    diagnostics: List[Diagnostic]
    patterns_detected: List[str]
    test_result: Optional[TestExecutionResult] = None
    elapsed_ms: float = 0.0


class RefactoringOracle:
    """Thread-safe execution oracle seam for refactoring evaluation."""

    def __init__(
        self,
        *,
        timeout_s: float = 5.0,
        max_memory_mb: int = 512,
        block_network: bool = True,
        block_database: bool = True,
        allow_sqlite_memory: bool = False,
        sandbox_provider: Optional[Any] = None,
        gate: Optional[SandboxedExecutionGate] = None,
    ) -> None:
        self.gate = gate or SandboxedExecutionGate(
            timeout_s=timeout_s,
            max_memory_mb=max_memory_mb,
            block_network=block_network,
            block_database=block_database,
            allow_sqlite_memory=allow_sqlite_memory,
            sandbox_provider=sandbox_provider,
        )
        self._lock = threading.Lock()
        self.n_calls = 0
        self.wall_s = 0.0

    def analyze(
        self,
        files: Mapping[str, str],
        *,
        tests: Optional[str | Mapping[str, str] | Sequence[str]] = None,
        mock_tests: Optional[str | Mapping[str, str] | Sequence[str]] = None,
        target_interfaces: Optional[Sequence[str]] = None,
        pattern: Optional[str] = None,
    ) -> Tuple[List[Diagnostic], RefactoringAstSummary, TestExecutionResult]:
        """Run AST analysis and sandboxed test execution gate."""
        t0 = time.monotonic()
        with self._lock:
            self.n_calls += 1

        diagnostics: List[Diagnostic] = []

        # 1. AST Analysis
        ast_summary = analyze_refactoring_ast(files)
        for syn_err in ast_summary.syntax_errors:
            diagnostics.append(
                Diagnostic(
                    code="REFACTOR_SYNTAX_ERROR",
                    line=1,
                    col=1,
                    message=syn_err,
                    offset=0,
                    source="refactoring_ast",
                    severity=1,
                )
            )

        # Check for target interfaces
        if target_interfaces:
            for iface in target_interfaces:
                if iface not in ast_summary.interfaces and iface not in ast_summary.classes:
                    diagnostics.append(
                        Diagnostic(
                            code="REFACTOR_MISSING_INTERFACE",
                            line=1,
                            col=1,
                            message=f"Expected interface or protocol '{iface}' was not found",
                            offset=0,
                            source="refactoring_ast",
                            severity=2,
                        )
                    )

        # Check for hardcoded I/O
        for h_io in ast_summary.hardcoded_io_calls:
            diagnostics.append(
                Diagnostic(
                    code="REFACTOR_HARDCODED_DEPENDENCY",
                    line=1,
                    col=1,
                    message=f"Hardcoded direct I/O call detected in class: {h_io}",
                    offset=0,
                    source="refactoring_ast",
                    severity=2,
                )
            )

        # If syntax errors present, don't execute broken code
        if ast_summary.syntax_errors:
            test_res = TestExecutionResult(
                passed=False,
                error="Syntax error prevents test execution",
                duration_s=time.monotonic() - t0,
            )
            with self._lock:
                self.wall_s += time.monotonic() - t0
            return diagnostics, ast_summary, test_res

        # 2. Sandboxed Test Execution Gate
        test_res = self.gate.run_tests(files, tests=tests, mock_tests=mock_tests)

        if test_res.timed_out:
            diagnostics.append(
                Diagnostic(
                    code="REFACTOR_TEST_TIMEOUT",
                    line=1,
                    col=1,
                    message=f"Test suite execution timed out: {test_res.error}",
                    offset=0,
                    source="refactoring_gate",
                    severity=1,
                )
            )
        elif not test_res.passed:
            diagnostics.append(
                Diagnostic(
                    code="REFACTOR_TEST_FAILURE",
                    line=1,
                    col=1,
                    message=f"Functional tests failed: {test_res.failed_tests} failed, {test_res.error_tests} errors ({test_res.error or 'failure'})",
                    offset=0,
                    source="refactoring_gate",
                    severity=1,
                )
            )

        # 3. Check Mockability & Leaky I/O Guards
        if test_res.real_socket_detected:
            diagnostics.append(
                Diagnostic(
                    code="REFACTOR_LEAKY_IO",
                    line=1,
                    col=1,
                    message=f"Real network socket opened during unit tests: {test_res.socket_details}",
                    offset=0,
                    source="refactoring_gate",
                    severity=1,
                )
            )

        if test_res.real_db_detected:
            diagnostics.append(
                Diagnostic(
                    code="REFACTOR_LEAKY_IO",
                    line=1,
                    col=1,
                    message=f"Real database connection opened during unit tests: {test_res.db_details}",
                    offset=0,
                    source="refactoring_gate",
                    severity=1,
                )
            )

        if mock_tests and not test_res.mockability_passed:
            diagnostics.append(
                Diagnostic(
                    code="REFACTOR_MOCKABILITY_FAILED",
                    line=1,
                    col=1,
                    message="Mockability verification suite failed to execute with mocked interfaces",
                    offset=0,
                    source="refactoring_gate",
                    severity=1,
                )
            )

        with self._lock:
            self.wall_s += time.monotonic() - t0

        return diagnostics, ast_summary, test_res

    def close(self) -> None:
        """Release any persistent gate resources."""
        if hasattr(self.gate, "sandbox_provider") and self.gate.sandbox_provider:
            try:
                self.gate.sandbox_provider.close()
            except Exception:
                pass


class RefactoringInvariantVerifier:
    """Verifier for behavioral invariance under refactoring & interface decoupling (#347).

    Reward Contract:
    - Base reward: 1.0 (clean)
    - 1.0 if and only if 100% of functional tests pass without regressions, mockability passes,
      and zero real network sockets or database connections are opened.
    - Test failures, timeouts, and leaky IO reduce reward according to severity weights.
    - Degenerate output (empty, whitespace, comment-only) short-circuits to -1.0.
    - Anti-Goodhart escape hatches (@unittest.skip, refactor-ignore, eval bypass) short-circuit to -1.0.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 5.0,
        max_memory_mb: int = 512,
        block_network: bool = True,
        block_database: bool = True,
        allow_sqlite_memory: bool = False,
        fail_fast: bool = True,
        code_overrides: Optional[Mapping[str, float]] = None,
        severity_weights: Mapping[int, float] = DEFAULT_SEVERITY_WEIGHTS,
        oracle: Optional[RefactoringOracle] = None,
        gate: Optional[SandboxedExecutionGate] = None,
        sandbox_provider: Optional[Any] = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_memory_mb = max_memory_mb
        self.block_network = block_network
        self.block_database = block_database
        self.allow_sqlite_memory = allow_sqlite_memory
        self.fail_fast = fail_fast
        self.code_overrides = dict(DEFAULT_REFACTORING_CODE_OVERRIDES)
        if code_overrides:
            self.code_overrides.update(code_overrides)
        self.severity_weights = severity_weights

        self.oracle = oracle or RefactoringOracle(
            timeout_s=timeout_s,
            max_memory_mb=max_memory_mb,
            block_network=block_network,
            block_database=block_database,
            allow_sqlite_memory=allow_sqlite_memory,
            sandbox_provider=sandbox_provider,
            gate=gate,
        )

        self._lock = threading.Lock()
        self._n_samples = 0
        self._n_clean = 0
        self._n_degenerate = 0
        self._n_hacked = 0
        self._n_test_failures = 0
        self._n_leaky_io = 0
        self._reward_total = 0.0
        self._elapsed_ms_total = 0.0

    def find_escape_hatches(self, text: str) -> List[str]:
        """Detect anti-Goodhart escape hatches or test bypass directives."""
        return find_refactoring_escape_hatches(text)

    def evaluate(
        self,
        completion: str | Mapping[str, str] | Sequence[dict],
        reference: Any = None,
        *,
        tests: Optional[str | Mapping[str, str] | Sequence[str]] = None,
        mock_tests: Optional[str | Mapping[str, str] | Sequence[str]] = None,
        prompt: str = "",
        pattern: Optional[str | RefactoringPattern] = None,
        target_interfaces: Optional[Sequence[str]] = None,
        entrypoint: Optional[str] = None,
        default_filename: str = "refactored.py",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate completion against behavioral tests, mockability, and interface decoupling."""
        t0 = time.monotonic()

        if isinstance(completion, str):
            raw_text = completion
        elif isinstance(completion, Mapping):
            raw_text = "\n".join(str(v) for v in completion.values())
        else:
            raw_text = str(completion)

        degen_reason = is_refactor_degenerate_output(raw_text)
        is_degenerate = degen_reason is not None

        hatches = self.find_escape_hatches(raw_text)
        is_hacked = len(hatches) > 0

        # Short-circuit on degenerate or hacked outputs
        if self.fail_fast and (is_degenerate or is_hacked):
            reward = -1.0
            elapsed = (time.monotonic() - t0) * 1000.0
            with self._lock:
                self._n_samples += 1
                if is_degenerate:
                    self._n_degenerate += 1
                if is_hacked:
                    self._n_hacked += 1
                self._reward_total += reward
                self._elapsed_ms_total += elapsed

            return {
                "reward": reward,
                "is_clean": False,
                "all_tests_passed": False,
                "functional_passed": False,
                "mockability_passed": False,
                "no_socket_opened": True,
                "no_db_opened": True,
                "is_degenerate": is_degenerate,
                "degenerate_reason": degen_reason,
                "is_hacked": is_hacked,
                "hatches": hatches,
                "diagnostics": [
                    Diagnostic(
                        code="REFACTOR_DEGENERATE_OUTPUT" if is_degenerate else "REFACTOR_ESCAPE_HATCH",
                        line=1,
                        col=1,
                        message=f"Degenerate: {degen_reason}" if is_degenerate else f"Escape hatch: {hatches}",
                        offset=0,
                        source="refactoring_verifier",
                        severity=1,
                    )
                ],
                "patterns_detected": [],
                "test_result": None,
                "elapsed_ms": elapsed,
            }

        # Parse files
        files = parse_multi_file_source(completion, default_filename=default_filename)

        # Run oracle analysis
        resolved_pattern = pattern.value if isinstance(pattern, RefactoringPattern) else pattern
        diagnostics, ast_summary, test_result = self.oracle.analyze(
            files,
            tests=tests,
            mock_tests=mock_tests,
            target_interfaces=target_interfaces,
            pattern=resolved_pattern,
        )

        reward = diagnostics_to_reward(
            diagnostics,
            hacked=is_hacked,
            degenerate=is_degenerate,
            severity_weights=self.severity_weights,
            code_overrides=self.code_overrides,
        )

        # Invariant: Reward 1.0 if and only if 100% tests pass and clean
        no_socket = not test_result.real_socket_detected
        no_db = not test_result.real_db_detected
        all_passed = bool(test_result.passed)

        is_clean = (
            (reward == 1.0)
            and all_passed
            and no_socket
            and no_db
            and not is_hacked
            and not is_degenerate
            and len(ast_summary.syntax_errors) == 0
        )

        if not is_clean and reward == 1.0:
            reward = 0.50  # Cap reward if not 100% clean

        elapsed = (time.monotonic() - t0) * 1000.0

        with self._lock:
            self._n_samples += 1
            if is_clean:
                self._n_clean += 1
            if not all_passed:
                self._n_test_failures += 1
            if not no_socket or not no_db:
                self._n_leaky_io += 1
            self._reward_total += reward
            self._elapsed_ms_total += elapsed

        return {
            "reward": reward,
            "is_clean": is_clean,
            "all_tests_passed": all_passed,
            "functional_passed": test_result.functional_passed,
            "mockability_passed": test_result.mockability_passed,
            "no_socket_opened": no_socket,
            "no_db_opened": no_db,
            "is_degenerate": is_degenerate,
            "degenerate_reason": degen_reason,
            "is_hacked": is_hacked,
            "hatches": hatches,
            "diagnostics": diagnostics,
            "patterns_detected": ast_summary.patterns_detected,
            "test_result": test_result,
            "elapsed_ms": elapsed,
        }

    def reward(
        self,
        completion: str | Mapping[str, str] | Sequence[dict],
        reference: Any = None,
        *,
        prompt: str = "",
        **kwargs: Any,
    ) -> float:
        """Score one completion against behavioral invariance and decoupling."""
        eval_res = self.evaluate(completion, reference=reference, prompt=prompt, **kwargs)
        return float(eval_res["reward"])

    def telemetry(self) -> Dict[str, Any]:
        """Aggregate telemetric metrics."""
        with self._lock:
            n = self._n_samples
            return {
                "verifier": "RefactoringInvariantVerifier",
                "n_samples": n,
                "n_clean": self._n_clean,
                "n_hacked": self._n_hacked,
                "n_degenerate": self._n_degenerate,
                "n_test_failures": self._n_test_failures,
                "n_leaky_io": self._n_leaky_io,
                "mean_reward": (self._reward_total / n) if n else 0.0,
                "mean_elapsed_ms": (self._elapsed_ms_total / n) if n else 0.0,
                "oracle_calls": self.oracle.n_calls,
                "oracle_wall_s": self.oracle.wall_s,
            }

    def close(self) -> None:
        """Release oracle resources."""
        self.oracle.close()
