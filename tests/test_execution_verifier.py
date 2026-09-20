"""Unit tests for isolated execution runner and mutation testing reward engine (#361).

Tests SandboxedCodeVerifier, mutation operators, deterministic kill scoring,
and RLVR integration without external network access.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

from src.train.verifiers.execution import (
    SandboxedCodeVerifier,
    resolve_execution_toolchain,
)
from src.train.verifiers.mutations import (
    AlterArithmetic,
    InvertConditionals,
    ModifyBoundaryConstants,
    MutationEngine,
    score_mutation,
)


def test_toolchain_resolution():
    """Verify python/node execution toolchain resolution."""
    assert resolve_execution_toolchain() is True


def test_sandboxed_verifier_terminates_infinite_loop():
    """Verify SandboxedCodeVerifier terminates infinite loops within configured timeout."""
    verifier = SandboxedCodeVerifier(timeout_s=0.5, memory_limit_mb=256)
    infinite_loop_code = """
def solution():
    while True:
        pass

solution()
"""
    res = verifier.execute_with_tests(infinite_loop_code, ["assert True"], timeout=0.5)
    assert res.timed_out is True
    assert res.pass_fraction == 0.0
    assert not res.memory_exceeded


def test_sandboxed_verifier_terminates_memory_allocation():
    """Verify SandboxedCodeVerifier terminates excessive memory allocations."""
    verifier = SandboxedCodeVerifier(timeout_s=2.0, memory_limit_mb=100)
    # Allocate ~250MB which exceeds the 100MB limit
    mem_alloc_code = """
import time
data = bytearray(250 * 1024 * 1024)
time.sleep(2)
"""
    res = verifier.execute_with_tests(mem_alloc_code, ["assert True"], timeout=2.0)
    assert res.memory_exceeded is True
    assert res.pass_fraction == 0.0


def test_sandboxed_verifier_blocks_restricted_system_calls():
    """Verify SandboxedCodeVerifier restricts network connections and unauthorized calls."""
    verifier = SandboxedCodeVerifier(timeout_s=2.0, restricted_system_calls=True)
    network_code = """
import socket

def solution():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("1.1.1.1", 80))

solution()
"""
    res = verifier.execute_with_tests(network_code, ["assert True"])
    assert res.restricted_blocked is True
    assert res.pass_fraction == 0.0


def test_python_pytest_runner_execution():
    """Verify Python execution and pass fraction computation via pytest."""
    verifier = SandboxedCodeVerifier(timeout_s=2.0)
    code = """
def add(a: int, b: int) -> int:
    return a + b
"""
    # 3 passing tests, 1 failing test
    tests = [
        "assert add(1, 2) == 3",
        "assert add(0, 0) == 0",
        "assert add(-1, 1) == 0",
        "assert add(2, 2) == 5",  # failing test
    ]
    res = verifier.execute_with_tests(code, tests, language="python", runner="pytest")
    assert res.passed_count == 3
    assert res.failed_count == 1
    assert res.pass_fraction == 0.75
    assert not res.timed_out
    assert not res.memory_exceeded


def test_ts_node_test_runner_execution():
    """Verify TypeScript execution and pass fraction computation via node:test."""
    verifier = SandboxedCodeVerifier(timeout_s=2.0)
    code = """
export function multiply(a: number, b: number): number {
    return a * b;
}
"""
    tests = [
        "assert.strictEqual(multiply(2, 3), 6);",
        "assert.strictEqual(multiply(0, 5), 0);",
        "assert.strictEqual(multiply(2, 2), 5);",  # failing
    ]
    res = verifier.execute_with_tests(code, tests, language="typescript", runner="node:test")
    assert res.passed_count == 2
    assert res.failed_count == 1
    assert res.pass_fraction == pytest.approx(2 / 3, rel=1e-2)
    assert not res.timed_out
    assert not res.memory_exceeded


def test_mutation_operators_python():
    """Verify mutation operators inject synthetic faults into Python AST."""
    code = """
def evaluate(a: int, b: int) -> bool:
    if a > 0 and b <= 10:
        return True
    return False

def compute(x: int) -> int:
    return x + 1
"""
    tree = ast.parse(code)

    # 1. Invert Conditionals
    op_cond = InvertConditionals()
    mutants_cond = op_cond.generate_python_mutants(tree, code)
    assert len(mutants_cond) >= 3
    descriptions = [m.description for m in mutants_cond]
    assert any("Invert comparison" in d for d in descriptions)
    assert any("Invert logical operator" in d for d in descriptions)
    assert any("Invert boolean constant" in d for d in descriptions)

    # 2. Alter Arithmetic
    op_arith = AlterArithmetic()
    mutants_arith = op_arith.generate_python_mutants(tree, code)
    assert len(mutants_arith) >= 1
    assert any("Alter arithmetic" in m.description for m in mutants_arith)

    # 3. Modify Boundary Constants
    op_bound = ModifyBoundaryConstants()
    mutants_bound = op_bound.generate_python_mutants(tree, code)
    assert len(mutants_bound) >= 2
    assert any("Modify boundary" in m.description for m in mutants_bound)


def test_mutation_operators_typescript():
    """Verify mutation operators inject synthetic faults into TypeScript code."""
    ts_code = """
export function checkAndCompute(a: number, b: number): number {
    if (a > 0 && b <= 10) {
        return a + b;
    } else if (a === 0) {
        return 1;
    }
    return 0;
}
"""
    engine = MutationEngine(max_mutants=20)
    mutants = engine.generate_mutants(ts_code, language="typescript")
    assert len(mutants) >= 5

    operators = {m.operator for m in mutants}
    assert "invert_conditionals" in operators
    assert "alter_arithmetic" in operators
    assert "modify_boundary_constants" in operators


def test_deterministic_kill_scoring():
    """Verify deterministic mutation kill scoring across test suites."""
    engine = MutationEngine(max_mutants=10)
    code = """
def abs_val(x: int) -> int:
    if x < 0:
        return -x
    return x
"""
    mutants_1 = engine.generate_mutants(code, language="python")
    mutants_2 = engine.generate_mutants(code, language="python")

    # Mutation generation must be strictly deterministic
    assert len(mutants_1) == len(mutants_2)
    for m1, m2 in zip(mutants_1, mutants_2):
        assert m1.id == m2.id
        assert m1.mutated_code == m2.mutated_code
        assert m1.description == m2.description

    # Score calculation math: R = R_pass * (1 + gamma * R_mutation)
    # Case 1: pass_rate = 1.0, 4/5 mutants killed, gamma = 0.5 -> 1.0 * (1 + 0.5 * 0.8) = 1.4
    s1 = score_mutation(pass_rate=1.0, mutants_tested=5, mutants_killed=4, gamma=0.5)
    assert s1 == pytest.approx(1.4)

    # Case 2: pass_rate = 0.0 -> reward is 0.0 regardless of mutants
    s2 = score_mutation(pass_rate=0.0, mutants_tested=5, mutants_killed=5, gamma=0.5)
    assert s2 == 0.0

    # Case 3: 0 mutants tested -> returns pass_rate
    s3 = score_mutation(pass_rate=0.8, mutants_tested=0, mutants_killed=0, gamma=0.5)
    assert s3 == pytest.approx(0.8)


def test_verifier_reward_completions():
    """Verify SandboxedCodeVerifier rewards completions that pass tests and kill mutants."""
    verifier = SandboxedCodeVerifier(timeout_s=2.0, gamma=0.5, max_mutants=5)

    code = """
def is_even(n: int) -> bool:
    return n % 2 == 0
"""
    # Comprehensive test suite killing mutants
    strong_tests = [
        "assert is_even(0) is True",
        "assert is_even(1) is False",
        "assert is_even(2) is True",
        "assert is_even(-2) is True",
        "assert is_even(-1) is False",
    ]
    r_strong = verifier.reward(code, tests=strong_tests, language="python")
    assert r_strong > 1.0  # Boosted above 1.0 by mutant kills

    # Flawed code passing only 2 of 5 tests receives partial credit (0.4) without mutation boost
    flawed_code = """
def is_even(n: int) -> bool:
    return False
"""
    r_flawed = verifier.reward(flawed_code, tests=strong_tests, language="python")
    assert r_flawed == pytest.approx(0.4)

    # Completely broken code failing all tests receives 0.0
    broken_code = """
def is_even(n: int) -> bool:
    raise RuntimeError("broken")
"""
    r_broken = verifier.reward(broken_code, tests=strong_tests, language="python")
    assert r_broken == 0.0

    # Degenerate code receives -1.0
    r_degen = verifier.reward("   // just a comment\n", tests=strong_tests)
    assert r_degen == -1.0


def test_verifier_telemetry_tracking():
    """Verify verifier telemetry records assertion pass rates, mutation scores, and timeouts."""
    verifier = SandboxedCodeVerifier(timeout_s=1.0, gamma=0.5, max_mutants=3)

    code = "def f(x): return x + 1"
    tests = ["assert f(1) == 2"]
    verifier.reward(code, tests=tests, language="python")

    # Time out rollout
    timeout_code = "while True: pass"
    verifier.reward(timeout_code, tests=["assert True"], language="python")

    # Degenerate rollout
    verifier.reward("", tests=tests)

    telem = verifier.telemetry()
    assert telem["n_samples"] == 3
    assert telem["n_pass"] == 1
    assert telem["n_timeouts"] == 1
    assert telem["n_degenerate"] == 1
    assert telem["n_mutants_tested"] > 0
    assert telem["wall_s"] > 0.0
    assert "assertion_pass_rate" in telem
    assert "mutation_score" in telem


def test_rlvr_smoke_10_steps(tmp_path: Path):
    """Acceptance: scripts/rlvr.py --reward execution completes 10 optimization steps."""
    import importlib.util

    if importlib.util.find_spec("mlx") is None and importlib.util.find_spec("torch") is None:
        pytest.skip("test requires either mlx or torch backend")

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "toy-mhm.yaml"

    from src.model.backend import get_backend
    from src.model.blocks import load_config

    cfg = load_config(str(cfg_path))
    backend = get_backend()
    model = backend.model_cls(cfg)

    init_path = tmp_path / "weights.safetensors"
    model.save(str(init_path))

    problems_path = tmp_path / "problems.jsonl"
    problems = [
        {
            "prompt": "def add(a: int, b: int) -> int:\n    return a + b\n",
            "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
            "language": "python",
        },
        {
            "prompt": "def sub(a: int, b: int) -> int:\n    return a - b\n",
            "tests": ["assert sub(3, 1) == 2", "assert sub(0, 0) == 0"],
            "language": "python",
        },
    ]
    with open(problems_path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    out_dir = tmp_path / "rlvr_exec_out"

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "rlvr.py"),
        "--config", str(cfg_path),
        "--init", str(init_path),
        "--problems", str(problems_path),
        "--reward", "execution",
        "--steps", "10",
        "--group-size", "2",
        "--verifier-workers", "1",
        "--byte-fallback",
        "--max-new-tokens", "4",
        "--out", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    assert res.returncode == 0, f"rlvr failed: STDOUT: {res.stdout}\nSTDERR: {res.stderr}"

    assert (out_dir / "weights.safetensors").exists()
    assert (out_dir / "telemetry.json").exists()

    telem = json.loads((out_dir / "telemetry.json").read_text(encoding="utf-8"))
    assert "assertion_pass_rate" in telem
    assert "mutation_score" in telem
    assert "n_timeouts" in telem
    assert telem["n_samples"] >= 10
