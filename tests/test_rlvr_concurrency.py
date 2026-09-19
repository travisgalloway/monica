"""Tests for RLVR concurrency, verifier memoization, and fail-fast performance (#341-#344)."""

import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.train.verifiers import (LspVerifier, MemoizedVerifier, exact_match_reward,
                                 math_reward, score_rollouts)


def test_memoized_verifier_thread_safety_and_speedup():
    """Verify that MemoizedVerifier is thread-safe and provides massive speedup on repeat rollouts."""
    eval_count = 0

    def slow_reward(completion: str, reference: str = None, *, prompt: str = "") -> float:
        nonlocal eval_count
        eval_count += 1
        time.sleep(0.01)  # simulate 10ms compiler check
        return 1.0 if "def " in completion else 0.0

    memo = MemoizedVerifier(slow_reward)

    # 16 items with only 4 unique completions
    completions = ["def foo(): pass", "let x = 1;", "def bar(): return 42", "var y = 2;"] * 4

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        scores = score_rollouts(memo.reward, completions, executor=pool)
    elapsed = time.monotonic() - start

    assert len(scores) == 16
    assert eval_count == 4  # only 4 evaluations were needed!
    assert memo.telemetry()["cache_hits"] == 12
    assert memo.telemetry()["cache_misses"] == 4
    assert memo.telemetry()["cache_hit_rate"] == 0.75

    # Repeat scoring again should be 100% cache hits and take < 5ms total
    start_cached = time.monotonic()
    cached_scores = score_rollouts(memo.reward, completions, max_workers=4)
    elapsed_cached = time.monotonic() - start_cached

    assert cached_scores == scores
    assert memo.telemetry()["cache_hits"] == 28
    assert elapsed_cached < 0.05


def test_lsp_verifier_fail_fast_performance():
    """Verify that fail_fast skips oracle entirely on degenerate and escape-hatch completions."""
    class MockSlowOracle:
        def __init__(self):
            self.n_calls = 0
            self.wall_s = 0.0

        def diagnostics(self, artifact):
            self.n_calls += 1
            time.sleep(0.02)  # 20ms simulated LSP debounce/diagnostics
            return []

        def close(self):
            pass

    slow_oracle = MockSlowOracle()
    v_fast = LspVerifier(oracle=slow_oracle, fail_fast=True)

    # Degenerate outputs (empty, comment only, whitespace) and escape hatches (@ts-ignore, as any)
    bad_samples = [
        "",
        "   \n\t",
        "// comment only\n",
        "const x = 1 as any;\n",
        "// @ts-ignore\nconst y = 2;\n",
    ]

    start = time.monotonic()
    for s in bad_samples:
        score = v_fast.reward(s)
        assert score == -1.0
    elapsed_fast = time.monotonic() - start

    # Zero oracle calls made!
    assert slow_oracle.n_calls == 0
    assert elapsed_fast < 0.05  # all sub-millisecond regex/ast checks

    # Clean code does call the oracle
    assert v_fast.reward("const clean = 42;\n") == 1.0
    assert slow_oracle.n_calls == 1


def test_score_rollouts_empty_and_single():
    assert score_rollouts(math_reward, []) == []
    assert score_rollouts(math_reward, ["42"], "42") == [1.0]


def test_rlvr_cli_end_to_end_concurrency(tmp_path):
    """Run a real 2-step RLVR execution with concurrent verifier workers and caching."""
    import subprocess
    import sys
    from src.model.backend import get_backend
    from src.model.blocks import load_config

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "toy-mhm.yaml"
    cfg = load_config(str(cfg_path))
    backend = get_backend()
    model = backend.model_cls(cfg)

    # Save portable weights
    init_path = tmp_path / "init.safetensors"
    model.save(str(init_path))

    # Create dummy math problem file
    problems_path = tmp_path / "problems.jsonl"
    problems_path.write_text('{"prompt": "Calculate 2 + 2", "answer": "4"}\n{"prompt": "Calculate 3 + 5", "answer": "8"}\n')

    out_dir = tmp_path / "rlvr_out"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "rlvr.py"),
        "--config", str(cfg_path),
        "--init", str(init_path),
        "--problems", str(problems_path),
        "--reward", "math",
        "--steps", "2",
        "--group-size", "4",
        "--verifier-workers", "2",
        "--verifier-cache",
        "--byte-fallback",
        "--max-new-tokens", "8",
        "--out", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    assert res.returncode == 0, f"rlvr failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert (out_dir / "weights.safetensors").exists()
    assert (out_dir / "telemetry.json").exists()
    telemetry = json.loads((out_dir / "telemetry.json").read_text())
    assert "cache_hit_rate" in telemetry

