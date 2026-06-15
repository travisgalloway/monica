"""Verifiable rewards for RLVR / GRPO (#78).

The cleanest post-training stage on licensing: the reward comes from a **verifier**
(exact-match, compiler, test runner) — not licensed data — so only problems + tests are
needed; the model generates the solutions (docs/design/08-corpus-pipeline.md lines 120-123).
**Math first** (exact-match, no sandbox — the cheapest clean reward loop), then a guarded
code path.

ABOVE THE SEAM — stdlib only, no backend. `exact_match_reward` / `math_reward` are pure and
safe. `CodeVerifier` **executes untrusted model output** and is therefore disabled by
default (`enabled=False`) and never run in CI; real TS/Rust/SQL grading uses an external
sandbox (SandboxFusion), out of scope here.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import List, Optional, Sequence

_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_text(s: str) -> str:
    """Lowercase, trim, collapse internal whitespace."""
    return _WS.sub(" ", (s or "").strip().lower())


def exact_match_reward(answer: str, gold: str) -> float:
    """1.0 if `answer` matches `gold` after normalization, else 0.0."""
    return 1.0 if normalize_text(answer) == normalize_text(gold) else 0.0


def extract_final_number(s: str) -> Optional[float]:
    """The final numeric answer in `s` (GSM8K-style: prefer text after `####`, else the
    last number). Strips thousands separators. Returns None if there is no number."""
    if not s:
        return None
    txt = s.split("####")[-1] if "####" in s else s
    nums = _NUM.findall(txt.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def math_reward(answer: str, gold: str, *, tol: float = 1e-6) -> float:
    """1.0 if the final number in `answer` equals the gold answer within `tol`, else 0.0.
    `gold` may be a bare number or a full solution string (its final number is used)."""
    a, g = extract_final_number(answer), extract_final_number(gold)
    if a is None or g is None:
        return 0.0
    return 1.0 if abs(a - g) <= tol else 0.0


class CodeVerifier:
    """Run candidate code against a test suite, rewarding the **fraction of tests passing**
    (partial credit; use >=5 tests per problem so a thin suite can't be gamed — the main
    RLVR failure mode).

    UNSAFE: executes untrusted model output in a subprocess. Disabled by default; set
    `enabled=True` to opt in (never in CI). Python only — TS/Rust/SQL belong in an external
    sandbox.
    """

    def __init__(self, *, timeout: float = 5.0, enabled: bool = False):
        self.timeout = timeout
        self.enabled = enabled

    def reward(self, code: str, tests: Sequence[str]) -> float:
        if not self.enabled:
            raise RuntimeError(
                "CodeVerifier is disabled (it executes untrusted code). "
                "Pass enabled=True to opt in — never in CI.")
        if not tests:
            return 0.0
        passed = 0
        for t in tests:
            program = f"{code}\n{t}\n"
            try:
                # Discard untrusted stdout/stderr (a candidate can print unbounded data);
                # only the exit code matters.
                r = subprocess.run([sys.executable, "-c", program],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=self.timeout)
                passed += int(r.returncode == 0)
            except subprocess.TimeoutExpired:
                pass
        return passed / len(tests)
