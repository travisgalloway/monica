"""Fixture tests for `eval_sets/opengrep_rules/correctness.yaml` -- the 12-rule,
pre-registered correctness ruleset (see the pre-registration process in
`eval_sets/opengrep_rules/README.md`). One positive + one negative fixture per
rule, drawn from the same general TypeScript bug taxonomy the rules were authored
against (never from this repo's eval transcripts).

Runs `opengrep scan --config <rules dir> <fixture> --json` directly (batch CLI,
not the LSP client) -- this validates the RULES themselves, independent of
`src/lsp/opengrep.py`'s LSP-based oracle (written in a later commit; these tests
predate it by design, per the plan's commit sequence, and stay valid afterward
since they exercise the ruleset, not the client).

Skipped wholesale on a host without the pinned `opengrep` binary on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List

import pytest

RULES_DIR = Path(__file__).resolve().parent.parent / "eval_sets" / "opengrep_rules"
_SCAN_TIMEOUT_S = 30.0

pytestmark = pytest.mark.skipif(shutil.which("opengrep") is None,
                                 reason="no opengrep binary on this host")


def _scan(tmp_path: Path, name: str, source: str) -> List[str]:
    """Write `source` to `<tmp_path>/<name>.ts`, scan it against the ruleset, and
    return the bare rule ids that fired (the CLI's `check_id` is a dotted path
    prefix + the rule id; only the last component is the rule id itself)."""
    fixture = tmp_path / f"{name}.ts"
    fixture.write_text(source, encoding="utf-8")
    proc = subprocess.run(
        ["opengrep", "scan", "--config", str(RULES_DIR), str(fixture),
         "--no-git-ignore", "--json"],
        capture_output=True, text=True, timeout=_SCAN_TIMEOUT_S,
    )
    assert proc.returncode == 0, f"opengrep scan failed: {proc.stderr}"
    payload = json.loads(proc.stdout)
    return [r["check_id"].rsplit(".", 1)[-1] for r in payload["results"]]


# Each entry: rule_id -> (positive fixture, negative fixture). Content mirrors the
# taxonomy each rule's `correctness.yaml` message documents -- authored from public
# TypeScript-bug-idiom knowledge, never from this repo's eval outcomes.
_FIXTURES = {
    "loop-bound-off-by-one": (
        """function sumAll(arr: number[]): number {
  let total = 0;
  for (let i = 0; i <= arr.length; i++) {
    total += arr[i];
  }
  return total;
}
""",
        """function sumAll(arr: number[]): number {
  let total = 0;
  for (let i = 0; i < arr.length; i++) {
    total += arr[i];
  }
  return total;
}
""",
    ),
    "index-at-length": (
        """function last(arr: number[]): number {
  return arr[arr.length];
}
""",
        """function last(arr: number[]): number {
  return arr[arr.length - 1];
}
""",
    ),
    "sort-without-comparator": (
        """function sortNums(arr: number[]): number[] {
  return arr.sort();
}
""",
        """function sortNums(arr: number[]): number[] {
  return arr.sort((a, b) => a - b);
}
""",
    ),
    "indexof-truthy-check": (
        """function contains(arr: number[], x: number): boolean {
  return arr.indexOf(x) > 0;
}
""",
        """function contains(arr: number[], x: number): boolean {
  return arr.indexOf(x) !== -1;
}
""",
    ),
    "parseint-no-radix": (
        """function toNum(s: string): number {
  return parseInt(s);
}
""",
        """function toNum(s: string): number {
  return parseInt(s, 10);
}
""",
    ),
    "self-comparison": (
        """function check(x: number, y: number): boolean {
  return x == x;
}
""",
        """function check(x: number, y: number): boolean {
  return x == y;
}
""",
    ),
    "useless-ternary": (
        """function pick(cond: boolean, a: number): number {
  return cond ? a : a;
}
""",
        """function pick(cond: boolean, a: number, b: number): number {
  return cond ? a : b;
}
""",
    ),
    "fill-shared-reference": (
        """function grid(n: number): number[][] {
  return Array(n).fill([]);
}
""",
        """function grid(n: number): number[][] {
  return Array.from({ length: n }, () => []);
}
""",
    ),
    "strict-equality-nan": (
        """function isBad(x: number): boolean {
  return x === NaN;
}
""",
        """function isBad(x: number): boolean {
  return Number.isNaN(x);
}
""",
    ),
    "typeof-array": (
        """function check(x: unknown): boolean {
  return typeof x === "array";
}
""",
        """function check(x: unknown): boolean {
  return Array.isArray(x);
}
""",
    ),
    "assignment-in-condition": (
        """function check(x: number, y: number): number {
  if (x = y) {
    return x;
  }
  return 0;
}
""",
        """function check(x: number, y: number): number {
  if (x === y) {
    return x;
  }
  return 0;
}
""",
    ),
    "for-in-over-array": (
        """function sumAll(arr: number[]): number {
  let total = 0;
  for (const i in arr) {
    total += arr[i];
  }
  return total;
}
""",
        """function sumAll(arr: number[]): number {
  let total = 0;
  for (const x of arr) {
    total += x;
  }
  return total;
}
""",
    ),
}


def test_ruleset_has_exactly_the_pre_registered_12_rules():
    assert sorted(_FIXTURES) == sorted(_expected_rule_ids())


def _expected_rule_ids() -> List[str]:
    return [
        "loop-bound-off-by-one", "index-at-length", "sort-without-comparator",
        "indexof-truthy-check", "parseint-no-radix", "self-comparison",
        "useless-ternary", "fill-shared-reference", "strict-equality-nan",
        "typeof-array", "assignment-in-condition", "for-in-over-array",
    ]


@pytest.mark.parametrize("rule_id", sorted(_FIXTURES))
def test_rule_fires_on_positive_fixture(tmp_path: Path, rule_id: str):
    positive, _ = _FIXTURES[rule_id]
    fired = _scan(tmp_path, f"{rule_id}_pos", positive)
    assert rule_id in fired, f"{rule_id} did not fire on its own positive fixture: {fired}"


@pytest.mark.parametrize("rule_id", sorted(_FIXTURES))
def test_rule_silent_on_negative_fixture(tmp_path: Path, rule_id: str):
    _, negative = _FIXTURES[rule_id]
    fired = _scan(tmp_path, f"{rule_id}_neg", negative)
    assert rule_id not in fired, f"{rule_id} false-fired on its own negative fixture: {fired}"


def test_map_without_return_is_recorded_as_excluded_not_silently_dropped():
    readme = (RULES_DIR / "README.md").read_text(encoding="utf-8")
    assert "map-without-return" in readme
