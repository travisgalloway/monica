"""The HumanEval-TS loader emits records the scorer accepts (#199 F1).

No network: builds a record from a fake MultiPL-E row and asserts it (a) satisfies the
#194 7-key schema so `validate_record`/`score_record` accept it, and (b) carries the F1
extras (`tests`, `stop_tokens`, `name`) that the driver needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.eval.ts_error_eval import validate_record


def _record_from_row(row: dict) -> dict:
    """Mirror scripts/build_humaneval_ts_set.py's per-row mapping (kept in sync by hand;
    the script's dataset dependency makes importing it here not worth the network)."""
    return {
        "id": row["name"],
        "error_class": "clean_control",
        "expected_diagnostic": "",
        "prompt": row["prompt"],
        "gold_completion": "\n",
        "error_completion": "",
        "notes": f"MultiPL-E humaneval-ts {row['name']}",
        "name": row["name"],
        "tests": row["tests"],
        "stop_tokens": row["stop_tokens"],
    }


FAKE_ROW = {
    "name": "HumanEval_0_has_close_elements",
    "prompt": "function has_close_elements(numbers: number[], threshold: number): boolean {\n",
    "tests": "assert(has_close_elements([1.0, 2.0], 0.5) === false);\n",
    "stop_tokens": ["\nfunction ", "\n/*", "\n//", "\nclass"],
}


def test_record_satisfies_the_194_schema():
    rec = _record_from_row(FAKE_ROW)
    validate_record(rec)                       # raises on any schema violation
    assert rec["error_class"] == "clean_control"
    assert rec["expected_diagnostic"] == ""    # clean_control => excluded from error_avoidance_rate


def test_record_carries_f1_extras():
    rec = _record_from_row(FAKE_ROW)
    assert rec["stop_tokens"] == FAKE_ROW["stop_tokens"]
    assert rec["tests"] == FAKE_ROW["tests"]
    assert rec["name"] == FAKE_ROW["name"]


def test_record_scores_without_keyerror():
    """A built record must round-trip through score_record (which reads only the 7 keys)."""
    from src.eval.lsp_eval import score_record

    @dataclass
    class _Result:
        prompt: str
        completion: str
        n_rollbacks: int = 0
        no_progress: bool = False
        events: tuple = ()
        n_soft_repairs: int = 0

        @property
        def artifact(self) -> str:
            return self.prompt + self.completion

    rec = _record_from_row(FAKE_ROW)
    res = _Result(prompt=rec["prompt"], completion="  return false;\n}\n")
    scored = score_record(rec, res, diagnose=lambda src: [])   # no tsc: pretend clean
    assert scored["id"] == FAKE_ROW["name"]
    assert scored["is_error_row"] is False       # clean_control
    assert scored["clean"] is True
