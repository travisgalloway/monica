"""Loader/validator for the TS error-injection eval set (#194).

`eval_sets/ts_error_injection/eval.jsonl` is a labeled, held-out set of TypeScript
"error-injected completion" examples: a compilable `prompt` paired with a correct
`gold_completion` and a wrong reference `error_completion` that deliberately triggers
a specific, real `tsc` diagnostic. It is the ground truth **#199**'s LSP-harness
scores against (diagnostic-clean rate, error-induced pass rate) — this module only
loads and schema-validates the set; it does not run `tsc` (that stays
`scripts/validate_ts_error_set.py`'s `tsc_diagnostics()`, reusable as #199's
`diagnose_fn`).

Three error families map to real `tsc` diagnostic codes (`EXPECTED_DIAGNOSTIC`),
plus `clean_control` rows (already-correct completions, testing over-repair / false
positives): see `eval_sets/ts_error_injection/README.md` for the full schema and
provenance.

ABOVE THE SEAM — stdlib only (`json`, `pathlib`). No `mlx`/`torch` import anywhere in
this module (guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

ERROR_CLASSES = (
    "unfamiliar_member_access",
    "undefined_name",
    "arity_mismatch",
    "clean_control",
)

EXPECTED_DIAGNOSTIC = {
    "unfamiliar_member_access": "TS2339",
    "undefined_name": "TS2304",
    "arity_mismatch": "TS2554",
    "clean_control": "",
}

DEFAULT_SET_PATH = (Path(__file__).resolve().parent.parent.parent
                     / "eval_sets" / "ts_error_injection" / "eval.jsonl")

_REQUIRED_KEYS = ("id", "error_class", "expected_diagnostic", "prompt",
                   "gold_completion", "error_completion", "notes")


def validate_record(rec: dict) -> None:
    """Raise `ValueError` if `rec` doesn't satisfy the eval-set schema.

    Checks: all required keys present; `error_class` is a known class; `prompt` and
    `gold_completion` are non-empty; `expected_diagnostic` equals the class's mapped
    `tsc` code; `error_completion` is non-empty for non-`clean_control` rows and
    empty for `clean_control` rows (the schema treats it as required-but-empty
    there, so downstream consumers can rely on it never being set).
    """
    missing = [k for k in _REQUIRED_KEYS if k not in rec]
    if missing:
        raise ValueError(f"record {rec.get('id', '<no id>')!r} missing keys: {missing}")

    error_class = rec["error_class"]
    if error_class not in ERROR_CLASSES:
        raise ValueError(f"record {rec['id']!r} has unknown error_class: {error_class!r}")

    if not rec["prompt"]:
        raise ValueError(f"record {rec['id']!r} has empty prompt")
    if not rec["gold_completion"]:
        raise ValueError(f"record {rec['id']!r} has empty gold_completion")

    expected = EXPECTED_DIAGNOSTIC[error_class]
    if rec["expected_diagnostic"] != expected:
        raise ValueError(
            f"record {rec['id']!r} has error_class {error_class!r} but "
            f"expected_diagnostic {rec['expected_diagnostic']!r} (expected {expected!r})")

    if error_class != "clean_control" and not rec["error_completion"]:
        raise ValueError(f"record {rec['id']!r} ({error_class}) has empty error_completion")
    if error_class == "clean_control" and rec["error_completion"]:
        raise ValueError(f"record {rec['id']!r} (clean_control) must have empty error_completion")


def load_ts_error_set(path: Path = DEFAULT_SET_PATH) -> List[dict]:
    """Read, schema-validate, and return the JSONL records at `path`.

    Raises `ValueError` if the file is empty, any record fails `validate_record`, or
    two records share an `id`.
    """
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        raise ValueError(f"no records found in {path}")

    seen_ids = set()
    for rec in records:
        validate_record(rec)
        if rec["id"] in seen_ids:
            raise ValueError(f"duplicate id in {path}: {rec['id']!r}")
        seen_ids.add(rec["id"])

    return records
