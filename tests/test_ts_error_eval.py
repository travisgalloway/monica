"""Tests for the TS error-injection eval set loader (#194).

Portable schema/loader tests run everywhere. One test additionally shells out to a
real `tsc` to sanity-check a couple of records against the actual compiler; it is
skipped (not module-level, so the rest of the file still runs) on a node-less host.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.ts_error_eval import (DEFAULT_SET_PATH, EXPECTED_DIAGNOSTIC,
                                     load_ts_error_set, validate_record)

_BASE_RECORD = {
    "id": "base-001",
    "error_class": "unfamiliar_member_access",
    "expected_diagnostic": "TS2339",
    "prompt": "interface User { name: string; }\nconst u: User = { name: \"a\" };\nconst v = u.",
    "gold_completion": "name;\n",
    "error_completion": "nope;\n",
    "notes": "test fixture",
}


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "set.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return p


def test_default_set_loads_clean():
    records = load_ts_error_set(DEFAULT_SET_PATH)
    assert len(records) == 96


def test_default_set_is_balanced():
    records = load_ts_error_set(DEFAULT_SET_PATH)
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["error_class"]] = counts.get(rec["error_class"], 0) + 1
    assert counts == {
        "unfamiliar_member_access": 28,
        "undefined_name": 28,
        "arity_mismatch": 28,
        "clean_control": 12,
    }


def test_default_set_has_unique_ids():
    records = load_ts_error_set(DEFAULT_SET_PATH)
    ids = [rec["id"] for rec in records]
    assert len(ids) == len(set(ids))


def test_valid_record_passes():
    validate_record(_BASE_RECORD)  # must not raise


def test_valid_clean_control_record_passes():
    rec = dict(_BASE_RECORD, error_class="clean_control", expected_diagnostic="",
               error_completion="")
    validate_record(rec)  # must not raise


def test_missing_key_raises(tmp_path):
    rec = dict(_BASE_RECORD)
    del rec["notes"]
    with pytest.raises(ValueError):
        load_ts_error_set(_write_jsonl(tmp_path, [rec]))


def test_unknown_error_class_raises(tmp_path):
    rec = dict(_BASE_RECORD, error_class="bogus_class")
    with pytest.raises(ValueError):
        load_ts_error_set(_write_jsonl(tmp_path, [rec]))


def test_diagnostic_class_mismatch_raises(tmp_path):
    rec = dict(_BASE_RECORD, expected_diagnostic="TS9999")
    with pytest.raises(ValueError):
        load_ts_error_set(_write_jsonl(tmp_path, [rec]))


def test_duplicate_ids_raises(tmp_path):
    with pytest.raises(ValueError):
        load_ts_error_set(_write_jsonl(tmp_path, [_BASE_RECORD, dict(_BASE_RECORD)]))


def test_empty_file_raises(tmp_path):
    with pytest.raises(ValueError):
        load_ts_error_set(_write_jsonl(tmp_path, []))


def test_non_clean_control_requires_error_completion(tmp_path):
    rec = dict(_BASE_RECORD, error_completion="")
    with pytest.raises(ValueError):
        load_ts_error_set(_write_jsonl(tmp_path, [rec]))


@pytest.mark.skipif(shutil.which("node") is None, reason="no node toolchain on this host")
def test_real_tsc_confirms_a_few_labels():
    from scripts.validate_ts_error_set import (DEFAULT_TSCONFIG_PATH, resolve_tsc,
                                                tsc_diagnostics)

    tsc_argv = resolve_tsc()
    if tsc_argv is None:
        pytest.skip("no tsc toolchain resolvable (run `npm install` in eval_sets/ts_error_injection)")

    records = load_ts_error_set(DEFAULT_SET_PATH)
    non_clean = [r for r in records if r["error_class"] != "clean_control"][:2]
    assert non_clean, "expected at least one non-clean_control record"

    for rec in non_clean:
        gold_codes = tsc_diagnostics(rec["prompt"] + rec["gold_completion"],
                                      DEFAULT_TSCONFIG_PATH, tsc_argv)
        assert gold_codes == [], f"{rec['id']}: gold produced {gold_codes}"

        error_codes = tsc_diagnostics(rec["prompt"] + rec["error_completion"],
                                       DEFAULT_TSCONFIG_PATH, tsc_argv)
        expected = EXPECTED_DIAGNOSTIC[rec["error_class"]]
        assert expected in error_codes, f"{rec['id']}: error produced {error_codes}, expected {expected}"
