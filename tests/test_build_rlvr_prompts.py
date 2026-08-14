"""#230 -- split determinism / disjointness / manifest tests for
`scripts/build_rlvr_prompts.py`. Pure Python (stdlib only), no toolchain, no
backend -- exercises the real checked-in eval sets, not fixtures, since the
whole point is a reproducible split OVER those files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_rlvr_prompts.py"
_spec = importlib.util.spec_from_file_location("build_rlvr_prompts", _SCRIPT_PATH)
build_rlvr_prompts = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = build_rlvr_prompts
_spec.loader.exec_module(build_rlvr_prompts)

build_manifest = build_rlvr_prompts.build_manifest
load_prompt_records = build_rlvr_prompts.load_prompt_records
split_records = build_rlvr_prompts.split_records


@pytest.fixture(scope="module")
def records():
    return load_prompt_records()


def test_load_prompt_records_no_duplicate_ids(records):
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids))
    assert len(records) > 0
    for r in records:
        assert set(r) == {"id", "prompt", "answer", "error_class"}


def test_split_same_seed_is_identical(records):
    train_a, val_a = split_records(records, seed=0)
    train_b, val_b = split_records(records, seed=0)
    assert [r["id"] for r in train_a] == [r["id"] for r in train_b]
    assert [r["id"] for r in val_a] == [r["id"] for r in val_b]


def test_split_different_seed_differs(records):
    train_0, val_0 = split_records(records, seed=0)
    train_1, val_1 = split_records(records, seed=1)
    assert {r["id"] for r in val_0} != {r["id"] for r in val_1}


def test_split_train_val_ids_disjoint(records):
    train, val = split_records(records, seed=0)
    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {r["id"] for r in records}


def test_split_every_error_class_present_on_both_sides(records):
    train, val = split_records(records, seed=0)
    all_classes = {r["error_class"] for r in records}
    train_classes = {r["error_class"] for r in train}
    val_classes = {r["error_class"] for r in val}
    assert train_classes == all_classes
    assert val_classes == all_classes


def test_split_respects_val_fraction_roughly(records):
    train, val = split_records(records, seed=0, val_fraction=0.2)
    frac = len(val) / (len(train) + len(val))
    assert 0.1 <= frac <= 0.35   # stratified per-class rounding, not exact


def test_manifest_sha256_stable_and_recomputable(records):
    train, val = split_records(records, seed=0)
    sources = ["eval.jsonl", "clean_prefixes.jsonl"]
    m1 = build_manifest(train, val, seed=0, val_fraction=0.2, sources=sources)
    m2 = build_manifest(train, val, seed=0, val_fraction=0.2, sources=sources)
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["train_ids_sha256"] == m2["train_ids_sha256"]
    assert m1["val_ids_sha256"] == m2["val_ids_sha256"]
    assert m1["n_train"] == len(train)
    assert m1["n_val"] == len(val)
    # per_class_counts sums back to the real split.
    total_train = sum(c["train"] for c in m1["per_class_counts"].values())
    total_val = sum(c["val"] for c in m1["per_class_counts"].values())
    assert total_train == len(train)
    assert total_val == len(val)


def test_manifest_id_hash_changes_if_split_changes(records):
    train_0, val_0 = split_records(records, seed=0)
    train_1, val_1 = split_records(records, seed=1)
    m0 = build_manifest(train_0, val_0, seed=0, val_fraction=0.2, sources=[])
    m1 = build_manifest(train_1, val_1, seed=1, val_fraction=0.2, sources=[])
    assert m0["val_ids_sha256"] != m1["val_ids_sha256"]


def test_include_humaneval_adds_clean_control_rows():
    without = load_prompt_records(include_humaneval=False)
    with_he = load_prompt_records(include_humaneval=True)
    assert len(with_he) > len(without)
    new_ids = {r["id"] for r in with_he} - {r["id"] for r in without}
    assert all(rid.startswith("HumanEval_") for rid in new_ids)


def test_main_writes_train_val_and_manifest(tmp_path):
    out_dir = tmp_path / "rlvr_ts"
    argv = ["build_rlvr_prompts.py", "--out-dir", str(out_dir), "--seed", "0"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        build_rlvr_prompts.main()
    finally:
        sys.argv = old_argv

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    manifest_path = out_dir / "rlvr_split_manifest.json"
    assert train_path.exists()
    assert val_path.exists()
    assert manifest_path.exists()

    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_train_lines = sum(1 for ln in train_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    n_val_lines = sum(1 for ln in val_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    assert manifest["n_train"] == n_train_lines
    assert manifest["n_val"] == n_val_lines
    assert "contamination_note" in manifest
