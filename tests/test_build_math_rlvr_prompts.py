"""#103 -- tests for scripts/build_math_rlvr_prompts.py and eval_sets/math."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_math_rlvr_prompts.py"
_spec = importlib.util.spec_from_file_location("build_math_rlvr_prompts", _SCRIPT_PATH)
build_math_rlvr_prompts = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = build_math_rlvr_prompts
_spec.loader.exec_module(build_math_rlvr_prompts)

build_manifest = build_math_rlvr_prompts.build_manifest
load_math_records = build_math_rlvr_prompts.load_math_records
split_records = build_math_rlvr_prompts.split_records


@pytest.fixture(scope="module")
def math_records():
    return load_math_records()


def test_load_math_records_no_duplicate_ids(math_records):
    ids = [r["id"] for r in math_records]
    assert len(ids) == len(set(ids))
    assert len(math_records) >= 20
    for r in math_records:
        assert set(r) == {"id", "prompt", "answer", "error_class", "domain"}
        assert r["domain"] == "math"
        assert r["error_class"] in ("gsm8k", "math")


def test_split_same_seed_is_identical(math_records):
    train_a, val_a = split_records(math_records, seed=0)
    train_b, val_b = split_records(math_records, seed=0)
    assert [r["id"] for r in train_a] == [r["id"] for r in train_b]
    assert [r["id"] for r in val_a] == [r["id"] for r in val_b]


def test_split_train_val_ids_disjoint(math_records):
    train, val = split_records(math_records, seed=0)
    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {r["id"] for r in math_records}


def test_split_every_error_class_present_on_both_sides(math_records):
    train, val = split_records(math_records, seed=0)
    classes = {r["error_class"] for r in math_records}
    assert {r["error_class"] for r in train} == classes
    assert {r["error_class"] for r in val} == classes


def test_manifest_structure_and_hash(math_records):
    train, val = split_records(math_records, seed=0)
    manifest = build_manifest(train, val, seed=0, val_fraction=0.2,
                              sources=["eval_sets/math/gsm8k.jsonl", "eval_sets/math/math.jsonl"])
    assert manifest["n_train"] == len(train)
    assert manifest["n_val"] == len(val)
    assert "train_ids_sha256" in manifest
    assert "val_ids_sha256" in manifest
    assert "manifest_sha256" in manifest


def test_math_verifier_scores_gold_answers(math_records):
    from src.train.verifiers import MathVerifier
    verifier = MathVerifier(use_sympy=True)
    for r in math_records:
        reward = verifier.reward(r["answer"], r["answer"])
        assert reward == 1.0, f"gold answer failed verification: {r['id']}: {r['answer']}"
