"""Tests for `src/eval/ssi_contract.py` -- the #225 M1-M5 measurement contract."""

from __future__ import annotations

import pytest

from src.eval.ssi_contract import (ArmSpec, ContractViolation, RepoSplit, assert_disjoint,
                                    contract_report, deletion_of_target, find_escape_hatches,
                                    has_escape_hatch, per_seed_compare, pooled_compare,
                                    repo_of, sign_test_p, split_by_repo, split_manifest,
                                    summarize_arm, validate_arms, wilcoxon_signed_rank_p)


def _arm(name, variable, baseline, *, signal_available=False, signal_used=False,
         seeds=(1, 2, 3)):
    return ArmSpec(name=name, variable=variable, baseline=baseline,
                    signal_available=signal_available, signal_used=signal_used, seeds=seeds)


# --------------------------------------------------------------------------- #
# validate_arms -- M1, M2, M4
# --------------------------------------------------------------------------- #

def test_validate_arms_rejects_empty_set():
    with pytest.raises(ContractViolation):
        validate_arms([])


def test_validate_arms_rejects_fewer_than_three_seeds():
    arms = [_arm("baseline", "control", "baseline"),
            _arm("treatment", "masking", "baseline", seeds=(1, 2))]
    with pytest.raises(ContractViolation):
        validate_arms(arms)


def test_validate_arms_rejects_unknown_baseline():
    arms = [_arm("treatment", "masking", "nonexistent")]
    with pytest.raises(ContractViolation):
        validate_arms(arms)


def test_validate_arms_rejects_empty_variable():
    arms = [_arm("baseline", "control", "baseline"),
            ArmSpec(name="treatment", variable="", baseline="baseline",
                    signal_available=False, signal_used=False, seeds=(1, 2, 3))]
    with pytest.raises(ContractViolation):
        validate_arms(arms)


def test_validate_arms_rejects_two_same_baseline_arms_sharing_a_variable():
    arms = [
        _arm("baseline", "control", "baseline"),
        _arm("treatment_a", "masking", "baseline"),
        _arm("treatment_b", "masking", "baseline"),
    ]
    with pytest.raises(ContractViolation):
        validate_arms(arms)


def test_validate_arms_rejects_signal_used_arm_with_no_null_sibling():
    arms = [
        _arm("baseline", "control", "baseline"),
        _arm("treatment", "masking", "baseline", signal_available=True, signal_used=True),
    ]
    with pytest.raises(ContractViolation):
        validate_arms(arms)


def test_validate_arms_accepts_wellformed_baseline_null_treatment_set():
    arms = [
        _arm("baseline", "control", "baseline"),
        _arm("null", "masking", "baseline", signal_available=True, signal_used=False),
        _arm("treatment", "masking", "baseline", signal_available=True, signal_used=True),
    ]
    validate_arms(arms)  # must not raise


# --------------------------------------------------------------------------- #
# per_seed_compare / pooled_compare
# --------------------------------------------------------------------------- #

def _rec(rid, clean):
    return {"id": rid, "clean": clean}


def test_per_seed_compare_raises_on_mismatched_seed_sets():
    baseline_by_seed = {1: [_rec("a", True)], 2: [_rec("a", True)]}
    other_by_seed = {1: [_rec("a", True)]}
    with pytest.raises(ContractViolation):
        per_seed_compare(baseline_by_seed, other_by_seed, key="clean")


def test_per_seed_compare_raises_on_mismatched_record_id_order():
    baseline_by_seed = {1: [_rec("a", True), _rec("b", False)]}
    other_by_seed = {1: [_rec("b", True), _rec("a", False)]}
    with pytest.raises(ContractViolation):
        per_seed_compare(baseline_by_seed, other_by_seed, key="clean")


def test_per_seed_compare_returns_expected_2x2_table():
    # Seed 1: a both-true, b baseline-false/other-true (a "flip to true").
    baseline_by_seed = {1: [_rec("a", True), _rec("b", False)]}
    other_by_seed = {1: [_rec("a", True), _rec("b", True)]}
    result = per_seed_compare(baseline_by_seed, other_by_seed, key="clean")
    assert set(result) == {1}
    table = result[1]["table"]
    assert table == {"both_true": 1, "baseline_true_other_false": 0,
                      "baseline_false_other_true": 1, "both_false": 0}


def test_pooled_compare_concatenates_all_seeds():
    baseline_by_seed = {1: [_rec("a", True)], 2: [_rec("b", False)]}
    other_by_seed = {1: [_rec("a", True)], 2: [_rec("b", True)]}
    pooled = pooled_compare(baseline_by_seed, other_by_seed, key="clean")
    assert pooled["n"] == 2
    assert pooled["table"]["both_true"] == 1
    assert pooled["table"]["baseline_false_other_true"] == 1


# --------------------------------------------------------------------------- #
# sign_test_p
# --------------------------------------------------------------------------- #

def test_sign_test_p_known_values():
    assert sign_test_p(3, 0) == pytest.approx(0.25)
    assert sign_test_p(0, 0) == 1.0


def test_sign_test_p_symmetric():
    assert sign_test_p(4, 1) == pytest.approx(sign_test_p(1, 4))


# --------------------------------------------------------------------------- #
# wilcoxon_signed_rank_p
# --------------------------------------------------------------------------- #

def test_wilcoxon_rejects_more_than_20_nonzero_deltas():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank_p(list(range(1, 22)))


def test_wilcoxon_all_positive_gives_minimum_attainable_p():
    deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = wilcoxon_signed_rank_p(deltas)
    n = len(deltas)
    assert p == pytest.approx(2 / (2 ** n))


def test_wilcoxon_all_zero_deltas_give_one():
    assert wilcoxon_signed_rank_p([0.0, 0.0, 0.0]) == 1.0


# --------------------------------------------------------------------------- #
# summarize_arm
# --------------------------------------------------------------------------- #

def test_summarize_arm_reports_consistent_direction():
    per_seed = {
        1: {"baseline_rate": 0.5, "other_rate": 0.6, "mcnemar_p": 0.5},
        2: {"baseline_rate": 0.4, "other_rate": 0.5, "mcnemar_p": 0.5},
    }
    out = summarize_arm(per_seed, sign_test_p(2, 0), {"n": 2})
    assert out["consistent_direction"] is True
    assert out["n_seeds"] == 2


def test_summarize_arm_reports_inconsistent_direction():
    per_seed = {
        1: {"baseline_rate": 0.5, "other_rate": 0.6, "mcnemar_p": 0.5},
        2: {"baseline_rate": 0.5, "other_rate": 0.4, "mcnemar_p": 0.5},
    }
    out = summarize_arm(per_seed, sign_test_p(1, 1), {"n": 2})
    assert out["consistent_direction"] is False


# --------------------------------------------------------------------------- #
# repo_of / split_by_repo / assert_disjoint / split_manifest
# --------------------------------------------------------------------------- #

def _repo_rec(repo, i):
    return {"id": f"{repo}:{i}", "meta": {"repo": repo} if repo is not None else {}}


def test_repo_of_reads_meta_repo_and_returns_none_when_absent():
    assert repo_of({"meta": {"repo": "org/name"}}) == "org/name"
    assert repo_of({"meta": {}}) is None
    assert repo_of({}) is None


def test_split_by_repo_whole_repos_land_on_one_side():
    records = [_repo_rec(f"org/repo{i}", j) for i in range(20) for j in range(3)]
    split = split_by_repo(records, eval_fraction=0.3, seed=42)
    assert_disjoint(split.train, split.eval)  # must not raise


def test_split_by_repo_stable_across_processes():
    # Hard-coded expected assignment for seed=42 -- a salted `hash()` based
    # implementation would fail this (it isn't stable across interpreters).
    records = [_repo_rec("org/alpha", 0), _repo_rec("org/beta", 0),
               _repo_rec("org/gamma", 0)]
    split = split_by_repo(records, eval_fraction=0.5, seed=42)
    eval_repos = sorted({repo_of(r) for r in split.eval})
    train_repos = sorted({repo_of(r) for r in split.train})
    assert eval_repos == ["org/alpha", "org/beta"]
    assert train_repos == ["org/gamma"]


def test_split_by_repo_on_unknown_raises_by_default():
    records = [_repo_rec(None, 0)]
    with pytest.raises(ContractViolation):
        split_by_repo(records, eval_fraction=0.5, seed=1)


def test_split_by_repo_on_unknown_quarantine_buckets_and_counts():
    records = [_repo_rec(None, 0), _repo_rec("org/alpha", 0)]
    split = split_by_repo(records, eval_fraction=0.5, seed=1, on_unknown="quarantine")
    assert len(split.quarantine) == 1
    manifest = split_manifest(split)
    assert manifest["n_quarantine"] == 1


def test_split_manifest_byte_reproducible():
    records = [_repo_rec(f"org/repo{i}", 0) for i in range(10)]
    split_a = split_by_repo(records, eval_fraction=0.4, seed=7)
    split_b = split_by_repo(records, eval_fraction=0.4, seed=7)
    assert split_manifest(split_a)["manifest_sha256"] == split_manifest(split_b)["manifest_sha256"]


def test_split_manifest_changes_when_eval_repos_change():
    records = [_repo_rec(f"org/repo{i}", 0) for i in range(10)]
    split_a = split_by_repo(records, eval_fraction=0.4, seed=7)
    manifest_a = split_manifest(split_a)

    more_records = records + [_repo_rec("org/extra", 0)]
    split_b = split_by_repo(more_records, eval_fraction=0.9, seed=7)
    manifest_b = split_manifest(split_b)

    assert manifest_a["eval_repos_sha256"] != manifest_b["eval_repos_sha256"]


def test_assert_disjoint_raises_on_overlap():
    a = [_repo_rec("org/shared", 0)]
    b = [_repo_rec("org/shared", 1)]
    with pytest.raises(ContractViolation):
        assert_disjoint(a, b)


# --------------------------------------------------------------------------- #
# Glue -- re-exports + contract_report
# --------------------------------------------------------------------------- #

def test_ssi_contract_reexports_the_m5_gate():
    assert has_escape_hatch("const v = u as any;")
    assert "as_any" in find_escape_hatches("const v = u as any;")
    assert deletion_of_target("function f() {}\n", "\n", anchors=["function f"]) == ["function f"]


def test_contract_report_assembles_arms_splits_stats():
    arms = [_arm("baseline", "control", "baseline")]
    records = [_repo_rec("org/repo0", 0)]
    split = split_by_repo(records, eval_fraction=0.0, seed=1)
    report = contract_report(arms, {"main": split}, {"baseline_vs_null": {"n": 1}})
    assert report["arms"][0]["name"] == "baseline"
    assert "main" in report["splits"]
    assert report["stats"]["baseline_vs_null"]["n"] == 1
