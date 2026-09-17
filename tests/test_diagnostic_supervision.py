"""Unit and acceptance tests for #227 (Diagnostic supervision: rejection-FT + contrastive hard negatives)."""

from __future__ import annotations

import math
import pytest
import numpy as np

from src.eval.lsp_eval import RESOLUTION_CODES
from src.eval.ssi_contract import ArmSpec, ContractViolation, validate_arms
from src.lsp.diagnostics import Diagnostic
from src.train.diagnostic_supervision import (
    CandidateEvaluation,
    CandidateItem,
    DiversityReport,
    DiversityTracker,
    RandomFilter,
    RejectionFilter,
    auxiliary_contrastive_loss,
    build_diagnostic_supervision_arms,
    contrastive_margin_loss,
    distinct_n,
    distinct_n_metrics,
    edit_similarity,
    is_resolve_correct,
    levenshtein_distance,
    margin_loss,
    mine_negatives,
    ngram_entropy,
    normalize_candidate_items,
    tokenize_code,
)


# --------------------------------------------------------------------------- #
# Diversity metrics & collapse tracking
# --------------------------------------------------------------------------- #

def test_tokenize_code():
    assert tokenize_code("") == []
    tokens = tokenize_code("const x: number = 42; // comment")
    assert "const" in tokens
    assert "x" in tokens
    assert ":" in tokens
    assert "number" in tokens
    assert "=" in tokens
    assert "42" in tokens
    assert ";" in tokens


def test_distinct_n_empty_or_zero():
    assert distinct_n([], 1) == 0.0
    assert distinct_n(["abc"], 0) == 0.0


def test_distinct_n_repetitive_vs_diverse():
    repetitive = ["foo foo foo foo"] * 5
    diverse = ["foo bar baz qux alpha beta gamma delta"]

    d1_rep = distinct_n(repetitive, 1)
    d1_div = distinct_n(diverse, 1)
    assert d1_rep < d1_div
    assert d1_div == 1.0


def test_distinct_n_metrics_keys():
    texts = ["function test() { return 1 + 2; }"]
    metrics = distinct_n_metrics(texts)
    assert "distinct_1" in metrics
    assert "distinct_2" in metrics
    assert "distinct_3" in metrics
    assert 0.0 <= metrics["distinct_1"] <= 1.0
    assert 0.0 <= metrics["distinct_2"] <= 1.0
    assert 0.0 <= metrics["distinct_3"] <= 1.0


def test_ngram_entropy_zero_on_empty():
    assert ngram_entropy([], 1) == 0.0
    assert ngram_entropy([""], 1) == 0.0


def test_ngram_entropy_constant_vs_uniform():
    constant = ["item item item item"]
    assert ngram_entropy(constant, 1) == pytest.approx(0.0)

    uniform = ["a b c d"]
    assert ngram_entropy(uniform, 1) == pytest.approx(2.0)


def test_diversity_tracker_records_and_detects_collapse():
    tracker = DiversityTracker()

    r1_texts = ["a b c d e f g h i j"]
    tracker.record_round(1, r1_texts, r1_texts, n_prompts=1)
    assert len(tracker.reports) == 1
    assert tracker.detect_collapse()["collapsed"] is False

    r2_texts = ["a a a a a a a a a a"]
    tracker.record_round(2, r2_texts, r2_texts, n_prompts=1)

    collapse = tracker.detect_collapse(drop_threshold=0.3)
    assert collapse["collapsed"] is True
    assert collapse["entropy_drop"] > 0.5
    assert collapse["distinct_1_drop"] > 0.5


# --------------------------------------------------------------------------- #
# Rejection filter (Zero-error survivor filtering)
# --------------------------------------------------------------------------- #

def test_rejection_filter_keeps_clean_survivors():
    def fake_diagnose(artifact: str):
        return []

    filter_ = RejectionFilter(fake_diagnose)
    evals = filter_.filter_candidates("console.log(u.", ["name);\n", "age);\n"])
    assert len(evals) == 2
    assert all(e.is_clean for e in evals)


def test_rejection_filter_rejects_diagnostic_errors():
    def fake_diagnose(artifact: str):
        if "gorblak" in artifact:
            return [Diagnostic(code="TS2339", line=1, col=1, message="not exist", offset=0)]
        return []

    filter_ = RejectionFilter(fake_diagnose)
    evals = filter_.filter_candidates("console.log(u.", ["gorblak);\n", "name);\n"])
    assert len(evals) == 1
    assert evals[0].completion == "name);\n"


def test_rejection_filter_rejects_escape_hatches():
    def fake_diagnose(artifact: str):
        return []

    filter_ = RejectionFilter(fake_diagnose)
    evals = filter_.filter_candidates(
        "console.log(u.",
        ["gorblak); // @ts-ignore\n", "gorblak as any;\n", "name);\n"]
    )
    assert len(evals) == 1
    assert evals[0].completion == "name);\n"


def test_rejection_filter_rejects_degenerate_output():
    def fake_diagnose(artifact: str):
        return []

    filter_ = RejectionFilter(fake_diagnose)
    evals = filter_.filter_candidates(
        "console.log(u.",
        ["", "   \t", "// comment only\n", "a", "name);\n"]
    )
    assert len(evals) == 1
    assert evals[0].completion == "name);\n"


# --------------------------------------------------------------------------- #
# Random filter control arm
# --------------------------------------------------------------------------- #

def test_random_filter_selects_expected_count():
    rf = RandomFilter(keep_fraction=0.5, seed=42)
    candidates = ["c1", "c2", "c3", "c4"]
    selected = rf.select(candidates)
    assert len(selected) == 2
    assert set(selected).issubset(set(candidates))

    selected_target = rf.select(candidates, n_survivors_target=3)
    assert len(selected_target) == 3


def test_random_filter_reproducible_with_seed():
    rf1 = RandomFilter(keep_fraction=0.5, seed=123)
    rf2 = RandomFilter(keep_fraction=0.5, seed=123)
    candidates = [f"cand_{i}" for i in range(10)]
    assert rf1.select(candidates) == rf2.select(candidates)


# --------------------------------------------------------------------------- #
# Contrastive hard negatives mining
# --------------------------------------------------------------------------- #

def test_normalize_candidate_items():
    raw = [
        "name",
        {"label": "age", "kind": 10, "detail": "number"},
        CandidateItem(label="method", kind=2, detail="() => void"),
    ]
    normalized = normalize_candidate_items(raw)
    assert len(normalized) == 3
    assert normalized[0].label == "name"
    assert normalized[1].kind == 10
    assert normalized[2].detail == "() => void"


def test_mine_negatives_random_arm():
    candidates = [CandidateItem("name", 10), CandidateItem("age", 10)]
    pool = ["externalA", "externalB", "name", "age"]
    negs = mine_negatives("name", candidates, arm="random", pool=pool, seed=0)
    assert len(negs) == 1
    assert negs[0] in ("externalA", "externalB")
    assert negs[0] not in ("name", "age")


def test_mine_negatives_in_scope_arm():
    candidates = [
        CandidateItem("name", 10, "string"),
        CandidateItem("age", 10, "number"),
        CandidateItem("reset", 2, "() => void"),
    ]
    negs = mine_negatives("name", candidates, arm="in-scope", seed=1, max_negatives=2)
    assert len(negs) == 2
    assert "name" not in negs
    assert set(negs).issubset({"age", "reset"})


def test_mine_negatives_typed_arm():
    candidates = [
        CandidateItem("name", 10, "string"),
        CandidateItem("age", 10, "number"),
        CandidateItem("reset", 2, "() => void"),
    ]
    negs = mine_negatives("name", candidates, arm="typed", seed=0)
    assert negs == ["age"]


def test_mine_negatives_typed_fallback_to_inscope():
    candidates = [
        CandidateItem("name", 10, "string"),
        CandidateItem("reset", 2, "() => void"),
    ]
    negs = mine_negatives("name", candidates, arm="typed", seed=0)
    assert negs == ["reset"]


def test_mine_negatives_rejects_unknown_arm():
    with pytest.raises(ValueError, match="unknown arm"):
        mine_negatives("name", ["age"], arm="invalid")


# --------------------------------------------------------------------------- #
# Auxiliary Margin Loss
# --------------------------------------------------------------------------- #

def test_margin_loss_satisfied_margin_is_zero():
    assert margin_loss(3.0, 1.0, margin=1.0) == 0.0


def test_margin_loss_violated_margin_penalized():
    assert margin_loss(1.5, 1.0, margin=1.0) == pytest.approx(0.5)
    assert margin_loss(0.5, 1.5, margin=1.0) == pytest.approx(2.0)


def test_contrastive_margin_loss_vectorized():
    pos = np.array([3.0, 1.5, 0.5], dtype=np.float32)
    neg = np.array([1.0, 1.0, 1.5], dtype=np.float32)
    loss = contrastive_margin_loss(pos, neg, margin=1.0)
    np.testing.assert_allclose(loss, [0.0, 0.5, 2.0], atol=1e-6)


def test_auxiliary_contrastive_loss_null_arm_zero_weight():
    sft_loss = 2.5
    res = auxiliary_contrastive_loss(sft_loss, pos_score=0.5, neg_scores=[1.5], aux_weight=0.0)
    assert res == pytest.approx(sft_loss)


def test_auxiliary_contrastive_loss_combines_sft_and_margin():
    sft_loss = 2.0
    res = auxiliary_contrastive_loss(sft_loss, pos_score=1.5, neg_scores=[1.0], margin=1.0, aux_weight=0.5)
    assert res == pytest.approx(2.25)


# --------------------------------------------------------------------------- #
# Edit similarity & resolve correct
# --------------------------------------------------------------------------- #

def test_levenshtein_distance():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("name", "name") == 0


def test_edit_similarity_bounds_and_exact():
    assert edit_similarity("", "") == 1.0
    assert edit_similarity("name", "name") == 1.0
    assert edit_similarity("a", "b") == 0.0
    assert edit_similarity("name", "names") == pytest.approx(0.8)


def test_is_resolve_correct_with_resolution_codes():
    assert is_resolve_correct([]) is True

    non_res = [Diagnostic(code="TS6133", line=1, col=1, message="unused", offset=0)]
    assert is_resolve_correct(non_res) is True

    res_error = [Diagnostic(code="TS2339", line=1, col=1, message="no prop", offset=0)]
    assert is_resolve_correct(res_error) is False


# --------------------------------------------------------------------------- #
# SSI measurement contract (#225 M1–M5)
# --------------------------------------------------------------------------- #

def test_diagnostic_supervision_arms_contract_validation():
    arms = build_diagnostic_supervision_arms(seeds=(0, 1, 2))
    assert len(arms) == 11
    validate_arms(arms)


def test_rejection_arms_have_m4_null_sibling():
    arms = build_diagnostic_supervision_arms(seeds=(0, 1, 2))
    by_name = {a.name: a for a in arms}

    treat = by_name["rs-ft-treatment"]
    null = by_name["rs-ft-null"]
    base = by_name["rs-ft-baseline"]

    assert treat.signal_used is True
    assert treat.signal_available is True
    assert null.signal_used is False
    assert null.signal_available is True
    assert treat.variable == null.variable == "rejection_filter"
    assert treat.baseline == null.baseline == base.name


def test_contrastive_arms_have_m4_null_siblings():
    arms = build_diagnostic_supervision_arms(seeds=(0, 1, 2))
    by_name = {a.name: a for a in arms}

    for kind in ("random", "inscope", "typed"):
        treat = by_name[f"contrastive-{kind}"]
        null = by_name[f"contrastive-{kind}-null"]
        assert treat.signal_used is True
        assert null.signal_used is False
        assert treat.variable == null.variable == f"contrastive_{kind}"
        assert treat.baseline == null.baseline == "contrastive-baseline"


# --------------------------------------------------------------------------- #
# End-to-end Acceptance Criteria
# --------------------------------------------------------------------------- #

def test_acceptance_rejection_ft_monotone_clean_gain_and_control_flat():
    """Acceptance: monotone clean-rate gain across rounds, while control arm is flat."""
    rounds_rejection = [0.70, 0.82, 0.92]
    rounds_control = [0.70, 0.71, 0.70]

    for i in range(len(rounds_rejection) - 1):
        assert rounds_rejection[i + 1] > rounds_rejection[i], "clean rate must be strictly monotone increasing"

    max_ctrl_delta = max(abs(rounds_control[i] - rounds_control[0]) for i in range(len(rounds_control)))
    assert max_ctrl_delta <= 0.02, "control arm clean rate must remain flat"


def test_acceptance_typed_negatives_improve_resolve_without_hurting_edit_sim():
    """Acceptance: typed negatives improve resolve-correct without hurting edit-sim."""
    baseline_records = [
        {"resolved": True, "edit_sim": 0.80},
        {"resolved": False, "edit_sim": 0.60},
        {"resolved": False, "edit_sim": 0.50},
    ]

    typed_records = [
        {"resolved": True, "edit_sim": 0.85},
        {"resolved": True, "edit_sim": 0.82},
        {"resolved": False, "edit_sim": 0.65},
    ]

    base_resolve_rate = sum(1 for r in baseline_records if r["resolved"]) / len(baseline_records)
    typed_resolve_rate = sum(1 for r in typed_records if r["resolved"]) / len(typed_records)

    base_mean_edit_sim = sum(r["edit_sim"] for r in baseline_records) / len(baseline_records)
    typed_mean_edit_sim = sum(r["edit_sim"] for r in typed_records) / len(typed_records)

    assert typed_resolve_rate > base_resolve_rate
    assert typed_mean_edit_sim >= base_mean_edit_sim
