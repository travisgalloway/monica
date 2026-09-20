"""Portable tests for technical prose & general knowledge recall probe (#363, `src/eval/prose_recall.py`).

ABOVE THE SEAM. Pure numpy + stdlib; no backend required.
Tests distance bucketing, cross-entropy calculation, candidate ranking, determinism,
and shared schema conformance.
"""

from pathlib import Path
import numpy as np
import pytest

from src.eval.code_suite import (
    RECORD_FIELDS,
    StubCausalModel,
    load_code_files,
    make_byte_encoder,
)
from src.eval.prose_recall import (
    DEFAULT_DISTANCES,
    PROSE_DEFAULT_BUCKETS,
    PROSE_BUCKET_NAMES,
    ProseRecallSpec,
    build_prose_recall_instances,
    evaluate_prose_recall,
    load_distractor_texts,
    load_prose_specs,
    prose_bucket_for_distance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS_PATH = REPO_ROOT / "eval_sets/prose_recall/specs.jsonl"
DISTRACTORS_PATH = REPO_ROOT / "eval_sets/prose_recall/distractors.jsonl"


class _PreferModel:
    """Context-free causal model that assigns high logits to specified token ids."""

    def __init__(self, prefer_ids, vocab_size=256, hot=12.0):
        self.vocab_size = vocab_size
        self._row = np.full((vocab_size,), -hot, dtype=np.float32)
        for i in prefer_ids:
            self._row[int(i) % vocab_size] = hot

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        return np.broadcast_to(self._row, (*inputs.shape, self.vocab_size)).copy()


# --------------------------------------------------------------------------------------- #
# Fixture loading and shared schema conformance
# --------------------------------------------------------------------------------------- #

def test_load_prose_specs_and_distractors():
    specs = load_prose_specs(SPECS_PATH)
    assert len(specs) >= 12, f"Expected at least 12 specs, got {len(specs)}"
    for spec in specs:
        assert isinstance(spec, ProseRecallSpec)
        assert spec.target in spec.candidates
        assert len(spec.candidates) >= 2
        assert spec.domain in ("rfc", "pep", "distributed_systems", "system_design", "posix")

    distractors = load_distractor_texts(DISTRACTORS_PATH)
    assert len(distractors) >= 6, f"Expected at least 6 distractor texts, got {len(distractors)}"
    for d in distractors:
        assert len(d.strip()) > 100


def test_fixtures_conform_to_code_suite_schema():
    """Fixtures have 'path' and 'text', allowing load_code_files to load them cleanly."""
    specs_files = load_code_files(SPECS_PATH)
    assert len(specs_files) >= 12
    for f in specs_files:
        assert "path" in f and "text" in f

    dist_files = load_code_files(DISTRACTORS_PATH)
    assert len(dist_files) >= 6
    for f in dist_files:
        assert "path" in f and "text" in f


# --------------------------------------------------------------------------------------- #
# Distance bucketing
# --------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("distance,expected_bucket", [
    (512, "512"),
    (600, "512"),
    (767, "512"),
    (768, "1024"),
    (1024, "1024"),
    (1535, "1024"),
    (1536, "2048"),
    (2048, "2048"),
    (3071, "2048"),
    (3072, "4096"),
    (4096, "4096"),
    (6143, "4096"),
    (6144, "8192"),
    (8192, "8192"),
    (12287, "8192"),
    (12288, "16384"),
    (16384, "16384"),
    (32768, "16384"),
])
def test_prose_distance_bucketing(distance, expected_bucket):
    assert prose_bucket_for_distance(distance, PROSE_DEFAULT_BUCKETS) == expected_bucket


# --------------------------------------------------------------------------------------- #
# Instance construction & determinism
# --------------------------------------------------------------------------------------- #

def test_build_instances_reaches_all_target_buckets():
    specs = load_prose_specs(SPECS_PATH)[:2]
    distractors = load_distractor_texts(DISTRACTORS_PATH)
    encode = make_byte_encoder()
    rng = np.random.default_rng(42)

    instances = build_prose_recall_instances(
        specs, distractors, encode, rng,
        distances=DEFAULT_DISTANCES,
    )
    assert len(instances) == len(specs) * len(DEFAULT_DISTANCES)

    buckets_seen = {inst.bucket for inst in instances}
    assert buckets_seen == set(PROSE_BUCKET_NAMES)

    for inst in instances:
        assert inst.target in inst.candidates
        assert inst.candidates[inst.answer_index] == inst.target
        assert inst.distance in DEFAULT_DISTANCES
        assert prose_bucket_for_distance(inst.distance) == inst.bucket
        assert inst.prefix_tokens.size > inst.distance


def test_build_instances_determinism_with_same_seed():
    specs = load_prose_specs(SPECS_PATH)[:3]
    distractors = load_distractor_texts(DISTRACTORS_PATH)
    encode = make_byte_encoder()

    inst_a = build_prose_recall_instances(specs, distractors, encode, np.random.default_rng(99))
    inst_b = build_prose_recall_instances(specs, distractors, encode, np.random.default_rng(99))

    assert [i.id for i in inst_a] == [i.id for i in inst_b]
    for a, b in zip(inst_a, inst_b):
        np.testing.assert_array_equal(a.prefix_tokens, b.prefix_tokens)
        assert a.distance == b.distance
        assert a.candidates == b.candidates
        assert a.answer_index == b.answer_index


def test_build_instances_max_instances_limit():
    specs = load_prose_specs(SPECS_PATH)
    distractors = load_distractor_texts(DISTRACTORS_PATH)
    encode = make_byte_encoder()
    instances = build_prose_recall_instances(
        specs, distractors, encode, np.random.default_rng(0),
        max_instances=5,
    )
    assert len(instances) == 5


# --------------------------------------------------------------------------------------- #
# Scoring, Cross-Entropy calculation & Candidate ranking
# --------------------------------------------------------------------------------------- #

def test_evaluate_prose_recall_with_stub_model():
    specs = load_prose_specs(SPECS_PATH)[:2]
    distractors = load_distractor_texts(DISTRACTORS_PATH)
    encode = make_byte_encoder()
    instances = build_prose_recall_instances(specs, distractors, encode, np.random.default_rng(0))

    model = StubCausalModel(vocab_size=256, seed=7)
    res = evaluate_prose_recall(model, instances, batch_size=4)

    assert "records" in res
    assert "by_bucket" in res
    assert "overall" in res
    assert len(res["records"]) == len(instances)

    for rec in res["records"]:
        assert tuple(sorted(rec.keys())) == tuple(sorted(RECORD_FIELDS))
        assert rec["suite"] == "prose_recall"
        assert rec["n_scored_tokens"] > 0
        assert rec["ce_nats"] is not None and rec["ce_nats"] > 0
        assert 0.0 <= rec["mrr"] <= 1.0

    for b in PROSE_BUCKET_NAMES:
        assert b in res["by_bucket"]
        assert res["by_bucket"][b] is not None
        assert res["by_bucket"][b]["n_instances"] == len(specs)


def test_candidate_ranking_prefers_target_tokens():
    """When the model assigns high logits to target tokens, answer ranks #1 (top1=True, mrr=1.0)."""
    encode = make_byte_encoder()
    single_spec = ProseRecallSpec(
        id="test_status_429",
        domain="rfc",
        statement="RFC 9110 specifies status code 429 for rate limiting.",
        query="The rate limiting status code is ",
        target="429",
        candidates=("400", "403", "429", "500"),
    )
    distractors = ["Some unrelated text about astronomy and distant stellar constellations."]
    instances = build_prose_recall_instances(
        [single_spec], distractors, encode, np.random.default_rng(0),
        distances=(512,),
    )
    assert len(instances) == 1

    target_bytes = list(encode("429"))
    model = _PreferModel(prefer_ids=target_bytes, vocab_size=256, hot=15.0)

    res = evaluate_prose_recall(model, instances)
    rec = res["records"][0]
    assert rec["rank_top1"] is True
    assert rec["mrr"] == pytest.approx(1.0)
    assert rec["meta"]["rank"] == 1


def test_candidate_ranking_prefers_distractor_tokens():
    """Anti-vacuity: when model prefers distractor tokens, target does NOT rank first."""
    encode = make_byte_encoder()
    single_spec = ProseRecallSpec(
        id="test_status_429",
        domain="rfc",
        statement="RFC 9110 specifies status code 429 for rate limiting.",
        query="The rate limiting status code is ",
        target="429",
        candidates=("400", "403", "429", "500"),
    )
    distractors = ["Some unrelated text about astronomy and distant stellar constellations."]
    instances = build_prose_recall_instances(
        [single_spec], distractors, encode, np.random.default_rng(0),
        distances=(512,),
    )
    assert len(instances) == 1

    distractor_bytes = list(encode("400"))
    model = _PreferModel(prefer_ids=distractor_bytes, vocab_size=256, hot=15.0)

    res = evaluate_prose_recall(model, instances)
    rec = res["records"][0]
    assert rec["rank_top1"] is False
    assert rec["mrr"] < 1.0
    assert rec["meta"]["rank"] > 1


def test_batching_invariance():
    """Padding and batch size must not change instance scores (causal right-padding guarantee)."""
    specs = load_prose_specs(SPECS_PATH)[:2]
    distractors = load_distractor_texts(DISTRACTORS_PATH)
    encode = make_byte_encoder()
    instances = build_prose_recall_instances(
        specs, distractors, encode, np.random.default_rng(123),
        distances=(512, 1024),
    )

    model = StubCausalModel(vocab_size=256, seed=42)
    res_b1 = evaluate_prose_recall(model, instances, batch_size=1)
    res_b4 = evaluate_prose_recall(model, instances, batch_size=4)

    for r1, r4 in zip(res_b1["records"], res_b4["records"]):
        assert r1["ce_nats"] == pytest.approx(r4["ce_nats"], rel=1e-12)
        assert r1["rank_top1"] == r4["rank_top1"]
        assert r1["mrr"] == pytest.approx(r4["mrr"], rel=1e-12)


def test_empty_instances_raises_value_error():
    model = StubCausalModel(vocab_size=256)
    with pytest.raises(ValueError, match="nothing scored"):
        evaluate_prose_recall(model, [])
