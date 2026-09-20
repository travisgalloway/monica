"""Portable tests for the shared code-eval-suite plumbing (#221, `src/eval/code_suite.py`).

No backend. The models here are causal by construction — either context-free (logits depend
only on the token at each position) or position-scripted — which is exactly the property the
right-padding batching relies on.
"""

import json

import numpy as np
import pytest

from src.eval.code_suite import (
    RECORD_FIELDS,
    ScoreRow,
    StubCausalModel,
    bucket_for_distance,
    dumps_canonical,
    format_bucket_table,
    load_code_files,
    make_byte_encoder,
    make_record,
    read_jsonl,
    score_rows,
    sha256_file,
    sha256_text,
    summarize_bucketed,
    summarize_records,
    write_jsonl,
)

VOCAB = 64


class _ScriptedModel:
    """`forward(inputs) -> (B, L, V)`; the prediction at position `t` is `script[t]`.

    Depends on position only — never on any input token — so it is trivially causal, and it
    lets a test assert an exact CE of ~0 over one chosen span and nothing else.
    """

    def __init__(self, script, vocab_size=VOCAB, hot=50.0):
        self.script = np.asarray(script, dtype=np.int64)
        self.vocab_size = vocab_size
        self.hot = hot

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        b, length = inputs.shape
        out = np.full((b, length, self.vocab_size), -self.hot, dtype=np.float32)
        for t in range(length):
            if t < self.script.size:
                out[:, t, int(self.script[t]) % self.vocab_size] = self.hot
        return out


# --------------------------------------------------------------------------------------- #
# Record schema + canonical IO
# --------------------------------------------------------------------------------------- #

def test_make_record_always_carries_every_field():
    rec = make_record(suite="s", id="i", bucket="short", distance=3)
    assert tuple(sorted(rec)) == tuple(sorted(RECORD_FIELDS))
    # Missing metrics are explicit None, never absent keys — an absent key reads downstream
    # exactly like a zero, which is the failure this schema exists to prevent.
    assert rec["ce_nats"] is None and rec["rank_top1"] is None and rec["mrr"] is None
    assert rec["n_scored_tokens"] == 0 and rec["meta"] == {}


def test_canonical_dump_is_key_order_independent():
    a = dumps_canonical({"b": 1, "a": 2})
    b = dumps_canonical({"a": 2, "b": 1})
    assert a == b == '{"a":2,"b":1}'


def test_canonical_dump_handles_numpy_scalars():
    blob = dumps_canonical({"i": np.int64(3), "f": np.float32(0.5), "b": np.bool_(True)})
    assert json.loads(blob) == {"i": 3, "f": 0.5, "b": True}


def test_write_jsonl_is_byte_reproducible_and_preserves_order(tmp_path):
    records = [make_record(suite="s", id=f"i{i}", bucket="short", distance=i) for i in (2, 0, 1)]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert write_jsonl(records, a) == 3
    write_jsonl(records, b)
    assert a.read_bytes() == b.read_bytes()
    # The writer never sorts: a probe with a nondeterministic order must show up in the diff,
    # not be silently normalised away.
    assert [r["id"] for r in read_jsonl(a)] == ["i2", "i0", "i1"]


def test_sha256_helpers(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello\n")
    assert sha256_file(p) == sha256_text("hello\n")


# --------------------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------------------- #

def _rec(bucket, ce, n, **kw):
    return make_record(suite="s", id=f"{bucket}-{ce}-{n}", bucket=bucket, distance=0,
                       n_scored_tokens=n, ce_nats=ce, token_accuracy=1.0, exact_match=1.0, **kw)


def test_summarize_is_token_weighted_not_instance_weighted():
    rows = [_rec("short", 1.0, 10), _rec("short", 3.0, 30)]
    agg = summarize_records(rows)
    assert agg["ce"] == pytest.approx((1.0 * 10 + 3.0 * 30) / 40)
    assert agg["n_tokens"] == 40 and agg["n_instances"] == 2


def test_summarize_raises_rather_than_reporting_a_perfect_model():
    with pytest.raises(ValueError, match="nothing scored"):
        summarize_records([])
    with pytest.raises(ValueError, match="nothing scored"):
        summarize_records([make_record(suite="s", id="a", bucket="short", distance=0)])


def test_empty_bucket_is_none_not_a_zero_row():
    res = summarize_bucketed([_rec("short", 2.0, 4)], ["short", "medium", "long"])
    assert res["by_bucket"]["short"]["ce"] == pytest.approx(2.0)
    assert res["by_bucket"]["medium"] is None and res["by_bucket"]["long"] is None
    assert "no instances in this bucket" in format_bucket_table("t", res)


def test_unlisted_buckets_are_appended_never_dropped():
    res = summarize_bucketed([_rec("short", 1.0, 2), _rec("surprise", 1.0, 2)], ["short"])
    assert list(res["by_bucket"]) == ["short", "surprise"]


def test_rank_fields_are_none_when_no_record_carries_them():
    agg = summarize_records([_rec("short", 1.0, 2)])
    assert agg["rank_top1_rate"] is None and agg["mrr"] is None


def test_rank_fields_aggregate_when_present():
    agg = summarize_records([_rec("short", 1.0, 2, rank_top1=True, mrr=1.0),
                             _rec("short", 1.0, 3, rank_top1=False, mrr=0.5)])
    assert agg["rank_top1_rate"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.75)


@pytest.mark.parametrize("value,expected", [(0, "short"), (255, "short"), (256, "medium"),
                                            (1024, "long")])
def test_bucket_for_distance_matches_fim_edges(value, expected):
    assert bucket_for_distance(value) == expected


# --------------------------------------------------------------------------------------- #
# The shared span scorer
# --------------------------------------------------------------------------------------- #

def test_score_row_rejects_a_span_that_cannot_be_predicted():
    toks = np.arange(10)
    with pytest.raises(ValueError):
        ScoreRow(tokens=toks, span_start=0, span_len=2)     # nothing predicts position 0
    with pytest.raises(ValueError):
        ScoreRow(tokens=toks, span_start=1, span_len=0)
    with pytest.raises(ValueError):
        ScoreRow(tokens=toks, span_start=5, span_len=99)    # runs past the row


def test_mask_covers_exactly_the_requested_span():
    """A model that is perfect on the span and wrong everywhere else still scores CE ~ 0 —
    which only holds if the mask is exactly the span."""
    toks = np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    model = _ScriptedModel(toks[1:])          # position t predicts toks[t+1]
    [res] = score_rows(model, [ScoreRow(tokens=toks, span_start=3, span_len=4)])
    assert res["n_scored_tokens"] == 4
    assert res["ce_nats"] < 1e-6
    assert res["token_accuracy"] == 1.0 and res["exact_match"] == 1.0
    assert res["total_ce_nats"] == pytest.approx(res["ce_nats"] * 4)


def test_a_span_shifted_by_one_would_be_caught():
    """Anti-vacuity for the test above: the same model over a span shifted by one is
    catastrophically wrong, so 'CE ~ 0' is an assertion about the mask, not the model."""
    toks = np.array([5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
    model = _ScriptedModel(toks[2:])          # deliberately off by one
    [res] = score_rows(model, [ScoreRow(tokens=toks, span_start=3, span_len=4)])
    assert res["ce_nats"] > 1.0


def test_padding_does_not_change_a_short_instances_score():
    """THE correctness test for the batching scheme: right-padding is only safe because the
    model is causal."""
    model = StubCausalModel(vocab_size=VOCAB, seed=1)
    short = ScoreRow(tokens=np.arange(1, 13) % VOCAB, span_start=4, span_len=3)
    long = ScoreRow(tokens=np.arange(1, 300) % VOCAB, span_start=100, span_len=20)

    batched = score_rows(model, [short, long], batch_size=2)
    alone = score_rows(model, [short], batch_size=1)
    plus = score_rows(model, [long], batch_size=1)
    assert batched[0]["ce_nats"] == pytest.approx(alone[0]["ce_nats"], rel=1e-12)
    assert batched[1]["ce_nats"] == pytest.approx(plus[0]["ce_nats"], rel=1e-12)


def test_score_rows_preserves_input_order_across_batches():
    model = StubCausalModel(vocab_size=VOCAB, seed=2)
    rows = [ScoreRow(tokens=np.arange(1, 20 + i) % VOCAB, span_start=3, span_len=2)
            for i in range(7)]
    one = score_rows(model, rows, batch_size=1)
    many = score_rows(model, rows, batch_size=3)
    for a, b in zip(one, many):
        assert a["ce_nats"] == pytest.approx(b["ce_nats"], rel=1e-12)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        score_rows(StubCausalModel(), [ScoreRow(tokens=np.arange(8), span_start=2, span_len=2)],
                   batch_size=0)


# --------------------------------------------------------------------------------------- #
# Stub model + encoders + fixture IO
# --------------------------------------------------------------------------------------- #

def test_stub_model_is_causal_and_seed_reproducible():
    a, b = StubCausalModel(vocab_size=VOCAB, seed=3), StubCausalModel(vocab_size=VOCAB, seed=3)
    x = np.array([[1, 2, 3, 4]])
    np.testing.assert_array_equal(a.forward(x), b.forward(x))
    # Changing a LATER token must not move an earlier position's logits.
    y = np.array([[1, 2, 3, 9]])
    np.testing.assert_array_equal(a.forward(x)[0, :3], a.forward(y)[0, :3])


def test_byte_encoder_round_trips_utf8_lengths():
    encode = make_byte_encoder()
    text = "const x = 1; // héllo"
    assert list(encode(text)) == list(text.encode("utf-8"))
    # Byte counts, never len(str) — non-ASCII would otherwise be under-counted.
    assert len(list(encode(text))) == len(text.encode("utf-8")) != len(text)


def test_load_code_files_sorts_and_validates(tmp_path):
    p = tmp_path / "f.jsonl"
    write_jsonl([{"path": "b.ts", "text": "x"}, {"path": "a.ts", "text": "y"}], p)
    assert [f["path"] for f in load_code_files(p)] == ["a.ts", "b.ts"]

    bad = tmp_path / "bad.jsonl"
    write_jsonl([{"path": "a.ts"}], bad)
    with pytest.raises(ValueError, match="needs 'path' and 'text'"):
        load_code_files(bad)


# --------------------------------------------------------------------------------------- #
# Repository recall probe (#359)
# --------------------------------------------------------------------------------------- #

def test_in_context_recall_model_is_causal_and_seed_reproducible():
    from src.eval.code_suite import InContextRecallModel

    a = InContextRecallModel(vocab_size=VOCAB, seed=4)
    b = InContextRecallModel(vocab_size=VOCAB, seed=4)
    x = np.array([[1, 2, 3, 4]])
    np.testing.assert_array_equal(a.forward(x), b.forward(x))

    # Causality: modifying a later token must not move earlier positions' logits
    y = np.array([[1, 2, 3, 9]])
    np.testing.assert_array_equal(a.forward(x)[0, :3], a.forward(y)[0, :3])


def test_in_context_recall_model_padding_invariance():
    from src.eval.code_suite import InContextRecallModel

    model = InContextRecallModel(vocab_size=VOCAB, seed=5)
    short = ScoreRow(tokens=np.arange(1, 13) % VOCAB, span_start=4, span_len=3)
    long = ScoreRow(tokens=np.arange(1, 100) % VOCAB, span_start=50, span_len=10)

    batched = score_rows(model, [short, long], batch_size=2)
    alone = score_rows(model, [short], batch_size=1)
    plus = score_rows(model, [long], batch_size=1)
    assert batched[0]["ce_nats"] == pytest.approx(alone[0]["ce_nats"], rel=1e-6)
    assert batched[1]["ce_nats"] == pytest.approx(plus[0]["ce_nats"], rel=1e-6)


def test_topological_sort_shows_higher_symbol_retrieval_than_random():
    from src.eval.code_suite import InContextRecallModel, evaluate_repo_recall

    files = [
        {"path": "src/units.ts",
         "text": "export function metersToFeet(m: number): number {\n  return m * 3.28084;\n}\n"
                 "export function feetToMeters(f: number): number {\n  return f / 3.28084;\n}\n"},
        {"path": "src/calc.ts",
         "text": "import { metersToFeet } from \"./units\";\n"
                 "export function computeElevation(elevMeters: number): number {\n"
                 "  return metersToFeet(elevMeters);\n}\n"},
        {"path": "src/app.ts",
         "text": "import { computeElevation } from \"./calc\";\n"
                 "export function main(): void {\n"
                 "  const res = computeElevation(100);\n}\n"}
    ]
    model = InContextRecallModel(vocab_size=256, seed=0)
    encode = make_byte_encoder()
    rng = np.random.default_rng(42)

    res = evaluate_repo_recall(model, files, encode, rng=rng, n_random_orders=10)

    # Acceptance Criterion 4: Topologically sorted sequences show higher top-1
    # symbol retrieval accuracy compared to randomized file ordering.
    assert res["topo_top1_accuracy"] > res["random_top1_accuracy"]
    assert res["topo_mrr"] > res["random_mrr"]
    assert res["topo_top1_accuracy"] == pytest.approx(1.0)
    assert res["random_top1_accuracy"] < 1.0


def test_repo_recall_records_conform_to_shared_schema_and_buckets():
    from src.eval.code_suite import InContextRecallModel, evaluate_repo_recall

    files = load_code_files("eval_sets/code_recall/fixture_repo.jsonl")
    model = InContextRecallModel(vocab_size=256, seed=0)
    res = evaluate_repo_recall(model, files, make_byte_encoder(), rng=np.random.default_rng(0), n_random_orders=3)

    assert res["records"]
    for rec in res["records"]:
        assert tuple(sorted(rec)) == tuple(sorted(RECORD_FIELDS))
        assert rec["suite"] == "repo_recall"
        assert rec["bucket"] in ("topological", "random")
        assert rec["n_scored_tokens"] > 0
    assert set(res["by_bucket"]) == {"topological", "random"}
    assert res["topo_top1_accuracy"] > res["random_top1_accuracy"]


def test_evaluate_data_contracts_records_and_buckets():
    from src.eval.code_suite import DATA_CONTRACT_BUCKETS, evaluate_data_contracts

    test_cases = [
        # SQL clean & faulty
        {
            "id": "sql-clean-1",
            "contract": "sql",
            "code": "SELECT id, name FROM users WHERE id > 0;",
            "expected_clean": True,
        },
        {
            "id": "sql-cartesian-1",
            "contract": "sql",
            "code": "SELECT * FROM users, orders;",
            "expected_clean": False,
        },
        # Data Pipeline clean & faulty
        {
            "id": "pipe-clean-1",
            "contract": "data_pipeline",
            "code": "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2]})\nout = df[['a']]",
            "expected_clean": True,
        },
        {
            "id": "pipe-iter-1",
            "contract": "data_pipeline",
            "code": "for idx, row in df.iterrows():\n    print(row)",
            "expected_clean": False,
        },
        # OpenAPI clean & faulty
        {
            "id": "openapi-clean-1",
            "contract": "openapi",
            "code": "openapi: 3.1.0\ninfo:\n  title: T\n  version: 1.0\npaths:\n  /users:\n    get:\n      responses:\n        '200':\n          description: ok\n          content:\n            application/json:\n              schema:\n                type: string\n                maxLength: 100",
            "expected_clean": True,
        },
        {
            "id": "openapi-untyped-1",
            "contract": "openapi",
            "code": "openapi: 3.1.0\ninfo:\n  title: T\n  version: 1.0\npaths:\n  /users:\n    get:\n      responses:\n        '200':\n          content:\n            application/json:\n              schema: {}",
            "expected_clean": False,
        },
        # GraphQL clean & faulty
        {
            "id": "graphql-clean-1",
            "contract": "graphql",
            "code": "type User {\n  id: ID!\n  name: String!\n}",
            "expected_clean": True,
        },
        {
            "id": "graphql-any-1",
            "contract": "graphql",
            "code": "scalar Any\ntype User {\n  payload: Any\n}",
            "expected_clean": False,
        },
        # Protobuf clean & faulty
        {
            "id": "proto-clean-1",
            "contract": "protobuf",
            "code": 'syntax = "proto3";\nmessage User {\n  string id = 1;\n  string name = 2;\n}',
            "expected_clean": True,
        },
        {
            "id": "proto-missing-1",
            "contract": "protobuf",
            "code": 'syntax = "proto3";\nmessage User {\n  string id;\n}',
            "expected_clean": False,
        },
        # Clean Architecture clean & faulty
        {
            "id": "arch-clean-1",
            "contract": "clean_architecture",
            "code": "class Entity:\n    def __init__(self, id: str):\n        self.id = id",
            "expected_clean": True,
        },
        {
            "id": "arch-leaky-1",
            "contract": "clean_architecture",
            "code": "from fastapi import FastAPI\napp = FastAPI()",
            "expected_clean": False,
        },
    ]

    res = evaluate_data_contracts(test_cases)

    assert res["records"]
    assert len(res["records"]) == 12
    assert res["n_cases"] == 12
    assert res["n_passed"] == 12
    assert res["accuracy"] == 1.0

    for rec in res["records"]:
        assert tuple(sorted(rec)) == tuple(sorted(RECORD_FIELDS))
        assert rec["suite"] == "data_contracts"
        assert rec["bucket"] in DATA_CONTRACT_BUCKETS
        assert rec["rank_top1"] is True

    for bucket in DATA_CONTRACT_BUCKETS:
        assert bucket in res["by_bucket"]
        assert res["by_bucket"][bucket]["n_instances"] == 2
        assert res["by_bucket"][bucket]["rank_top1_rate"] == 1.0


def test_evaluate_refactoring_schema_and_metrics():
    from src.eval.code_suite import REFACTORING_BUCKETS, evaluate_refactoring

    test_cases = [
        {
            "id": "refactor-strat-clean",
            "pattern": "strategy",
            "code": "class S: pass\nclass A(S): pass",
            "tests": "import unittest\nclass T(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)",
            "expected_clean": True,
        },
        {
            "id": "refactor-strat-faulty",
            "pattern": "strategy",
            "code": "class S: pass",
            "tests": "import unittest\nclass T(unittest.TestCase):\n    def test_fail(self): self.assertTrue(False)",
            "expected_clean": False,
        },
    ]

    res = evaluate_refactoring(test_cases)
    assert res["records"]
    assert len(res["records"]) == 2
    assert res["n_cases"] == 2
    assert res["n_passed"] == 2
    assert res["accuracy"] == 1.0

    for rec in res["records"]:
        assert tuple(sorted(rec)) == tuple(sorted(RECORD_FIELDS))
        assert rec["suite"] == "refactoring"
        assert rec["bucket"] in REFACTORING_BUCKETS
        assert rec["rank_top1"] is True
