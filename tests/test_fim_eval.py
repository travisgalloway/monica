"""Portable tests for the distance-bucketed FIM eval (#215, `src/eval/fim_eval.py`).

No backend: the model is a fake whose logits depend only on the token at each position, which
makes it causal by construction — exactly what the right-padding batching scheme relies on.
"""

import numpy as np
import pytest

from src.eval.fim_eval import (
    DEFAULT_BUCKETS,
    FIMExample,
    FIMSentinels,
    aggregate_by,
    aggregate_rows,
    attention_after_suffix,
    bucket_of,
    build_fim_examples,
    documents_from_shards,
    evaluate_fim,
    evaluate_fim_multi_key,
    format_fim_multi_table,
    format_fim_table,
    make_fim_example,
)

VOCAB = 32


class _FakeModel:
    """`forward(inputs) -> (B, L, V)`; logits at position t depend ONLY on `inputs[b, t]`.

    That is a strictly causal model (each position ignores everything after it), which is the
    property `evaluate_fim`'s right-padding assumes. A model that peeked at the padded tail would
    break `test_padding_does_not_change_a_short_examples_score`.
    """

    def __init__(self, vocab_size=VOCAB, n_layers=4, attn_every=2):
        self.vocab_size = vocab_size

        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.n_layers = n_layers
        self.config.attn_every = attn_every
        # A fixed random embedding table -> deterministic, non-degenerate logits.
        self._table = np.random.default_rng(0).standard_normal((vocab_size, vocab_size))

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        return self._table[inputs % self.vocab_size].astype(np.float32)


class _OracleModel:
    """Predicts the next id as `(current + 1) % vocab` with huge confidence.

    Fed a document of consecutive ids this is a perfect predictor *inside* the middle span, but
    the first middle token is predicted from the `fim_middle` sentinel, which carries no such
    relation — so `bridge` overrides that one row (sentinel id -> the actual first middle token).
    Still causal: every position depends only on its own input id.
    """

    def __init__(self, vocab_size=VOCAB, bridge=None):
        self.vocab_size = vocab_size
        self._table = np.full((vocab_size, vocab_size), -50.0, dtype=np.float32)
        for i in range(vocab_size):
            self._table[i, (i + 1) % vocab_size] = 50.0
        for src, dst in (bridge or {}).items():
            self._table[src] = -50.0
            self._table[src, dst] = 50.0

    def forward(self, inputs):
        return self._table[np.asarray(inputs) % self.vocab_size]


def _doc(n, start=6):
    """A document of consecutive ids, all above the 0..5 special range."""
    return np.array([(start + i) % VOCAB for i in range(n)], dtype=np.int64)


# --------------------------------------------------------------------------------------- #
# PSM layout
# --------------------------------------------------------------------------------------- #

def test_psm_layout_and_reassembly():
    s = FIMSentinels()
    doc = _doc(10)
    ex = make_fim_example(doc, 3, 7, s, doc_index=4)

    assert ex.tokens[0] == s.prefix
    assert ex.tokens[1 + 3] == s.suffix
    assert ex.middle_start == 1 + 3 + 1 + 3
    assert ex.tokens[ex.middle_start] == s.middle
    assert (ex.prefix_len, ex.middle_len, ex.suffix_len) == (3, 4, 3)
    assert ex.doc_index == 4
    assert ex.recall_distance == ex.middle_start == ex.prefix_len + ex.suffix_len + 2

    prefix = ex.tokens[1:1 + ex.prefix_len]
    suffix = ex.tokens[2 + ex.prefix_len:2 + ex.prefix_len + ex.suffix_len]
    middle = ex.tokens[ex.middle_start + 1:]
    np.testing.assert_array_equal(np.concatenate([prefix, middle, suffix]), doc)

    for sentinel in s.all_sentinels:
        assert int((ex.tokens == sentinel).sum()) == 1


@pytest.mark.parametrize("a,b", [(0, 0), (0, 10), (10, 10), (4, 4)])
def test_degenerate_cuts_still_reassemble(a, b):
    doc = _doc(10)
    ex = make_fim_example(doc, a, b)
    prefix = ex.tokens[1:1 + ex.prefix_len]
    suffix = ex.tokens[2 + ex.prefix_len:2 + ex.prefix_len + ex.suffix_len]
    middle = ex.tokens[ex.middle_start + 1:]
    np.testing.assert_array_equal(np.concatenate([prefix, middle, suffix]), doc)


def test_invalid_cuts_raise():
    with pytest.raises(ValueError):
        make_fim_example(_doc(10), 7, 3)
    with pytest.raises(ValueError):
        make_fim_example(_doc(10), 0, 11)


# --------------------------------------------------------------------------------------- #
# Buckets
# --------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("value,expected", [
    (0, "short"), (255, "short"), (256, "medium"), (1023, "medium"),
    (1024, "long"), (10 ** 6, "long"),
])
def test_bucket_boundaries(value, expected):
    assert bucket_of(value, DEFAULT_BUCKETS) == expected


def test_bucket_of_raises_when_nothing_matches():
    with pytest.raises(ValueError):
        bucket_of(500, (("tiny", 0, 10),))


# --------------------------------------------------------------------------------------- #
# Construction determinism
# --------------------------------------------------------------------------------------- #

def test_same_generator_seed_yields_identical_examples():
    docs = [_doc(20), _doc(35, start=9), _doc(12, start=11)]
    a = build_fim_examples(docs, np.random.default_rng(7), n_per_doc=3)
    b = build_fim_examples(docs, np.random.default_rng(7), n_per_doc=3)
    assert len(a) == len(b) == 9
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.tokens, y.tokens)
        assert (x.middle_start, x.middle_len, x.doc_index) == (y.middle_start, y.middle_len,
                                                               y.doc_index)


def test_build_skips_docs_that_cannot_yield_a_middle():
    exs = build_fim_examples([_doc(0), _doc(1), _doc(9)], np.random.default_rng(0),
                             n_per_doc=1, min_middle=2)
    assert [e.doc_index for e in exs] == [2]
    assert all(e.middle_len >= 2 for e in exs)


# --------------------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------------------- #

def test_mask_covers_exactly_the_middle_span():
    """A model that is perfect on the middle span and WRONG everywhere else still scores CE ~ 0.

    That only holds if the mask covers the middle span and nothing else: the oracle mispredicts
    every prefix/suffix/sentinel position, so a mask that were one position too wide would blow
    the CE up to ~100 nats.
    """
    doc = _doc(24)
    ex = make_fim_example(doc, 5, 15)
    oracle = _OracleModel(bridge={FIMSentinels().middle: int(doc[5])})
    res = evaluate_fim(oracle, [ex], batch_size=1)
    rec = res["records"][0]
    assert rec["n_middle_tokens"] == ex.middle_len
    assert rec["ce_nats"] < 1e-6
    assert rec["token_accuracy"] == 1.0
    assert rec["exact_match"] == 1.0
    assert res["overall"]["perplexity"] == pytest.approx(1.0, abs=1e-5)
    assert res["overall"]["exact_match_rate"] == 1.0


def test_mask_one_position_wider_would_be_caught():
    """Anti-vacuity for the test above: the same oracle scored over a span shifted by one is
    catastrophically wrong, so 'CE ~ 0' is a real assertion about the mask, not about the model."""
    doc = _doc(24)
    ex = make_fim_example(doc, 5, 15)
    oracle = _OracleModel(bridge={FIMSentinels().middle: int(doc[5])})
    shifted = FIMExample(tokens=ex.tokens, middle_start=ex.middle_start - 1,
                         prefix_len=ex.prefix_len, middle_len=ex.middle_len,
                         suffix_len=ex.suffix_len, doc_index=0)
    assert evaluate_fim(oracle, [shifted], batch_size=1)["records"][0]["ce_nats"] > 1.0


def test_aggregate_is_the_token_weighted_mean_of_the_records():
    docs = [_doc(30), _doc(24, start=8)]
    exs = build_fim_examples(docs, np.random.default_rng(3), n_per_doc=2, min_middle=2)
    res = evaluate_fim(_FakeModel(), exs, batch_size=3)
    rows = [r for r in res["records"] if r["n_middle_tokens"] > 0]
    n = sum(r["n_middle_tokens"] for r in rows)
    expected = sum(r["ce_nats"] * r["n_middle_tokens"] for r in rows) / n
    assert res["overall"]["ce"] == pytest.approx(expected, rel=1e-12)
    assert res["overall"]["n_tokens"] == n
    assert res["overall"]["n_examples"] == len(rows)


def test_padding_does_not_change_a_short_examples_score():
    """THE key correctness test for the batching scheme: batching a short example with a much
    longer one must not move its score. Right-padding is only safe because the model is causal."""
    short = make_fim_example(_doc(12), 3, 8, doc_index=0)
    long = make_fim_example(_doc(400, start=7), 100, 300, doc_index=1)
    model = _FakeModel()

    batched = evaluate_fim(model, [short, long], batch_size=2)["records"]
    alone = evaluate_fim(model, [short], batch_size=1)["records"]
    plus = evaluate_fim(model, [long], batch_size=1)["records"]

    assert batched[0]["ce_nats"] == pytest.approx(alone[0]["ce_nats"], rel=1e-12)
    assert batched[1]["ce_nats"] == pytest.approx(plus[0]["ce_nats"], rel=1e-12)
    assert batched[0]["token_accuracy"] == alone[0]["token_accuracy"]


def test_records_carry_both_distances_and_bucket_by_the_key():
    """`key` selects what gets bucketed; every record carries both distances regardless, so #221
    can bucket on recall distance without a second module."""
    ex = make_fim_example(_doc(600), 300, 400)     # prefix_len 300, recall_distance 502
    model = _FakeModel()

    by_prefix = evaluate_fim(model, [ex], batch_size=1)["records"][0]
    by_recall = evaluate_fim(model, [ex], batch_size=1,
                             key=lambda e: e.recall_distance)["records"][0]

    assert by_prefix["prefix_len"] == 300 and by_prefix["recall_distance"] == 502
    assert by_prefix["bucket"] == "medium"          # 300 -> [256, 1024)
    assert by_recall["bucket"] == "medium"          # 502 -> [256, 1024)

    far = make_fim_example(_doc(2400), 1200, 1400)
    assert evaluate_fim(model, [far], batch_size=1)["records"][0]["bucket"] == "long"


def test_empty_bucket_is_none_not_a_crash():
    exs = [make_fim_example(_doc(20), 4, 12)]      # short bucket only
    res = evaluate_fim(_FakeModel(), exs, batch_size=1)
    assert res["by_bucket"]["short"] is not None
    assert res["by_bucket"]["medium"] is None
    assert res["by_bucket"]["long"] is None
    assert "no examples in this bucket" in format_fim_table(res)


def test_no_examples_raises_rather_than_reporting_a_perfect_model():
    with pytest.raises(ValueError, match="no middle tokens scored"):
        evaluate_fim(_FakeModel(), [], batch_size=2)


def test_empty_middle_is_recorded_but_not_scored():
    empty = make_fim_example(_doc(10), 4, 4)       # zero-length middle
    real = make_fim_example(_doc(10), 2, 8)
    res = evaluate_fim(_FakeModel(), [empty, real], batch_size=2)
    assert res["records"][0]["n_middle_tokens"] == 0
    assert res["records"][0]["ce_nats"] is None
    assert res["overall"]["n_examples"] == 1       # the empty one contributes nothing


def test_all_empty_middles_raises():
    exs = [make_fim_example(_doc(10), 4, 4), make_fim_example(_doc(10), 0, 0)]
    with pytest.raises(ValueError, match="no middle tokens scored"):
        evaluate_fim(_FakeModel(), exs, batch_size=2)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        evaluate_fim(_FakeModel(), [make_fim_example(_doc(10), 2, 6)], batch_size=0)


# --------------------------------------------------------------------------------------- #
# Advisory (requirement 5 — documentation + config guidance, no model code)
# --------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_layers,attn_every,expected", [
    (4, 2, True),        # config/toy-hybrid.yaml — [Mamba, Attn, Mamba, Attn]
    (28, 8, False),      # config/student-1b.yaml — attention at 7/15/23, top block is Mamba
    (24, None, False),   # pure Mamba
    (24, 0, False),
    (0, 2, False),
    (24, 8, True),
])
def test_attention_after_suffix_truth_table(n_layers, attn_every, expected):
    assert attention_after_suffix(n_layers, attn_every) is expected


def test_advisory_present_only_when_the_top_block_is_not_attention():
    ex = make_fim_example(_doc(20), 4, 12)
    ok = evaluate_fim(_FakeModel(n_layers=4, attn_every=2), [ex], batch_size=1)
    bad = evaluate_fim(_FakeModel(n_layers=28, attn_every=8), [ex], batch_size=1)
    assert ok["advisory"] is None
    assert "final block is not attention" in bad["advisory"]
    assert "advisory:" in format_fim_table(bad)


def test_advisory_never_raises_on_a_config_less_model():
    class _Bare:
        def forward(self, inputs):
            return np.zeros((*np.asarray(inputs).shape, VOCAB), dtype=np.float32)

    res = evaluate_fim(_Bare(), [make_fim_example(_doc(20), 4, 12)], batch_size=1)
    assert res["advisory"] is None


# --------------------------------------------------------------------------------------- #
# Shard bridge
# --------------------------------------------------------------------------------------- #

def test_documents_from_shards_round_trips_doc_boundaries(tmp_path):
    from src.data.shard import pack_sequences

    docs = [[6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21]]
    pack_sequences(iter(docs), tmp_path, seq_len=4, shard_size_mb=1)
    recovered = documents_from_shards(tmp_path, min_len=1)
    assert len(recovered) == len(docs)
    assert list(recovered[0]) == docs[0]


def test_documents_from_shards_honours_max_docs(tmp_path):
    from src.data.shard import pack_sequences

    docs = [[6, 7, 8, 9] for _ in range(6)]
    pack_sequences(iter(docs), tmp_path, seq_len=4, shard_size_mb=1)
    assert len(documents_from_shards(tmp_path, max_docs=2)) == 2


# --------------------------------------------------------------------------------------- #
# Multi-key aggregation (#221's additive extension)
# --------------------------------------------------------------------------------------- #

class _CountingModel(_FakeModel):
    """Counts `forward` calls, so 'one forward pass, both keyings' is an assertion rather
    than a comment."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_forward = 0

    def forward(self, inputs):
        self.n_forward += 1
        return super().forward(inputs)


def _multi_key_examples():
    # prefix_len and recall_distance land in DIFFERENT buckets for these, which is the only
    # way the two keyings can be told apart.
    return [
        make_fim_example(_doc(700), 100, 200, doc_index=0),    # prefix 100 (short), recall 602
        make_fim_example(_doc(2600, start=7), 200, 400, doc_index=1),  # prefix 200, recall 2402
        make_fim_example(_doc(60, start=9), 10, 30, doc_index=2),      # both short
    ]


def test_aggregate_rows_is_the_token_weighted_mean():
    rows = [{"n_middle_tokens": 10, "ce_nats": 1.0, "token_accuracy": 1.0, "exact_match": 1.0},
            {"n_middle_tokens": 30, "ce_nats": 3.0, "token_accuracy": 0.0, "exact_match": 0.0}]
    agg = aggregate_rows(rows)
    assert agg["ce"] == pytest.approx((1.0 * 10 + 3.0 * 30) / 40)
    assert agg["token_accuracy"] == pytest.approx(10 / 40)
    assert agg["exact_match_rate"] == pytest.approx(0.5)      # per-example, not token-weighted


def test_aggregate_by_rebuckets_without_touching_the_model():
    records = evaluate_fim(_FakeModel(), _multi_key_examples(), batch_size=2)["records"]
    by_prefix = aggregate_by(records, key_field="prefix_len")
    by_recall = aggregate_by(records, key_field="recall_distance")
    assert by_prefix["short"] is not None and by_prefix["long"] is None
    assert by_recall["long"] is not None                       # 2402 tokens of recall distance
    assert by_prefix != by_recall


def test_multi_key_matches_two_separate_single_key_runs():
    examples = _multi_key_examples()
    model = _FakeModel()
    multi = evaluate_fim_multi_key(model, examples, batch_size=2)
    by_prefix = evaluate_fim(model, examples, batch_size=2,
                             key=lambda e: e.prefix_len)["by_bucket"]
    by_recall = evaluate_fim(model, examples, batch_size=2,
                             key=lambda e: e.recall_distance)["by_bucket"]
    assert multi["by_key"]["prefix_len"] == by_prefix
    assert multi["by_key"]["recall_distance"] == by_recall
    assert multi["overall"] == evaluate_fim(model, examples, batch_size=2)["overall"]


def test_multi_key_scores_once_not_once_per_key():
    examples = _multi_key_examples()
    multi_model = _CountingModel()
    evaluate_fim_multi_key(multi_model, examples, batch_size=2)
    single_model = _CountingModel()
    evaluate_fim(single_model, examples, batch_size=2)
    # Same number of forward calls as ONE evaluate_fim — the point of the extension.
    assert multi_model.n_forward == single_model.n_forward


def test_multi_key_records_carry_both_distances():
    res = evaluate_fim_multi_key(_FakeModel(), _multi_key_examples(), batch_size=2)
    for rec in res["records"]:
        assert "prefix_len" in rec and "recall_distance" in rec


def test_multi_key_rejects_an_unknown_key():
    with pytest.raises(ValueError, match="unknown FIM bucket key"):
        evaluate_fim_multi_key(_FakeModel(), _multi_key_examples(), keys=("nonsense",))


def test_multi_key_raises_on_an_empty_set():
    with pytest.raises(ValueError, match="no middle tokens scored"):
        evaluate_fim_multi_key(_FakeModel(), [], batch_size=2)


def test_multi_key_table_labels_each_keying():
    res = evaluate_fim_multi_key(_FakeModel(n_layers=28, attn_every=8), _multi_key_examples(),
                                 batch_size=2)
    table = format_fim_multi_table(res)
    assert "FIM loss by prefix_len:" in table and "FIM loss by recall_distance:" in table
    assert "advisory:" in table and table.count("advisory:") == 1


def test_evaluate_fim_output_is_unchanged_by_the_refactor():
    """#215 owns `evaluate_fim`; #221's extension is additive only."""
    examples = _multi_key_examples()
    res = evaluate_fim(_FakeModel(), examples, batch_size=2)
    assert sorted(res) == ["advisory", "by_bucket", "overall", "records"]
    assert sorted(res["overall"]) == ["ce", "exact_match_rate", "n_examples", "n_tokens",
                                      "perplexity", "token_accuracy"]
    assert res["records"][0]["bucket"] == "short"


def test_fim_example_is_frozen():
    ex = make_fim_example(_doc(10), 2, 6)
    assert isinstance(ex, FIMExample)
    with pytest.raises(Exception):
        ex.middle_start = 0
