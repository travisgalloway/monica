"""Portable tests for the RULER-over-code needle probe (#221, `src/eval/code_needle.py`)."""

import numpy as np
import pytest

from src.eval.code_needle import (
    DEFAULT_DEPTHS,
    bucket_name,
    build_needle_instances,
    evaluate_code_needle,
    needle_text,
    query_text,
)
from src.eval.code_suite import StubCausalModel, load_code_files, make_byte_encoder

HAYSTACK = "eval_sets/code_needle/haystack.jsonl"


class _ScriptedModel:
    """Predicts `script[t]` at position `t` — position-only, hence causal. Used to prove the
    scored span is exactly the value's tokens."""

    def __init__(self, script, vocab_size=256, hot=50.0):
        self.script = np.asarray(script, dtype=np.int64)
        self.vocab_size = vocab_size
        self.hot = hot

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        b, length = inputs.shape
        out = np.full((b, length, self.vocab_size), -self.hot, dtype=np.float32)
        for t in range(min(length, self.script.size)):
            out[:, t, int(self.script[t]) % self.vocab_size] = self.hot
        return out


def _fillers():
    return load_code_files(HAYSTACK)


def _build(seed=0, **kw):
    kw.setdefault("context_lens", (512,))
    kw.setdefault("depths", (0.0, 0.5, 1.0))
    return build_needle_instances(_fillers(), make_byte_encoder(),
                                  np.random.default_rng(seed), **kw)


# --------------------------------------------------------------------------------------- #
# Grid construction
# --------------------------------------------------------------------------------------- #

def test_every_instance_is_exactly_context_len_tokens():
    """The token budget is exact — the haystack's last filler is truncated to hit it — so a
    depth comparison across cells is not confounded by different context lengths."""
    for inst in _build(context_lens=(512, 1024), depths=DEFAULT_DEPTHS):
        assert inst.tokens.size == inst.context_len


def test_the_needle_and_the_query_are_both_present():
    encode = make_byte_encoder()
    for inst in _build():
        text = bytes(int(t) for t in inst.tokens).decode("utf-8", "replace")
        assert needle_text(inst.key, inst.value).strip() in text
        assert query_text(inst.key).strip() in text
        # The scored span is the value's own tokens, at the very end of the row.
        assert list(inst.tokens[inst.span_start:]) == list(encode(inst.value))
        assert inst.span_len == len(list(encode(inst.value)))


def test_depth_moves_the_needle_monotonically():
    by_depth = {inst.depth: inst for inst in _build(context_lens=(1024,),
                                                    depths=(0.0, 0.5, 1.0))}
    # `distance` is measured from the end of the needle to the scored span, so a deeper
    # needle is a SHORTER distance. That ordering is the whole point of the grid.
    assert by_depth[0.0].distance > by_depth[0.5].distance > by_depth[1.0].distance


def test_grid_covers_every_requested_cell():
    ctxs, depths = (512, 1024), (0.0, 0.5)
    got = {i.bucket for i in _build(context_lens=ctxs, depths=depths)}
    assert got == {bucket_name(c, d) for c in ctxs for d in depths}


def test_multikey_plants_more_needles_without_changing_the_budget():
    single = _build(context_lens=(1024,), depths=(0.5,), variants=("single",))[0]
    multi = _build(context_lens=(1024,), depths=(0.5,), variants=("multikey",),
                   n_needles=4)[0]
    assert single.n_needles == 1 and multi.n_needles == 4
    assert single.tokens.size == multi.tokens.size == 1024
    text = bytes(int(t) for t in multi.tokens).decode("utf-8", "replace")
    assert text.count("MONICA_NEEDLE_") >= 4 + 1     # 4 planted + the query's reference


def test_a_context_too_small_for_the_needle_yields_no_instance():
    """An unreachable cell produces NO instance — it is reported as an empty bucket, never
    as a zero score."""
    assert _build(context_lens=(16,), depths=(0.5,)) == []


def test_same_seed_reproduces_the_grid_exactly():
    a, b = _build(seed=11), _build(seed=11)
    assert [i.id for i in a] == [i.id for i in b]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.tokens, y.tokens)
        assert (x.key, x.value, x.distance, x.span_start) == (y.key, y.value, y.distance,
                                                              y.span_start)


def test_different_seeds_draw_different_needles():
    assert {i.value for i in _build(seed=1)} != {i.value for i in _build(seed=2)}


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown needle variant"):
        _build(variants=("multivalue",))


def test_n_needles_must_be_positive():
    with pytest.raises(ValueError):
        _build(variants=("multikey",), n_needles=0)


# --------------------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------------------- #

def test_mask_covers_exactly_the_value_span():
    inst = _build(context_lens=(512,), depths=(0.5,))[0]
    model = _ScriptedModel(inst.tokens[1:])          # perfect next-token oracle, positionally
    res = evaluate_code_needle(model, [inst], batch_size=1, context_lens=(512,), depths=(0.5,))
    rec = res["records"][0]
    assert rec["n_scored_tokens"] == inst.span_len
    assert rec["ce_nats"] < 1e-6
    assert rec["exact_match"] == 1.0 and rec["token_accuracy"] == 1.0


def test_a_span_shifted_by_one_would_be_caught():
    inst = _build(context_lens=(512,), depths=(0.5,))[0]
    model = _ScriptedModel(inst.tokens[2:])          # off by one
    res = evaluate_code_needle(model, [inst], batch_size=1, context_lens=(512,), depths=(0.5,))
    assert res["records"][0]["ce_nats"] > 1.0
    assert res["records"][0]["exact_match"] == 0.0


def test_records_use_the_shared_schema_and_report_every_grid_cell():
    from src.eval.code_suite import RECORD_FIELDS

    ctxs, depths = (512,), (0.0, 0.5, 1.0)
    instances = _build(context_lens=ctxs, depths=depths)
    res = evaluate_code_needle(StubCausalModel(vocab_size=256, seed=0), instances,
                               batch_size=2, context_lens=ctxs, depths=depths)
    for rec in res["records"]:
        assert tuple(sorted(rec)) == tuple(sorted(RECORD_FIELDS))
        assert rec["suite"] == "code_needle"
    assert list(res["by_bucket"]) == [bucket_name(c, d) for c in ctxs for d in depths]


def test_an_unreachable_cell_is_reported_as_none_not_dropped():
    """A cell with no instances must stay visible in the table — a missing cell and a
    measured-zero cell must not look alike."""
    instances = _build(context_lens=(512,), depths=(0.5,))
    res = evaluate_code_needle(StubCausalModel(vocab_size=256, seed=0), instances,
                               batch_size=1, context_lens=(512, 99999), depths=(0.5,))
    assert res["by_bucket"][bucket_name(512, 0.5)] is not None
    assert res["by_bucket"][bucket_name(99999, 0.5)] is None


def test_batching_does_not_change_a_score():
    instances = _build(context_lens=(512, 1024), depths=(0.0, 1.0))
    model = StubCausalModel(vocab_size=256, seed=4)
    one = evaluate_code_needle(model, instances, batch_size=1, context_lens=(512, 1024),
                               depths=(0.0, 1.0))["records"]
    many = evaluate_code_needle(model, instances, batch_size=8, context_lens=(512, 1024),
                                depths=(0.0, 1.0))["records"]
    for a, b in zip(one, many):
        assert a["ce_nats"] == pytest.approx(b["ce_nats"], rel=1e-12)
