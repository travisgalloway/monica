"""MoE routing diagnostics (#217): per-domain expert histograms + the mid-training
kill-criterion. Portable — no backend; `expert_histograms` is exercised against a
duck-typed fake model (the `tests/test_moe_balance.py:233` `_FakeModel` pattern) over
tiny real packed files built with `src.data.pack.pack_ids` (the `tests/test_train_loop.py`
precedent)."""

import numpy as np
import pytest

from src.data.pack import pack_ids
from src.eval.moe_routing import (expert_histograms, format_routing_report,
                                  histogram_overlap, kill_check, specialization_report)


# --------------------------------------------------------------------------- #
# histogram_overlap
# --------------------------------------------------------------------------- #
def test_histogram_overlap_identical_histograms_is_one():
    h = [[10.0, 20.0, 30.0], [5.0, 5.0]]
    result = histogram_overlap(h, h)
    assert result["mean"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in result["per_layer"])


def test_histogram_overlap_disjoint_histograms_is_zero():
    a = [[10.0, 0.0], [0.0, 10.0]]
    b = [[0.0, 10.0], [10.0, 0.0]]
    result = histogram_overlap(a, b)
    assert result["mean"] == pytest.approx(0.0)
    assert result["per_layer"] == [pytest.approx(0.0), pytest.approx(0.0)]


def test_histogram_overlap_hand_worked_partial_case():
    # Layer 0: p = [0.5, 0.5], q = [1.0, 0.0] -> overlap = min(.5,1) + min(.5,0) = 0.5.
    a = [[5.0, 5.0]]
    b = [[10.0, 0.0]]
    result = histogram_overlap(a, b)
    assert result["per_layer"] == [pytest.approx(0.5)]
    assert result["mean"] == pytest.approx(0.5)


def test_histogram_overlap_per_layer_values_differ_across_layers():
    a = [[10.0, 0.0], [5.0, 5.0]]
    b = [[10.0, 0.0], [10.0, 0.0]]
    result = histogram_overlap(a, b)
    assert result["per_layer"][0] == pytest.approx(1.0)   # layer 0 identical
    assert result["per_layer"][1] == pytest.approx(0.5)   # layer 1 half-overlap
    assert result["mean"] == pytest.approx(0.75)


def test_histogram_overlap_layer_count_mismatch_raises():
    with pytest.raises(ValueError, match="layers"):
        histogram_overlap([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]])


def test_histogram_overlap_expert_count_mismatch_raises():
    with pytest.raises(ValueError, match="experts"):
        histogram_overlap([[1.0, 2.0]], [[1.0, 2.0, 3.0]])


def test_histogram_overlap_zero_total_layer_raises():
    with pytest.raises(ValueError, match="zero-total"):
        histogram_overlap([[0.0, 0.0]], [[1.0, 2.0]])


# --------------------------------------------------------------------------- #
# specialization_report
# --------------------------------------------------------------------------- #
def _hists_3domain():
    return {
        "a": [[10.0, 0.0]],
        "b": [[10.0, 0.0]],       # identical to a
        "c": [[0.0, 10.0]],       # disjoint from a and b
    }


def test_specialization_report_pair_keys_sorted():
    report = specialization_report(_hists_3domain())
    assert set(report["by_pair"]) == {"a|b", "a|c", "b|c"}


def test_specialization_report_mean_max_pair_correct():
    report = specialization_report(_hists_3domain())
    assert report["by_pair"]["a|b"]["mean"] == pytest.approx(1.0)
    assert report["by_pair"]["a|c"]["mean"] == pytest.approx(0.0)
    assert report["by_pair"]["b|c"]["mean"] == pytest.approx(0.0)
    assert report["mean_overlap"] == pytest.approx(1.0 / 3)
    assert report["max_pair"] == "a|b"
    assert report["max_overlap"] == pytest.approx(1.0)


def test_specialization_report_carries_unmeasured_domains():
    hists = {**_hists_3domain(), "d": None}
    report = specialization_report(hists)
    assert report["unmeasured_domains"] == ["d"]
    assert report["domains"] == ["a", "b", "c"]
    assert "d" not in report["by_pair"] and not any("d" in k for k in report["by_pair"])


def test_specialization_report_fewer_than_two_measured_raises():
    with pytest.raises(ValueError, match="at least 2"):
        specialization_report({"a": [[10.0, 0.0]], "b": None})
    with pytest.raises(ValueError, match="at least 2"):
        specialization_report({"a": None, "b": None})


# --------------------------------------------------------------------------- #
# kill_check
# --------------------------------------------------------------------------- #
def test_kill_check_triggered_above_threshold():
    report = specialization_report(_hists_3domain())
    kill = kill_check(report, pair=("a", "b"), threshold=0.90)
    assert kill["triggered"] is True
    assert kill["specializing"] is False
    assert kill["pair"] == "a|b"
    assert kill["overlap"] == pytest.approx(1.0)


def test_kill_check_not_triggered_below_threshold():
    report = specialization_report(_hists_3domain())
    kill = kill_check(report, pair=("a", "c"), threshold=0.90)
    assert kill["triggered"] is False
    assert kill["specializing"] is True


def test_kill_check_boundary_is_greater_equal():
    report = specialization_report(_hists_3domain())
    kill = kill_check(report, pair=("a", "b"), threshold=1.0)   # overlap == 1.0 exactly
    assert kill["triggered"] is True                            # >=, not >


def test_kill_check_missing_domain_raises():
    report = specialization_report(_hists_3domain())
    with pytest.raises(ValueError, match="not in the report"):
        kill_check(report, pair=("a", "nonexistent"))


def test_kill_check_unmeasured_domain_raises():
    hists = {**_hists_3domain(), "d": None}
    report = specialization_report(hists)
    with pytest.raises(ValueError, match="unmeasured"):
        kill_check(report, pair=("a", "d"))


def test_kill_check_specializing_is_not_triggered():
    report = specialization_report(_hists_3domain())
    for pair in (("a", "b"), ("a", "c")):
        kill = kill_check(report, pair=pair, threshold=0.5)
        assert kill["specializing"] == (not kill["triggered"])


# --------------------------------------------------------------------------- #
# format_routing_report
# --------------------------------------------------------------------------- #
def test_format_routing_report_names_every_pair_and_verdict():
    report = specialization_report(_hists_3domain())
    kill = kill_check(report, pair=("a", "b"), threshold=0.90)
    text = format_routing_report(report, kill)
    for key in report["by_pair"]:
        assert key in text
    assert "a|b" in text          # the kill verdict line


# --------------------------------------------------------------------------- #
# expert_histograms — duck-typed fake model + tiny real packed files
# --------------------------------------------------------------------------- #
class _FakeModel:
    """The slice of the backend model `expert_histograms` touches: `set_moe_load_counting`,
    `forward` (return value discarded, so any array works), and `pop_moe_load` — scripted
    to return one histogram per call, in call order. The first call is always the
    module's mandatory discard-pop before the first domain."""

    def __init__(self, script):
        self._script = list(script)
        self.counting = False

    def set_moe_load_counting(self, flag):
        self.counting = bool(flag)

    def forward(self, inputs):
        return np.asarray(inputs)

    def pop_moe_load(self):
        if not self._script:
            raise AssertionError("pop_moe_load() called more times than scripted")
        return self._script.pop(0)


def _packed(tmp_path, name, n_tokens, vocab=32, seed=0):
    path = tmp_path / f"{name}.bin"
    ids = np.random.default_rng(seed).integers(0, vocab, size=n_tokens)
    pack_ids(ids, path)
    return path


def test_expert_histograms_empty_domains_raises(tmp_path):
    model = _FakeModel([[[1.0]]])   # discard-pop only; never reached
    with pytest.raises(ValueError, match="no domains"):
        expert_histograms(model, {})


def test_expert_histograms_missing_packed_path_raises(tmp_path):
    model = _FakeModel([[[1.0, 1.0]]])   # non-empty discard-pop: model has MoE layers
    domains = {"missing": str(tmp_path / "does_not_exist.bin")}
    with pytest.raises(FileNotFoundError):
        expert_histograms(model, domains)


def test_expert_histograms_no_moe_model_raises(tmp_path):
    packed = _packed(tmp_path, "d", n_tokens=200, vocab=32)
    model = _FakeModel([[]])   # pop_moe_load() == [] -> no MoE layers
    with pytest.raises(ValueError, match="no MoE layers"):
        expert_histograms(model, {"d": str(packed)}, seq_len=8, batch_size=2)


def test_expert_histograms_all_zero_histogram_raises(tmp_path):
    packed = _packed(tmp_path, "d", n_tokens=200, vocab=32)
    # discard-pop (non-empty), then the domain's own pop returns an all-zero histogram.
    model = _FakeModel([[[1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]])
    with pytest.raises(ValueError, match="every expert count is zero"):
        expert_histograms(model, {"d": str(packed)}, seq_len=8, batch_size=2)


def test_expert_histograms_too_small_domain_records_none(tmp_path):
    # seq_len=8 needs 9 tokens/chunk; 4 tokens is too small for even one chunk.
    tiny = _packed(tmp_path, "tiny", n_tokens=4, vocab=32)
    ok = _packed(tmp_path, "ok", n_tokens=200, vocab=32)
    model = _FakeModel([[[1.0]], [[5.0, 3.0]]])   # discard-pop, then "ok" domain's pop
    out = expert_histograms(model, {"tiny": str(tiny), "ok": str(ok)},
                            seq_len=8, batch_size=2, max_batches=2)
    assert out["tiny"] is None
    assert out["ok"] == [[5.0, 3.0]]


def test_expert_histograms_all_none_raises(tmp_path):
    tiny_a = _packed(tmp_path, "a", n_tokens=4, vocab=32)
    tiny_b = _packed(tmp_path, "b", n_tokens=4, vocab=32)
    model = _FakeModel([[[1.0]]])   # discard-pop only; neither domain reaches a real pop
    with pytest.raises(ValueError, match="too small to score"):
        expert_histograms(model, {"a": str(tiny_a), "b": str(tiny_b)}, seq_len=8, batch_size=2)


def test_expert_histograms_returns_scripted_hist_per_domain(tmp_path):
    a = _packed(tmp_path, "a", n_tokens=200, vocab=32)
    b = _packed(tmp_path, "b", n_tokens=200, vocab=32)
    model = _FakeModel([[[1.0]], [[9.0, 1.0]], [[2.0, 8.0]]])
    out = expert_histograms(model, {"a": str(a), "b": str(b)},
                            seq_len=8, batch_size=2, max_batches=2)
    assert out == {"a": [[9.0, 1.0]], "b": [[2.0, 8.0]]}
    assert model.counting is True
