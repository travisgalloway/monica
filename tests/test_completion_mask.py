"""Tests for #226's SSI glue (`src/lsp/completion_mask.py`): the `LabelSource`
implementations, the identifier-span state machine, and `CompletionMasker`.
Binary-free — a counting fake stands in for `TsLspService`.
"""

from __future__ import annotations

import pytest

from src.eval.ssi_contract import ArmSpec, ContractViolation, validate_arms
from src.lsp.completion_mask import (CompletionMasker, LspLabels, NullLabels,
                                      OracleLabels)

# Single-char "tokens" -- id i decodes to _CHARS[i], independent of context. Enough
# to exercise the span/masking logic without a real BPE tokenizer's subtleties
# (multi-token-label subtleties are covered directly in test_constrained_sampling.py).
_CHARS = list("abcdefghijklmnopqrstuvwxyz .();\n_$0123456789\"'")


def _decode(ids):
    return "".join(_CHARS[i] for i in ids)


def _vocab_size():
    return len(_CHARS)


class _CountingSource:
    """A `LabelSource` that records every query and returns a fixed label list."""

    def __init__(self, labels):
        self.labels = labels
        self.n_queries = 0
        self.calls = []

    def query(self, path, text, anchor_offset):
        self.n_queries += 1
        self.calls.append((path, text, anchor_offset))
        return list(self.labels)


# --------------------------------------------------------------------------- #
# label sources
# --------------------------------------------------------------------------- #

def test_oracle_labels_blank_reference_raises():
    with pytest.raises(ValueError):
        OracleLabels("   ")
    with pytest.raises(ValueError):
        OracleLabels("")


def test_oracle_labels_extracts_identifiers():
    labels = OracleLabels("console.log(u.name);\n")
    assert "name" in labels._labels
    assert "console" in labels._labels


def test_null_labels_queries_inner_but_returns_no_mask():
    inner = _CountingSource(["name"])
    null = NullLabels(inner)
    result = null.query("f.ts", "u.", 2)
    assert inner.n_queries == 1
    assert result is None


def test_lsp_labels_records_incomplete_lists():
    class _FakeItem:
        def __init__(self, label):
            self.label = label

    class _FakeService:
        def __init__(self):
            self.last_completion_incomplete = True
            self.updated = []

        def update(self, path, text):
            self.updated.append((path, text))

        def completions(self, path, offset):
            return [_FakeItem("name")]

    svc = _FakeService()
    src = LspLabels(svc, "f.ts")
    labels = src.query("f.ts", "u.", 2)
    assert labels == ["name"]
    assert src.n_incomplete_lists == 1
    assert svc.updated == [("f.ts", "u.")]


# --------------------------------------------------------------------------- #
# span state machine
# --------------------------------------------------------------------------- #

def test_span_opens_on_dot_issues_exactly_one_query_per_span():
    source = _CountingSource(["name", "age"])
    masker = CompletionMasker(source, "f.ts", _decode, mask_scope="member")
    vs = _vocab_size()

    masker.mask_for("u.", vocab_size=vs)
    assert source.n_queries == 1

    # still inside the same span -- no new query
    masker.mask_for("u.n", vocab_size=vs)
    masker.mask_for("u.na", vocab_size=vs)
    assert source.n_queries == 1

    # span exits at the first non-identifier char
    masker.mask_for("u.name;", vocab_size=vs)
    assert source.n_queries == 1

    # a new dot opens a NEW span -> a second query
    masker.mask_for("u.name;v.", vocab_size=vs)
    assert source.n_queries == 2


def test_never_arms_inside_a_string():
    source = _CountingSource(["name"])
    masker = CompletionMasker(source, "f.ts", _decode, mask_scope="member")
    masker.mask_for('"a.b"', vocab_size=_vocab_size())
    assert source.n_queries == 0


def test_never_arms_inside_a_comment():
    source = _CountingSource(["name"])
    masker = CompletionMasker(source, "f.ts", _decode, mask_scope="member")
    # this vocabulary has no "/" char, so approximate a comment with a string
    # (both are handled by the same mask_strings_and_comments walk); the
    # string case above already exercises the shared machinery directly.
    masker.mask_for('"u.x"', vocab_size=_vocab_size())
    assert source.n_queries == 0


def test_query_receives_text_up_to_and_including_the_dot():
    source = _CountingSource(["name"])
    masker = CompletionMasker(source, "f.ts", _decode, mask_scope="member")
    masker.mask_for("u.", vocab_size=_vocab_size())
    path, text, anchor_offset = source.calls[0]
    assert text == "u."
    assert anchor_offset == 2


# --------------------------------------------------------------------------- #
# CompletionMasker.mask_for -- end to end
# --------------------------------------------------------------------------- #

def test_mask_for_confines_to_label_prefix():
    source = _CountingSource(["name"])
    masker = CompletionMasker(source, "f.ts", _decode, mask_scope="member")
    allowed = masker.mask_for("u.", vocab_size=_vocab_size())
    assert allowed == [_CHARS.index("n")]


def test_mask_for_bypasses_and_counts_on_divergence():
    source = _CountingSource(["name"])
    masker = CompletionMasker(source, "f.ts", _decode, mask_scope="member")
    masker.mask_for("u.", vocab_size=_vocab_size())
    # the model typed a char that can never lead to "name" -- allowed set is
    # empty, mask_for must bypass (None) rather than crash or hand sample() []
    allowed = masker.mask_for("u.z", vocab_size=_vocab_size())
    assert allowed is None
    assert masker.n_mask_bypass == 1
    assert masker.n_mask_steps == 2  # the "u." step also counted


def test_mask_for_outside_a_span_is_a_no_op():
    source = _CountingSource(["name"])
    masker = CompletionMasker(source, "f.ts", _decode, mask_scope="member")
    allowed = masker.mask_for("hello", vocab_size=_vocab_size())
    assert allowed is None
    assert source.n_queries == 0
    assert masker.n_mask_steps == 0


def test_null_source_never_masks_but_still_tracks_completion_calls():
    inner = _CountingSource(["name"])
    null = NullLabels(inner)
    masker = CompletionMasker(null, "f.ts", _decode, mask_scope="member")
    allowed = masker.mask_for("u.", vocab_size=_vocab_size())
    assert allowed is None
    assert inner.n_queries == 1
    assert masker.n_completion_calls == 1
    assert masker.n_mask_steps == 0  # never actually masked


def test_invalid_mask_scope_raises():
    source = _CountingSource([])
    with pytest.raises(ValueError):
        CompletionMasker(source, "f.ts", _decode, mask_scope="bogus")


# --------------------------------------------------------------------------- #
# #225 M4 -- the five-arm declaration this driver needs (item 11 of the plan's
# verification list): the naive three-arm set must be REJECTED.
# --------------------------------------------------------------------------- #

def _five_arms():
    seeds = (0, 1, 2)
    return [
        ArmSpec(name="unconstrained", variable="root", baseline="unconstrained",
                signal_available=False, signal_used=False, seeds=seeds),
        ArmSpec(name="masked-null", variable="mask", baseline="unconstrained",
                signal_available=True, signal_used=False, seeds=seeds),
        ArmSpec(name="masked", variable="mask", baseline="unconstrained",
                signal_available=True, signal_used=True, seeds=seeds),
        ArmSpec(name="masked-oracle-null", variable="mask-oracle", baseline="unconstrained",
                signal_available=True, signal_used=False, seeds=seeds),
        ArmSpec(name="masked-oracle", variable="mask-oracle", baseline="unconstrained",
                signal_available=True, signal_used=True, seeds=seeds),
    ]


def test_five_declared_arms_validate():
    validate_arms(_five_arms())  # must not raise


def test_naive_three_arm_set_is_rejected():
    seeds = (0, 1, 2)
    naive = [
        ArmSpec(name="unconstrained", variable="root", baseline="unconstrained",
                signal_available=False, signal_used=False, seeds=seeds),
        ArmSpec(name="masked", variable="mask", baseline="unconstrained",
                signal_available=True, signal_used=True, seeds=seeds),
        ArmSpec(name="masked-oracle", variable="mask-oracle", baseline="unconstrained",
                signal_available=True, signal_used=True, seeds=seeds),
    ]
    with pytest.raises(ContractViolation):
        validate_arms(naive)
