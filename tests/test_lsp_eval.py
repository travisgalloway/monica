"""Tests for `src/eval/lsp_eval.py` — pure scoring, no model, no `tsc`."""

from __future__ import annotations

from typing import List

import pytest

from src.eval.lsp_eval import compare, mcnemar_p_value, score_record, summarize
from src.lsp.diagnostics import Diagnostic
from src.lsp.harness import GenResult

_MEMBER_ACCESS_REC = {
    "id": "member-access-001",
    "error_class": "unfamiliar_member_access",
    "expected_diagnostic": "TS2339",
    "prompt": "console.log(u.",
    "gold_completion": "name);\n",
    "error_completion": "gorblak);\n",
    "notes": "test fixture",
}

_CLEAN_CONTROL_REC = {
    "id": "clean-control-001",
    "error_class": "clean_control",
    "expected_diagnostic": "",
    "prompt": "console.log(u.",
    "gold_completion": "name);\n",
    "error_completion": "",
    "notes": "test fixture",
}


def _result(completion: str, **kwargs) -> GenResult:
    prompt = kwargs.pop("prompt", "console.log(u.")
    defaults = dict(strategy="test", prompt=prompt, completion=completion,
                     context=prompt + completion)
    defaults.update(kwargs)
    return GenResult(**defaults)


def _no_diags(source: str) -> List[Diagnostic]:
    return []


def _flag_gorblak(source: str) -> List[Diagnostic]:
    idx = source.find("gorblak")
    if idx == -1:
        return []
    return [Diagnostic(code="TS2339", line=1, col=idx + 1, message="msg", offset=idx)]


# --------------------------------------------------------------------------- #
# score_record
# --------------------------------------------------------------------------- #

def test_score_record_clean_and_avoided():
    result = _result("name);")
    scored = score_record(_MEMBER_ACCESS_REC, result, _no_diags)
    assert scored["clean"] is True
    assert scored["avoided"] is True
    assert scored["exact_gold"] is False  # gold_completion has a trailing "\n", completion doesn't
    assert scored["truncated"] is False
    assert scored["suppression_hack"] is False


def test_score_record_exact_gold_match():
    result = _result("name);\n")
    scored = score_record(_MEMBER_ACCESS_REC, result, _no_diags)
    assert scored["exact_gold"] is True


def test_score_record_error_not_avoided():
    result = _result("gorblak);")
    scored = score_record(_MEMBER_ACCESS_REC, result, _flag_gorblak)
    assert scored["clean"] is False
    assert scored["avoided"] is False
    assert "TS2339" in scored["codes"]


def test_score_record_clean_control_has_no_avoided_verdict():
    result = _result("name);")
    scored = score_record(_CLEAN_CONTROL_REC, result, _no_diags)
    assert scored["is_error_row"] is False
    assert scored["avoided"] is None
    assert scored["clean"] is True


def test_score_record_truncated_forces_not_clean():
    # No statement boundary anywhere in the completion -> truncated, even though
    # the (fake) diagnose_fn reports nothing wrong with it.
    result = _result("name")  # no trailing `;` or `\n`
    scored = score_record(_MEMBER_ACCESS_REC, result, _no_diags)
    assert scored["truncated"] is True
    assert scored["clean"] is False


def test_score_record_suppression_hack_forces_not_clean():
    result = _result("gorblak); // @ts-ignore\n")
    scored = score_record(_MEMBER_ACCESS_REC, result, _no_diags)  # tsc silenced by the hack
    assert scored["suppression_hack"] is True
    assert scored["clean"] is False


def test_score_record_rolled_back_flag():
    result = _result("name);", n_rollbacks=2)
    scored = score_record(_MEMBER_ACCESS_REC, result, _no_diags)
    assert scored["rolled_back"] is True

    result2 = _result("name);", n_rollbacks=0)
    scored2 = score_record(_MEMBER_ACCESS_REC, result2, _no_diags)
    assert scored2["rolled_back"] is False


def test_score_record_suggestion_leak_from_events():
    events = [{"kind": "soft_repair", "code": "TS2304",
               "comment": "// tsc: TS2304: Cannot find name 'x'. Did you mean 'y'?"}]
    result = _result("name);", events=events, n_soft_repairs=1)
    scored = score_record(_MEMBER_ACCESS_REC, result, _no_diags)
    assert scored["suggestion_leak"] is True


def test_score_record_no_suggestion_leak_without_did_you_mean():
    events = [{"kind": "soft_repair", "code": "TS2304", "comment": "// tsc: TS2304: bad"}]
    result = _result("name);", events=events, n_soft_repairs=1)
    scored = score_record(_MEMBER_ACCESS_REC, result, _no_diags)
    assert scored["suggestion_leak"] is False


def test_score_record_cost_fields_carried_through():
    result = _result("name);", n_forward_tokens=10, n_forward_tokens_nocache=15,
                      n_generated_tokens=3, n_tsc_calls=2, wall_s=0.5)
    scored = score_record(_MEMBER_ACCESS_REC, result, _no_diags)
    assert scored["n_forward_tokens"] == 10
    assert scored["n_forward_tokens_nocache"] == 15
    assert scored["n_generated_tokens"] == 3
    assert scored["n_tsc_calls"] == 2
    assert scored["wall_s"] == 0.5


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #

def test_summarize_rates():
    error_result_avoided = _result("name);")
    error_result_not_avoided = _result("gorblak);")
    clean_control_result = _result("name);", n_rollbacks=1)

    scored = [
        score_record(_MEMBER_ACCESS_REC, error_result_avoided, _no_diags),
        score_record(dict(_MEMBER_ACCESS_REC, id="member-access-002"),
                     error_result_not_avoided, _flag_gorblak),
        score_record(_CLEAN_CONTROL_REC, clean_control_result, _no_diags),
    ]
    summary = summarize(scored)

    assert summary["n"] == 3
    assert summary["n_error_rows"] == 2
    assert summary["n_clean_control_rows"] == 1
    assert summary["error_avoidance_rate"] == pytest.approx(0.5)
    assert summary["over_repair_rate"] == pytest.approx(1.0)  # the 1 clean_control row rolled back
    assert summary["diagnostic_clean_rate"] == pytest.approx(2 / 3)


def test_summarize_empty_is_nan_not_crash():
    summary = summarize([])
    assert summary["n"] == 0


# --------------------------------------------------------------------------- #
# mcnemar_p_value
# --------------------------------------------------------------------------- #

def test_mcnemar_p_value_no_discordant_pairs_is_one():
    assert mcnemar_p_value(0, 0) == 1.0


def test_mcnemar_p_value_symmetric_discordance_is_high():
    # Equal numbers of flips each way -> maximally non-significant.
    assert mcnemar_p_value(5, 5) == pytest.approx(1.0)


def test_mcnemar_p_value_matches_hand_computed():
    # n=10, k=1 (one-sided tail), exact binomial two-sided p.
    from math import comb
    expected = min(1.0, 2 * (comb(10, 0) + comb(10, 1)) / (2 ** 10))
    assert mcnemar_p_value(9, 1) == pytest.approx(expected)


def test_mcnemar_p_value_is_symmetric_in_its_arguments():
    assert mcnemar_p_value(2, 8) == mcnemar_p_value(8, 2)


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

def test_compare_table_and_rates():
    baseline = [{"clean": True}, {"clean": True}, {"clean": False}, {"clean": False}]
    other = [{"clean": True}, {"clean": False}, {"clean": True}, {"clean": False}]
    # both_true=1, baseline_true_other_false=1, baseline_false_other_true=1, both_false=1
    result = compare(baseline, other, key="clean")

    assert result["table"] == {"both_true": 1, "baseline_true_other_false": 1,
                                "baseline_false_other_true": 1, "both_false": 1}
    assert result["baseline_rate"] == pytest.approx(0.5)
    assert result["other_rate"] == pytest.approx(0.5)
    assert result["flip_to_true_rate"] == pytest.approx(0.5)   # 1 of 2 baseline-False -> True
    assert result["flip_to_false_rate"] == pytest.approx(0.5)  # 1 of 2 baseline-True -> False
    assert result["mcnemar_p"] == mcnemar_p_value(1, 1)


def test_compare_all_repaired_no_regressions():
    baseline = [{"clean": False}] * 4
    other = [{"clean": True}] * 4
    result = compare(baseline, other, key="clean")
    assert result["flip_to_true_rate"] == 1.0
    assert result["flip_to_false_rate"] != result["flip_to_false_rate"]  # NaN: no baseline-True rows


def test_compare_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compare([{"clean": True}], [{"clean": True}, {"clean": False}], key="clean")
