"""Portable tests for the external-suite loaders/adapters (#221, `src/eval/external_sets.py`).

Everything here runs offline against the checked-in **synthetic** fixtures. The two
fail-loud behaviours are the important assertions: an unpinned live pull must raise, and a
schema drift must raise rather than yield a row with empty fields.
"""

import pytest

from src.eval.external_sets import (
    EXTERNAL_SETS,
    VALID_KINDS,
    external_sets_manifest,
    get_external_set,
    load_external,
    normalize_crosscodeeval,
    normalize_multipl_e,
    normalize_safim,
    revision_for,
)

EXPECTED = {"multipl-e-humaneval-ts", "multipl-e-mbpp-ts", "safim", "real-fim-eval",
            "crosscodeeval", "repobench", "mceval"}


def test_all_seven_named_suites_are_present():
    assert set(EXTERNAL_SETS) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_fixture_exists_and_normalizes(name):
    rows = load_external(name)
    assert rows, f"{name}: fixture is empty, so its adapter is untested"
    for row in rows:
        assert sorted(row) == ["answer", "id", "kind", "meta", "prompt", "suffix"]
        assert row["kind"] in VALID_KINDS
        assert row["id"] and row["prompt"]
        # `suffix` is non-None only for infill sets — the invariant the driver relies on
        # when it lays an infill row out PSM-style.
        if row["kind"] == "infill":
            assert row["suffix"] is not None
        else:
            assert row["suffix"] is None


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_entry_is_unpinned_and_says_so(name):
    """The pin table ships unpinned by design (a commit SHA cannot be resolved offline, and
    inventing one is worse than not having one). When a pin is filled this test is the thing
    that has to be updated deliberately — the status can never drift silently."""
    assert revision_for(name) is None
    assert external_sets_manifest()[name]["pinned"] is False


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_live_pull_without_a_pin_raises_and_names_the_fix(name):
    with pytest.raises(SystemExit, match="no pinned revision"):
        load_external(name, fixture_only=False)


def test_limit_caps_rows():
    assert len(load_external("safim", limit=2)) == 2


def test_unknown_set_is_rejected():
    with pytest.raises(SystemExit, match="unknown external set"):
        get_external_set("not-a-suite")
    with pytest.raises(SystemExit, match="unknown external set"):
        load_external("not-a-suite")


def test_manifest_echoes_everything_the_results_json_needs():
    manifest = external_sets_manifest()
    assert set(manifest) == EXPECTED
    for name, entry in manifest.items():
        assert entry["revision"] is None and entry["pinned"] is False
        assert entry["split"] and entry["hf_repo"]
        assert entry["n_fixture_rows"] > 0
        assert entry["fixture"].startswith("eval_sets/external/")
    # Only the MultiPL-E identifiers were confirmable offline; the rest must say so.
    assert manifest["multipl-e-humaneval-ts"]["repo_verified"] is True
    assert manifest["safim"]["repo_verified"] is False
    assert "NOT verified" in manifest["safim"]["note"]


# --------------------------------------------------------------------------------------- #
# Adapters fail loudly on schema drift
# --------------------------------------------------------------------------------------- #

def test_multipl_e_has_no_gold_answer():
    """MultiPL-E ships tests, not bodies — `answer` must be None so nothing downstream
    fabricates a teacher-forced score for it."""
    rows = load_external("multipl-e-humaneval-ts")
    assert all(r["answer"] is None for r in rows)
    assert all(r["meta"]["stop_tokens"] for r in rows)


def test_safim_carries_prefix_suffix_and_gold():
    row = normalize_safim({"task_id": "t", "prompt": "p", "suffix": "s", "ground_truth": "g"})
    assert (row["id"], row["prompt"], row["suffix"], row["answer"]) == ("t", "p", "s", "g")
    assert row["kind"] == "infill"


def test_crossfile_context_rides_in_meta_not_the_prompt():
    """How cross-file context is laid into the window is the caller's experimental choice,
    so the adapter must not silently concatenate it."""
    row = normalize_crosscodeeval({"task_id": "t", "prompt": "p", "groundtruth": "g",
                                   "crossfile_context": "CTX"})
    assert row["prompt"] == "p" and "CTX" not in row["prompt"]
    assert row["meta"]["crossfile_context"] == "CTX"


@pytest.mark.parametrize("normalize,row", [
    (normalize_multipl_e, {"prompt": "p"}),                       # missing `name`
    (normalize_safim, {"task_id": "t", "prompt": "p"}),           # missing suffix/gold
    (normalize_crosscodeeval, {"task_id": "t", "prompt": "p"}),   # missing groundtruth
])
def test_schema_drift_raises_rather_than_emitting_an_empty_row(normalize, row):
    with pytest.raises(ValueError, match="missing"):
        normalize(row)
