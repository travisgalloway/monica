"""Portable tests for the external-suite loaders/adapters (#221, #304,
`src/eval/external_sets.py`).

Everything here except the `MONICA_EXTERNAL_LIVE=1` block runs offline against the
checked-in **synthetic** fixtures and needs no network and no `datasets` install. The
fail-loud behaviours are the important assertions: an unpinned live pull must raise, a stale
pin must fail *by name*, and a schema drift — including a column that exists but holds the
wrong *type* — must raise rather than yield a row with a junk prompt.

**A fixture result is not evidence that a live pull works.** That is what the opt-in live
test at the bottom is for; it is deliberately not wired into any CI job (see
`eval_sets/external/README.md`), so CI never silently "proves" the live path.
"""

import os
import re
import sys
import types
from dataclasses import replace

import pytest

from src.eval.external_sets import (
    EXTERNAL_SETS,
    SAFIM_PLACEHOLDER,
    VALID_KINDS,
    external_sets_manifest,
    get_external_set,
    load_external,
    normalize_crosscodeeval,
    normalize_mceval,
    normalize_multipl_e,
    normalize_real_fim,
    normalize_repobench,
    normalize_safim,
    revision_for,
)

EXPECTED = {"multipl-e-humaneval-ts", "multipl-e-mbpp-ts", "safim", "real-fim-eval",
            "crosscodeeval", "repobench", "mceval"}

ROW_KEYS = ["answer", "id", "kind", "meta", "prompt", "suffix"]

LIVE = os.environ.get("MONICA_EXTERNAL_LIVE") == "1"
LIVE_REASON = "live Hugging Face pull; set MONICA_EXTERNAL_LIVE=1 to run (not wired into CI)"


def test_all_seven_named_suites_are_present():
    assert set(EXTERNAL_SETS) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_fixture_exists_and_normalizes(name):
    rows = load_external(name)
    assert rows, f"{name}: fixture is empty, so its adapter is untested"
    for row in rows:
        assert sorted(row) == ROW_KEYS
        assert row["kind"] in VALID_KINDS
        assert row["id"] and row["prompt"]
        # `suffix` is non-None only for infill sets — the invariant the driver relies on
        # when it lays an infill row out PSM-style.
        if row["kind"] == "infill":
            assert row["suffix"] is not None
        else:
            assert row["suffix"] is None


# --------------------------------------------------------------------------------------- #
# The pins (#304)
# --------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_entry_is_pinned_to_a_real_sha(name):
    """Every entry carries a real 40-hex HF commit SHA (#304). Before that, the table
    shipped unpinned and no live pull was possible at all."""
    revision = revision_for(name)
    assert revision is not None, f"{name}: still unpinned"
    assert re.fullmatch(r"[0-9a-f]{40}", revision), f"{name}: {revision!r} is not a commit SHA"
    assert external_sets_manifest()[name]["pinned"] is True


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_entry_records_how_its_identifier_was_confirmed(name):
    """`repo_verified` is not a boolean anyone should have to take on trust: the `note` must
    say how and when the identifier was checked."""
    spec = get_external_set(name)
    assert spec.repo_verified is True, f"{name}: identifier not confirmed"
    assert spec.note.strip(), f"{name}: no note recording the confirmation method"
    assert "2026-08-20" in spec.note, f"{name}: note does not date the confirmation"


def test_the_corrected_identifiers_say_why_they_changed():
    """Four ids/configs were wrong in the table #221 shipped. Each correction has to carry
    its reason, or the next reader re-litigates it."""
    for name in ("real-fim-eval", "crosscodeeval", "repobench"):
        assert "CORRECTED #304" in get_external_set(name).note
    for name in ("safim", "mceval"):
        assert "CONFIG CORRECTED #304" in get_external_set(name).note
    # The CrossCodeEval entry is a third-party mirror and must not pretend otherwise.
    assert "third-party mirror" in get_external_set("crosscodeeval").note
    assert "3356" in get_external_set("crosscodeeval").note
    # RepoBench v1.1 has no TypeScript, which is a measurement caveat, not a footnote.
    assert "Python" in get_external_set("repobench").note


def test_a_live_pull_without_a_pin_raises_and_names_the_fix(monkeypatch):
    """DoD item 4. Every shipped entry is now pinned, so this is proven against a
    *synthetic* unpinned entry — the guard has to keep working for whatever is added next.
    Needs no network: the check happens before `datasets` is even imported."""
    original = get_external_set("safim")
    monkeypatch.setitem(EXTERNAL_SETS, "safim", replace(original, revision=None))
    with pytest.raises(SystemExit, match="no pinned revision"):
        load_external("safim", fixture_only=False)
    monkeypatch.undo()
    assert revision_for("safim") == original.revision


def _fake_datasets(exc: Exception) -> types.ModuleType:
    """A stand-in `datasets` whose `load_dataset` fails, so the wrapper can be tested
    offline and without the optional `[data]` extra installed."""
    module = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        raise exc

    module.load_dataset = load_dataset
    return module


def test_a_stale_pin_fails_by_name_not_as_an_opaque_traceback(monkeypatch):
    """DoD item 5. A withdrawn revision / renamed / newly-gated repo must surface as a
    `SystemExit` naming the set and the whole pin, not a bare `datasets` traceback."""
    bogus = "0" * 40
    monkeypatch.setitem(EXTERNAL_SETS, "mceval",
                        replace(get_external_set("mceval"), revision=bogus))
    monkeypatch.setitem(sys.modules, "datasets",
                        _fake_datasets(FileNotFoundError("Revision not found")))
    with pytest.raises(SystemExit) as excinfo:
        load_external("mceval", fixture_only=False)
    message = str(excinfo.value)
    assert "mceval" in message
    assert bogus in message
    assert "Multilingual-Multimodal-NLP/McEval" in message
    assert "generation" in message and "test" in message
    assert "FileNotFoundError" in message          # the original cause is not swallowed
    assert excinfo.value.__cause__ is not None     # ... and it is chained


def test_data_files_is_threaded_through_to_load_dataset(monkeypatch):
    """The CrossCodeEval mirror declares no configs, so the file has to be named
    explicitly. Assert it actually reaches `load_dataset` rather than being carried in the
    dataclass and dropped."""
    seen = {}
    module = types.ModuleType("datasets")

    def load_dataset(repo, config, *, split, revision, **kwargs):
        seen.update(repo=repo, config=config, split=split, revision=revision, **kwargs)
        return []

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)

    assert load_external("crosscodeeval", fixture_only=False) == []
    spec = get_external_set("crosscodeeval")
    assert seen["data_files"] == spec.data_files
    assert seen["data_files"].endswith("typescript/line_completion_rg1_bm25.jsonl")
    assert (seen["repo"], seen["split"], seen["revision"]) == (
        spec.hf_repo, spec.split, spec.revision)

    seen.clear()
    assert load_external("mceval", fixture_only=False) == []
    assert "data_files" not in seen, "sets without a data_files pin must not pass one"


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
        assert entry["revision"] is not None and entry["pinned"] is True
        assert entry["repo_verified"] is True and entry["note"]
        assert entry["split"] and entry["hf_repo"]
        assert entry["n_fixture_rows"] > 0
        assert entry["fixture"].startswith("eval_sets/external/")
    # Two sets had `config=None`, which is invalid upstream — the manifest is where a run
    # would show that regression.
    assert manifest["safim"]["config"] == "block"
    assert manifest["mceval"]["config"] == "generation"
    assert manifest["crosscodeeval"]["data_files"]
    assert manifest["multipl-e-humaneval-ts"]["data_files"] is None


# --------------------------------------------------------------------------------------- #
# Adapters, against the REAL upstream column names (observed live 2026-08-20)
# --------------------------------------------------------------------------------------- #

def test_multipl_e_has_no_gold_answer():
    """MultiPL-E ships tests, not bodies — `answer` must be None so nothing downstream
    fabricates a teacher-forced score for it."""
    rows = load_external("multipl-e-humaneval-ts")
    assert all(r["answer"] is None for r in rows)
    assert all(r["meta"]["stop_tokens"] for r in rows)


def test_safim_splits_eval_prompt_on_the_completion_placeholder():
    """SAFIM has **no `suffix` column**, and its `prompt` is the NL problem statement. The
    prefix/suffix come from `eval_prompt` around `{{completion}}`."""
    row = normalize_safim({
        "task_id": "t", "lang": "typescript", "prompt": "Complete the block.",
        "eval_prompt": "const a = 1;\n" + SAFIM_PLACEHOLDER + "\nreturn a;\n",
        "ground_truth": "a += 1;", "unit_tests": "ok",
    })
    assert (row["id"], row["kind"], row["answer"]) == ("t", "infill", "a += 1;")
    assert row["prompt"] == "const a = 1;\n"
    assert row["suffix"] == "\nreturn a;\n"
    # The NL statement must not shadow the normalized (code) prompt.
    assert row["meta"]["problem_statement"] == "Complete the block."
    assert "Complete the block." not in row["prompt"]


def test_safim_raises_when_the_completion_placeholder_is_missing():
    """A silent `partition` miss would hand back an empty suffix, turning an infill instance
    into a trivially easy completion — a scoring bug that never announces itself."""
    with pytest.raises(ValueError, match="placeholder"):
        normalize_safim({"task_id": "t", "eval_prompt": "no hole here", "ground_truth": "g"})


def test_real_fim_builds_an_id_because_upstream_has_no_id_column():
    row = normalize_real_fim({
        "repo": "org/repo", "ref": "abc123", "path": "src/a.ts", "prompt": "const a = ",
        "suffix": ";\n", "canonical_solution": "1", "lang": "TypeScript", "timestamp": None,
    })
    assert row["id"] == "org/repo@abc123:src/a.ts"
    assert (row["prompt"], row["suffix"], row["answer"]) == ("const a = ", ";\n", "1")


def test_real_fim_allows_a_top_of_file_insertion_with_an_empty_prefix():
    """56 of Real-FIM-Eval `add`'s 17879 real rows insert at the top of a file, so the
    prefix is legitimately empty while the suffix carries the context. Rejecting those would
    drop real instances; rejecting an empty *pair* is what matters."""
    row = normalize_real_fim({
        "repo": "org/repo", "ref": "abc", "path": "src/a.ts", "prompt": "",
        "suffix": "export const a = 1;\n", "canonical_solution": "// header\n",
    })
    assert row["prompt"] == "" and row["suffix"]
    with pytest.raises(ValueError, match="prompt is empty"):
        normalize_real_fim({"repo": "r", "ref": "", "path": "p", "prompt": "",
                            "suffix": "", "canonical_solution": "x"})


def test_crossfile_context_rides_in_meta_not_the_prompt():
    """How cross-file context is laid into the window is the caller's experimental choice,
    so the adapter must not silently concatenate it."""
    row = normalize_crosscodeeval({
        "prompt": "p", "groundtruth": "g", "right_context": "\n}\n",
        "metadata": {"task_id": "project_cc_typescript/7", "repository": "org-repo-abc",
                     "file": "src/a.ts"},
        "crossfile_context": {"text": "CTX", "list": [{"retrieved_chunk": "CTX"}]},
    })
    assert row["prompt"] == "p" and "CTX" not in row["prompt"]
    assert row["id"] == "project_cc_typescript/7"
    assert row["meta"]["crossfile_context"] == "CTX"
    assert row["meta"]["right_context"] == "\n}\n"
    assert row["meta"]["repo"] == "org-repo-abc"


def test_repobench_takes_its_prompt_from_cropped_code_not_the_context_structs():
    """RepoBench v1.1's `context` is a **list of `{identifier, path, snippet}` structs**.
    `prompt=row["context"]` would pass a presence-only check and emit a non-string prompt —
    the fail-open this test exists to keep closed."""
    context = [{"identifier": "h.helper", "path": "src/h.ts", "snippet": "export const h = 1;"}]
    row = normalize_repobench({
        "repo_name": "org/repo", "file_path": "src/a.ts", "context": context,
        "import_statement": "import { h } from './h';", "token_num": 12,
        "cropped_code": "import { h } from './h';\n\nfunction f() {\n",
        "all_code": "...", "next_line": "  return h;", "gold_snippet_index": 0, "level": "2k",
    })
    assert row["prompt"] == "import { h } from './h';\n\nfunction f() {\n"
    assert isinstance(row["prompt"], str)
    assert row["meta"]["crossfile_snippets"] == context
    assert row["answer"] == "  return h;"


def test_a_struct_valued_prompt_is_rejected_rather_than_passed_through():
    """The guard has to actually fail on the bad shape, not just on a missing key. Feed the
    struct list in as the prompt column and it must raise."""
    context = [{"identifier": "h.helper", "path": "src/h.ts", "snippet": "export const h = 1;"}]
    with pytest.raises(ValueError, match="prompt must be a str"):
        normalize_repobench({
            "repo_name": "org/repo", "file_path": "src/a.ts", "context": context,
            "cropped_code": context,          # the old bug's shape, made explicit
            "next_line": "  return h;",
        })
    with pytest.raises(ValueError, match="prompt must be a str"):
        normalize_multipl_e({"name": "n", "prompt": ["not", "a", "string"]})


def test_mceval_derives_its_language_from_task_id():
    """McEval `generation` has **no `language` column**; `task_id` is `<Language>/<n>`."""
    row = normalize_mceval({"task_id": "TypeScript/3", "prompt": "function f() {\n",
                            "canonical_solution": "  return 1;\n}\n", "test": "t",
                            "entry_point": "f", "level": "easy"})
    assert row["id"] == "TypeScript/3"
    assert row["meta"]["language"] == "TypeScript"
    assert row["answer"] == "  return 1;\n}\n"
    with pytest.raises(ValueError, match="separator"):
        normalize_mceval({"task_id": "no-slash", "prompt": "p"})


@pytest.mark.parametrize("normalize,row", [
    (normalize_multipl_e, {"prompt": "p"}),                                   # missing `name`
    (normalize_safim, {"task_id": "t", "prompt": "p"}),                       # no eval_prompt
    (normalize_real_fim, {"repo": "r", "path": "p", "prompt": "x"}),          # no suffix/gold
    (normalize_crosscodeeval, {"prompt": "p", "groundtruth": "g"}),           # no metadata
    (normalize_repobench, {"repo_name": "r", "file_path": "f"}),              # no cropped_code
    (normalize_mceval, {"task_id": "TypeScript/1"}),                          # no prompt
])
def test_schema_drift_raises_rather_than_emitting_an_empty_row(normalize, row):
    with pytest.raises(ValueError, match="missing"):
        normalize(row)


# --------------------------------------------------------------------------------------- #
# Opt-in live pull. NOT wired into any CI job — see eval_sets/external/README.md.
# --------------------------------------------------------------------------------------- #

@pytest.mark.skipif(not LIVE, reason=LIVE_REASON)
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_live_pull_returns_documented_shape(name):
    spec = get_external_set(name)
    rows = load_external(name, fixture_only=False)
    assert rows, f"{name}: live pull returned zero rows"
    for row in rows:
        assert sorted(row) == ROW_KEYS
        assert row["kind"] == spec.kind
        assert isinstance(row["prompt"], str)
        assert str(row["id"])
        if row["kind"] == "infill":
            assert isinstance(row["suffix"], str)
            assert row["prompt"] or row["suffix"]
        else:
            assert row["suffix"] is None
            assert row["prompt"]


@pytest.mark.skipif(not LIVE, reason=LIVE_REASON)
def test_live_stale_pin_fails_by_name(monkeypatch):
    """The offline version of this stubs `datasets`; this one drives a genuinely bogus SHA
    through the real client, which is the failure an operator will actually meet."""
    bogus = "0" * 40
    monkeypatch.setitem(EXTERNAL_SETS, "multipl-e-humaneval-ts",
                        replace(get_external_set("multipl-e-humaneval-ts"), revision=bogus))
    with pytest.raises(SystemExit) as excinfo:
        load_external("multipl-e-humaneval-ts", fixture_only=False)
    assert "multipl-e-humaneval-ts" in str(excinfo.value) and bogus in str(excinfo.value)
