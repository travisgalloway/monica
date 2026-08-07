"""Portable tests for cross-file symbol-resolution recall (#221, `src/eval/code_recall.py`).

No backend. Two model shapes are used: `StubCausalModel` (context-free, causal by
construction) for the mechanical properties, and `_PreferModel` — which makes a chosen set
of token ids cheap — for the ranking assertions, so "the answer ranked first" is a real
claim about the ranking code rather than an accident of the fixture.
"""

import numpy as np
import pytest

from src.eval.code_recall import (
    build_recall_instances,
    evaluate_code_recall,
    exported_symbols,
    imported_symbols,
    resolve_module,
)
from src.eval.code_suite import StubCausalModel, load_code_files, make_byte_encoder

FIXTURE = "eval_sets/code_recall/fixture_repo.jsonl"


class _PreferModel:
    """Context-free (hence causal) model that assigns a high logit to `prefer` ids."""

    def __init__(self, prefer, vocab_size=256, hot=8.0):
        self.vocab_size = vocab_size
        self._row = np.full((vocab_size,), -hot, dtype=np.float32)
        for i in prefer:
            self._row[int(i) % vocab_size] = hot

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        return np.broadcast_to(self._row, (*inputs.shape, self.vocab_size)).copy()


# --------------------------------------------------------------------------------------- #
# Extraction is fail-closed
# --------------------------------------------------------------------------------------- #

def test_exports_are_extracted_by_kind():
    src = ("export function alpha(): void {}\n"
           "export const beta = 1;\n"
           "export class Gamma {}\n"
           "export interface Delta { x: number }\n"
           "export type Epsilon = string;\n"
           "export enum Zeta { A }\n"
           "export async function eta(): Promise<void> {}\n")
    assert exported_symbols(src) == {
        "alpha": "function", "beta": "const", "Gamma": "class", "Delta": "interface",
        "Epsilon": "type", "Zeta": "enum", "eta": "function",
    }


@pytest.mark.parametrize("src", [
    'export * from "./geometry";',                    # star re-export
    'export { slugify as toSlug } from "./strings";',  # aliased re-export
    'export { slugify } from "./strings";',            # plain re-export
    "export default function () { return 42; }",       # anonymous default
    "export default class {}",
])
def test_ambiguous_export_forms_are_skipped_never_guessed(src):
    assert exported_symbols(src) == {}


def test_a_name_exported_twice_is_dropped():
    src = "export const dup = 1;\nexport function dup(): void {}\n"
    assert "dup" not in exported_symbols(src)


def test_imports_are_extracted_and_ambiguous_specifiers_dropped():
    src = ('import { alpha, beta } from "./mod";\n'
           'import { gamma as g } from "./other";\n'
           'import defaultThing from "./third";\n'
           'import {\n  delta\n} from "./multiline";\n')
    got = imported_symbols(src)
    assert got == {"alpha": "./mod", "beta": "./mod"}
    for skipped in ("gamma", "g", "defaultThing", "delta"):
        assert skipped not in got


def test_a_name_imported_from_two_modules_is_dropped():
    src = 'import { x } from "./a";\nimport { x } from "./b";\n'
    assert imported_symbols(src) == {}


@pytest.mark.parametrize("spec,expected", [
    ("./geometry", "src/geometry.ts"),
    ("../src/geometry", "src/geometry.ts"),
    ("./geometry.ts", "src/geometry.ts"),
    ("react", None),                    # external package — no definition in the bundle
    ("./missing", None),
])
def test_resolve_module(spec, expected):
    known = ["src/geometry.ts", "src/report.ts"]
    assert resolve_module(spec, "src/report.ts", known) == expected


# --------------------------------------------------------------------------------------- #
# Instance construction
# --------------------------------------------------------------------------------------- #

def _fixture_instances(seed=0, **kw):
    files = load_code_files(FIXTURE)
    return build_recall_instances(files, make_byte_encoder(), np.random.default_rng(seed), **kw)


def test_the_checked_in_fixture_yields_instances_in_several_buckets():
    instances = _fixture_instances()
    assert instances, "the fixture must produce instances or the suite is vacuous"
    assert {i.bucket for i in instances} >= {"medium", "long"}
    # Every instance resolves a symbol its user file actually imports from its definer.
    for inst in instances:
        assert inst.definer != inst.user
        assert inst.symbol in inst.candidates
        assert inst.candidates[inst.answer_index] == inst.symbol
        assert inst.candidates == tuple(sorted(inst.candidates))


def test_the_ambiguous_fixture_files_never_become_a_definer():
    for inst in _fixture_instances():
        assert inst.definer not in ("src/reexport.ts", "src/anonymous.ts")


def test_same_seed_reproduces_the_instance_set_exactly():
    a, b = _fixture_instances(seed=7), _fixture_instances(seed=7)
    assert [i.id for i in a] == [i.id for i in b]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.prefix_tokens, y.prefix_tokens)
        assert (x.distance, x.candidates, x.n_distractor_files) == (
            y.distance, y.candidates, y.n_distractor_files)


def test_distance_is_measured_in_tokens_and_matches_its_bucket():
    from src.eval.code_suite import bucket_for_distance

    for inst in _fixture_instances():
        assert bucket_for_distance(inst.distance) == inst.bucket
        # The prefix is the definition head + everything between it and the use site.
        assert inst.prefix_tokens.size > inst.distance


def test_padding_files_only_ever_increase_the_distance():
    """The bucket search walks a growing prefix of distractor files, so within one
    (user, symbol) triple a larger bucket must never be reached with fewer files."""
    by_symbol = {}
    for inst in _fixture_instances():
        by_symbol.setdefault((inst.user, inst.symbol), []).append(inst)
    for group in by_symbol.values():
        ordered = sorted(group, key=lambda i: i.distance)
        assert [i.n_distractor_files for i in ordered] == sorted(
            i.n_distractor_files for i in ordered)


def test_n_candidates_must_leave_room_for_a_distractor():
    files = load_code_files(FIXTURE)
    with pytest.raises(ValueError):
        build_recall_instances(files, make_byte_encoder(), np.random.default_rng(0),
                               n_candidates=1)


def test_max_instances_caps_construction():
    assert len(_fixture_instances(max_instances=3)) == 3


# --------------------------------------------------------------------------------------- #
# Scoring + ranking
# --------------------------------------------------------------------------------------- #

_MINI = [
    {"path": "src/def.ts", "text": "export function aaaa(): number {\n  return 1;\n}\n"},
    {"path": "src/other.ts", "text": "export function zzzz(): number {\n  return 2;\n}\n"},
    {"path": "src/use.ts",
     "text": 'import { aaaa } from "./def";\n\nfunction run(): number {\n  return aaaa();\n}\n'},
]


def _mini_instances():
    """Exactly two exported symbols exist (`aaaa`, `zzzz`), so the candidate set is fixed
    regardless of the draw — which is what makes the ranking assertions below deterministic."""
    return build_recall_instances(_MINI, make_byte_encoder(), np.random.default_rng(0),
                                  n_candidates=2)


def test_the_mini_repo_has_a_fixed_two_candidate_set():
    instances = _mini_instances()
    assert instances
    for inst in instances:
        assert inst.candidates == ("aaaa", "zzzz")
        assert inst.symbol == "aaaa"


def test_the_answer_ranks_first_when_the_model_prefers_its_tokens():
    instances = _mini_instances()
    model = _PreferModel(set(b"a"))
    res = evaluate_code_recall(model, instances, batch_size=2)
    assert res["records"]
    for rec in res["records"]:
        assert rec["rank_top1"] is True
        assert rec["mrr"] == pytest.approx(1.0)
        assert rec["meta"]["rank"] == 1


def test_and_ranks_last_when_the_model_prefers_a_distractors_tokens():
    """Anti-vacuity for the test above — the ranking must be able to be WRONG."""
    instances = _mini_instances()
    model = _PreferModel(set(b"z"))
    res = evaluate_code_recall(model, instances, batch_size=2)
    for rec in res["records"]:
        assert rec["rank_top1"] is False
        assert rec["mrr"] == pytest.approx(0.5)


def test_records_carry_the_shared_schema_and_aggregate_by_bucket():
    from src.eval.code_suite import RECORD_FIELDS

    res = evaluate_code_recall(StubCausalModel(vocab_size=256, seed=0), _fixture_instances(),
                               batch_size=4)
    assert res["records"]
    for rec in res["records"]:
        assert tuple(sorted(rec)) == tuple(sorted(RECORD_FIELDS))
        assert rec["suite"] == "code_recall"
        assert rec["n_scored_tokens"] > 0
    assert res["overall"]["n_instances"] == len(res["records"])
    assert set(res["by_bucket"]) >= {"short", "medium", "long"}


def test_batching_does_not_change_a_score():
    instances = _fixture_instances()[:4]
    model = StubCausalModel(vocab_size=256, seed=5)
    one = evaluate_code_recall(model, instances, batch_size=1)["records"]
    many = evaluate_code_recall(model, instances, batch_size=8)["records"]
    for a, b in zip(one, many):
        assert a["ce_nats"] == pytest.approx(b["ce_nats"], rel=1e-12)
        assert a["rank_top1"] == b["rank_top1"]


def test_no_instances_raises_rather_than_reporting_a_perfect_model():
    with pytest.raises(ValueError, match="nothing scored"):
        evaluate_code_recall(StubCausalModel(), [], batch_size=2)


def test_a_repo_with_no_resolvable_import_yields_no_instances():
    files = [{"path": "a.ts", "text": "export function alpha(): void {}\n"},
             {"path": "b.ts", "text": 'import { alpha } from "react";\nalpha();\n'}]
    assert build_recall_instances(files, make_byte_encoder(), np.random.default_rng(0)) == []
