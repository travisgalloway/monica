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


# --------------------------------------------------------------------------------------- #
# Synthetic Multi-File Repository Fixtures & #345 Symbol Grounding / Blast Radius Tests
# --------------------------------------------------------------------------------------- #

from src.eval.code_suite import (
    REPO_GROUNDING_SYMBOL_KINDS,
    build_information_gain_instances,
    build_repo_grounding_instances,
    evaluate_blast_radius,
    evaluate_information_gain,
    evaluate_repository_symbol_recall,
)
from src.train.verifiers.repository_context import (
    BlastRadiusVerifier,
    CallSiteLocation,
    inprocess_blast_radius_diagnostics,
    score_blast_radius_prediction,
)

SYNTHETIC_REPO = [
    {
        "path": "src/types.ts",
        "text": (
            "export interface UserProfile {\n"
            "  id: string;\n"
            "  name: string;\n"
            "  role: string;\n"
            "}\n\n"
            "export interface DatabaseConfig {\n"
            "  host: string;\n"
            "  port: number;\n"
            "}\n\n"
            "export type AccountStatus = 'active' | 'suspended' | 'pending';\n"
        ),
    },
    {
        "path": "src/utils.ts",
        "text": (
            "import { DatabaseConfig } from './types';\n\n"
            "export function buildConnectionString(cfg: DatabaseConfig): string {\n"
            "  return cfg.host + ':' + cfg.port;\n"
            "}\n\n"
            "export function sanitizeIdentifier(raw: string): string {\n"
            "  return raw.trim().toLowerCase();\n"
            "}\n"
        ),
    },
    {
        "path": "src/services/user_service.ts",
        "text": (
            "import { UserProfile, AccountStatus } from '../types';\n"
            "import { sanitizeIdentifier } from '../utils';\n\n"
            "export function createUser(id: string, name: string): UserProfile {\n"
            "  return { id, name: sanitizeIdentifier(name), role: 'user' };\n"
            "}\n\n"
            "export function deactivateUser(user: UserProfile, status: AccountStatus): boolean {\n"
            "  return status === 'suspended';\n"
            "}\n"
        ),
    },
    {
        "path": "src/controllers/user_controller.ts",
        "text": (
            "import { UserProfile } from '../types';\n"
            "import { createUser, deactivateUser } from '../services/user_service';\n\n"
            "export function registerAccount(id: string, name: string): UserProfile {\n"
            "  const user: UserProfile = createUser(id, name);\n"
            "  return user;\n"
            "}\n\n"
            "export function suspendAccount(user: UserProfile): boolean {\n"
            "  return deactivateUser(user, 'suspended');\n"
            "}\n"
        ),
    },
    {
        "path": "src/distractors/analytics.ts",
        "text": (
            "export function recordMetric(event: string, value: number): void {\n"
            "  // Analytics tracking distractor module\n"
            "  const payload = { event, value, timestamp: 1234567890 };\n"
            "}\n"
        ),
    },
    {
        "path": "src/distractors/reporting.ts",
        "text": (
            "export function generateReportSummary(items: string[]): string {\n"
            "  // Distractor reporting pipeline\n"
            "  return items.join(', ');\n"
            "}\n"
        ),
    },
]


def test_synthetic_repo_grounding_extracts_all_three_symbol_kinds():
    """Synthetic multi-file fixture yields type_annotation, imported_symbol, and function_invocation."""
    encode = make_byte_encoder()
    instances = build_repo_grounding_instances(SYNTHETIC_REPO, encode, context_lengths=(128,))
    assert instances, "Grounding instances must not be empty"

    extracted_kinds = {inst.symbol_kind for inst in instances}
    for expected_kind in REPO_GROUNDING_SYMBOL_KINDS:
        assert expected_kind in extracted_kinds, f"Expected {expected_kind} in extracted kinds {extracted_kinds}"

    # Verify instance properties
    for inst in instances:
        assert inst.symbol in inst.candidates
        assert inst.candidates[inst.answer_index] == inst.symbol
        assert inst.definer != inst.consumer
        assert inst.prefix_tokens.size > 0


def test_grounding_instances_reach_8k_16k_32k_context_buckets():
    """Distractor expansion reaches the required 8k-32k token context horizons (#345)."""
    encode = make_byte_encoder()
    target_contexts = (8192, 16384, 32768)
    instances = build_repo_grounding_instances(
        SYNTHETIC_REPO,
        encode,
        context_lengths=target_contexts,
        max_instances=15,
    )
    assert instances

    buckets_seen = {inst.context_bucket for inst in instances}
    assert {"8k", "16k", "32k"}.issubset(buckets_seen)

    for inst in instances:
        if inst.context_bucket == "8k":
            assert inst.context_length >= 1000
        elif inst.context_bucket == "16k":
            assert inst.context_length > 8192
        elif inst.context_bucket == "32k":
            assert inst.context_length > 16384


def test_evaluate_repository_symbol_recall_records_and_ranking():
    """evaluate_repository_symbol_recall computes Top-1, MRR, and shared schema records."""
    from src.eval.code_suite import RECORD_FIELDS

    encode = make_byte_encoder()
    instances = build_repo_grounding_instances(SYNTHETIC_REPO, encode, context_lengths=(64,))[:6]
    model = StubCausalModel(vocab_size=256, seed=42)

    res = evaluate_repository_symbol_recall(model, instances, batch_size=2)
    assert "records" in res
    assert "top1_recall" in res
    assert "mrr" in res
    assert "by_symbol_kind" in res
    assert "by_context_bucket" in res

    for rec in res["records"]:
        assert tuple(sorted(rec.keys())) == tuple(sorted(RECORD_FIELDS))
        assert rec["suite"] == "repo_symbol_grounding"
        assert rec["rank_top1"] in (True, False)
        assert 0.0 <= rec["mrr"] <= 1.0


def test_symbol_recall_top1_when_model_prefers_target_tokens():
    """Grounding candidate ranking ranks answer first when model prefers target symbol tokens."""
    encode = make_byte_encoder()
    instances = build_repo_grounding_instances(SYNTHETIC_REPO, encode, context_lengths=(64,))
    # Filter to an instance with symbol 'createUser'
    user_instances = [inst for inst in instances if inst.symbol == "createUser"][:1]
    assert user_instances

    inst = user_instances[0]
    target_bytes = list(encode(inst.symbol))
    model = _PreferModel(prefer=target_bytes, vocab_size=256, hot=10.0)

    res = evaluate_repository_symbol_recall(model, user_instances)
    assert res["records"][0]["rank_top1"] is True
    assert res["records"][0]["mrr"] == pytest.approx(1.0)


def test_information_gain_probe_extracts_and_evaluates_delta_ce():
    """Information-gain probe measures Delta CE on implementation tokens with vs without interface context (#345)."""
    from src.eval.code_suite import RECORD_FIELDS

    encode = make_byte_encoder()
    instances = build_information_gain_instances(SYNTHETIC_REPO, encode)
    assert instances, "Should extract information-gain instances"

    for inst in instances:
        assert inst.with_context_tokens.size > inst.without_context_tokens.size
        assert inst.span_len > 0
        assert inst.interface_files

    model = StubCausalModel(vocab_size=256, seed=12)
    res = evaluate_information_gain(model, instances, batch_size=2)

    assert "mean_delta_ce" in res
    assert "records" in res
    for rec in res["records"]:
        assert tuple(sorted(rec.keys())) == tuple(sorted(RECORD_FIELDS))
        assert rec["suite"] == "repo_information_gain"
        assert "delta_ce" in rec["meta"]
        assert "ce_with" in rec["meta"]
        assert "ce_without" in rec["meta"]


# --------------------------------------------------------------------------------------- #
# BlastRadiusVerifier Unit & Acceptance Tests
# --------------------------------------------------------------------------------------- #

def test_blast_radius_signature_mutation_detection():
    """In-process compiler diagnostics accurately pinpoint call-sites when function signature changes (#345)."""
    repo_dict = {f["path"]: f["text"] for f in SYNTHETIC_REPO}
    # Mutate createUser in src/services/user_service.ts: add required parameter 'tenantId: string'
    mutated_service = (
        "import { UserProfile, AccountStatus } from '../types';\n"
        "import { sanitizeIdentifier } from '../utils';\n\n"
        "export function createUser(id: string, name: string, tenantId: string): UserProfile {\n"
        "  return { id, name: sanitizeIdentifier(name), role: 'user' };\n"
        "}\n\n"
        "export function deactivateUser(user: UserProfile, status: AccountStatus): boolean {\n"
        "  return status === 'suspended';\n"
        "}\n"
    )

    diags = inprocess_blast_radius_diagnostics(
        repo=repo_dict,
        mutated_file="src/services/user_service.ts",
        mutated_content=mutated_service,
    )

    assert len(diags) == 1
    diag = diags[0]
    assert diag.file == "src/controllers/user_controller.ts"
    assert diag.line == 5
    assert diag.code == "TS2554"
    assert "Expected 3 arguments, but got 2" in diag.message


def test_blast_radius_scoring_runs_under_150ms():
    """Scoring precision and recall against compiler diagnostic locations completes deterministically in <150ms (#345)."""
    predicted = [
        CallSiteLocation(file="src/controllers/user_controller.ts", line=5, code="TS2554"),
        CallSiteLocation(file="src/controllers/user_controller.ts", line=10, code="TS2554"),
    ]
    ground_truth = [
        CallSiteLocation(file="src/controllers/user_controller.ts", line=5, code="TS2554"),
    ]

    res = score_blast_radius_prediction(predicted, ground_truth)
    assert res["elapsed_ms"] < 150.0, f"Scoring took {res['elapsed_ms']}ms, expected < 150ms"
    assert res["precision"] == pytest.approx(0.5)
    assert res["recall"] == pytest.approx(1.0)
    assert res["f1"] == pytest.approx(2.0 * 0.5 * 1.0 / 1.5)
    assert res["file_precision"] == pytest.approx(1.0)
    assert res["file_recall"] == pytest.approx(1.0)


def test_blast_radius_verifier_full_flow():
    """BlastRadiusVerifier parses predictions from JSON and plain text, awarding 1.0 for exact matches."""
    repo_dict = {f["path"]: f["text"] for f in SYNTHETIC_REPO}
    mutated_service = (
        "import { UserProfile, AccountStatus } from '../types';\n"
        "import { sanitizeIdentifier } from '../utils';\n\n"
        "export function createUser(id: string, name: string, tenantId: string): UserProfile {\n"
        "  return { id, name: sanitizeIdentifier(name), role: 'user' };\n"
        "}\n\n"
        "export function deactivateUser(user: UserProfile, status: AccountStatus): boolean {\n"
        "  return status === 'suspended';\n"
        "}\n"
    )

    verifier = BlastRadiusVerifier()

    # 1. Exact JSON match -> reward 1.0
    json_pred = '[{"file": "src/controllers/user_controller.ts", "line": 5, "code": "TS2554"}]'
    res = verifier.evaluate(
        json_pred,
        repo=repo_dict,
        mutated_file="src/services/user_service.ts",
        mutated_content=mutated_service,
    )
    assert res["reward"] == 1.0
    assert res["is_clean"] is True
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0

    # 2. Plain text / markdown output -> reward 1.0
    text_pred = "- src/controllers/user_controller.ts:5: error TS2554: Expected 3 arguments"
    reward_text = verifier.reward(
        text_pred,
        repo=repo_dict,
        mutated_file="src/services/user_service.ts",
        mutated_content=mutated_service,
    )
    assert reward_text == 1.0

    # 3. Partial prediction (hallucinated extra site) -> reward < 1.0
    partial_pred = (
        "- src/controllers/user_controller.ts:5\n"
        "- src/controllers/user_controller.ts:99\n"
    )
    reward_partial = verifier.reward(
        partial_pred,
        repo=repo_dict,
        mutated_file="src/services/user_service.ts",
        mutated_content=mutated_service,
    )
    assert 0.0 < reward_partial < 1.0

    # 4. Telemetry
    telem = verifier.telemetry()
    assert telem["n_samples"] >= 3
    assert telem["n_clean"] >= 2
    assert telem["mean_elapsed_ms"] < 150.0


def test_blast_radius_anti_goodhart_and_degeneracy():
    """BlastRadiusVerifier penalizes empty, comment-only, and escape-hatch completions with -1.0."""
    verifier = BlastRadiusVerifier()
    ground_truth = [CallSiteLocation(file="src/a.ts", line=10)]

    # Empty
    assert verifier.reward("", reference=ground_truth) == -1.0
    # Whitespace
    assert verifier.reward("   \n\t  ", reference=ground_truth) == -1.0
    # Comment-only
    assert verifier.reward("// No impacted files found in repository\n", reference=ground_truth) == -1.0
    # Escape hatch (@ts-ignore)
    assert verifier.reward("@ts-ignore src/a.ts:10", reference=ground_truth) == -1.0
    # Escape hatch (eval injection)
    assert verifier.reward("eval('src/a.ts:10')", reference=ground_truth) == -1.0


def test_evaluate_blast_radius_suite():
    """evaluate_blast_radius in code_suite.py aggregates test cases across BLAST_RADIUS_BUCKETS."""
    cases = [
        {
            "id": "case-sig-1",
            "bucket": "signature_mutation",
            "completion": '[{"file": "src/controllers/user_controller.ts", "line": 5}]',
            "ground_truth": [{"file": "src/controllers/user_controller.ts", "line": 5}],
            "expected_clean": True,
        },
        {
            "id": "case-sig-2",
            "bucket": "required_param_added",
            "completion": "src/controllers/user_controller.ts:5",
            "ground_truth": [{"file": "src/controllers/user_controller.ts", "line": 5}],
            "expected_clean": True,
        },
    ]
    res = evaluate_blast_radius(cases)
    assert res["accuracy"] == 1.0
    assert res["n_cases"] == 2
    assert res["n_passed"] == 2
    assert "signature_mutation" in res["by_bucket"]
    assert "required_param_added" in res["by_bucket"]
