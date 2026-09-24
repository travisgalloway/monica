"""Tests for pre-tokenization filtering, PII scrubbing, Prettier formatting,
and AST decontamination run (#419).

Exercises the decontamination loader, zero-overlap verifier, manifest generation,
and Prettier formatting without requiring full Datatrove cluster infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.datatrove_pipeline import (
    generate_pretokenization_manifest,
    load_eval_decontaminator,
    verify_zero_decontamination_overlap,
)
from src.data.dedup import Decontaminator
from src.lsp.prettier import format_source, resolve_prettier


def test_load_eval_decontaminator_from_blocklist():
    """Verify loading pre-built decontamination blocklist for humaneval_ts and ts_error_injection."""
    decon, path, sets = load_eval_decontaminator()
    assert decon is not None
    assert path is not None
    assert path.exists()
    assert len(decon.ngrams) > 1000
    assert "eval_sets/humaneval_ts" in sets
    assert "eval_sets/ts_error_injection" in sets


def test_load_eval_decontaminator_dynamic_from_eval_sets(tmp_path):
    """Verify dynamic construction of decontaminator from eval sets directly when blocklist path is absent."""
    fake_eval_dir = tmp_path / "eval_sets" / "fake_ts"
    fake_eval_dir.mkdir(parents=True)
    eval_file = fake_eval_dir / "eval.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "prompt": "export function evaluateComplexTensorGraph(rootNode: GraphNode): EvaluationResult { return rootNode.eval(); }",
            "gold_completion": "const computedResult = evaluateComplexTensorGraph(testNode); assert(computedResult.isValid);",
        }) + "\n")

    decon, path, sets = load_eval_decontaminator(
        blocklist_path=tmp_path / "nonexistent.txt",
        eval_sets=[str(fake_eval_dir)],
    )
    assert path is None
    assert decon is not None
    assert len(decon.ngrams) > 0
    # Overlapping document must be flagged as contaminated
    assert decon.contaminated("export function evaluateComplexTensorGraph(rootNode: GraphNode): EvaluationResult { return rootNode.eval(); }")
    # Clean doc must pass
    assert not decon.contaminated("const completelyUnrelatedValue = 42 * 1337;")


def test_verify_zero_decontamination_overlap(tmp_path):
    """Verify zero overlap checker flags contaminated lines and confirms zero-leak corpus (#419)."""
    decon = Decontaminator.from_texts([
        "export function binarySearchTreeInsert(rootNode: TreeNode, val: number): TreeNode { return rootNode; }",
    ], ngram_sizes=(13, 7))

    clean_file = tmp_path / "clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "const simpleValue = 12345;\n"}) + "\n")
        f.write(json.dumps({"text": "export interface Config { timeoutMs: number; }\n"}) + "\n")

    overlap_count, samples = verify_zero_decontamination_overlap(clean_file, decon)
    assert overlap_count == 0
    assert len(samples) == 0

    leaked_file = tmp_path / "leaked.jsonl"
    with open(leaked_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "const simpleValue = 12345;\n"}) + "\n")
        f.write(json.dumps({"text": "export function binarySearchTreeInsert(rootNode: TreeNode, val: number): TreeNode { return rootNode; }\n"}) + "\n")

    overlap_count, samples = verify_zero_decontamination_overlap(leaked_file, decon)
    assert overlap_count == 1
    assert len(samples) == 1


def test_generate_pretokenization_manifest_schema(tmp_path):
    """Verify manifest.json contains all required clean-rate, deduplication, and decontamination metrics (#419)."""
    out_dir = tmp_path / "manifest_test"
    out_dir.mkdir(parents=True)
    cleaned_file = out_dir / "cleaned.jsonl"
    with open(cleaned_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "const x = 1;\n", "id": "1", "metadata": {"lang": "typescript", "source": "stack-v2"}}) + "\n")
        f.write(json.dumps({"text": "A brief clean article about computers.\n", "id": "2", "metadata": {"lang": "en", "source": "essential-web"}}) + "\n")

    manifest = generate_pretokenization_manifest(
        str(out_dir),
        n_source=5,
        cleaned_jsonl_name="cleaned.jsonl",
        dedup_stats={
            "applied": True,
            "method": "minhash_lsh",
            "n_before_dedup": 4,
            "n_after_dedup": 2,
            "dropped_duplicates": 2,
            "drop_rate": 0.5,
            "repos_before_dedup": 3,
            "repos_after_dedup": 2,
            "duplicate_repos_removed": 1,
            "duplicate_files_removed": 2,
        },
        decontam_stats={
            "applied": True,
            "eval_sets": ["eval_sets/humaneval_ts", "eval_sets/ts_error_injection"],
            "blocklist": "eval_sets/decontam/blocklist.txt",
            "ngram_sizes": [13, 7],
            "overlap_count": 0,
            "zero_overlap_verified": True,
        },
        scrub_stats={"applied": True, "docs_scrubbed": 1, "secrets_scrubbed": 2},
        prettier_stats={"applied": True, "formatted_docs": 1},
    )

    # Acceptance Criteria validations
    assert manifest["clean_rate"]["n_source"] == 5
    assert manifest["clean_rate"]["n_cleaned"] == 2
    assert manifest["clean_rate"]["n_dropped"] == 3
    assert manifest["clean_rate"]["drop_rate"] == 0.6

    assert manifest["deduplication"]["applied"] is True
    assert manifest["deduplication"]["dropped_duplicates"] == 2
    assert manifest["deduplication"]["duplicate_repos_removed"] == 1

    assert manifest["decontamination"]["zero_overlap_verified"] is True
    assert manifest["decontamination"]["overlap_count"] == 0
    assert "eval_sets/humaneval_ts" in manifest["decontamination"]["eval_sets"]
    assert "eval_sets/ts_error_injection" in manifest["decontamination"]["eval_sets"]

    # Manifest file written to disk
    manifest_disk = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_disk["document_count"] == 2
    assert manifest_disk["clean_rate"]["drop_rate"] == 0.6


def test_prettier_ast_normalization_minimizes_token_entropy():
    """Verify Prettier formats irregular AST / syntax into canonical form (#419)."""
    argv = resolve_prettier()
    if argv is None:
        pytest.skip("Prettier not installed on host")

    raw_code = "function    calculateTotal(  price:number,  tax:number):number{return price+(price*tax);}"
    formatted = format_source(raw_code, argv)
    assert formatted == "function calculateTotal(price: number, tax: number): number {\n  return price + price * tax;\n}\n"


def test_build_corpus_cli_pretokenize_help():
    """Verify build_corpus.py --help exposes --pretokenize and --from-raw (#419)."""
    import subprocess
    import sys
    res = subprocess.run([sys.executable, "scripts/build_corpus.py", "--help"],
                         capture_output=True, text=True)
    assert res.returncode == 0
    assert "--pretokenize" in res.stdout
    assert "--from-raw" in res.stdout
    assert "--prettier" in res.stdout
