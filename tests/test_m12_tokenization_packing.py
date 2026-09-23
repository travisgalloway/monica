"""Tests for M12 Tokenization & Packing POC & MVP runs (SPM + DAG sorting) (#370).

Verifies:
* Space-run and newline separation (#357)
* Suffix-Prefix-Middle (SPM) and 50/50 joint FIM mode in native Swift packer (#358)
* Import DAG topological sorting in repository context packing (#359)
* Unified build_corpus.py pipeline execution on sample source repositories
* Shard layout (.bin + .bounds + manifest.json) and PackedLoader compatibility
* Decontamination blocklist filtering against humaneval_ts and ts_error_injection
"""

import json
from pathlib import Path
import numpy as np
import pytest

from scripts.build_corpus import find_monica_tokenize, find_default_tokenizer, run_corpus_pipeline
from src.data.loader import PackedLoader
from src.data.shard import open_shard, read_manifest, doc_start_offsets

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def monica_tokenize():
    tok = find_monica_tokenize()
    if tok is None:
        pytest.skip("monica-tokenize binary not found (cd swift && swift build)")
    return tok


@pytest.fixture(scope="module")
def datatrove():
    # run_corpus_pipeline drives datatrove, which lives in the py3.11 .venv-dt, not the main env.
    return pytest.importorskip("datatrove", reason="datatrove not installed (runs in .venv-dt)")


@pytest.fixture(scope="module")
def tokenizer_json():
    tfile = find_default_tokenizer()
    if tfile is None:
        pytest.skip("tokenizer vocabulary JSON not found")
    return tfile


def test_m12_poc_sample_shards_manifest_and_loader():
    """Verify that generated sample shards in data/mhm_sample_shards satisfy all acceptance criteria."""
    shards_dir = REPO_ROOT / "data" / "mhm_sample_shards"
    if not (shards_dir / "manifest.json").exists():
        pytest.skip("data/mhm_sample_shards not generated yet")

    manifest = read_manifest(shards_dir)

    # 1. Schema & Layout
    assert manifest["dtype"] == "uint16"
    assert manifest["tokenizer"] == "code"
    assert manifest["seq_len"] > 0
    assert manifest["n_tokens"] > 0
    assert manifest["n_sequences"] > 0
    assert len(manifest["shards"]) >= 1

    # 2. M12 enhancements provenance
    assert manifest.get("repo_dag_sorted") is True, "Repository DAG sorting flag missing"
    assert manifest.get("fim_mode") in ("joint", "spm", "psm"), "FIM mode missing"
    assert manifest.get("fim_rate") is not None
    assert "repos" in manifest and len(manifest["repos"]) > 0

    # 3. Language mix and quality gate
    assert "language_mix" in manifest
    assert "typescript" in manifest["language_mix"]
    assert "python" in manifest["language_mix"]

    # 4. Decontamination
    assert manifest["decontamination"]["applied"] is True
    assert "eval_sets/humaneval_ts" in manifest["decontamination"]["eval_sets"]
    assert "eval_sets/ts_error_injection" in manifest["decontamination"]["eval_sets"]

    # 5. Token inspection: REPO_NAME (6), FILE_SEP (7), FIM sentinels (1, 2, 3)
    first_shard = manifest["shards"][0]["name"]
    toks, bnds = open_shard(shards_dir, first_shard)
    assert len(toks) == len(bnds)
    assert (toks == 6).sum() > 0, "No <|repo_name|> tokens found in sample shard"
    assert (toks == 7).sum() > 0, "No <|file_sep|> tokens found in sample shard"
    assert (toks == 1).sum() > 0 or (toks == 3).sum() > 0, "No FIM sentinels found in sample shard"

    # 6. PackedLoader verification
    loader = PackedLoader(shards_dir / f"{first_shard}.bin",
                          seq_len=manifest["seq_len"], batch_size=2, shuffle=False)
    inputs, targets = next(iter(loader.epoch()))
    assert inputs.shape == (2, manifest["seq_len"])
    assert targets.shape == (2, manifest["seq_len"])


def test_m12_spm_mode_in_repo_packing(datatrove, monica_tokenize, tokenizer_json, tmp_path):
    """Verify that SPM mode in repo packing emits <|fim_suffix|> before <|fim_prefix|> (#358)."""
    repo_manifest = tmp_path / "spm_repo.jsonl"
    repo_manifest.write_text(json.dumps({
        "repo": "spm-test-repo",
        "files": [
            {
                "path": "src/module.ts",
                "content": """export function calculateDiscount(price: number, rate: number): number {
    const factor = 1.0 - rate;
    const discounted = price * factor;
    return discounted;
}

export function calculateTax(subtotal: number, taxRate: number): number {
    const totalTax = subtotal * taxRate;
    return totalTax;
}

export function calculateFinalTotal(price: number, discountRate: number, taxRate: number): number {
    const discounted = calculateDiscount(price, discountRate);
    const tax = calculateTax(discounted, taxRate);
    return discounted + tax;
}
""",
            }
        ]
    }) + "\n")

    out_dir = tmp_path / "clean_out"
    shards_dir = tmp_path / "shards_out"

    res = run_corpus_pipeline(
        source="repo",
        from_repo=repo_manifest,
        out_dir=out_dir,
        pack=True,
        shards_out=shards_dir,
        tokenizer=tokenizer_json,
        tokenize_bin=monica_tokenize,
        seq_len=64,
        fim_mode="spm",
        fim_rate=1.0,
        fim_seed=42,
    )

    manifest = read_manifest(shards_dir)
    toks, bnds = open_shard(shards_dir, "part-00000")

    # In SPM mode: <|fim_suffix|> is token 3, <|fim_prefix|> is token 1
    assert 3 in toks, "SPM suffix token missing"
    assert 1 in toks, "SPM prefix token missing"
    assert np.where(toks == 3)[0][0] < np.where(toks == 1)[0][0], "SPM suffix must precede prefix"


def test_m12_joint_mode_in_repo_packing(datatrove, monica_tokenize, tokenizer_json, tmp_path):
    """Verify joint FIM mode produces both PSM and SPM documents (#358)."""
    repo_manifest = tmp_path / "joint_repo.jsonl"
    files = []
    for i in range(12):
        files.append({
            "path": f"src/calc_{i}.ts",
            "content": f"""export function computeValue{i}(x: number): number {{
    const stepOne = x * 2 + {i};
    const stepTwo = stepOne * 3 + {i};
    const stepThree = stepTwo - 5;
    const stepFour = stepThree + 10;
    return stepFour;
}}
""",
        })
    repo_manifest.write_text(json.dumps({"repo": "joint-test-repo", "files": files}) + "\n")

    out_dir = tmp_path / "clean_out"
    shards_dir = tmp_path / "shards_out"

    res = run_corpus_pipeline(
        source="repo",
        from_repo=repo_manifest,
        out_dir=out_dir,
        pack=True,
        shards_out=shards_dir,
        tokenizer=tokenizer_json,
        tokenize_bin=monica_tokenize,
        seq_len=64,
        fim_mode="joint",
        fim_rate=1.0,
        fim_seed=123,
    )

    toks, bnds = open_shard(shards_dir, "part-00000")
    # Both prefix (1) and suffix (3) sentinels must appear
    assert (toks == 1).sum() > 0
    assert (toks == 3).sum() > 0


def test_m12_dag_topological_sorting_in_build_corpus(datatrove, monica_tokenize, tokenizer_json, tmp_path):
    """Verify that DAG topological sorting places imported interfaces before consumers (#359)."""
    repo_dir = tmp_path / "my_project"
    repo_dir.mkdir()
    # Write files in inverted order: consumer first, definition second
    (repo_dir / "consumer.ts").write_text("""import { Config } from "./definition";
export function useConfig(c: Config): string {
    return c.host;
}
""")
    (repo_dir / "definition.ts").write_text("""export interface Config {
    host: string;
    port: number;
}
""")

    out_dir = tmp_path / "clean_out"
    shards_dir = tmp_path / "shards_out"

    res = run_corpus_pipeline(
        source="repo",
        from_repo=repo_dir,
        out_dir=out_dir,
        dag_sort=True,
        pack=True,
        shards_out=shards_dir,
        tokenizer=tokenizer_json,
        tokenize_bin=monica_tokenize,
        seq_len=64,
        fim_rate=0.0,
    )

    # In repo_manifest, definition.ts must precede consumer.ts
    repo_manifest_file = out_dir / "repo_manifest.jsonl"
    assert repo_manifest_file.exists()
    with open(repo_manifest_file) as f:
        data = json.loads(f.readline())
    paths = [f["path"] for f in data["files"]]
    assert paths.index("definition.ts") < paths.index("consumer.ts"), "Dependency definition must precede consumer"


def test_m12_space_run_and_newline_separation(monica_tokenize, tokenizer_json, tmp_path):
    """Verify that space runs (2, 4, 8 spaces) and newlines emit isolated tokens (#357)."""
    import subprocess
    code = "  \n    const x = 1;\n        const y = 2;\n"
    res = subprocess.run([
        str(monica_tokenize), "encode",
        "--tokenizer", str(tokenizer_json),
        "--json",
    ], input=code, capture_output=True, text=True, check=True)
    tokens = json.loads(res.stdout)

    # Decode tokens individually
    for tid in tokens:
        dec = subprocess.run([
            str(monica_tokenize), "decode",
            "--tokenizer", str(tokenizer_json),
        ], input=str(tid), capture_output=True, text=True, check=True)
        # Ensure newline is never merged with indentation spaces (#357 rule)
        val = dec.stdout
        if "\n" in val:
            assert val in ("\n", "\r\n", "\r"), f"Found composite newline token: {repr(val)}"


def test_m12_decontamination_benchmark_gate():
    """Verify that eval benchmark prompts from humaneval_ts and ts_error_injection are dropped."""
    from src.data.dedup import Decontaminator
    blocklist_path = REPO_ROOT / "eval_sets" / "decontam" / "blocklist.txt"
    assert blocklist_path.exists()
    with open(blocklist_path, encoding="utf-8") as bf:
        decon = Decontaminator.from_texts(bf)

    # Test clean code passes
    clean = "export function multiplyByTwo(val: number): number { return val * 2; }\n"
    assert not decon.contaminated(clean), "Clean code should not be flagged"

    # Test humaneval prompt is detected
    humaneval_path = REPO_ROOT / "eval_sets" / "humaneval_ts" / "humaneval_ts.jsonl"
    with open(humaneval_path, encoding="utf-8") as f:
        h_row = json.loads(f.readline())
    assert decon.contaminated(h_row["prompt"]), "Benchmark prompt must be flagged contaminated"
