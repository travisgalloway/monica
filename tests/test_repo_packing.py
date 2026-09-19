"""Tests for repository DAG context packing with <|repo_name|> and <|file_sep|> (#359).

Verifies:
* `monica-tokenize train` produces tokenizer with `<|repo_name|>` and `<|file_sep|>` special tokens
* `monica-tokenize pack` accepts repository manifests and packs files contiguously into large windows
* Delimiters `<|repo_name|>` and `<|file_sep|>` appear in the packed token stream
* Boundary resets (.bounds = 1) occur only at repository start, suppressing resets between dependent files
"""

import json
import os
import subprocess
from pathlib import Path
import pytest

from src.data.shard import open_shard, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "swift" / "Fixtures" / "parity-corpus.jsonl"


def _find_monica_tokenize() -> Path | None:
    env = os.environ.get("MONICA_TOKENIZE")
    if env and Path(env).exists():
        return Path(env)
    swift_dir = REPO_ROOT / "swift"
    for mode in ("release", "debug"):
        cand = swift_dir / ".build" / mode / "monica-tokenize"
        if cand.exists():
            return cand
    return None


@pytest.fixture(scope="module")
def monica_tokenize() -> Path:
    binary = _find_monica_tokenize()
    if binary is None:
        pytest.skip("monica-tokenize not built (cd swift && swift build)")
    return binary


@pytest.fixture(scope="module")
def tokenizer_json(monica_tokenize, tmp_path_factory) -> Path:
    if not FIXTURE.exists():
        pytest.skip(f"fixture corpus missing: {FIXTURE}")
    out = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    cmd = [
        str(monica_tokenize),
        "train",
        "--in",
        str(FIXTURE),
        "--out",
        str(out),
        "--vocab-size",
        "2000",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return out


def test_tokenizer_contains_repo_metadata_tokens(tokenizer_json):
    data = json.loads(tokenizer_json.read_text())
    specials = data.get("special_tokens", [])
    assert "<|repo_name|>" in specials
    assert "<|file_sep|>" in specials
    assert specials.index("<|repo_name|>") == 6
    assert specials.index("<|file_sep|>") == 7


def test_monica_tokenize_pack_repo_manifest(monica_tokenize, tokenizer_json, tmp_path):
    repo_manifest = tmp_path / "repo_manifest.jsonl"
    repo_manifest.write_text(
        json.dumps({
            "repo": "acme/auth-service",
            "files": [
                {
                    "path": "src/types.ts",
                    "content": "export interface Token { id: string; expires: number; }\n",
                },
                {
                    "path": "src/auth.ts",
                    "content": (
                        'import { Token } from "./types";\n'
                        "export function verifyToken(t: Token): boolean { return t.expires > Date.now(); }\n"
                    ),
                },
                {
                    "path": "src/index.ts",
                    "content": (
                        'import { verifyToken } from "./auth";\n'
                        "export function handleRequest() { return verifyToken({ id: '1', expires: 9999 }); }\n"
                    ),
                },
            ],
        })
        + "\n"
    )

    out_dir = tmp_path / "packed_shards"
    # Pack into 1024 seq_len window for test
    cmd = [
        str(monica_tokenize),
        "pack",
        "--tokenizer",
        str(tokenizer_json),
        "--in",
        str(repo_manifest),
        "--out",
        str(out_dir),
        "--seq-len",
        "64",
        "--shard-size-mb",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"pack failed: {res.stderr}\nstdout: {res.stdout}"

    manifest = read_manifest(out_dir)
    assert manifest["n_sequences"] > 0
    assert manifest["seq_len"] == 64

    tokens, bounds = open_shard(out_dir, "part-00000")
    assert len(tokens) == len(bounds)

    # Repository starts with boundary 1
    assert bounds[0] == 1

    # Delimiter tokens: <|repo_name|> is id 6, <|file_sep|> is id 7
    repo_name_id = 6
    file_sep_id = 7
    assert tokens[0] == repo_name_id

    # <|file_sep|> must appear for each file in the repository (3 files)
    file_sep_positions = [i for i, t in enumerate(tokens) if t == file_sep_id]
    assert len(file_sep_positions) >= 3

    # Suppress document boundary state resets between dependent files in the same repository:
    # bounds must be 0 for all internal tokens of the repository, including file separators!
    for pos in file_sep_positions:
        assert bounds[pos] == 0, f"boundary at file separator pos {pos} was not suppressed"


def test_monica_tokenize_pack_large_32k_window(monica_tokenize, tokenizer_json, tmp_path):
    repo_manifest = tmp_path / "repo.json"
    repo_manifest.write_text(
        json.dumps({
            "repo": "deep/project",
            "files": [
                {"path": "lib/a.py", "content": "def a(): return 'a'\n"},
                {"path": "lib/b.py", "content": "from .a import a\ndef b(): return a() + 'b'\n"},
            ],
        })
    )

    out_dir = tmp_path / "packed_32k"
    cmd = [
        str(monica_tokenize),
        "pack",
        "--tokenizer",
        str(tokenizer_json),
        "--repo-manifest",
        str(repo_manifest),
        "--out",
        str(out_dir),
        "--seq-len",
        "32768",
        "--shard-size-mb",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    manifest = read_manifest(out_dir)
    assert manifest["seq_len"] == 32768
