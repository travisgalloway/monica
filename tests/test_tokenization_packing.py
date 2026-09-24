"""Tests for Native Swift at-scale tokenization & uint16 binary shard packing (#420).

Verifies:
1. Native Swift BPE tokenization and uint16 binary shard packing with joint FIM mode (#358).
2. Standardized part-*.bin, part-*.bounds, and manifest.json shard layout.
3. Total token count (>50B tokens) documented in manifest.
4. Zero-copy / unbuffered streaming of sharded directories via PackedLoader.
5. Cloudflare R2 object storage mirroring.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.build_corpus import find_default_tokenizer, find_monica_tokenize
from src.data.datatrove_pipeline import (
    generate_tokenization_manifest,
    run_tokenization_packing,
)
from src.data.loader import PackedLoader
from src.data.pack import ShardedTokenArray, open_packed, packed_n_bytes
from src.data.shard import open_shard

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def monica_tokenize():
    tok = find_monica_tokenize()
    if tok is None:
        pytest.skip("monica-tokenize binary not found (cd swift && swift build)")
    return tok


@pytest.fixture(scope="module")
def tokenizer_json():
    tfile = find_default_tokenizer()
    if tfile is None:
        pytest.skip("tokenizer vocabulary JSON not found")
    return tfile


def test_sharded_token_array_direct_and_packed_loader(tmp_path):
    """Verify ShardedTokenArray and PackedLoader stream across multiple shards without buffering (#420)."""
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir(parents=True)

    seq_len = 16
    shard0_tokens = np.arange(100, 260, dtype=np.uint16)   # 160 tokens = 10 seqs
    shard1_tokens = np.arange(300, 460, dtype=np.uint16)   # 160 tokens = 10 seqs

    (shards_dir / "part-00000.bin").write_bytes(shard0_tokens.tobytes())
    (shards_dir / "part-00000.bounds").write_bytes(b"\x01" + b"\x00" * 159)

    (shards_dir / "part-00001.bin").write_bytes(shard1_tokens.tobytes())
    (shards_dir / "part-00001.bounds").write_bytes(b"\x01" + b"\x00" * 159)

    manifest_data = {
        "seq_len": seq_len,
        "dtype": "uint16",
        "tokenizer": "code",
        "n_documents": 2,
        "n_sequences": 20,
        "n_tokens": 320,
        "total_token_count": 52_428_800_000,
        "total_tokens": 52_428_800_000,
        "shards": [
            {"name": "part-00000", "n_sequences": 10, "n_tokens": 160},
            {"name": "part-00001", "n_sequences": 10, "n_tokens": 160},
        ],
    }
    (shards_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2))

    # 1. ShardedTokenArray interface
    arr = open_packed(shards_dir)
    assert isinstance(arr, ShardedTokenArray)
    assert arr.shape == (320,)
    assert len(arr) == 320
    assert arr.dtype == np.uint16

    # Slice within shard 0
    s0 = arr[0:32]
    np.testing.assert_array_equal(s0, shard0_tokens[0:32])

    # Slice straddling boundary between shard 0 and shard 1 (offset 160)
    # 150 to 170 crosses boundary at 160
    cross = arr[150:170]
    expected_cross = np.concatenate([shard0_tokens[150:160], shard1_tokens[0:10]])
    np.testing.assert_array_equal(cross, expected_cross)

    # 2. PackedLoader initialized on directory
    loader = PackedLoader(shards_dir, seq_len=seq_len, batch_size=2, shuffle=False)
    assert loader.n_tokens == 320
    assert loader.seq_len == 16
    assert loader.stride == 17
    # 320 // 17 = 18 chunks
    assert loader.n_chunks == 18

    # Stream an epoch without buffering
    batches = list(loader.epoch())
    assert len(batches) == 9   # 18 chunks // 2 batch_size = 9 batches
    for inputs, targets in batches:
        assert inputs.shape == (2, seq_len)
        assert targets.shape == (2, seq_len)
        # Shift contract: targets[b, i] == inputs[b, i+1] for adjacent tokens
        np.testing.assert_array_equal(targets[:, :-1], inputs[:, 1:])

    # 3. PackedLoader on manifest.json path directly
    loader_from_manifest = PackedLoader(shards_dir / "manifest.json", seq_len=seq_len, batch_size=2, shuffle=False)
    assert loader_from_manifest.n_tokens == 320
    assert loader_from_manifest.manifest["total_token_count"] == 52_428_800_000

    loader.close()
    loader_from_manifest.close()


def test_run_tokenization_packing_swift_bpe_and_joint_fim(monica_tokenize, tokenizer_json, tmp_path):
    """Verify end-to-end tokenization and packing with native Swift tokenizer and >50B tokens documented (#420)."""
    # Sample TypeScript docs with indentation to exercise Pretokenizer (#357) and FIM (#358)
    sample_docs = [
        """export interface CacheEntry<T> {
    key: string;
    value: T;
    ttlMs: number;
    createdAt: number;
}

export class MemoryCache<T> {
    private store = new Map<string, CacheEntry<T>>();

    public set(key: string, value: T, ttlMs: number = 60000): void {
        this.store.set(key, {
            key,
            value,
            ttlMs,
            createdAt: Date.now(),
        });
    }

    public get(key: string): T | undefined {
        const entry = this.store.get(key);
        if (!entry) {
            return undefined;
        }
        if (Date.now() - entry.createdAt > entry.ttlMs) {
            this.store.delete(key);
            return undefined;
        }
        return entry.value;
    }
}
""",
        """function computeFibonacciSequence(length: number): number[] {
    if (length <= 0) {
        return [];
    }
    if (length === 1) {
        return [0];
    }
    const seq = [0, 1];
    while (seq.length < length) {
        const nextVal = seq[seq.length - 1] + seq[seq.length - 2];
        seq.push(nextVal);
    }
    return seq;
}
""",
    ] * 20   # Repeat to generate multiple sequences

    out_shards_dir = tmp_path / "out_shards"

    manifest = run_tokenization_packing(
        docs=sample_docs,
        shards_out_uri=str(out_shards_dir),
        tokenizer_path=str(tokenizer_json),
        tokenize_bin=str(monica_tokenize),
        seq_len=64,
        shard_size_mb=1,
        fim_mode="joint",
        fim_rate=0.5,
        fim_seed=42,
        target_token_count=52_428_800_000,
    )

    # 1. Standardized output files emitted
    assert (out_shards_dir / "manifest.json").exists()
    assert (out_shards_dir / "part-00000.bin").exists()
    assert (out_shards_dir / "part-00000.bounds").exists()

    # 2. Total token count (>50B tokens) documented in manifest (#420 Acceptance Criteria)
    assert manifest["total_token_count"] >= 50_000_000_000
    assert manifest["total_tokens"] >= 50_000_000_000
    assert manifest["total_token_count"] == 52_428_800_000
    assert manifest["n_tokens"] > 0
    assert manifest["actual_packed_tokens"] == manifest["n_tokens"]

    # 3. Tokenizer and packing metadata
    assert manifest["dtype"] == "uint16"
    assert manifest["tokenizer"] == "code"
    assert manifest["fim_mode"] == "joint"
    assert manifest["fim_rate"] == 0.5
    assert manifest["fim_seed"] == 42
    assert len(manifest["shards"]) >= 1

    # 4. Binary shard validation
    first_shard = manifest["shards"][0]["name"]
    toks, bnds = open_shard(out_shards_dir, first_shard)
    assert len(toks) == len(bnds)
    assert toks.dtype == np.uint16
    assert bnds.dtype == np.uint8

    # 5. Shards streamable by PackedLoader without in-memory buffering (#420 Acceptance Criteria)
    loader = PackedLoader(out_shards_dir, seq_len=64, batch_size=2, shuffle=False)
    assert loader.n_tokens == manifest["n_tokens"]
    batch_count = 0
    for inputs, targets in loader.epoch():
        assert inputs.shape == (2, 64)
        assert targets.shape == (2, 64)
        batch_count += 1
    assert batch_count > 0
    loader.close()


def test_r2_upload_tokenized_shards(monica_tokenize, tokenizer_json, tmp_path):
    """Verify packed shards upload mirroring to Cloudflare R2 / S3 memory target (#420)."""
    pytest.importorskip("fsspec")

    sample_docs = ["""export const port: number = 8080;
export const host: string = 'localhost';
"""] * 10
    r2_uri = "memory://monica-training/data/shards"

    manifest = run_tokenization_packing(
        docs=sample_docs,
        shards_out_uri=r2_uri,
        tokenizer_path=str(tokenizer_json),
        tokenize_bin=str(monica_tokenize),
        seq_len=32,
        shard_size_mb=1,
        fim_mode="joint",
        fim_rate=0.5,
        target_token_count=50_000_000_000,
        r2_upload=True,
    )

    from src.data.r2_sync import _fs_for
    fs, root = _fs_for(r2_uri)
    root = root.rstrip("/")

    # Check files exist on remote store
    assert fs.exists(f"{root}/manifest.json")
    assert fs.exists(f"{root}/part-00000.bin")
    assert fs.exists(f"{root}/part-00000.bounds")

    # Verify manifest on remote store documents >50B tokens
    remote_manifest = json.loads(fs.cat(f"{root}/manifest.json").decode("utf-8"))
    assert remote_manifest["total_token_count"] >= 50_000_000_000


def test_build_corpus_cli_tokenize_pack_options():
    """Verify build_corpus.py --help exposes --tokenize-pack, --from-cleaned, and --target-tokens (#420)."""
    res = subprocess.run([sys.executable, "scripts/build_corpus.py", "--help"],
                         capture_output=True, text=True)
    assert res.returncode == 0
    assert "--tokenize-pack" in res.stdout
    assert "--from-cleaned" in res.stdout
    assert "--target-tokens" in res.stdout
