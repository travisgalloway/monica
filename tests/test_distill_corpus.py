"""Distillation-corpus driver (#92): the clean -> Qwen3-tokenize -> uint32-pack orchestrator
that produces the frozen `poc-distill/corpus/` artifact the teacher precompute (#94) and every
student sweep (#98) consume.

Pure numpy + pyarrow (no backend); fully offline via the byte-fallback tokenizer. Skips cleanly
where pyarrow is absent, mirroring test_corpus.py.
"""

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from src.data.distill_corpus import build_distill_corpus, tokenized_subdir
from src.data.corpus import ingest_dummy
from src.data.pack import packing_dtype_for
from src.data.shard import open_shard, read_manifest


def test_tokenized_subdir():
    # The exact dir name the student manifests reference (config/manifests/student-1b-*.yaml).
    assert tokenized_subdir("qwen3", 8192) == "qwen3-8k"
    assert tokenized_subdir("olmo", 1024) == "olmo-1k"


def test_qwen25_packs_uint32():
    # The Qwen2.5 vocab (151,646) exceeds the uint16 ceiling -> uint32 packing (#90). Locks the
    # uint32 claim without needing the HF tokenizer offline.
    assert packing_dtype_for(151646) == np.dtype(np.uint32)
    assert packing_dtype_for(50280) == np.dtype(np.uint16)        # OLMo POC path stays uint16


def test_build_distill_corpus_byte_fallback(tmp_path):
    # End-to-end on synthetic docs, byte fallback (offline). The label is still "qwen3" — the
    # manifest records the short name, not the (absent) HF repo path.
    manifest = build_distill_corpus(ingest_dummy(400, seed=1), tmp_path,
                                    tokenizer="qwen3", byte_fallback=True, seq_len=1024)

    # Stage 1: cleaned Parquet shards exist.
    cleaned = list((tmp_path / "corpus" / "cleaned").glob("*.parquet"))
    assert cleaned, "no cleaned text shards written"

    # Stage 2: tokenized shards + sidecars + per-dir manifest under the tokenized prefix.
    tok_dir = tmp_path / "corpus" / "tokenized" / tokenized_subdir("qwen3", 1024)
    assert (tok_dir / "manifest.json").exists()
    assert list(tok_dir.glob("part-*.bin")) and list(tok_dir.glob("part-*.bounds"))

    # Pack manifest records the acceptance fields: tokenizer (short), seq_len, dtype, token count.
    pack_manifest = read_manifest(tok_dir)
    assert pack_manifest["tokenizer"] == "qwen3"
    assert pack_manifest["seq_len"] == 1024
    assert pack_manifest["dtype"] == "uint16"            # byte fallback vocab 256 -> uint16
    assert pack_manifest["n_tokens"] > 0

    # Document boundaries present so a downstream reset/atomic-pack consumer can honor them.
    name = pack_manifest["shards"][0]["name"]
    toks, bnds = open_shard(tok_dir, name)
    assert len(toks) == pack_manifest["shards"][0]["n_tokens"]
    assert int(bnds.sum()) == pack_manifest["n_documents"]   # one flag per emitted doc-start

    # Corpus-level manifest: the two-stage summary, pointing at the tokenized prefix.
    assert manifest["tokenizer"] == "qwen3" and manifest["byte_fallback"] is True
    assert manifest["seq_len"] == 1024 and manifest["dtype"] == "uint16"
    assert manifest["n_tokens"] == pack_manifest["n_tokens"]
    assert manifest["n_documents"] == pack_manifest["n_documents"]
    assert manifest["n_cleaned_shards"] == len(cleaned)
    assert manifest["tokenized_dir"].endswith(tokenized_subdir("qwen3", 1024))


def test_seq_len_8k_subdir_matches_manifests(tmp_path):
    # At the real seq_len the tokenized dir is exactly the path the student manifests name.
    manifest = build_distill_corpus(ingest_dummy(2000, seed=2), tmp_path,
                                    tokenizer="qwen3", byte_fallback=True, seq_len=8192)
    assert manifest["tokenized_dir"].endswith("corpus/tokenized/qwen3-8k")


def test_cli_smoke_byte_fallback(tmp_path):
    # The `python -m src.data.distill_corpus` entrypoint runs offline end to end.
    out_root = tmp_path / "pd"
    r = subprocess.run(
        [sys.executable, "-m", "src.data.distill_corpus", "--source", "dummy",
         "--max-docs", "300", "--byte-fallback", "--tokenizer", "qwen3",
         "--seq-len", "1024", "--out-root", str(out_root)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "distill corpus:" in r.stdout
    assert (out_root / "corpus" / "manifest.json").exists()
    assert (out_root / "corpus" / "tokenized" / "qwen3-1k" / "manifest.json").exists()
