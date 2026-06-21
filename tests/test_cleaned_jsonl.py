"""Cleaned-text JSONL bridge (#80): the datatrove clean pass writes JSON-lines shards, and the
existing tokenize/pack stages must consume them (datatrove -> trainer shards). Portable: gzip +
json + numpy only, no datatrove, so it runs in the main env.
"""

import gzip
import json

import numpy as np
import pytest

from src.data.corpus import has_jsonl_shards, iter_jsonl_texts


def _write_jsonl_gz(path, texts):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"text": t, "id": str(i), "metadata": {"source": "fineweb-edu"}}) + "\n")


def test_iter_jsonl_texts_reads_gzip_shards(tmp_path):
    _write_jsonl_gz(tmp_path / "00000.jsonl.gz", ["alpha beta", "gamma delta"])
    _write_jsonl_gz(tmp_path / "00001.jsonl.gz", ["epsilon zeta"])
    assert has_jsonl_shards(tmp_path) is True
    assert list(iter_jsonl_texts(tmp_path)) == ["alpha beta", "gamma delta", "epsilon zeta"]


def test_has_jsonl_shards_false_for_parquet_dir(tmp_path):
    pytest.importorskip("pyarrow")
    from src.data.corpus import Record, write_shards
    write_shards([Record(text="x y z", source="s")], str(tmp_path))
    assert has_jsonl_shards(tmp_path) is False         # parquet present -> not the jsonl path


def test_jsonl_feeds_tokenize_pack_uint32(tmp_path):
    # The datatrove-output format tokenizes + packs through the existing stage to uint32 shards.
    from src.data.shard import open_shard, pack_sequences
    from src.data.tokenize import ByteTokenizer, tokenize_docs

    _write_jsonl_gz(tmp_path / "00000.jsonl.gz", ["the quick brown fox " * 20] * 8)
    tok = ByteTokenizer()
    docs = tokenize_docs(iter_jsonl_texts(tmp_path), tok)
    out = tmp_path / "tok"
    manifest = pack_sequences(docs, out, seq_len=64, tokenizer="qwen25", dtype=np.uint32)
    assert manifest["dtype"] == "uint32" and manifest["n_tokens"] > 0
    toks, bnds = open_shard(out, manifest["shards"][0]["name"])
    assert toks.dtype == np.uint32
