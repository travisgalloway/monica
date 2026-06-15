"""Corpus pipeline skeleton (#69): schema, local Parquet sharded IO, and the local
gate — the shards compose with the existing tokenize/pack/split stages.

Pure numpy + pyarrow (no backend). Skips cleanly where pyarrow is absent.
"""

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from src.data import corpus
from src.data.corpus import (Record, build_corpus, ingest_dummy, ingest_text_file,
                             iter_shard_texts, normalize, read_shards, write_shards)
from src.data.tokenize import ByteTokenizer, tokenize_texts
from src.data.pack import pack_ids, open_packed
from src.data.split import split_packed
from src.data.loader import PackedLoader


def test_record_schema():
    r = Record(text="hi", source="dummy", meta={"k": 1})
    assert r.to_dict() == {"text": "hi", "source": "dummy", "lang": "en",
                           "license": "unknown", "meta": {"k": 1}}
    assert corpus.RECORD_FIELDS == ("text", "source", "lang", "license", "meta")


def test_normalize_collapses_and_drops_empty():
    recs = [Record("a\n  b\tc", "s"), Record("   ", "s"), Record("x", "s")]
    out = list(normalize(recs))
    assert [r.text for r in out] == ["a b c", "x"]      # whitespace collapsed, empty dropped


def test_write_read_roundtrip_schema_and_meta(tmp_path):
    recs = [Record(f"doc {i}", "src", lang="en", license="MIT", meta={"i": i})
            for i in range(5)]
    shards = write_shards(recs, tmp_path / "cleaned", shard_size_mb=128)
    assert len(shards) == 1                              # large budget -> one shard
    back = list(read_shards(tmp_path / "cleaned"))
    assert [r.text for r in back] == [f"doc {i}" for i in range(5)]
    assert back[0].license == "MIT" and back[2].meta == {"i": 2}   # meta JSON round-trips


def test_few_large_shards_rolls_by_size(tmp_path):
    # ~1 KB docs with a tiny 0-MB-ish budget -> each doc rolls its own shard; a big
    # budget keeps them in one. Proves the few-large-shard rolling logic both ways.
    recs = [Record("x" * 1000, "src") for _ in range(4)]
    many = write_shards(recs, tmp_path / "many", shard_size_mb=0, prefix="part")
    assert len(many) == 4
    one = write_shards(recs, tmp_path / "one", shard_size_mb=64)
    assert len(one) == 1
    assert {r.text for r in read_shards(tmp_path / "many")} == {"x" * 1000}


def test_build_corpus_composes_with_tokenize_pack_split(tmp_path):
    """The local gate: dummy -> Parquet shards -> the EXISTING tokenize/pack/split."""
    shards = build_corpus(ingest_dummy(200, seed=0), tmp_path / "cleaned")
    assert shards and all(s.endswith(".parquet") for s in shards)

    # shards -> text -> tokenize (byte fallback) -> pack -> split, all offline.
    ids = tokenize_texts(iter_shard_texts(tmp_path / "cleaned"), ByteTokenizer())
    packed = tmp_path / "packed.bin"
    n = pack_ids(ids, packed)
    assert n > 0 and open_packed(packed).shape[0] == n

    split_packed(packed, tmp_path / "split", val_tokens=64)
    loader = PackedLoader(tmp_path / "split" / "train.bin", seq_len=16, batch_size=2,
                          shuffle=False, seed=0)
    inputs, targets = next(iter(loader.epoch()))
    assert inputs.shape == (2, 16) and targets.shape == (2, 16)


def test_cli_dummy_then_tokenize_consumes_shards(tmp_path):
    """End-to-end at the CLI: `python -m src.data.corpus` writes shards that
    `python -m src.data.tokenize --in <dir>` consumes directly."""
    cleaned = tmp_path / "cleaned"
    r1 = subprocess.run([sys.executable, "-m", "src.data.corpus", "--source", "dummy",
                         "--out", str(cleaned), "--max-docs", "100"],
                        capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    assert list(cleaned.glob("*.parquet"))

    packed = tmp_path / "packed.bin"
    r2 = subprocess.run([sys.executable, "-m", "src.data.tokenize", "--in", str(cleaned),
                         "--out", str(packed), "--byte-fallback"],
                        capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    assert packed.exists() and open_packed(packed).shape[0] > 0
