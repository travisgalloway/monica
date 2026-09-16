"""Corpus pipeline skeleton (#69): schema, local Parquet sharded IO, and the local
gate — the shards compose with the existing tokenize/pack/split stages.

Pure numpy + pyarrow (no backend). Skips cleanly where pyarrow is absent.
"""

import json
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


# --- MHM Training Corpus Sample Shards & Decontamination (#252) -----------------------

def test_decontamination_eval_sets():
    """Verify decontamination logic against both humaneval_ts and ts_error_injection eval sets."""
    import json
    from pathlib import Path
    from src.data.dedup import Decontaminator

    blocklist_path = Path(__file__).resolve().parents[1] / "eval_sets/decontam/blocklist.txt"
    assert blocklist_path.exists(), f"missing decontamination blocklist: {blocklist_path}"

    with open(blocklist_path, encoding="utf-8") as f:
        decon = Decontaminator.from_texts(f)

    # 1. Probe humaneval_ts: prompt from benchmark must be detected as contaminated
    humaneval_path = Path(__file__).resolve().parents[1] / "eval_sets/humaneval_ts/humaneval_ts.jsonl"
    assert humaneval_path.exists()
    with open(humaneval_path, encoding="utf-8") as f:
        h_row = json.loads(f.readline())
    assert decon.contaminated(h_row["prompt"]), "humaneval_ts benchmark prompt was not flagged contaminated"

    # 2. Probe ts_error_injection: prompt from benchmark must be detected as contaminated
    ts_error_path = Path(__file__).resolve().parents[1] / "eval_sets/ts_error_injection/eval.jsonl"
    assert ts_error_path.exists()
    with open(ts_error_path, encoding="utf-8") as f:
        ts_row = json.loads(f.readline())
    assert decon.contaminated(ts_row["prompt"]), "ts_error_injection prompt was not flagged contaminated"

    # 3. Clean synthetic code must pass uncontaminated
    clean_code = "export function multiplyByTwo(val: number): number { return val * 2; }\n"
    assert not decon.contaminated(clean_code), "clean code was falsely flagged as contaminated"


def test_mhm_sample_shards_layout_and_loader():
    """Verify that sample shards match src/data/shard.py layout and are readable by PackedLoader."""
    from pathlib import Path
    from src.data.shard import open_shard, read_manifest

    shards_dir = Path(__file__).resolve().parents[1] / "data/mhm_sample_shards"
    if not (shards_dir / "manifest.json").exists():
        pytest.skip("data/mhm_sample_shards not generated yet; run scripts/build_corpus.py --pack")

    manifest = read_manifest(shards_dir)

    # Schema & layout assertions
    assert manifest["dtype"] == "uint16"
    assert manifest["tokenizer"] == "code"
    assert manifest["seq_len"] > 0
    assert manifest["n_tokens"] > 0
    assert manifest["n_documents"] > 0
    assert manifest["n_sequences"] > 0
    assert len(manifest["shards"]) >= 1

    # Manifest records per-language mix and filter rate (#252 acceptance)
    assert "language_mix" in manifest and len(manifest["language_mix"]) > 0
    assert "filter_rate" in manifest
    assert "drop_rate" in manifest["filter_rate"]
    assert "decontamination" in manifest
    assert manifest["decontamination"]["applied"] is True

    # Shard binary and boundary files match
    first_shard = manifest["shards"][0]["name"]
    toks, bnds = open_shard(shards_dir, first_shard)
    assert toks.dtype == np.uint16
    assert bnds.dtype == np.uint8
    assert len(toks) == len(bnds)
    assert len(toks) == manifest["shards"][0]["n_tokens"]

    # PackedLoader directly consumes shard without code changes
    loader = PackedLoader(shards_dir / f"{first_shard}.bin",
                          seq_len=manifest["seq_len"], batch_size=2, shuffle=False)
    inputs, targets = next(iter(loader.epoch()))
    assert inputs.shape == (2, manifest["seq_len"])
    assert targets.shape == (2, manifest["seq_len"])


def test_mhm_sample_shards_split_val_tokens(tmp_path):
    """Verify split_shards carves out a val split readable by PackedLoader."""
    from pathlib import Path
    from src.data.shard import pack_sequences
    from src.data.split import split_shards

    # Create a small multi-shard test corpus if data/mhm_sample_shards is not present
    shards_dir = Path(__file__).resolve().parents[1] / "data/mhm_sample_shards"
    if not (shards_dir / "manifest.json").exists():
        shards_dir = tmp_path / "shards"
        docs = [[i % 1000 + 1 for i in range(256)] for _ in range(8)]
        pack_sequences(docs, shards_dir, seq_len=64, shard_size_mb=1, tokenizer="code")

    split_dir = tmp_path / "split"
    val_tokens = 512
    tr_path, va_path = split_shards(shards_dir, split_dir, val_tokens=val_tokens)

    assert tr_path.exists() and va_path.exists()
    assert (split_dir / "val.meta.json").exists()

    val_meta = json.loads((split_dir / "val.meta.json").read_text())
    assert val_meta["n_tokens"] == val_tokens

    # Both train and val are readable by PackedLoader
    tr_loader = PackedLoader(tr_path, seq_len=64, batch_size=2, shuffle=False)
    va_loader = PackedLoader(va_path, seq_len=64, batch_size=2, shuffle=False)
    t_in, t_tgt = next(iter(tr_loader.epoch()))
    v_in, v_tgt = next(iter(va_loader.epoch()))
    assert t_in.shape == (2, 64) and v_in.shape == (2, 64)
