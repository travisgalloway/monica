"""Mamba-aware pack/shard with document-boundary sidecar (#74).

Pure numpy + stdlib. The boundary sidecar (consumed by #68) is the contract here.
"""

import numpy as np
import pytest

from src.data.shard import (doc_start_offsets, open_shard, pack_sequences, read_manifest,
                            segment_ids)


def test_pack_sequences_boundaries_and_roundtrip(tmp_path):
    # 3 docs of lengths 4, 6, 5 = 15 tokens; seq_len 5 -> 3 full sequences (15 tokens).
    docs = [[1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    manifest = pack_sequences(docs, tmp_path, seq_len=5, shard_size_mb=512, tokenizer="byte")
    assert manifest["n_sequences"] == 3 and manifest["n_tokens"] == 15
    assert manifest["n_documents"] == 3 and len(manifest["shards"]) == 1

    toks, bnds = open_shard(tmp_path, manifest["shards"][0]["name"])
    assert list(toks) == list(range(1, 16))
    # doc starts at global offsets 0, 4, 10
    assert doc_start_offsets(bnds) == [0, 4, 10]
    assert bnds.sum() == 3


def test_pack_sequences_drops_final_partial(tmp_path):
    # 13 tokens, seq_len 5 -> 2 full sequences (10 tokens); 3 trailing dropped.
    docs = [list(range(1, 14))]
    manifest = pack_sequences(docs, tmp_path, seq_len=5)
    assert manifest["n_sequences"] == 2 and manifest["n_tokens"] == 10
    toks, _ = open_shard(tmp_path, manifest["shards"][0]["name"])
    assert len(toks) == 10


def test_n_documents_counts_only_emitted_starts(tmp_path):
    # doc A fills seq 0 (5 tok); doc B starts at offset 5 inside the dropped partial.
    manifest = pack_sequences([[1, 2, 3, 4, 5], [6, 7, 8]], tmp_path, seq_len=5)
    assert manifest["n_sequences"] == 1 and manifest["n_documents"] == 1   # B not counted
    _, bnds = open_shard(tmp_path, manifest["shards"][0]["name"])
    assert int(np.asarray(bnds).sum()) == manifest["n_documents"]          # matches sidecar


def test_pack_sequences_rolls_few_large_shards(tmp_path):
    # tiny budget -> one shard per (sequence-aligned) budget; proves few-large rolling.
    docs = [list(range(1, 5)) for _ in range(8)]      # 8 docs x 4 = 32 tokens
    manifest = pack_sequences(docs, tmp_path, seq_len=4, shard_size_mb=0)  # budget -> seq_len
    assert manifest["n_sequences"] == 8 and len(manifest["shards"]) == 8
    # boundary count across all shards equals doc count
    total_bounds = 0
    for s in manifest["shards"]:
        _, bnds = open_shard(tmp_path, s["name"])
        total_bounds += int(np.asarray(bnds).sum())
    assert total_bounds == 8


def test_pack_sequences_uint16_guard(tmp_path):
    with pytest.raises((OverflowError, ValueError)):
        pack_sequences([[1, 2, 70000]], tmp_path, seq_len=2)   # 70000 > uint16 max


def test_chunk_align_pads_docs_to_chunk_starts(tmp_path):
    # chunk_align=4: doc len 3 -> 4 (1 pad), doc len 5 -> 8 (3 pad); starts at 0 and 4.
    manifest = pack_sequences([[1, 2, 3], [4, 5, 6, 7, 8]], tmp_path, seq_len=4,
                              chunk_align=4, pad_id=0)
    toks, bnds = open_shard(tmp_path, manifest["shards"][0]["name"])
    assert doc_start_offsets(bnds) == [0, 4]           # both starts are multiples of 4
    assert list(toks) == [1, 2, 3, 0, 4, 5, 6, 7, 8, 0, 0, 0]
    assert list(segment_ids(bnds)) == [0, 0, 0, 0] + [1] * 8


def test_chunk_align_requires_divisible_seq_len(tmp_path):
    with pytest.raises(ValueError):
        pack_sequences([[1, 2]], tmp_path, seq_len=6, chunk_align=4)   # 6 % 4 != 0


def test_manifest_persisted(tmp_path):
    pack_sequences([[1, 2, 3, 4]], tmp_path, seq_len=2, tokenizer="starcoder2")
    m = read_manifest(tmp_path)
    assert m["seq_len"] == 2 and m["tokenizer"] == "starcoder2" and m["dtype"] == "uint16"
