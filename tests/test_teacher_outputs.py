"""Cached teacher top-k format + DistillLoader (#94). Portable — no backend import.

Covers the write/read round-trip, positional alignment to the packed corpus tokens, exact
shuffle/skip parity with PackedLoader (so training resume stays bit-exact), the k-subslice,
and the vocab guard.
"""

import numpy as np
import pytest

from src.data.loader import PackedLoader
from src.data.pack import pack_ids
from src.data.teacher_outputs import (DistillLoader, read_teacher_meta, topk_outputs_paths,
                                      write_manifest, write_teacher_topk)

VOCAB = 256


def _build_packed(tmp_path, n_chunks, seq_len, split="train"):
    """Write a flat packed token file whose token at position i is `i % VOCAB`, so the
    loader's chunking is verifiable by value."""
    stride = seq_len + 1
    n_tokens = n_chunks * stride + 2          # +2 leftover tokens the loader ignores
    ids = (np.arange(n_tokens) % VOCAB).astype(np.uint16)
    path = tmp_path / f"{split}.bin"
    pack_ids(ids, path, dtype=np.uint16)
    return path


def _write_aligned_topk(tmp_path, out_dir, n_chunks, seq_len, k, split="train"):
    """Cache top-k where row r has idx[r,0] == vals[r,0] == r (the global teacher-row index),
    so alignment (chunk c, position p) -> row c*seq_len+p can be asserted exactly."""
    n_rows = n_chunks * seq_len
    rows = np.arange(n_rows)
    vals = np.zeros((n_rows, k), dtype=np.float32)
    idx = np.zeros((n_rows, k), dtype=np.int64)
    vals[:, 0] = rows
    idx[:, 0] = rows % VOCAB
    # remaining columns: arbitrary but distinct, in range
    for c in range(1, k):
        vals[:, c] = -rows - c
        idx[:, c] = (rows + c) % VOCAB
    meta = write_teacher_topk(out_dir, split, blocks=[(vals, idx)], n_chunks=n_chunks,
                              seq_len=seq_len, vocab_size=VOCAB,
                              src_packed=str(tmp_path / f"{split}.bin"), src_n_tokens=n_rows)
    return meta


def test_write_read_roundtrip(tmp_path):
    n_chunks, seq_len, k = 6, 4, 3
    out = tmp_path / "teacher-outputs"
    meta = _write_aligned_topk(tmp_path, out, n_chunks, seq_len, k)
    assert meta["k"] == k
    assert meta["n_rows"] == n_chunks * seq_len
    assert meta["vals_dtype"] == "float16" and meta["idx_dtype"] == "uint32"

    read = read_teacher_meta(out, "train")
    assert read == meta
    paths = topk_outputs_paths(out, "train")
    assert paths["vals"].exists() and paths["idx"].exists() and paths["meta"].exists()
    # raw fp16 dump: file size == n_rows * k * 2 bytes
    assert paths["vals"].stat().st_size == n_chunks * seq_len * k * 2


def test_write_rejects_wrong_row_count(tmp_path):
    out = tmp_path / "teacher-outputs"
    bad = (np.zeros((5, 3), np.float32), np.zeros((5, 3), np.int64))
    with pytest.raises(ValueError, match="expected n_chunks"):
        write_teacher_topk(out, "train", blocks=[bad], n_chunks=6, seq_len=4,
                           vocab_size=VOCAB, src_packed="x")


def test_distill_loader_shapes_and_alignment(tmp_path):
    n_chunks, seq_len, k, B = 6, 4, 3, 2
    _build_packed(tmp_path, n_chunks, seq_len)
    out = tmp_path / "teacher-outputs"
    _write_aligned_topk(tmp_path, out, n_chunks, seq_len, k)

    loader = DistillLoader(tmp_path / "train.bin", out, "train", seq_len, B,
                           shuffle=False, drop_last=True)
    # Walk a no-shuffle epoch: chunk indices come out 0,1,2,... so we can predict every value.
    seen = 0
    for inputs, targets, vals, idx in loader.epoch():
        assert inputs.shape == (B, seq_len) and targets.shape == (B, seq_len)
        assert vals.shape == (B, seq_len, k) and idx.shape == (B, seq_len, k)
        for b in range(B):
            c = seen + b
            # token i == i % VOCAB; chunk c starts at c*(seq_len+1)
            base = c * (seq_len + 1)
            assert np.array_equal(inputs[b], (base + np.arange(seq_len)) % VOCAB)
            assert np.array_equal(targets[b], (base + 1 + np.arange(seq_len)) % VOCAB)
            # teacher row for (chunk c, position p) is c*seq_len + p (encoded in column 0)
            expected_rows = c * seq_len + np.arange(seq_len)
            assert np.array_equal(vals[b, :, 0], expected_rows.astype(np.float32))
        seen += B


def test_shuffle_and_skip_match_packed_loader(tmp_path):
    """DistillLoader must visit chunks in the SAME order as PackedLoader for the same seed,
    and skip_batches must drop the same prefix — this is what keeps training resume exact."""
    n_chunks, seq_len, k, B, seed = 12, 4, 3, 2, 7
    _build_packed(tmp_path, n_chunks, seq_len)
    out = tmp_path / "teacher-outputs"
    _write_aligned_topk(tmp_path, out, n_chunks, seq_len, k)

    pk = PackedLoader(tmp_path / "train.bin", seq_len, B, shuffle=True, seed=seed)
    dl = DistillLoader(tmp_path / "train.bin", out, "train", seq_len, B, shuffle=True, seed=seed)
    for (pi, pt), (di, dt, _, _) in zip(pk.epoch(), dl.epoch()):
        assert np.array_equal(pi, di) and np.array_equal(pt, dt)

    # skip_batches=2 drops the same two leading batches as a fresh reseeded epoch's tail.
    pk_full = list(PackedLoader(tmp_path / "train.bin", seq_len, B, shuffle=True, seed=seed)
                   .epoch(reseed=seed))
    dl_skip = list(DistillLoader(tmp_path / "train.bin", out, "train", seq_len, B,
                                 shuffle=True, seed=seed).epoch(reseed=seed, skip_batches=2))
    assert len(dl_skip) == len(pk_full) - 2
    for (pi, _), (di, _, _, _) in zip(pk_full[2:], dl_skip):
        assert np.array_equal(pi, di)


def test_k_subslice(tmp_path):
    n_chunks, seq_len, k, B = 4, 4, 5, 2
    _build_packed(tmp_path, n_chunks, seq_len)
    out = tmp_path / "teacher-outputs"
    _write_aligned_topk(tmp_path, out, n_chunks, seq_len, k)

    dl = DistillLoader(tmp_path / "train.bin", out, "train", seq_len, B, k=2, shuffle=False)
    _, _, vals, idx = next(iter(dl.epoch()))
    assert vals.shape[-1] == 2 and idx.shape[-1] == 2
    with pytest.raises(ValueError, match="exceeds stored k"):
        DistillLoader(tmp_path / "train.bin", out, "train", seq_len, B, k=99)


def test_alignment_mismatch_rejected(tmp_path):
    """A teacher-outputs file built for a different n_chunks must be rejected."""
    n_chunks, seq_len, k, B = 6, 4, 3, 2
    _build_packed(tmp_path, n_chunks, seq_len)
    out = tmp_path / "teacher-outputs"
    # write meta claiming a different n_chunks
    _write_aligned_topk(tmp_path, out, n_chunks - 1, seq_len, k)
    with pytest.raises(ValueError, match="not aligned"):
        DistillLoader(tmp_path / "train.bin", out, "train", seq_len, B)


def test_manifest_written(tmp_path):
    n_chunks, seq_len, k = 4, 4, 3
    out = tmp_path / "teacher-outputs"
    _write_aligned_topk(tmp_path, out, n_chunks, seq_len, k)
    m = write_manifest(out, k=k, seq_len=seq_len, effective_vocab_size=VOCAB,
                       corpus_manifest="data/poc-distill", teacher={"model_id": "x"},
                       splits=["train"])
    assert m["n_rows_total"] == n_chunks * seq_len
    assert (out / "manifest.json").exists()
