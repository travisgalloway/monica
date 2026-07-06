"""Unit tests for the #177 append-merge (scripts/append_new_chunks.py), Tier-1 fix.

Drives `merge_teacher_shards_stream_to_r2` directly against synthetic, hand-built
`teacher-{train,val}.topk_vals`/`.topk_idx`/`.meta.json` files in a temp dir — no real R2, no
GPU, no network. `_fs_for` (src/data/r2_sync.py) resolves any non-`s3://` URI to the local
filesystem, so a bare local path exercises the exact same code path a real R2 prefix would.

The bug this guards: Step 3 (`precompute_teacher.py`) only ever precomputes a **train-only**
shard-1 for the new chunks (the frozen base `val` chunks have no extension counterpart). The old
`merge_teacher_shards_stream_to_r2` unconditionally read `shard1_local/teacher-{split}.meta.json`
for every split in `--splits` (default `train,val`), so the default invocation crashed with
`FileNotFoundError` on `val` — after the expensive precompute already ran. The fix passes
shard-1-less splits through unchanged from shard-0 instead of requiring a merge counterpart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.append_new_chunks import FINEWEB_N_CHUNKS, merge_teacher_shards_stream_to_r2


def _write_topk(out_dir: Path, split: str, *, n_rows: int, n_chunks: int, k: int = 4,
                seq_len: int = 4, vocab_size: int = 32, seed: int = 0) -> dict:
    """Hand-build one split's `teacher-<split>.{topk_vals,topk_idx,meta.json}` trio, matching
    the on-disk schema `src/data/teacher_outputs.py` reads/writes (fp16 vals, uint32 idx; meta
    fields `split/k/n_rows/n_chunks/seq_len/vals_dtype/idx_dtype/vocab_size/src_packed/
    src_n_tokens`). `n_rows` intentionally need not equal `n_chunks*seq_len` here — the merge
    function only reads/propagates the meta fields, it does not cross-check them against actual
    array shape (that positional cross-check is `DistillLoader`'s job, exercised elsewhere)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vals = rng.random((n_rows, k), dtype=np.float32).astype(np.float16)
    idx = rng.integers(0, vocab_size, size=(n_rows, k)).astype(np.uint32)
    vals.tofile(out_dir / f"teacher-{split}.topk_vals")
    idx.tofile(out_dir / f"teacher-{split}.topk_idx")
    meta = {"split": split, "k": k, "n_rows": n_rows, "n_chunks": n_chunks, "seq_len": seq_len,
            "vals_dtype": "float16", "idx_dtype": "uint32", "vocab_size": vocab_size,
            "src_packed": "fineweb/train.bin", "src_n_tokens": n_rows}
    (out_dir / f"teacher-{split}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def test_default_splits_train_val_with_train_only_shard1_does_not_crash(tmp_path):
    """(a) shard0 has both train+val; shard1_local has ONLY train (no val files) — the real
    Step-3 shape. Calling the merge with the default splits=["train", "val"] must not raise, and
    must emit both `teacher-train.*` and `teacher-val.*` at the push destination."""
    shard0_dir = tmp_path / "shard0"
    shard1_dir = tmp_path / "shard1_local"
    push_dir = tmp_path / "push"

    _write_topk(shard0_dir, "train", n_rows=8, n_chunks=FINEWEB_N_CHUNKS)
    _write_topk(shard0_dir, "val", n_rows=6, n_chunks=FINEWEB_N_CHUNKS, seed=1)
    _write_topk(shard1_dir, "train", n_rows=3, n_chunks=17, seed=2)
    # Deliberately no teacher-val.* under shard1_dir.

    merge_teacher_shards_stream_to_r2(str(shard0_dir), shard1_dir, ["train", "val"], str(push_dir))

    for split in ("train", "val"):
        assert (push_dir / f"teacher-{split}.topk_vals").exists()
        assert (push_dir / f"teacher-{split}.topk_idx").exists()
        assert (push_dir / f"teacher-{split}.meta.json").exists()
    assert (push_dir / "manifest.json").exists()


def test_val_passthrough_is_byte_identical_to_shard0(tmp_path):
    """(b) the passed-through `val` bytes + meta at the push destination equal shard0's
    original val files unchanged (proving passthrough, not corruption or accidental merge)."""
    shard0_dir = tmp_path / "shard0"
    shard1_dir = tmp_path / "shard1_local"
    push_dir = tmp_path / "push"

    _write_topk(shard0_dir, "train", n_rows=8, n_chunks=FINEWEB_N_CHUNKS)
    shard0_val_meta = _write_topk(shard0_dir, "val", n_rows=6, n_chunks=FINEWEB_N_CHUNKS, seed=1)
    _write_topk(shard1_dir, "train", n_rows=3, n_chunks=17, seed=2)

    merge_teacher_shards_stream_to_r2(str(shard0_dir), shard1_dir, ["train", "val"], str(push_dir))

    for kind in ("topk_vals", "topk_idx"):
        src_bytes = (shard0_dir / f"teacher-val.{kind}").read_bytes()
        dst_bytes = (push_dir / f"teacher-val.{kind}").read_bytes()
        assert dst_bytes == src_bytes, f"val {kind} passthrough is not byte-identical to shard0"

    dst_meta = json.loads((push_dir / "teacher-val.meta.json").read_text())
    assert dst_meta == shard0_val_meta
    # Explicitly not summed with any shard-1 val (there is none): counts are shard0's own.
    assert dst_meta["n_rows"] == 6
    assert dst_meta["n_chunks"] == FINEWEB_N_CHUNKS


def test_train_still_merges_shard0_plus_shard1(tmp_path):
    """(c) `train` (which DOES have a shard1 counterpart) still merges: n_rows/n_chunks in the
    merged output equal the shard0+shard1 sums (existing merge behavior preserved)."""
    shard0_dir = tmp_path / "shard0"
    shard1_dir = tmp_path / "shard1_local"
    push_dir = tmp_path / "push"

    _write_topk(shard0_dir, "train", n_rows=8, n_chunks=FINEWEB_N_CHUNKS)
    _write_topk(shard0_dir, "val", n_rows=6, n_chunks=FINEWEB_N_CHUNKS, seed=1)
    _write_topk(shard1_dir, "train", n_rows=3, n_chunks=17, seed=2)

    merge_teacher_shards_stream_to_r2(str(shard0_dir), shard1_dir, ["train", "val"], str(push_dir))

    dst_meta = json.loads((push_dir / "teacher-train.meta.json").read_text())
    assert dst_meta["n_rows"] == 8 + 3
    assert dst_meta["n_chunks"] == FINEWEB_N_CHUNKS + 17
    assert dst_meta["merged_from_shards"] == 2

    # The merged train topk_vals/idx bytes are shard0's prefix followed by shard1's (streamed
    # concatenation, not a passthrough).
    expected_vals = ((shard0_dir / "teacher-train.topk_vals").read_bytes()
                     + (shard1_dir / "teacher-train.topk_vals").read_bytes())
    assert (push_dir / "teacher-train.topk_vals").read_bytes() == expected_vals

    manifest = json.loads((push_dir / "manifest.json").read_text())
    assert manifest["splits"] == ["train", "val"]
    assert manifest["n_rows_total"] == (8 + 3) + 6


def test_splits_train_only_still_works_without_val(tmp_path):
    """`--splits train` (no val at all requested) still works — the pre-existing single-split
    path is unaffected by the passthrough branch."""
    shard0_dir = tmp_path / "shard0"
    shard1_dir = tmp_path / "shard1_local"
    push_dir = tmp_path / "push"

    _write_topk(shard0_dir, "train", n_rows=8, n_chunks=FINEWEB_N_CHUNKS)
    _write_topk(shard1_dir, "train", n_rows=3, n_chunks=17, seed=2)

    merge_teacher_shards_stream_to_r2(str(shard0_dir), shard1_dir, ["train"], str(push_dir))

    assert (push_dir / "teacher-train.meta.json").exists()
    assert not (push_dir / "teacher-val.meta.json").exists()


def test_shard0_n_chunks_mismatch_aborts(tmp_path):
    """A shard-0 whose recorded `n_chunks` doesn't match the frozen `FINEWEB_N_CHUNKS` must abort
    (corpus drift guard) rather than silently merging against a misaligned base cache."""
    shard0_dir = tmp_path / "shard0"
    shard1_dir = tmp_path / "shard1_local"
    push_dir = tmp_path / "push"

    _write_topk(shard0_dir, "train", n_rows=8, n_chunks=FINEWEB_N_CHUNKS - 1)
    _write_topk(shard1_dir, "train", n_rows=3, n_chunks=17, seed=2)

    with pytest.raises(SystemExit, match="n_chunks"):
        merge_teacher_shards_stream_to_r2(str(shard0_dir), shard1_dir, ["train"], str(push_dir))


def test_local_shard0_path_is_reused_without_recopy(tmp_path, monkeypatch):
    """Optional hardening: when `--topk-dir` (shard0_r2) is already a local filesystem path (not
    an `s3://` URI), the merge must NOT re-download/re-copy it into `_shard0_cache` — it should
    read straight from the given local path. Assert `download_dir` is never called in this case."""
    import scripts.append_new_chunks as anc

    shard0_dir = tmp_path / "shard0"
    shard1_dir = tmp_path / "shard1_local"
    push_dir = tmp_path / "push"

    _write_topk(shard0_dir, "train", n_rows=8, n_chunks=FINEWEB_N_CHUNKS)
    _write_topk(shard1_dir, "train", n_rows=3, n_chunks=17, seed=2)

    called = {"download_dir": False}

    def _fail_if_called(*args, **kwargs):
        called["download_dir"] = True
        raise AssertionError("download_dir should not be called for a local shard0 path")

    monkeypatch.setattr("src.data.r2_sync.download_dir", _fail_if_called)

    anc.merge_teacher_shards_stream_to_r2(str(shard0_dir), shard1_dir, ["train"], str(push_dir))

    assert not called["download_dir"]
    assert (push_dir / "teacher-train.meta.json").exists()
