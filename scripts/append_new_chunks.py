"""Phase B' append-merge recipe (#65): append the new-source distillation-extension corpus
onto the existing 566 GB FineWeb teacher top-k cache, without re-precomputing the unchanged
FineWeb chunks (`.claude/plans/issue-65.md`).

**Pre-requisite**: run `scripts/verify_teacher_alignment.py` against the regenerated FineWeb
`train.bin` FIRST and confirm it passes — this script assumes that safety gate already passed
and does not re-run it.

Recipe:
  1. Regenerate FineWeb's flat `train.bin` (`split.py`'s `split_shards`, `--val-tokens
     10_000_000`) and assert it still yields the frozen `n_chunks == 230318` — the corpus MUST
     NOT have silently drifted under the existing 566 GB cache's positional binding.
  2. Trim `train.bin` to exactly `230318 * 8193 * 4 = 7,547,981,496` bytes via copy-prefix-
     then-rename (NOT `os.truncate` — RunPod's `/vol` MooseFS mount silently no-ops truncate,
     corrupting resumed writes; see commit 06f2853).
  3. Build a flat `new/train.bin` (+ `.meta.json` sidecar) from the extension corpus's
     tokenized shard (`part-*.bin`), concatenated in shard order.
  4. `cat` the trimmed FineWeb prefix + the new flat file into the combined `train.bin`, and
     write the combined `.meta.json` (summed `n_tokens`).
  5. Merge teacher top-k shards via the stream-to-R2 pattern (download frozen shard-0 from R2,
     keep the freshly-precomputed shard-1 local, concatenate streaming straight to a new R2
     prefix — avoiding materializing the ~1.57 TB combined set locally); verify shard-0's
     `n_chunks == 230318` with no stray `start_chunk` offset. Splits with no shard-1 counterpart
     (Step 3 only ever precomputes `train` for the new chunks) are passed through unchanged from
     shard-0 rather than merged.
  6. Verify `DistillLoader` opens the combined `(train.bin, merged teacher-outputs)` pair.

Pod-only (network + big local volumes); authored here (Mac, Phase A') for Phase B' to run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

#: The frozen FineWeb cache's chunk count / geometry (positionally binds the existing
#: 566 GB teacher-outputs cache — see teacher_outputs.py's DistillLoader n_chunks assert).
FINEWEB_N_CHUNKS = 230318
SEQ_LEN = 8192
STRIDE = SEQ_LEN + 1
UINT32_BYTES = 4
FINEWEB_TRIM_BYTES = FINEWEB_N_CHUNKS * STRIDE * UINT32_BYTES  # 7,547,981,496


def trim_file_prefix_copy(path: Path, keep_bytes: int) -> None:
    """Trim `path` to its first `keep_bytes` bytes via copy-prefix-then-rename — safe on any
    POSIX filesystem regardless of truncate semantics. `os.truncate()` silently no-ops on
    RunPod's `/vol` MooseFS mount (returns 0 but leaves the file's original size), corrupting
    any subsequent append; see commit 06f2853 for the confirmed incident."""
    tmp = path.with_suffix(path.suffix + ".trim_tmp")
    try:
        with open(path, "rb") as src, open(tmp, "wb") as dst:
            remaining = keep_bytes
            while remaining > 0:
                buf = src.read(min(8 * 1024 * 1024, remaining))
                if not buf:
                    break
                dst.write(buf)
                remaining -= len(buf)
        os.replace(tmp, path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"failed to trim {path} to {keep_bytes} bytes: {e}") from e


def regenerate_fineweb_train(shard_dir: Path, out_dir: Path,
                             val_tokens: int = 10_000_000) -> Tuple[Path, Path]:
    """Step 1: rebuild FineWeb's flat `train.bin`/`val.bin` via `split.split_shards` and assert
    the frozen chunk count. Raises `SystemExit` (abort -> full re-precompute) on drift. Returns
    `(train_path, val_path)` — `val_path` is the base cache's positional partner for the `val`
    split (never trimmed/combined; the extension contributes train data only)."""
    from src.data.pack import open_packed
    from src.data.split import split_shards

    train_path, val_path = split_shards(shard_dir, out_dir, val_tokens)
    n_tokens = int(open_packed(train_path).shape[0])
    n_chunks = n_tokens // STRIDE
    if n_chunks != FINEWEB_N_CHUNKS:
        raise SystemExit(
            f"regenerated FineWeb train.bin has n_chunks={n_chunks}, expected "
            f"{FINEWEB_N_CHUNKS} — the corpus has drifted from the frozen 566 GB cache's "
            "positional binding; abort and run a full re-precompute instead of an append")
    return train_path, val_path


def trim_to_frozen_chunks(train_path: Path) -> None:
    """Step 2: trim the regenerated `train.bin` down to exactly the frozen 230,318 chunks."""
    from src.data.pack import packed_dtype

    dtype = packed_dtype(train_path)
    if dtype.itemsize != UINT32_BYTES:
        raise SystemExit(f"expected uint32 packed dtype, got {dtype.name}")
    trim_file_prefix_copy(train_path, FINEWEB_TRIM_BYTES)
    train_path.with_suffix(".meta.json").write_text(
        json.dumps({"dtype": dtype.name, "n_tokens": FINEWEB_N_CHUNKS * SEQ_LEN}))


def build_flat_new_train(tokenized_shard_dir: Path, out_path: Path) -> dict:
    """Step 3: flatten the extension corpus's tokenized shard (`part-*.bin`) into one flat
    `new/train.bin` + `.meta.json` sidecar, in shard order."""
    from src.data.shard import open_shard, read_manifest

    man = read_manifest(tokenized_shard_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = 0
    with open(out_path, "wb") as f:
        for sh in man["shards"]:
            toks, _bounds = open_shard(tokenized_shard_dir, sh["name"])
            toks.tofile(f)
            n_tokens += int(toks.shape[0])
    meta = {"dtype": man["dtype"], "n_tokens": n_tokens}
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return meta


def cat_combined_train(fineweb_train: Path, new_train: Path, combined_path: Path) -> dict:
    """Step 4: concatenate the trimmed FineWeb prefix + the new flat file into one combined
    `train.bin`, writing the summed `.meta.json`."""
    from src.data.pack import packed_dtype

    fw_dtype = packed_dtype(fineweb_train)
    new_dtype = packed_dtype(new_train)
    if fw_dtype != new_dtype:
        raise SystemExit(f"dtype mismatch: fineweb={fw_dtype.name} new={new_dtype.name}")

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = 0
    with open(combined_path, "wb") as dst:
        for src_path in (fineweb_train, new_train):
            with open(src_path, "rb") as src:
                while True:
                    buf = src.read(256 * 1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)
            n_tokens += json.loads(src_path.with_suffix(".meta.json").read_text())["n_tokens"]

    meta = {"dtype": fw_dtype.name, "n_tokens": n_tokens}
    combined_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return meta


def _remote_topk_paths(prefix: str, split: str) -> dict:
    base = f"{prefix.rstrip('/')}/teacher-{split}"
    return {"vals": f"{base}.topk_vals", "idx": f"{base}.topk_idx", "meta": f"{base}.meta.json"}


def _stream_copy(src: Path, fs, dst_path: str) -> None:
    """Stream-copy one local file to an fsspec destination in bounded-memory chunks."""
    with fs.open(dst_path, "wb") as dst, open(src, "rb") as src_f:
        while True:
            buf = src_f.read(256 * 1024 * 1024)
            if not buf:
                break
            dst.write(buf)


def merge_teacher_shards_stream_to_r2(shard0_r2: str, shard1_local: Path, splits: List[str],
                                      push: str) -> None:
    """Step 5: merge the frozen shard-0 (566 GB, downloaded from R2) with the freshly
    precomputed shard-1 (new chunks, local) by concatenating streaming straight to `push` — no
    ~1.57 TB local materialization. Verifies shard-0 is the whole unshifted frozen cache (no
    stray `start_chunk`, `n_chunks == FINEWEB_N_CHUNKS`) before merging.

    Splits with no shard-1 counterpart are **passed through unchanged** rather than merged: Step
    3 only ever precomputes `train` for the new chunks (the frozen base `val` chunks have no
    extension counterpart), so a shard-1-less split streams shard-0's `topk_vals`/`topk_idx`
    straight to `push` and copies its `.meta.json` verbatim — counts stay shard-0's, not summed.
    This makes the default `--splits train,val` complete out of the box: `train` merges,
    `val` passes through from the base cache.
    """
    from src.data.r2_sync import _fs_for, download_dir

    if str(shard0_r2).startswith("s3://"):
        local_root = shard1_local.parent / "_shard0_cache"
        download_dir(shard0_r2, str(local_root))
    else:
        # `--topk-dir` is already a local filesystem path (e.g. Step 1 already downloaded it for
        # the alignment gate) — `_fs_for` resolves any non-`s3://` URI to the local FS, so reuse
        # the path directly instead of an ~566 GB local->local re-copy into `local_root`.
        local_root = Path(shard0_r2)

    fs, root = _fs_for(push)
    root = root.rstrip("/")
    fs.makedirs(root, exist_ok=True)

    n_rows_total = 0
    shard0_meta = None
    for split in splits:
        shard0_meta = json.loads((local_root / f"teacher-{split}.meta.json").read_text())
        if shard0_meta.get("start_chunk") not in (None, 0):
            raise SystemExit(
                f"shard-0 {split} has a stray start_chunk={shard0_meta['start_chunk']}")

        # Use the protocol-stripped `root` (from `_fs_for(push)` above), not the raw `push`
        # URI -- matches the established pattern elsewhere in this codebase (e.g.
        # src/data/corpus.py's shard writer) for building fs.open() destinations.
        out_paths = _remote_topk_paths(root, split)
        shard1_meta_path = shard1_local / f"teacher-{split}.meta.json"

        if not shard1_meta_path.exists():
            # Passthrough: no shard-1 counterpart for this split. No FINEWEB_N_CHUNKS check
            # here -- that invariant binds only the frozen FineWeb TRAIN prefix (the thing a new
            # shard-1 gets appended after); `val` is a separate, much smaller held-out cache
            # (e.g. 305 chunks, not 230,318) with no shard-1 counterpart to align against, so
            # there is nothing to verify beyond the start_chunk check above. Stream shard-0's
            # files unchanged and copy its .meta.json verbatim (counts = shard-0's own).
            for kind, out_path in (("vals", out_paths["vals"]), ("idx", out_paths["idx"])):
                _stream_copy(local_root / f"teacher-{split}.topk_{kind}", fs, out_path)
            with fs.open(out_paths["meta"], "w") as f:
                f.write((local_root / f"teacher-{split}.meta.json").read_text())
            n_rows_total += shard0_meta["n_rows"]
            continue

        # Merging: this split's shard-0 IS the frozen FineWeb prefix a new shard-1 is being
        # appended after, so it must still be exactly the frozen chunk count before concatenating.
        if shard0_meta["n_chunks"] != FINEWEB_N_CHUNKS:
            raise SystemExit(
                f"shard-0 {split} n_chunks={shard0_meta['n_chunks']} != {FINEWEB_N_CHUNKS}")

        shard1_meta = json.loads(shard1_meta_path.read_text())
        total_rows = shard0_meta["n_rows"] + shard1_meta["n_rows"]
        total_chunks = shard0_meta["n_chunks"] + shard1_meta["n_chunks"]
        n_rows_total += total_rows

        for kind, out_path in (("vals", out_paths["vals"]), ("idx", out_paths["idx"])):
            src_paths = [local_root / f"teacher-{split}.topk_{kind}",
                        shard1_local / f"teacher-{split}.topk_{kind}"]
            with fs.open(out_path, "wb") as dst:
                for sp in src_paths:
                    with open(sp, "rb") as src:
                        while True:
                            buf = src.read(256 * 1024 * 1024)
                            if not buf:
                                break
                            dst.write(buf)

        meta = {"split": split, "k": shard0_meta["k"], "n_rows": total_rows,
                "n_chunks": total_chunks, "seq_len": shard0_meta["seq_len"],
                "vals_dtype": shard0_meta["vals_dtype"], "idx_dtype": shard0_meta["idx_dtype"],
                "vocab_size": shard0_meta["vocab_size"],
                "src_packed": shard0_meta.get("src_packed"), "src_n_tokens": None,
                "merged_from_shards": 2}
        with fs.open(out_paths["meta"], "w") as f:
            f.write(json.dumps(meta, indent=2))

    manifest = {"k": shard0_meta["k"], "seq_len": shard0_meta["seq_len"],
               "vals_dtype": shard0_meta["vals_dtype"], "idx_dtype": shard0_meta["idx_dtype"],
               "effective_vocab_size": shard0_meta["vocab_size"], "corpus_manifest": None,
               "teacher": None, "splits": splits, "n_rows_total": n_rows_total}
    with fs.open(f"{root}/manifest.json", "w") as f:
        f.write(json.dumps(manifest, indent=2))


def verify_combined(train_path: Path, topk_dir, split: str, seq_len: int,
                    batch_size: int = 4) -> None:
    """Step 6: open the combined pair through `DistillLoader` end to end (one batch is enough
    to prove alignment)."""
    from src.data.teacher_outputs import DistillLoader

    loader = DistillLoader(train_path, topk_dir, split, seq_len, batch_size, shuffle=False)
    inputs, targets, vals, idx = next(loader.epoch())
    assert inputs.shape == targets.shape == (batch_size, seq_len)
    assert vals.shape[:2] == idx.shape[:2] == (batch_size, seq_len)
    print(f"verify_combined: OK ({inputs.shape}, k={vals.shape[-1]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fineweb-shards", type=Path, required=True,
                    help="the original FineWeb tokenized shard dir (shard.py output)")
    ap.add_argument("--extension-shards", type=Path, required=True,
                    help="the new-source distillation-extension tokenized shard dir")
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="scratch dir for the regenerated/trimmed/combined train.bin")
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--topk-dir", required=True,
                    help="the frozen topk-logits-merged dir (R2 prefix; shard-0 for the merge). "
                         "Kept as a plain str, not Path — Path() collapses 's3://' to 's3:/', "
                         "breaking the s3:// vs. local-path detection in "
                         "merge_teacher_shards_stream_to_r2.")
    ap.add_argument("--shard1-local", type=Path, required=True,
                    help="the freshly precomputed teacher-outputs shard for the new chunks")
    ap.add_argument("--push", required=True, help="R2 prefix for the merged teacher-outputs")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = ap.parse_args()

    splits = [s for s in args.splits.split(",") if s]

    train_path, val_path = regenerate_fineweb_train(args.fineweb_shards, args.work_dir,
                                                     args.val_tokens)
    trim_to_frozen_chunks(train_path)

    new_train = args.work_dir / "new" / "train.bin"
    build_flat_new_train(args.extension_shards, new_train)

    combined_path = args.work_dir / "combined" / "train.bin"
    cat_combined_train(train_path, new_train, combined_path)

    merge_teacher_shards_stream_to_r2(str(args.topk_dir), args.shard1_local, splits, args.push)

    local_merged = args.work_dir / "_merged_meta_check"
    from src.data.r2_sync import download_dir
    download_dir(args.push, str(local_merged))
    if "train" in splits:
        verify_combined(combined_path, local_merged, "train", args.seq_len)
    if "val" in splits:
        # `val` is passed through unchanged from the base cache (no extension counterpart), so
        # its positional partner is the base val.bin regenerate_fineweb_train wrote — NOT the
        # combined train.bin.
        verify_combined(val_path, local_merged, "val", args.seq_len)

    print(f"append complete: combined train.bin -> {combined_path}, merged teacher -> {args.push}")


if __name__ == "__main__":
    main()
