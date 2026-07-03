"""Merge sharded teacher top-k outputs from a cluster precompute run into one coherent set.

Each pod in the cluster ran `precompute_teacher.py --shard-id N --num-shards M`, writing its
shard to `<push>/shard-N/`.  This script downloads all shards from R2, concatenates the binary
files in shard order (preserving positional alignment with the corpus), and writes the merged
teacher-outputs dir that `DistillLoader` / `distill.py` expect.

`val` is handled as a shard-0-only passthrough (the cluster launcher only computes val on
shard-0 — cheap, not worth sharding), so the default `--splits train,val` works without every
shard needing a `teacher-val.meta.json`.

    # After all 4 pods finish (shards 0-3 on R2):
    python scripts/merge_teacher_shards.py \\
        --source s3://monica-training/poc-distill/teacher-outputs/topk-logits \\
        --num-shards 4 \\
        --out /vol/teacher-outputs/topk-logits \\
        --push s3://monica-training/poc-distill/teacher-outputs/topk-logits-merged

The merged dir is a normal teacher-outputs layout that DistillLoader reads without any changes.
Merge is fast (all shards already on disk after download; cat is I/O bound).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="R2 prefix holding shard-0/, shard-1/, ... (e.g. s3://.../topk-logits)")
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True, help="local output dir for merged result")
    ap.add_argument("--splits", default="train,val",
                    help="splits to merge (default: train,val)")
    ap.add_argument("--push", default=None,
                    help="after merging, push --out to this R2 prefix")
    return ap.parse_args()


def _download_shard(source: str, shard_id: int, local_root: Path) -> Path:
    from src.data.r2_sync import download_dir
    shard_uri = f"{source.rstrip('/')}/shard-{shard_id}"
    shard_local = local_root / f"shard-{shard_id}"
    shard_local.mkdir(parents=True, exist_ok=True)
    download_dir(shard_uri, str(shard_local))
    return shard_local


def _cat_binary(src_paths: list[Path], dst: Path) -> None:
    """Concatenate binary files in order to dst (streaming, 256 MB at a time)."""
    CHUNK = 256 * 1024 * 1024
    with open(dst, "wb") as out:
        for p in src_paths:
            with open(p, "rb") as f:
                while True:
                    buf = f.read(CHUNK)
                    if not buf:
                        break
                    out.write(buf)


def main() -> None:
    args = _parse_args()
    from src.data.teacher_outputs import topk_outputs_paths, write_manifest

    splits = [s for s in args.splits.split(",") if s]
    local_root = args.out.parent / "_shard_cache"
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.num_shards} shards from {args.source} ...")
    shard_dirs = []
    for i in range(args.num_shards):
        print(f"  shard {i}/{args.num_shards} ...", end=" ", flush=True)
        d = _download_shard(args.source, i, local_root)
        shard_dirs.append(d)
        print("done")

    teacher_info = None
    corpus_manifest = None
    k, seq_len, vocab_size = None, None, None

    for split in splits:
        print(f"Merging split '{split}' ...")

        if split == "val":
            # The cluster launcher only computes val on shard-0 (cheap; not worth sharding
            # across pods) — merge it as a single-shard passthrough rather than requiring
            # every shard to have a teacher-val.meta.json (they don't).
            meta_path = shard_dirs[0] / "teacher-val.meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"shard 0 missing {meta_path} (expected val on shard-0)")
            m = json.loads(meta_path.read_text())
            if k is None:
                k, seq_len, vocab_size = m["k"], m["seq_len"], m["vocab_size"]
            elif m["k"] != k or m["seq_len"] != seq_len or m["vocab_size"] != vocab_size:
                raise ValueError("shard 0 val meta mismatch: k/seq_len/vocab_size differ from train")

            out_paths = topk_outputs_paths(args.out, split)
            print(f"  copying {m['n_rows']:,} rows ({m['n_chunks']} chunks) from shard-0 ...")
            _cat_binary([shard_dirs[0] / "teacher-val.topk_vals"], out_paths["vals"])
            _cat_binary([shard_dirs[0] / "teacher-val.topk_idx"], out_paths["idx"])
            meta = {"split": split, "k": k, "n_rows": m["n_rows"], "n_chunks": m["n_chunks"],
                    "seq_len": seq_len, "vals_dtype": "float16", "idx_dtype": "uint32",
                    "vocab_size": vocab_size, "src_packed": m.get("src_packed", ""),
                    "src_n_tokens": m.get("src_n_tokens"), "merged_from_shards": 1}
            out_paths["meta"].write_text(json.dumps(meta, indent=2))
            print(f"  {split}: {m['n_rows']:,} rows written to {args.out} (shard-0 only)")

            if teacher_info is None:
                m0_manifest = shard_dirs[0] / "manifest.json"
                if m0_manifest.exists():
                    m0 = json.loads(m0_manifest.read_text())
                    teacher_info = m0.get("teacher")
                    corpus_manifest = m0.get("corpus_manifest")
            continue

        # Collect per-shard meta and verify consistency
        shard_metas = []
        for i, sd in enumerate(shard_dirs):
            meta_path = sd / f"teacher-{split}.meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"shard {i} missing {meta_path}")
            m = json.loads(meta_path.read_text())
            shard_metas.append(m)
            if k is None:
                k, seq_len, vocab_size = m["k"], m["seq_len"], m["vocab_size"]
            else:
                if m["k"] != k or m["seq_len"] != seq_len or m["vocab_size"] != vocab_size:
                    raise ValueError(f"shard {i} meta mismatch: k/seq_len/vocab_size differ")

        total_rows = sum(m["n_rows"] for m in shard_metas)
        total_chunks = sum(m["n_chunks"] for m in shard_metas)

        # Verify shard order is contiguous
        expected_start = 0
        for i, m in enumerate(shard_metas):
            sc = m.get("start_chunk", None)
            if sc is not None and sc != expected_start:
                raise ValueError(f"shard {i} start_chunk={sc}, expected {expected_start}")
            expected_start += m["n_chunks"]

        # Concatenate binary files
        out_paths = topk_outputs_paths(args.out, split)
        vals_src = [sd / f"teacher-{split}.topk_vals" for sd in shard_dirs]
        idx_src = [sd / f"teacher-{split}.topk_idx" for sd in shard_dirs]
        print(f"  concatenating {total_rows:,} rows ({total_chunks} chunks) ...")
        _cat_binary(vals_src, out_paths["vals"])
        _cat_binary(idx_src, out_paths["idx"])

        # Write combined meta (strip shard-specific fields)
        src_packed = shard_metas[0].get("src_packed", "")
        src_n_tokens = shard_metas[0].get("src_n_tokens")
        meta = {"split": split, "k": k, "n_rows": total_rows, "n_chunks": total_chunks,
                "seq_len": seq_len, "vals_dtype": "float16", "idx_dtype": "uint32",
                "vocab_size": vocab_size, "src_packed": src_packed,
                "src_n_tokens": src_n_tokens, "merged_from_shards": args.num_shards}
        out_paths["meta"].write_text(json.dumps(meta, indent=2))
        print(f"  {split}: {total_rows:,} rows written to {args.out}")

        # Grab run-level info from shard-0 manifest if present
        m0_manifest = shard_dirs[0] / "manifest.json"
        if m0_manifest.exists() and teacher_info is None:
            m0 = json.loads(m0_manifest.read_text())
            teacher_info = m0.get("teacher")
            corpus_manifest = m0.get("corpus_manifest")

    write_manifest(args.out, k=k, seq_len=seq_len, effective_vocab_size=vocab_size,
                   corpus_manifest=corpus_manifest, teacher=teacher_info, splits=splits)
    print(f"Merged manifest written: {args.out}/manifest.json")

    if args.push:
        from src.data.r2_sync import upload_dir
        written = upload_dir(str(args.out), args.push)
        print(f"Pushed {len(written)} file(s) -> {args.push}")


if __name__ == "__main__":
    main()
