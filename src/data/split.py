"""Hold out a small validation shard from the packed stream.

The validation shard MUST NOT overlap the training stream — held-out perplexity on
it is the primary pipeline-health signal (see eval/val_loss). We split by a single
contiguous cut so train and val token ranges are provably disjoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from .pack import open_packed, packed_dtype, packed_n_bytes


def split_packed(packed_path: Path, out_dir: Path, val_tokens: int,
                 from_end: bool = True) -> Tuple[Path, Path]:
    """Cut `val_tokens` contiguous tokens off the packed file into a val shard.

    Returns (train_path, val_path). The two ranges are disjoint by construction.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = open_packed(packed_path)
    dtype = packed_dtype(packed_path)          # uint16 (POC) / uint32 (Qwen3) — preserved
    src_bytes = packed_n_bytes(packed_path)     # UTF-8 byte count if recorded, else None (#192)
    n = data.shape[0]
    if val_tokens >= n:
        raise ValueError(f"val_tokens={val_tokens} >= total tokens={n}")

    if from_end:
        train, val = data[: n - val_tokens], data[n - val_tokens:]
    else:
        val, train = data[:val_tokens], data[val_tokens:]

    train_path, val_path = out_dir / "train.bin", out_dir / "val.bin"
    np.asarray(train, dtype=dtype).tofile(train_path)
    np.asarray(val, dtype=dtype).tofile(val_path)
    # Propagate the corpus byte count proportionally to the token cut, preserving the
    # uniform bytes/token ratio the bits-per-byte metric assumes (#192). None on legacy
    # (no-n_bytes) packed files — val_bpb is simply omitted downstream.
    val_bytes = train_bytes = None
    if src_bytes is not None and n > 0:
        val_bytes = round(src_bytes * (val_tokens / n))
        train_bytes = src_bytes - val_bytes
    for p, a, nb in ((train_path, train, train_bytes), (val_path, val, val_bytes)):
        meta = {"dtype": dtype.name, "n_tokens": int(a.shape[0])}
        if nb is not None:
            meta["n_bytes"] = int(nb)
        with open(p.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f)
    return train_path, val_path


def split_shards(shard_dir: Path, out_dir: Path, val_tokens: int) -> Tuple[Path, Path]:
    """Make a `train.bin`/`val.bin` split (the `PackedLoader` format) from a tokenized SHARD
    directory (`shard.py`'s `part-*.bin` + `manifest.json`) — the bridge from the datatrove/scale
    corpus to the trainer, with NO re-tokenize. The shards are a flat token stream; the last
    `val_tokens` are held out for val, the rest concatenated into train (disjoint by construction).

    Streams shard-by-shard (memmap -> file), so it is bounded in memory regardless of corpus size.
    Drops the `.bounds` sidecars — `PackedLoader` packs flat `seq_len` windows and ignores doc
    boundaries (the #68 boundary-aware path is separate).

    Out of scope for #192: the shard manifest does not record UTF-8 byte counts, so the
    outputs here carry no `n_bytes` and `val_bpb` is simply omitted downstream (graceful
    degradation). Recording per-shard bytes in `shard.py`'s manifest is the follow-up needed
    to light up BPB on this path (likely #193)."""
    from .shard import open_shard, read_manifest

    shard_dir = Path(shard_dir)
    man = read_manifest(shard_dir)
    total, dtype = int(man["n_tokens"]), np.dtype(man["dtype"])
    if val_tokens >= total:
        raise ValueError(f"val_tokens={val_tokens} >= total tokens={total}")
    train_n = total - val_tokens                       # val is the contiguous tail

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out_dir / "train.bin", out_dir / "val.bin"
    written = 0
    with open(train_path, "wb") as ftr, open(val_path, "wb") as fva:
        for sh in man["shards"]:
            toks, _ = open_shard(shard_dir, sh["name"])     # uint32 memmap (dtype from manifest)
            arr = np.asarray(toks)
            if written >= train_n:                          # whole shard is val
                arr.tofile(fva)
            elif written + len(arr) <= train_n:             # whole shard is train
                arr.tofile(ftr)
            else:                                           # shard straddles the cut
                cut = train_n - written
                arr[:cut].tofile(ftr)
                arr[cut:].tofile(fva)
            written += len(arr)
    for p, n in ((train_path, train_n), (val_path, val_tokens)):
        with open(p.with_suffix(".meta.json"), "w") as f:
            json.dump({"dtype": dtype.name, "n_tokens": int(n)}, f)
    return train_path, val_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--packed", type=Path, help="a single packed .bin (pack.py output)")
    src.add_argument("--shards", type=Path, help="a tokenized shard dir (shard.py output)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-tokens", type=int, required=True)
    args = ap.parse_args()
    if args.shards:
        tr, va = split_shards(args.shards, args.out, args.val_tokens)
    else:
        tr, va = split_packed(args.packed, args.out, args.val_tokens)
    print(f"train -> {tr}\nval   -> {va}")


if __name__ == "__main__":
    main()
