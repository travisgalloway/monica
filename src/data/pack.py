"""Pack token-id streams into a flat memory-mapped token file.

The packed dtype follows the tokenizer vocab: **uint16** for the original POC path
(OLMo, vocab ~50k < 65536) and **uint32** for the distillation student (Qwen3, vocab
151,669 — see #90 and docs/reserve/10-distillation.md). `packing_dtype_for` picks it; the
dtype is recorded in the `<name>.meta.json` sidecar so the loader reads the file back
correctly with no JSON parsing during training. Defaults preserve the uint16 behavior, so
existing artifacts are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

#: Default packed dtype (the POC/legacy path). uint32 is opt-in via `dtype=`.
DTYPE = np.uint16

#: The uint16 ceiling — vocabs below this pack as uint16, at/above as uint32.
UINT16_CEILING = 1 << 16

#: array-module typecodes per packed dtype (used by the streaming writer in shard.py).
_TYPECODE = {np.dtype(np.uint16): "H", np.dtype(np.uint32): "I"}


def packing_dtype_for(vocab_or_max_id: int) -> np.dtype:
    """Smallest unsigned dtype that holds token ids for this vocab / max id: uint16 if it
    fits under 65536, else uint32 (the only two we pack)."""
    return np.dtype(np.uint16) if vocab_or_max_id < UINT16_CEILING else np.dtype(np.uint32)


def typecode_for(dtype) -> str:
    """`array` module typecode for a packed dtype ('H' uint16 / 'I' uint32)."""
    try:
        return _TYPECODE[np.dtype(dtype)]
    except KeyError:
        raise ValueError(
            f"unsupported packing dtype {np.dtype(dtype).name}; use uint16 or uint32")


def pack_ids(ids: Iterable[int] | np.ndarray, out_path: Path,
             chunk: int = 1 << 20, dtype=DTYPE, n_bytes: int | None = None) -> int:
    """Write a flat token file in `dtype` (uint16 or uint32). Returns the tokens written.

    Validates the ORIGINAL ids against `dtype`'s range before casting (casting first would
    silently wrap out-of-range / negative ids). The chosen dtype is recorded in the
    `.meta.json` sidecar so `open_packed` reads the file back correctly.

    `n_bytes`, when given, is the UTF-8 byte count of the source text (#192, for the
    tokenizer-invariant bits-per-byte metric) and is recorded in the sidecar too. Omitted
    by default so legacy meta files stay byte-identical."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    hi = int(np.iinfo(dtype).max)

    if isinstance(ids, np.ndarray):
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"ids array must have an integer dtype, got {ids.dtype} "
                             "(float ids would silently truncate/wrap on cast)")
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) > hi):
            raise ValueError(f"token id out of range for {dtype.name} [0, {hi}]")
        arr = ids.astype(dtype, copy=False)
        arr.tofile(out_path)
        n = arr.size
    else:
        n = 0
        with open(out_path, "wb") as f:
            buf = []
            for tid in ids:
                if tid < 0 or tid > hi:
                    raise ValueError(f"token id out of range for {dtype.name} [0, {hi}]")
                buf.append(tid)
                if len(buf) >= chunk:
                    np.asarray(buf, dtype=dtype).tofile(f)
                    n += len(buf)
                    buf = []
            if buf:
                np.asarray(buf, dtype=dtype).tofile(f)
                n += len(buf)

    meta = {"dtype": dtype.name, "n_tokens": int(n)}
    if n_bytes is not None:
        meta["n_bytes"] = int(n_bytes)
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f)
    return n


def packed_dtype(path: Path) -> np.dtype:
    """The packed dtype recorded in `<name>.meta.json` (fallback uint16 for legacy files)."""
    meta_path = Path(path).with_suffix(".meta.json")
    if meta_path.exists():
        return np.dtype(json.loads(meta_path.read_text()).get("dtype", "uint16"))
    return np.dtype(DTYPE)


def packed_n_bytes(path: Path | str) -> int | None:
    """UTF-8 byte count recorded in `<name>.meta.json` or manifest.json, or None if absent."""
    path = Path(path)
    if path.is_dir():
        mf = path / "manifest.json"
        if mf.exists():
            v = json.loads(mf.read_text())
            val = v.get("n_bytes") or v.get("total_raw_bytes") or v.get("total_volume_bytes")
            return int(val) if val is not None else None
        return None
    if path.name == "manifest.json":
        v = json.loads(path.read_text())
        val = v.get("n_bytes") or v.get("total_raw_bytes") or v.get("total_volume_bytes")
        return int(val) if val is not None else None
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        v = json.loads(meta_path.read_text()).get("n_bytes")
        return int(v) if v is not None else None
    return None


def set_packed_n_bytes(path: Path, n_bytes: int) -> None:
    """Patch the `n_bytes` field into `<name>.meta.json` after the fact (#192).

    For streaming callers (tokenize.py's `.bin` path) the byte count is only known once
    the token generator has been fully drained BY `pack_ids` — it can't be passed as a
    `pack_ids(..., n_bytes=...)` kwarg, because Python evaluates call arguments (including
    a `stats["n_bytes"]` lookup) before the function body runs and drains the generator,
    which would capture 0. Call this only after the `pack_ids` call that wrote `path`
    returns."""
    meta_path = Path(path).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["n_bytes"] = int(n_bytes)
    meta_path.write_text(json.dumps(meta))


class ShardedTokenArray:
    """Memory-mapped array across multiple token shards with zero in-memory buffering.

    Exposes a unified 1D array interface (shape, dtype, slicing) over a sequence of
    memory-mapped .bin shard files. Slices are read directly from the OS page cache
    without copying or buffering the full shard data into process memory.
    """

    def __init__(self, shards: Sequence[Path | str], dtype: np.dtype | str = DTYPE,
                 shard_lengths: Optional[Sequence[int]] = None,
                 manifest: Optional[dict] = None):
        self.shard_paths = [Path(p) for p in shards]
        self.dtype = np.dtype(dtype)
        self.manifest = manifest
        self._mmaps: list[np.memmap] = []
        self._lengths: list[int] = []

        for idx, p in enumerate(self.shard_paths):
            if not p.exists():
                raise FileNotFoundError(f"Shard file not found: {p}")
            n = shard_lengths[idx] if shard_lengths is not None else None
            m = np.memmap(p, dtype=self.dtype, mode="r", shape=(n,) if n is not None else None)
            self._mmaps.append(m)
            self._lengths.append(int(m.shape[0]))

        self._offsets = np.zeros(len(self._lengths) + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(self._lengths)
        self._total_tokens = int(self._offsets[-1])
        self.shape = (self._total_tokens,)

    @property
    def size(self) -> int:
        return self._total_tokens

    @property
    def ndim(self) -> int:
        return 1

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self._total_tokens * self.itemsize

    def __len__(self) -> int:
        return self._total_tokens

    def close(self) -> None:
        for m in self._mmaps:
            if hasattr(m, "_mmap") and m._mmap is not None:
                m._mmap.close()

    def __getitem__(self, item) -> np.ndarray | int | np.integer:
        if isinstance(item, slice):
            start, stop, step = item.indices(self._total_tokens)
            if step != 1:
                indices = range(start, stop, step)
                return np.asarray([self[i] for i in indices], dtype=self.dtype)
            if start >= stop:
                return np.empty(0, dtype=self.dtype)

            from bisect import bisect_right
            start_shard = bisect_right(self._offsets, start) - 1
            stop_shard = bisect_right(self._offsets, stop - 1) - 1

            if start_shard == stop_shard:
                local_start = start - self._offsets[start_shard]
                local_stop = stop - self._offsets[start_shard]
                return self._mmaps[start_shard][local_start:local_stop]

            parts = []
            for s_idx in range(start_shard, stop_shard + 1):
                s_off = self._offsets[s_idx]
                e_off = self._offsets[s_idx + 1]
                p_start = max(start, s_off) - s_off
                p_stop = min(stop, e_off) - s_off
                parts.append(self._mmaps[s_idx][p_start:p_stop])
            return np.concatenate(parts)

        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += self._total_tokens
            if idx < 0 or idx >= self._total_tokens:
                raise IndexError(f"index {item} out of bounds for axis 0 with size {self._total_tokens}")
            from bisect import bisect_right
            s_idx = bisect_right(self._offsets, idx) - 1
            local_idx = idx - self._offsets[s_idx]
            return self._mmaps[s_idx][local_idx]

        raise TypeError(f"Invalid index type: {type(item)}")


def open_packed(path: Path | str | Sequence[Path | str]) -> np.memmap | ShardedTokenArray:
    """Memory-map a packed file or shard directory read-only at the dtype its sidecar records (uint16 legacy)."""
    if isinstance(path, (list, tuple)):
        return ShardedTokenArray(path)

    path = Path(path)
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            dtype = np.dtype(manifest.get("dtype", "uint16"))
            shards_info = manifest.get("shards", [])
            if shards_info:
                shard_paths = [path / f"{s['name']}.bin" for s in shards_info]
                shard_lengths = [s.get("n_tokens") for s in shards_info]
                return ShardedTokenArray(shard_paths, dtype=dtype, shard_lengths=shard_lengths, manifest=manifest)
        bin_files = sorted(path.glob("part-*.bin")) or sorted(path.glob("*.bin"))
        if not bin_files:
            raise FileNotFoundError(f"No .bin shard files found in directory {path}")
        return ShardedTokenArray(bin_files)

    if path.name == "manifest.json":
        manifest = json.loads(path.read_text())
        parent_dir = path.parent
        dtype = np.dtype(manifest.get("dtype", "uint16"))
        shards_info = manifest.get("shards", [])
        if shards_info:
            shard_paths = [parent_dir / f"{s['name']}.bin" for s in shards_info]
            shard_lengths = [s.get("n_tokens") for s in shards_info]
            return ShardedTokenArray(shard_paths, dtype=dtype, shard_lengths=shard_lengths, manifest=manifest)
        bin_files = sorted(parent_dir.glob("part-*.bin")) or sorted(parent_dir.glob("*.bin"))
        return ShardedTokenArray(bin_files, dtype=dtype, manifest=manifest)

    meta_path = path.with_suffix(".meta.json")
    n = None
    dtype = np.dtype(DTYPE)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        n = meta["n_tokens"]
        dtype = np.dtype(meta.get("dtype", "uint16"))
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) if n else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help=".npy uint16/uint32 ids")
    ap.add_argument("--out", type=Path, required=True, help="packed .bin")
    ap.add_argument("--dtype", choices=("auto", "uint16", "uint32"), default="auto",
                    help="packed dtype; 'auto' picks uint16/uint32 from the max id")
    args = ap.parse_args()
    ids = np.load(args.inp)
    dtype = (packing_dtype_for(int(ids.max()) + 1 if ids.size else 0)
             if args.dtype == "auto" else np.dtype(args.dtype))
    n = pack_ids(ids, args.out, dtype=dtype)
    print(f"packed {n} tokens ({dtype.name}) -> {args.out}")


if __name__ == "__main__":
    main()
