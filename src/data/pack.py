"""Pack token-id streams into a flat memory-mapped token file.

The packed dtype follows the tokenizer vocab: **uint16** for the original POC path
(OLMo, vocab ~50k < 65536) and **uint32** for the distillation student (Qwen2.5, vocab
151,646 — see #90 and docs/design/10-distillation.md). `packing_dtype_for` picks it; the
dtype is recorded in the `<name>.meta.json` sidecar so the loader reads the file back
correctly with no JSON parsing during training. Defaults preserve the uint16 behavior, so
existing artifacts are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

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
    return _TYPECODE[np.dtype(dtype)]


def pack_ids(ids: Iterable[int] | np.ndarray, out_path: Path,
             chunk: int = 1 << 20, dtype=DTYPE) -> int:
    """Write a flat token file in `dtype` (uint16 or uint32). Returns the tokens written.

    Validates the ORIGINAL ids against `dtype`'s range before casting (casting first would
    silently wrap out-of-range / negative ids). The chosen dtype is recorded in the
    `.meta.json` sidecar so `open_packed` reads the file back correctly."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    hi = int(np.iinfo(dtype).max)

    if isinstance(ids, np.ndarray):
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

    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump({"dtype": dtype.name, "n_tokens": int(n)}, f)
    return n


def packed_dtype(path: Path) -> np.dtype:
    """The packed dtype recorded in `<name>.meta.json` (fallback uint16 for legacy files)."""
    meta_path = Path(path).with_suffix(".meta.json")
    if meta_path.exists():
        return np.dtype(json.loads(meta_path.read_text()).get("dtype", "uint16"))
    return np.dtype(DTYPE)


def open_packed(path: Path) -> np.memmap:
    """Memory-map a packed file read-only at the dtype its sidecar records (uint16 legacy)."""
    path = Path(path)
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
