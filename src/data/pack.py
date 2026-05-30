"""Pack token-id streams into a flat memory-mapped uint16 file.

uint16 because the OLMo vocab (~50k) fits under 65536 — confirm the actual vocab
before committing (see MambaConfig.validate / tokenize.load_olmo_tokenizer). The
loader reads this format directly at train time; no JSON parsing during training.

A small sidecar `<name>.meta.json` records dtype and token count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

DTYPE = np.uint16


def pack_ids(ids: Iterable[int] | np.ndarray, out_path: Path,
             chunk: int = 1 << 20) -> int:
    """Write a flat uint16 token file. Returns the number of tokens written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(ids, np.ndarray):
        arr = ids.astype(DTYPE, copy=False)
        if arr.max(initial=0) >= 65536:
            raise ValueError("token id >= 65536 does not fit uint16")
        arr.tofile(out_path)
        n = arr.size
    else:
        n = 0
        with open(out_path, "wb") as f:
            buf = []
            for tid in ids:
                if tid >= 65536:
                    raise ValueError("token id >= 65536 does not fit uint16")
                buf.append(tid)
                if len(buf) >= chunk:
                    np.asarray(buf, dtype=DTYPE).tofile(f)
                    n += len(buf)
                    buf = []
            if buf:
                np.asarray(buf, dtype=DTYPE).tofile(f)
                n += len(buf)

    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump({"dtype": "uint16", "n_tokens": int(n)}, f)
    return n


def open_packed(path: Path) -> np.memmap:
    """Memory-map a packed file read-only as uint16."""
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    n = None
    if meta_path.exists():
        n = json.loads(meta_path.read_text())["n_tokens"]
    return np.memmap(path, dtype=DTYPE, mode="r", shape=(n,) if n else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help=".npy uint16 ids")
    ap.add_argument("--out", type=Path, required=True, help="packed .bin")
    args = ap.parse_args()
    ids = np.load(args.inp)
    n = pack_ids(ids, args.out)
    print(f"packed {n} tokens -> {args.out}")


if __name__ == "__main__":
    main()
