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

from .pack import open_packed, DTYPE


def split_packed(packed_path: Path, out_dir: Path, val_tokens: int,
                 from_end: bool = True) -> Tuple[Path, Path]:
    """Cut `val_tokens` contiguous tokens off the packed file into a val shard.

    Returns (train_path, val_path). The two ranges are disjoint by construction.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = open_packed(packed_path)
    n = data.shape[0]
    if val_tokens >= n:
        raise ValueError(f"val_tokens={val_tokens} >= total tokens={n}")

    if from_end:
        train, val = data[: n - val_tokens], data[n - val_tokens:]
    else:
        val, train = data[:val_tokens], data[val_tokens:]

    train_path, val_path = out_dir / "train.bin", out_dir / "val.bin"
    np.asarray(train, dtype=DTYPE).tofile(train_path)
    np.asarray(val, dtype=DTYPE).tofile(val_path)
    for p, a in ((train_path, train), (val_path, val)):
        with open(p.with_suffix(".meta.json"), "w") as f:
            json.dump({"dtype": "uint16", "n_tokens": int(a.shape[0])}, f)
    return train_path, val_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-tokens", type=int, required=True)
    args = ap.parse_args()
    tr, va = split_packed(args.packed, args.out, args.val_tokens)
    print(f"train -> {tr}\nval   -> {va}")


if __name__ == "__main__":
    main()
