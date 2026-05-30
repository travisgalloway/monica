"""Pull a small slice of Dolma (or generate synthetic text for offline testing).

Target for the POC run: ~2-5B tokens total (~10-20GB raw text; packed files are
several GB on their own — plan disk accordingly). For the smoke test a few million
tokens is plenty.

Before downloading raw text, CHECK whether AI2 publishes a pre-tokenized Dolma
subset (e.g. on the HuggingFace Hub). If so, prefer it and skip `tokenize.py`
entirely.

Network access in some environments is restricted; `--dummy` produces synthetic
text so the rest of the pipeline can be exercised without a download.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterator


def dummy_texts(n_docs: int = 1000, seed: int = 0) -> Iterator[str]:
    """Synthetic documents for offline pipeline testing (no network)."""
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(256)]
    for _ in range(n_docs):
        n = rng.randint(20, 200)
        yield " ".join(rng.choice(vocab) for _ in range(n))


def download_dolma_slice(out_dir: Path, max_docs: int) -> Path:
    """Stream a Dolma slice to `out_dir` as line-delimited text shards.

    Implemented against `datasets`/HuggingFace at run time on a networked host.
    """
    raise NotImplementedError(
        "Wire to AI2 Dolma via `datasets` on a networked host; check first for a "
        "pre-tokenized subset to skip tokenize.py. Use --dummy for offline testing."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--max-docs", type=int, default=10000)
    ap.add_argument("--dummy", action="store_true", help="synthetic text, no network")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dummy:
        path = args.out / "dummy.txt"
        with open(path, "w") as f:
            for doc in dummy_texts(args.max_docs):
                f.write(doc + "\n")
        print(f"wrote synthetic text -> {path}")
    else:
        download_dolma_slice(args.out, args.max_docs)


if __name__ == "__main__":
    main()
