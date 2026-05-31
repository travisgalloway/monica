"""Pull a small slice of FineWeb-Edu (or generate synthetic text for offline testing).

Target for the POC run: ~2-5B tokens total (~10-20GB raw text; packed files are
several GB on their own — plan disk accordingly). For the smoke test a few million
tokens is plenty.

Corpus (issue #4): ``HuggingFaceFW/fineweb-edu`` (ODC-By), using a ready sample
subset (e.g. ``sample-10BT``) streamed via `datasets`. NOTE: there is no
compatibly-licensed pre-tokenized small-vocab (< 65536) subset on the HF Hub — AI2's
dolma3 mixes are raw text at the OLMo-2 100278 vocab, and third-party tokenized sets
use Llama/Pile tokenizers — so we tokenize raw text ourselves via `tokenize.py`.

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


def download_fineweb_edu_slice(out_dir: Path, max_docs: int) -> Path:
    """Stream a FineWeb-Edu slice to `out_dir` as line-delimited text shards.

    Implemented against `datasets`/HuggingFace at run time on a networked host (#10):
    stream `HuggingFaceFW/fineweb-edu` (e.g. the `sample-10BT` subset) and write the
    `text` field line by line. No compatible pre-tokenized subset exists, so the raw
    text feeds `tokenize.py`.
    """
    raise NotImplementedError(
        "Wire to HuggingFaceFW/fineweb-edu (sample-10BT) via `datasets` on a networked "
        "host (#10), writing the `text` field as line-delimited shards. Use --dummy for "
        "offline testing."
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
        download_fineweb_edu_slice(args.out, args.max_docs)


if __name__ == "__main__":
    main()
