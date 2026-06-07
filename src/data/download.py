"""Pull a small slice of FineWeb-Edu (or generate synthetic text for offline testing).

Target for the POC run: ~2-5B tokens total (~10-20GB raw text; packed files are
several GB on their own — plan disk accordingly). For the smoke test a few million
tokens is plenty.

Corpus (issue #4): ``HuggingFaceFW/fineweb-edu`` (ODC-By), using a ready sample
subset (e.g. ``sample-10BT``) streamed via `datasets` to a single line-delimited
text file (one normalized document per line). NOTE: there is no
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


def _normalize_doc(text: str) -> str:
    """Collapse all whitespace (incl. internal newlines) so each doc is one line.

    The pipeline is one-doc-per-line and `tokenize.py` appends EOS *per line*; a raw
    document's internal newlines would otherwise inject spurious EOS boundaries
    mid-document. Collapsing runs of whitespace to single spaces is lossy on layout
    but correct for the perplexity-curve POC.
    """
    return " ".join(text.split())


def download_fineweb_edu_slice(
    out_dir: Path, max_docs: int, subset: str = "sample-10BT"
) -> Path:
    """Stream a FineWeb-Edu slice to a single line-delimited text file. Returns its path.

    Streams `HuggingFaceFW/fineweb-edu` (the `subset` config, default `sample-10BT`)
    via `datasets`, writing each document's normalized `text` field as one line to
    `out_dir / "fineweb-edu.txt"`. No compatible pre-tokenized small-vocab subset
    exists, so the raw text feeds `tokenize.py` (OLMo tokenizer). Stops after
    `max_docs` non-empty documents.
    """
    from datasets import load_dataset  # imported lazily (optional `data` extra)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fineweb-edu.txt"
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True
    )
    written = 0
    with open(path, "w") as f:
        for ex in ds:
            doc = _normalize_doc(ex.get("text", ""))
            if not doc:
                continue
            f.write(doc + "\n")
            written += 1
            if written % 100_000 == 0:
                print(f"  ... {written} docs")
            if written >= max_docs:
                break
    print(f"wrote {written} docs -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--max-docs", type=int, default=10000)
    ap.add_argument("--subset", default="sample-10BT", help="FineWeb-Edu config name")
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
        download_fineweb_edu_slice(args.out, args.max_docs, args.subset)


if __name__ == "__main__":
    main()
