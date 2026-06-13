"""Pull a slice of an English corpus (or generate synthetic text for offline testing).

Sources (`--source`):

  * ``fineweb``   — ``HuggingFaceFW/fineweb-edu`` (ODC-By), the original web-text path.
  * ``wikipedia`` — ``wikimedia/structured-wikipedia`` (``enwiki_namespace_0``,
    CC BY-SA 4.0): clean encyclopedic English. Records are structured; we extract
    ``name`` + ``abstract`` + section prose into one normalized line per article.
  * ``instruct``  — ``databricks/databricks-dolly-15k`` (CC BY-SA 3.0): human
    prompt/response pairs, formatted with the shared instruction template
    (``src.data.instruct_format``) so the model learns a prompt->response shape. Tiny
    relative to the pretraining corpus, so ``--repeat`` oversamples it.

All sources write one normalized document per line (``tokenize.py`` appends EOS per
line). The text extractors are pure functions over an injected record iterable, so the
parsing logic is unit-testable without any network or `datasets` dependency.

Network access in some environments is restricted; ``--dummy`` produces synthetic
text so the rest of the pipeline can be exercised without a download.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

from .instruct_format import format_example


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


# --- FineWeb-Edu (web text) ---------------------------------------------------------

def download_fineweb_edu_slice(
    out: Path, max_docs: int, subset: str = "sample-10BT"
) -> Path:
    """Stream a FineWeb-Edu slice to a single line-delimited text file. Returns its path."""
    from datasets import load_dataset  # imported lazily (optional `data` extra)

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True
    )
    texts = (_normalize_doc(ex.get("text", "")) for ex in ds)
    return _write_lines(texts, out, max_docs)


# --- Wikipedia (structured) ---------------------------------------------------------

def _walk_section_prose(node: object) -> Iterator[str]:
    """Yield paragraph text from a structured-wikipedia section tree.

    Records store ``sections`` as a tree of nodes; prose lives in nodes typed
    ``"paragraph"`` with a string ``value``. We recurse ``has_parts`` and skip
    everything else (tables, references, infoboxes, images) by simply not collecting
    their values. Tolerant of schema drift: any unrecognized node is just recursed.
    """
    if isinstance(node, dict):
        if node.get("type") == "paragraph" and isinstance(node.get("value"), str):
            yield node["value"]
        parts = node.get("has_parts")
        if isinstance(parts, (list, tuple)):
            for part in parts:
                yield from _walk_section_prose(part)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_section_prose(item)


def wikipedia_doc_text(record: dict) -> str:
    """Extract one normalized prose line from a structured-wikipedia record.

    Uses ``name`` (title) + ``abstract`` (clean lead summary) + section paragraph
    prose. ``sections`` is a JSON-encoded string on the real dataset; we accept either
    a string (decoded here) or an already-parsed list (for tests). Falls back to title
    + abstract alone when section parsing yields nothing or errors.
    """
    title = (record.get("name") or "").strip()
    abstract = (record.get("abstract") or "").strip()

    sections = record.get("sections")
    if isinstance(sections, str):
        try:
            sections = json.loads(sections)
        except (ValueError, TypeError):
            sections = None
    body = " ".join(_walk_section_prose(sections)) if sections is not None else ""

    return _normalize_doc(" ".join(p for p in (title, abstract, body) if p))


def iter_wikipedia_texts(records: Iterable[dict]) -> Iterator[str]:
    """Yield normalized non-empty prose lines from structured-wikipedia records."""
    for rec in records:
        doc = wikipedia_doc_text(rec)
        if doc:
            yield doc


def download_wikipedia_slice(out: Path, max_docs: int) -> Path:
    """Stream English structured-wikipedia prose to a line-delimited text file."""
    from datasets import load_dataset  # imported lazily (optional `data` extra)

    ds = load_dataset(
        "wikimedia/structured-wikipedia", "enwiki_namespace_0",
        split="train", streaming=True,
    )
    return _write_lines(iter_wikipedia_texts(ds), out, max_docs)


# --- Dolly (instruction pairs) ------------------------------------------------------

def iter_instruct_texts(records: Iterable[dict], repeat: int = 1) -> Iterator[str]:
    """Yield template-formatted instruction docs, each repeated `repeat` times.

    Dolly fields: ``instruction``, ``response``, optional ``context``. Repeating
    oversamples the tiny instruction set so its format is actually learned; emitting
    the copies adjacently is fine since the loader chunk-shuffles at train time.
    """
    if repeat < 1:
        raise ValueError(f"repeat={repeat} must be >= 1")
    for rec in records:
        instruction = (rec.get("instruction") or "").strip()
        response = (rec.get("response") or "").strip()
        if not instruction or not response:
            continue
        doc = _normalize_doc(
            format_example(instruction, response, rec.get("context") or "")
        )
        for _ in range(repeat):
            yield doc


def download_instruct(out: Path, repeat: int = 1) -> Path:
    """Stream Dolly-15k, formatted with the instruction template, to a text file."""
    from datasets import load_dataset  # imported lazily (optional `data` extra)

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    # No doc cap: the full 15k (times `repeat`) is the point.
    return _write_lines(iter_instruct_texts(ds, repeat), out, max_docs=None)


# --- shared writer ------------------------------------------------------------------

def _write_lines(texts: Iterable[str], out: Path, max_docs: int | None) -> Path:
    """Write non-empty normalized lines to `out`, stopping after `max_docs` (or all).

    Explicit UTF-8 + "\\n" newlines: the corpus is UTF-8, and forcing the line
    terminator avoids Windows \\r\\n translation that downstream tokenization would
    otherwise mis-strip.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for doc in texts:
            if not doc:
                continue
            f.write(doc + "\n")
            written += 1
            if written % 100_000 == 0:
                print(f"  ... {written} docs")
            if max_docs is not None and written >= max_docs:
                break
    print(f"wrote {written} docs -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=("fineweb", "wikipedia", "instruct"),
                    default="fineweb", help="corpus source (ignored with --dummy)")
    ap.add_argument("--out", type=Path, default=Path("data/raw"),
                    help="output file (or directory for --dummy/legacy default)")
    ap.add_argument("--max-docs", type=int, default=10000,
                    help="cap documents (ignored for --source instruct)")
    ap.add_argument("--subset", default="sample-10BT", help="FineWeb-Edu config name")
    ap.add_argument("--repeat", type=int, default=1,
                    help="oversample factor for --source instruct")
    ap.add_argument("--dummy", action="store_true", help="synthetic text, no network")
    args = ap.parse_args()

    if args.dummy:
        out = args.out
        if out.is_dir() or out.suffix == "":   # --out may be a dir (legacy) or a file
            out = out / "dummy.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            for doc in dummy_texts(args.max_docs):
                f.write(doc + "\n")
        print(f"wrote synthetic text -> {out}")
        return

    # If --out is an existing dir (or the legacy default), pick a source-named file in it.
    out = args.out
    if out.is_dir() or out.suffix == "":
        out = out / {"fineweb": "fineweb-edu.txt", "wikipedia": "wiki.txt",
                     "instruct": "instruct.txt"}[args.source]

    if args.source == "fineweb":
        download_fineweb_edu_slice(out, args.max_docs, args.subset)
    elif args.source == "wikipedia":
        download_wikipedia_slice(out, args.max_docs)
    else:
        download_instruct(out, args.repeat)


if __name__ == "__main__":
    main()
