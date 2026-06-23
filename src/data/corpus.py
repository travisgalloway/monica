"""Corpus pipeline skeleton (#69): the staged flow that turns raw sources into cleaned,
re-mixable text shards — runnable locally on the Mac before any cloud exists.

This is the laptop-scale, pyarrow-native version of the datatrove pipeline in
docs/design/08-corpus-pipeline.md (Stage 2, Normalize). Every source maps into one
COMMON SCHEMA record `{text, source, lang, license, meta}`; the flow is
ingest -> normalize -> filter -> write Parquet shards. At scale (#80) the same shape
runs on HF `datatrove` with the writer pointed at R2 via `s3fs` — the `out_uri` seam
below (an fsspec URI) is exactly where that swap happens: `file://` now, `s3://` later.

Under the distillation-first plan (docs/design/10-distillation.md) this corpus builds the
**teacher corpus + the production-reserve from-scratch data (#75)**, not the distillation
student's training data — the student consumes pre-tokenized Qwen3 artifacts + teacher
top-k logits (#92/#94), which are precomputed once, not re-derived through these stages.

ABOVE THE SEAM — no `mlx`/`torch`. Heavy data deps (pyarrow, fsspec) are imported
LAZILY inside the IO functions, mirroring download.py/tokenize.py, so importing this
module stays cheap and the seam guard (tests/test_import_guard.py) needs nothing extra.

CLI (mirrors download/tokenize/pack/split):
    python -m src.data.corpus --source dummy --in data/raw/dummy.txt --out data/cleaned
The Parquet shards it writes feed the existing tokenize/pack/split stages:
    python -m src.data.tokenize --in data/cleaned --out data/packed.bin --byte-fallback
    python -m src.data.split    --packed data/packed.bin --out data/split --val-tokens 2000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List

from .download import _normalize_doc, dummy_texts

#: The common-schema columns, in Parquet column order.
RECORD_FIELDS = ("text", "source", "lang", "license", "meta")


@dataclass
class Record:
    """One document in the common corpus schema. `meta` is free-form per-source
    metadata, stored as a JSON-string column in Parquet so the table schema is stable
    across sources."""

    text: str
    source: str
    lang: str = "en"
    license: str = "unknown"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "lang": self.lang,
                "license": self.license, "meta": dict(self.meta)}


# --------------------------------------------------------------------------- #
# Stages: ingest -> normalize -> filter
# --------------------------------------------------------------------------- #
def ingest_text_file(path, source: str, lang: str = "en",
                     license: str = "unknown") -> Iterator[Record]:
    """One raw Record per line of a UTF-8 text file (normalize() cleans it)."""
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            yield Record(text=line.rstrip("\r\n"), source=source, lang=lang, license=license)


def ingest_dummy(n_docs: int = 1000, seed: int = 0,
                 source: str = "dummy") -> Iterator[Record]:
    """Synthetic Records for offline testing (no network), via download.dummy_texts."""
    for text in dummy_texts(n_docs, seed):
        yield Record(text=text, source=source, lang="en", license="synthetic")


def normalize(records: Iterable[Record]) -> Iterator[Record]:
    """Collapse each doc's whitespace to one line (reuses download._normalize_doc) and
    drop docs that normalize to empty — the Stage-2 contract."""
    for r in records:
        text = _normalize_doc(r.text)
        if text:
            yield Record(text=text, source=r.source, lang=r.lang,
                         license=r.license, meta=r.meta)


def filter_records(records: Iterable[Record], min_chars: int = 1, *,
                   quality: bool = False, license_filter: bool = False,
                   drop_minified: bool = False, drop_autogen: bool = False,
                   scrub: bool = False, stats=None) -> Iterator[Record]:
    """Stage-3 filters (#72). With only `min_chars` set this is the original length
    heuristic; the keyword flags enable the real text-quality / permissive-license /
    minified-autogen / secret-scrub filters (see `src.data.filters`). Default-off keeps
    the #69 local-gate path byte-for-byte unchanged."""
    from .filters import filter_records as _filter
    yield from _filter(records, min_chars=min_chars, quality=quality,
                       license_filter=license_filter, drop_minified=drop_minified,
                       drop_autogen=drop_autogen, scrub=scrub, stats=stats)


# --------------------------------------------------------------------------- #
# Local sharded Parquet IO (the durable, re-mixable artifact)
# --------------------------------------------------------------------------- #
def _table_from_records(batch: List[Record]):
    import pyarrow as pa
    return pa.table({
        "text": [r.text for r in batch],
        "source": [r.source for r in batch],
        "lang": [r.lang for r in batch],
        "license": [r.license for r in batch],
        "meta": [json.dumps(r.meta, separators=(",", ":"), sort_keys=True) for r in batch],
    })


def write_shards(records: Iterable[Record], out_uri, shard_size_mb: int = 128,
                 prefix: str = "part", compression: str = "zstd") -> List[str]:
    """Write Records as Parquet shards under `out_uri` (an fsspec URI: a local path /
    `file://` now, `s3://` later via s3fs at #80 — same code path). Rolls a new shard
    once the buffered text passes `shard_size_mb`, so output is FEW LARGE files (the R2
    Class-A-ops constraint). Returns the shard paths written, in order.
    """
    import fsspec
    import pyarrow.parquet as pq

    fs, root = fsspec.core.url_to_fs(str(out_uri))
    root = root.rstrip("/")
    fs.makedirs(root, exist_ok=True)
    budget = shard_size_mb * (1 << 20)
    written: List[str] = []
    buf: List[Record] = []
    nbytes = 0
    idx = 0

    def flush() -> None:
        nonlocal buf, nbytes, idx
        if not buf:
            return
        shard = f"{root}/{prefix}-{idx:05d}.parquet"
        with fs.open(shard, "wb") as fo:
            pq.write_table(_table_from_records(buf), fo, compression=compression)
        written.append(shard)
        idx += 1
        buf, nbytes = [], 0

    for r in records:
        buf.append(r)
        nbytes += len(r.text.encode("utf-8"))
        if nbytes >= budget:
            flush()
    flush()
    return written


def _shard_paths(uri) -> List[str]:
    import fsspec
    fs, root = fsspec.core.url_to_fs(str(uri))
    if fs.isdir(root):
        return sorted(fs.glob(f"{root.rstrip('/')}/*.parquet"))
    return [root]


def read_shards(uri) -> Iterator[Record]:
    """Read Records back from the Parquet shard(s) at `uri` (reverses write_shards)."""
    import fsspec
    import pyarrow.parquet as pq

    fs, _ = fsspec.core.url_to_fs(str(uri))
    for shard in _shard_paths(uri):
        with fs.open(shard, "rb") as fo:
            tbl = pq.read_table(fo)
        d = tbl.to_pydict()
        for i in range(tbl.num_rows):
            yield Record(text=d["text"][i], source=d["source"][i], lang=d["lang"][i],
                         license=d["license"][i], meta=json.loads(d["meta"][i] or "{}"))


def iter_shard_texts(uri) -> Iterator[str]:
    """Just the `text` column from the shard(s) at `uri` — this is what feeds tokenize."""
    for r in read_shards(uri):
        yield r.text


def iter_jsonl_texts(uri) -> Iterator[str]:
    """The `text` field from JSON-lines shard(s) at `uri` — the format the datatrove clean pass
    (`datatrove_pipeline.py`, #80) writes (`*.jsonl` / gzipped `*.jsonl.gz`). Lets the existing
    tokenize/pack stages consume datatrove output. `uri` is an fsspec URI (local or `s3://`)."""
    import gzip
    import io

    import fsspec

    fs, root = fsspec.core.url_to_fs(str(uri))
    files = (sorted(fs.glob(f"{root.rstrip('/')}/*.jsonl*")) if fs.isdir(root) else [root])
    for fp in files:
        with fs.open(fp, "rb") as raw:
            stream = (gzip.open(raw, "rt", encoding="utf-8") if str(fp).endswith(".gz")
                      else io.TextIOWrapper(raw, encoding="utf-8"))
            try:                                   # close explicitly so gzip CRC/EOF is validated
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    text = json.loads(line).get("text")
                    if text:                       # skip missing/empty text (no empty docs injected)
                        yield text
            finally:
                stream.close()


def has_jsonl_shards(uri) -> bool:
    """True if `uri` is a directory of JSONL shards (datatrove output) and not Parquet."""
    import fsspec

    fs, root = fsspec.core.url_to_fs(str(uri))
    if not fs.isdir(root):
        return str(root).endswith((".jsonl", ".jsonl.gz"))
    root = root.rstrip("/")
    return bool(fs.glob(f"{root}/*.jsonl*")) and not fs.glob(f"{root}/*.parquet")


# --------------------------------------------------------------------------- #
# Orchestrator + CLI
# --------------------------------------------------------------------------- #
def build_corpus(records: Iterable[Record], out_uri, *, min_chars: int = 1,
                 quality: bool = False, license_filter: bool = False,
                 drop_minified: bool = False, drop_autogen: bool = False,
                 scrub: bool = False, dedup: str | None = None, near_threshold: float = 0.8,
                 decontaminator=None, shard_size_mb: int = 128,
                 stats=None, dedup_stats=None) -> List[str]:
    """The Stage 2–5 flow end to end: normalize -> filter -> (dedup) -> (decontaminate) ->
    write Parquet shards. Returns the shard paths written; pass a `filters.FilterStats` as
    `stats` and a `dedup.DedupStats` as `dedup_stats` to collect drop counts. All Stage 3–5
    work is opt-in (the #69 local gate is unchanged).

    `dedup`: None | "exact" | "minhash" ("minhash" runs exact first, then MinHash-LSH).
    `decontaminator`: an optional `dedup.Decontaminator` (benchmark n-gram stripping)."""
    cleaned = filter_records(normalize(records), min_chars=min_chars, quality=quality,
                             license_filter=license_filter, drop_minified=drop_minified,
                             drop_autogen=drop_autogen, scrub=scrub, stats=stats)
    if dedup in ("exact", "minhash"):
        from .dedup import exact_dedup, near_dedup
        cleaned = exact_dedup(cleaned, stats=dedup_stats)
        if dedup == "minhash":
            cleaned = near_dedup(cleaned, threshold=near_threshold, stats=dedup_stats)
    elif dedup is not None:
        raise ValueError(f"unknown dedup mode: {dedup!r} (use None/'exact'/'minhash')")
    if decontaminator is not None:
        from .dedup import decontaminate
        cleaned = decontaminate(cleaned, decontaminator, stats=dedup_stats)
    return write_shards(cleaned, out_uri, shard_size_mb=shard_size_mb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("dummy", "text"), default="text",
                    help="dummy: synthetic offline docs; text: a UTF-8 text file (--in)")
    ap.add_argument("--in", dest="inp", default=None, help="input text file (--source text)")
    ap.add_argument("--out", required=True, help="output dir / fsspec URI for Parquet shards")
    ap.add_argument("--source-name", default=None,
                    help="value for the schema `source` field (default: source type / filename)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--license", default="unknown")
    ap.add_argument("--max-docs", type=int, default=1000, help="--source dummy: #synthetic docs")
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--quality", action="store_true",
                    help="Stage-3 text-quality heuristics (drop symbol soup / boilerplate)")
    ap.add_argument("--license-filter", action="store_true",
                    help="permissive-license-only gate on code records")
    ap.add_argument("--drop-minified", action="store_true", help="drop minified code")
    ap.add_argument("--drop-autogen", action="store_true", help="drop autogenerated code")
    ap.add_argument("--scrub", action="store_true", help="scrub secrets/PII to placeholders")
    ap.add_argument("--dedup", choices=("exact", "minhash"), default=None,
                    help="Stage-4 dedup: exact-hash, or exact+MinHash-LSH near-dedup")
    ap.add_argument("--near-threshold", type=float, default=0.8,
                    help="MinHash-LSH Jaccard threshold for near-dedup")
    ap.add_argument("--decontam-file", default=None,
                    help="Stage-5 decontamination: text file of eval-benchmark lines to "
                         "strip (13-gram + 7-gram overlap)")
    ap.add_argument("--shard-size-mb", type=int, default=128,
                    help="roll a new shard past this size (few large shards)")
    args = ap.parse_args()

    if args.source == "dummy":
        records = ingest_dummy(args.max_docs, source=args.source_name or "dummy")
    else:
        if not args.inp:
            ap.error("--in is required for --source text")
        name = args.source_name or Path(args.inp).stem
        records = ingest_text_file(args.inp, source=name, lang=args.lang, license=args.license)

    from .filters import FilterStats
    stats = FilterStats()
    decon = None
    dedup_stats = None
    if args.decontam_file:
        from .dedup import Decontaminator, DedupStats
        with open(args.decontam_file, encoding="utf-8") as f:
            decon = Decontaminator.from_texts(f)
        dedup_stats = DedupStats()
    if args.dedup and dedup_stats is None:
        from .dedup import DedupStats
        dedup_stats = DedupStats()
    shards = build_corpus(records, args.out, min_chars=args.min_chars,
                          quality=args.quality, license_filter=args.license_filter,
                          drop_minified=args.drop_minified, drop_autogen=args.drop_autogen,
                          scrub=args.scrub, dedup=args.dedup, near_threshold=args.near_threshold,
                          decontaminator=decon, shard_size_mb=args.shard_size_mb,
                          stats=stats, dedup_stats=dedup_stats)
    extra = f"  dedup={dedup_stats.as_dict()}" if dedup_stats else ""
    print(f"wrote {len(shards)} shard(s) -> {args.out}  filter={stats.as_dict()}{extra}")


if __name__ == "__main__":
    main()
