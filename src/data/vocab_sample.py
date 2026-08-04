"""Build the tagged multilingual text sample the vocab-size sweep measures on (#251).

`DEFAULT_VOCAB_SIZE = 16384` was sized when #193 was scoped TypeScript-only; #198 rescoped
the corpus to general multilingual Essential-Web + Stack-v2. Re-deriving the vocab needs a
sample drawn from the *rescoped* distribution — prose plus several code languages, each
document tagged with its language so the sweep can report a per-language bytes/token
breakdown (the issue's acceptance criterion).

Three things here are not obvious and are load-bearing (all measured, see the #251 plan):

* **`datasets.load_dataset()` is unusable on Essential-Web** — the repo holds 99,896 parquet
  files and streaming init does not yield a first row in 7+ minutes. We open a *single*
  parquet file directly through the `hf://` filesystem instead (~2.5 s to open, ~8 MB/s).
  The same trick serves Stack-v2's metadata.
* **Stack-v2 carries metadata only** — content lives in Software Heritage's S3 bucket keyed
  by ``blob_id`` (see ``stack_v2.download_contents``). Serial fetching runs at 0.03 MB/s and
  is infeasible; a 64-thread pool reaches 1.9 MB/s, which is what makes this job finish.
* **``detected_licenses`` is empty on ~67% of Stack-v2 rows** and the corpus's permissive
  gate rejects an empty license for code. The license/vendor/generated/size filter therefore
  runs on the *metadata columns*, before any S3 GET — otherwise two thirds of the fetch is
  thrown away.

ABOVE THE SEAM — no ``mlx``/``torch``. ``pyarrow``, ``fsspec`` and ``boto3`` are imported
LAZILY inside functions (the ``stack_v2`` idiom), so importing this module needs none of them
(guarded by ``tests/test_import_guard.py``).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from .filters import is_permissive
from .stack_v2 import download_contents

#: HF repo paths. Opened file-by-file rather than through ``load_dataset`` (see module docstring).
ESSENTIAL_WEB_REPO = "datasets/EssentialAI/essential-web-v1.0"
ESSENTIAL_WEB_DIR = f"{ESSENTIAL_WEB_REPO}/data/crawl=CC-MAIN-2013-20"
STACK_V2_REPO = "datasets/bigcode/the-stack-v2-dedup"

#: Metadata pre-filter bounds. Below ``MIN_FILE_BYTES`` a file is mostly a stub/licence header
#: and contributes little vocabulary; above ``MAX_FILE_BYTES`` it is usually generated or data,
#: and one document would dominate a language's byte budget.
MIN_FILE_BYTES = 1024
MAX_FILE_BYTES = 131072


@dataclass(frozen=True)
class SliceSpec:
    """One language slice of the sample: where to read it and how many raw UTF-8 bytes to take.

    ``kind`` is ``"essential_web"`` (prose, public, read straight from parquet) or
    ``"stack_v2"`` (code metadata + a Software Heritage content fetch). ``source`` is the
    Stack-v2 language directory name for code slices, unused for prose.
    """

    lang: str
    kind: str
    target_bytes: int
    source: str = ""


#: The mixture. No blend weights are locked anywhere in the repo (checked doc 13 and #193), so
#: the sweep picks one and *records* it in the manifest. TypeScript is deliberately
#: over-weighted — the program is TypeScript-first. NOTE the per-language rows of the results
#: table are blend-independent (a language's bytes/token is computed only over its own
#: documents); only the "overall" row is blend-weighted.
DEFAULT_MIXTURE: List[SliceSpec] = [
    SliceSpec("en", "essential_web", 20 * 1024 * 1024),
    SliceSpec("typescript", "stack_v2", 8 * 1024 * 1024, "TypeScript"),
    SliceSpec("python", "stack_v2", 4 * 1024 * 1024, "Python"),
    SliceSpec("javascript", "stack_v2", 4 * 1024 * 1024, "JavaScript"),
    SliceSpec("java", "stack_v2", 3 * 1024 * 1024, "Java"),
    SliceSpec("go", "stack_v2", 2 * 1024 * 1024, "Go"),
    SliceSpec("rust", "stack_v2", 2 * 1024 * 1024, "Rust"),
    SliceSpec("cpp", "stack_v2", 2 * 1024 * 1024, "C++"),
    SliceSpec("markdown", "stack_v2", 2 * 1024 * 1024, "Markdown"),
]


def scale_mixture(mixture: Sequence[SliceSpec], scale: float) -> List[SliceSpec]:
    """Scale every slice's byte target by ``scale``, keeping every language.

    The escape hatch for a slow ingest: halving the targets still clears ~200k unique
    pre-tokens (4x the largest vocab swept), so the tail merges stay meaningful. Dropping
    *languages* would not be an acceptable trade — the per-language breakdown is the
    deliverable.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return [SliceSpec(s.lang, s.kind, max(1, int(s.target_bytes * scale)), s.source)
            for s in mixture]


# --------------------------------------------------------------------------- metadata filter


def metadata_ok(row: Dict[str, Any], *, min_bytes: int = MIN_FILE_BYTES,
                max_bytes: int = MAX_FILE_BYTES) -> bool:
    """Decide from Stack-v2 *metadata columns alone* whether a row is worth fetching.

    Applied before any S3 GET (finding 4 of the #251 plan: ~67% of rows have an empty
    ``detected_licenses`` and would be dropped by the permissive gate after being paid for).
    Skipping vendored/generated files matters specifically for a *vocab* measurement —
    minified bundles teach garbage merges and distort bytes/token upward.
    """
    licenses = row.get("detected_licenses") or []
    if isinstance(licenses, str):
        licenses = [licenses]
    if not licenses or not is_permissive(licenses[0]):
        return False
    if row.get("is_vendor") or row.get("is_generated"):
        return False
    n = row.get("length_bytes")
    if n is None or not (min_bytes <= n <= max_bytes):
        return False
    return bool(row.get("blob_id"))


# --------------------------------------------------------------------------- parquet readers


def _hf_parquet(path: str):
    """Open one HF-hosted parquet file directly (bypassing ``datasets``; see module docstring)."""
    import fsspec
    import pyarrow.parquet as pq

    return pq.ParquetFile(fsspec.filesystem("hf").open(path))


def _list_parquet(directory: str) -> List[str]:
    """Sorted parquet shard paths under an HF repo directory — a fixed, reproducible order."""
    import fsspec

    fs = fsspec.filesystem("hf")
    return sorted(p for p in fs.ls(directory, detail=False) if p.endswith(".parquet"))


def iter_essential_web(target_bytes: int, *, files: Optional[Sequence[str]] = None,
                       batch_rows: int = 1000,
                       open_parquet: Optional[Callable[[str], Any]] = None,
                       ) -> Iterator[str]:
    """Yield Essential-Web documents in parquet row order until ``target_bytes`` is reached.

    ``files`` defaults to the first shards of one crawl; ``open_parquet`` is injected by tests.
    Row order from a fixed starting file makes the sample reproducible.
    """
    opener = open_parquet or _hf_parquet
    paths = list(files) if files else [
        f"{ESSENTIAL_WEB_DIR}/train-{i:05d}-of-04233.parquet" for i in range(4)]
    total = 0
    for path in paths:
        if total >= target_bytes:
            return
        pf = opener(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if not text:
                    continue
                total += len(text.encode("utf-8"))
                yield text
                if total >= target_bytes:
                    return


def iter_stack_v2_metadata(lang_dir: str, *, batch_rows: int = 2000,
                           files: Optional[Sequence[str]] = None,
                           open_parquet: Optional[Callable[[str], Any]] = None,
                           list_parquet: Optional[Callable[[str], List[str]]] = None,
                           ) -> Iterator[Dict[str, Any]]:
    """Yield metadata rows for one Stack-v2 language that pass ``metadata_ok``, in row order."""
    opener = open_parquet or _hf_parquet
    lister = list_parquet or _list_parquet
    paths = list(files) if files else lister(f"{STACK_V2_REPO}/data/{lang_dir}")
    columns = ["blob_id", "src_encoding", "detected_licenses", "length_bytes",
               "is_vendor", "is_generated", "path", "repo_name"]
    for path in paths:
        pf = opener(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            for row in batch.to_pylist():
                if metadata_ok(row):
                    row["_parquet"] = path
                    yield row


# --------------------------------------------------------------------------- threaded fetch


@dataclass
class FetchStats:
    """Per-slice fetch accounting, recorded in the manifest so the sample is auditable."""

    fetched: int = 0
    failed: int = 0
    bytes: int = 0
    parquet_files: List[str] = field(default_factory=list)


def fetch_contents(rows: Iterable[Dict[str, Any]], target_bytes: int, *, s3_client,
                   threads: int = 64, stats: Optional[FetchStats] = None,
                   chunk: int = 512) -> Iterator[str]:
    """Fetch Software Heritage content for ``rows`` concurrently until ``target_bytes``.

    Serial fetching measures 0.03 MB/s against SWH — the pool is not an optimization, it is
    what makes the job finish (1.9 MB/s at 64 threads). Rows are submitted in chunks so a
    long metadata stream is not materialized, and results are consumed in submission order so
    the sample stays deterministic given the same metadata order. Decode/fetch failures are
    counted and skipped, never fatal: a handful of unreadable blobs must not lose the run.
    """
    st = stats if stats is not None else FetchStats()
    if target_bytes <= 0:
        return
    seen_files: set = set()

    def _one(row: Dict[str, Any]) -> Optional[str]:
        try:
            return download_contents(row["blob_id"], row.get("src_encoding") or "utf-8",
                                     s3_client=s3_client)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=threads) as pool:

        def _drain(batch: List[Dict[str, Any]]) -> Iterator[str]:
            """Fetch one chunk in parallel, yielding in submission order (determinism).

            Rows are submitted as individual futures (not ``pool.map``, which blocks until
            every future in the chunk is done) so that once ``target_bytes`` is crossed
            mid-chunk, futures that haven't started running yet can be cancelled instead of
            paying for the rest of the chunk (up to ``chunk`` wasted S3 fetches otherwise).
            """
            futures = [pool.submit(_one, row) for row in batch]
            for future, src in zip(futures, batch):
                text = future.result()
                path = src.get("_parquet")
                if path and path not in seen_files:
                    seen_files.add(path)
                    st.parquet_files.append(path)
                if text is None:
                    st.failed += 1
                    continue
                st.fetched += 1
                st.bytes += len(text.encode("utf-8"))
                yield text
                if st.bytes >= target_bytes:
                    for f in futures:
                        f.cancel()
                    return

        batch: List[Dict[str, Any]] = []
        for row in rows:
            batch.append(row)
            if len(batch) < chunk:
                continue
            yield from _drain(batch)
            batch = []
            if st.bytes >= target_bytes:
                return
        if batch:
            yield from _drain(batch)


# --------------------------------------------------------------------------- sample assembly


def _manifest_matches(manifest: Dict[str, Any], mixture: Sequence[SliceSpec]) -> bool:
    """True when a cached manifest was built from exactly this mixture.

    Compares ``(kind, source, target_bytes)`` per language, not just the byte target —
    two slices can share a language tag and byte budget while reading different sources
    (e.g. a `stack_v2` slice re-tagged from a different language directory, or a slice
    whose `kind` changed between `essential_web` and `stack_v2`), and reusing that cache
    would silently invalidate a sweep rerun. ``build_sample`` already writes `kind` and
    `source` into each manifest slice, so no manifest format change is needed here.
    """
    want = {s.lang: (s.kind, s.source, s.target_bytes) for s in mixture}
    got = {k: (v.get("kind"), v.get("source"), v.get("target_bytes"))
           for k, v in (manifest.get("slices") or {}).items()}
    return want == got


def build_sample(work_dir: Path, *, mixture: Sequence[SliceSpec] = tuple(DEFAULT_MIXTURE),
                 threads: int = 64, s3_client=None, rebuild: bool = False,
                 log: Callable[[str], None] = print) -> Path:
    """Build (or reuse) ``<work_dir>/sample/sample.jsonl`` — ``{"text", "lang"}`` rows.

    Reuse is the important resilience property: a re-run of the sweep after a `stats` bug
    costs zero network time. A cached sample is reused whenever its manifest records the same
    mixture, unless ``rebuild``.
    """
    sample_dir = Path(work_dir) / "sample"
    out_path = sample_dir / "sample.jsonl"
    manifest_path = sample_dir / "manifest.json"
    if not rebuild and out_path.exists() and manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            cached = {}
        if _manifest_matches(cached, mixture):
            log(f"reusing cached sample: {out_path} "
                f"({cached.get('total_bytes', 0) / 1e6:.1f} MB, "
                f"{cached.get('total_docs', 0)} docs)")
            return out_path

    sample_dir.mkdir(parents=True, exist_ok=True)
    slices: Dict[str, Any] = {}
    total_docs = 0
    total_bytes = 0
    started = time.time()

    with out_path.open("w", encoding="utf-8") as fh:
        for spec in mixture:
            t0 = time.time()
            stats = FetchStats()
            docs = 0
            written = 0
            if spec.kind == "essential_web":
                texts: Iterable[str] = iter_essential_web(spec.target_bytes)
            elif spec.kind == "stack_v2":
                if s3_client is None:
                    raise RuntimeError(
                        "a Software Heritage S3 client is required for stack_v2 slices — "
                        "run with AWS_PROFILE=monica-swh, or pass s3_client=")
                texts = fetch_contents(iter_stack_v2_metadata(spec.source), spec.target_bytes,
                                       s3_client=s3_client, threads=threads, stats=stats)
            else:
                raise ValueError(f"unknown slice kind {spec.kind!r}")
            for text in texts:
                fh.write(json.dumps({"text": text, "lang": spec.lang},
                                    ensure_ascii=False) + "\n")
                docs += 1
                written += len(text.encode("utf-8"))
            slices[spec.lang] = {
                "kind": spec.kind,
                "source": spec.source,
                "target_bytes": spec.target_bytes,
                "bytes": written,
                "docs": docs,
                "fetch_failed": stats.failed,
                "parquet_files": stats.parquet_files,
                "seconds": round(time.time() - t0, 1),
            }
            total_docs += docs
            total_bytes += written
            log(f"  {spec.lang:<11} {written / 1e6:6.2f} MB  {docs:6d} docs  "
                f"{time.time() - t0:6.1f}s"
                + (f"  ({stats.failed} fetch failures)" if stats.failed else ""))

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.time() - started, 1),
        "threads": threads,
        "filters": {"min_file_bytes": MIN_FILE_BYTES, "max_file_bytes": MAX_FILE_BYTES,
                    "permissive_license_required": True,
                    "skip_vendor": True, "skip_generated": True},
        "total_docs": total_docs,
        "total_bytes": total_bytes,
        "slices": slices,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"sample: {total_bytes / 1e6:.1f} MB, {total_docs} docs -> {out_path}")
    return out_path


def resolve_s3():
    """A pooled boto3 S3 client sized for the fetch pool, or ``None`` when unavailable.

    Distinct from ``stack_v2.resolve_swh_s3`` (which builds an unpooled default client and
    requires the key env vars): here the credentials usually come from ``AWS_PROFILE``, and
    the connection pool must be at least as large as the thread pool or boto3 serializes the
    requests it was given threads to parallelize.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        return None
    try:
        return boto3.client("s3", config=Config(max_pool_connections=96,
                                                retries={"max_attempts": 5,
                                                         "mode": "adaptive"}))
    except Exception:
        return None
