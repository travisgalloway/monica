"""Stage 1: stream permissively-licensed TypeScript from Stack v2 / Software Heritage (#193).

The Stack v2 (``bigcode/the-stack-v2-dedup``) HF dataset carries METADATA rows only — no
file content, by design (bigcode cannot redistribute the raw code). Content lives in
**Software Heritage**'s public S3 bucket, keyed by ``blob_id``, and must be fetched +
gzip-decompressed + decoded separately — the recipe on the dataset card
(https://huggingface.co/datasets/bigcode/the-stack-v2). ``resolve_swh_s3()`` +
``download_contents()`` implement that recipe; ``iter_stack_v2_ts`` composes it with the
HF metadata stream (or an injected ``rows`` iterable for offline tests / CI) into the
common ``Record`` schema, gated by the same permissive-license rule the rest of the
corpus pipeline uses (``filters.license_ok``).

ABOVE THE SEAM — no ``mlx``/``torch``. ``boto3`` (the ``stack-v2`` extra) and ``datasets``
(the ``data`` extra) are imported LAZILY inside functions, so importing this module — and
the ``--from-jsonl`` offline path in ``scripts/build_ts_clean_corpus.py`` — never requires
either dependency to be installed (guarded by ``tests/test_import_guard.py``).
"""

from __future__ import annotations

import gzip
import os
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .corpus import Record
from .filters import license_ok, normalize_license

#: The public Software Heritage content bucket (unauthenticated reads are NOT possible —
#: SWH requires a signed request even for public objects, hence real AWS creds below).
SWH_BUCKET = "softwareheritage"


def resolve_swh_s3():
    """Return a boto3 S3 client for the Software Heritage bucket, or ``None`` if ``boto3``
    isn't installed or no AWS credentials are configured (graceful-skip idiom, mirrors
    ``src.lsp.tsc.resolve_tsc``: callers check for ``None`` and skip Stage 1 cleanly rather
    than crashing on a missing optional dependency / missing creds)."""
    try:
        import boto3
    except ImportError:
        return None
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        return None
    try:
        return boto3.client("s3")
    except Exception:
        return None


def download_contents(blob_id: str, src_encoding: str, *, s3_client,
                       compression: str = ".gz") -> str:
    """Fetch ``s3://softwareheritage/content/{blob_id}``, gzip-decompress the body
    (``compression``), and decode it with ``src_encoding`` — the bigcode Stack v2
    dataset-card recipe. ``s3_client`` is injected (a real ``boto3`` client, or a fake
    object whose ``get_object`` returns ``{"Body": <readable>}``) so the decompress/decode
    path is unit-testable with a synthetic gzip blob and no network access.
    """
    key = f"content/{blob_id}"
    obj = s3_client.get_object(Bucket=SWH_BUCKET, Key=key)
    body = obj["Body"].read()
    if compression == ".gz":
        body = gzip.decompress(body)
    elif compression not in (None, ""):
        raise ValueError(f"unsupported compression: {compression!r}")
    return body.decode(src_encoding)


def _row_to_record(row: Dict[str, Any], content: str) -> Record:
    """Map an HF Stack v2 metadata row + its resolved content to the common corpus schema.

    ``detected_licenses`` on Stack v2 is a list (a file can carry more than one detected
    license); we keep the first entry as the record's license — good enough for the
    permissive-only gate downstream, which only needs to know the file is *unambiguously*
    permissive to keep it."""
    licenses = row.get("detected_licenses") or row.get("license") or []
    if isinstance(licenses, str):
        license_value = licenses
    elif licenses:
        license_value = licenses[0]
    else:
        license_value = ""
    return Record(
        text=content,
        source="stack-v2",
        lang="typescript",
        license=normalize_license(license_value),
        meta={
            "is_code": True,
            "repo": row.get("repo_name"),
            "blob_id": row.get("blob_id"),
            "path": row.get("path"),
        },
    )


def iter_stack_v2_ts(*, limit: int = -1, streaming: bool = True, s3_client=None,
                      dataset: str = "bigcode/the-stack-v2-dedup", config: str = "TypeScript",
                      rows: Optional[Iterable[Dict[str, Any]]] = None) -> Iterator[Record]:
    """Stream permissively-licensed TypeScript ``Record``s from Stack v2.

    Resolves each metadata row's file content from Software Heritage via ``s3_client``
    (defaulting to ``resolve_swh_s3()`` when not given) and applies the corpus's
    permissive-license gate (``filters.license_ok``).

    When ``rows`` is given, it is iterated directly instead of streaming from HF — this is
    the **offline-test / CI path**: no ``datasets`` and no HF network access, so a caller
    can exercise the whole content-resolution + license-gate logic against a handful of
    fixture dicts (still needs an ``s3_client`` for content, real or fake). When ``rows``
    is omitted, ``datasets.load_dataset(dataset, config, split="train",
    streaming=streaming)`` supplies the metadata rows and a resolvable ``s3_client`` (real
    creds) is required.
    """
    if rows is None:
        from datasets import load_dataset

        rows = load_dataset(dataset, config, split="train", streaming=streaming)
        if s3_client is None:
            s3_client = resolve_swh_s3()
        if s3_client is None:
            raise RuntimeError(
                "no Software Heritage S3 client resolvable — set AWS_ACCESS_KEY_ID / "
                "AWS_SECRET_ACCESS_KEY (and install the `stack-v2` extra), or pass "
                "s3_client=/rows= explicitly for offline use")

    n = 0
    for row in rows:
        if limit >= 0 and n >= limit:
            break
        content = download_contents(row["blob_id"], row.get("src_encoding") or "utf-8",
                                     s3_client=s3_client)
        record = _row_to_record(row, content)
        if not license_ok(record):
            continue
        n += 1
        yield record
