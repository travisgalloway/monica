"""Offline tests for the Stack v2 / Software Heritage reader (#193 Stage 1).

Everything here is driven with injected fixtures (a fake S3 client, an injected `rows`
iterable) -- no `boto3`, no `datasets`, no network. Mirrors `tests/test_download_sources.py`.
"""

from __future__ import annotations

import gzip
import io

from src.data.corpus import Record
from src.data.stack_v2 import _row_to_record, download_contents, iter_stack_v2_ts


class _FakeBody:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self) -> bytes:
        return self._buf.read()


class _FakeS3Client:
    """Serves canned gzip blobs for `blob_id`s registered in `blobs`."""

    def __init__(self, blobs: dict):
        self.blobs = blobs
        self.calls = []

    def get_object(self, Bucket: str, Key: str):
        self.calls.append((Bucket, Key))
        blob_id = Key.split("/")[-1]
        return {"Body": _FakeBody(self.blobs[blob_id])}


def _gzip_blob(text: str) -> bytes:
    return gzip.compress(text.encode("utf-8"))


# --- download_contents (decode path, no boto3/network) -------------------------------
def test_download_contents_decodes_gzip_blob():
    src = "export const x: number = 1;\n"
    client = _FakeS3Client({"abc123": _gzip_blob(src)})
    out = download_contents("abc123", "utf-8", s3_client=client)
    assert out == src
    assert client.calls == [("softwareheritage", "content/abc123")]


def test_download_contents_respects_src_encoding():
    src = "const s = 'café';"
    blob = gzip.compress(src.encode("latin-1"))
    client = _FakeS3Client({"blob2": blob})
    out = download_contents("blob2", "latin-1", s3_client=client)
    assert out == src


def test_download_contents_no_compression():
    src = "const y = 2;"
    client = _FakeS3Client({"raw1": src.encode("utf-8")})
    out = download_contents("raw1", "utf-8", s3_client=client, compression="")
    assert out == src


# --- _row_to_record (metadata row -> common schema) -----------------------------------
def test_row_to_record_maps_fields():
    row = {
        "blob_id": "abc123",
        "path": "src/index.ts",
        "repo_name": "octocat/hello-world",
        "detected_licenses": ["MIT"],
    }
    rec = _row_to_record(row, "export const x = 1;")
    assert isinstance(rec, Record)
    assert rec.text == "export const x = 1;"
    assert rec.source == "stack-v2"
    assert rec.lang == "typescript"
    assert rec.license == "mit"
    assert rec.meta == {"is_code": True, "repo": "octocat/hello-world",
                        "blob_id": "abc123", "path": "src/index.ts"}


def test_row_to_record_handles_missing_license():
    row = {"blob_id": "b2", "path": "a.ts", "repo_name": "r"}
    rec = _row_to_record(row, "x")
    assert rec.license == ""


def test_row_to_record_handles_string_license_field():
    row = {"blob_id": "b3", "path": "a.ts", "repo_name": "r", "license": "Apache-2.0"}
    rec = _row_to_record(row, "x")
    assert rec.license == "apache-2.0"


# --- iter_stack_v2_ts(rows=...) -- injected iterable, no HF/S3 ------------------------
def _rows():
    return [
        {"blob_id": "mit1", "path": "a.ts", "repo_name": "r1",
         "detected_licenses": ["MIT"], "src_encoding": "utf-8"},
        {"blob_id": "gpl1", "path": "b.ts", "repo_name": "r2",
         "detected_licenses": ["GPL-3.0"], "src_encoding": "utf-8"},
        {"blob_id": "apache1", "path": "c.ts", "repo_name": "r3",
         "detected_licenses": ["Apache-2.0"], "src_encoding": "utf-8"},
    ]


def _client_for(rows) -> _FakeS3Client:
    return _FakeS3Client({r["blob_id"]: _gzip_blob(f"// {r['blob_id']}\n") for r in rows})


def test_iter_stack_v2_ts_gates_on_permissive_license():
    rows = _rows()
    client = _client_for(rows)
    out = list(iter_stack_v2_ts(rows=rows, s3_client=client))
    # GPL-3.0 is dropped; MIT and Apache-2.0 are kept.
    assert {r.meta["blob_id"] for r in out} == {"mit1", "apache1"}
    assert all(r.lang == "typescript" and r.source == "stack-v2" for r in out)


def test_iter_stack_v2_ts_respects_limit():
    rows = _rows()
    client = _client_for(rows)
    out = list(iter_stack_v2_ts(rows=rows, s3_client=client, limit=1))
    assert len(out) == 1
    assert out[0].meta["blob_id"] == "mit1"   # first permissive row seen


def test_iter_stack_v2_ts_defaults_src_encoding_to_utf8():
    row = {"blob_id": "noenc", "path": "d.ts", "repo_name": "r4", "detected_licenses": ["MIT"]}
    client = _FakeS3Client({"noenc": _gzip_blob("const z = 1;")})
    out = list(iter_stack_v2_ts(rows=[row], s3_client=client))
    assert len(out) == 1
    assert out[0].text == "const z = 1;"


def test_iter_stack_v2_ts_yields_nothing_for_empty_rows():
    assert list(iter_stack_v2_ts(rows=[], s3_client=_FakeS3Client({}))) == []
