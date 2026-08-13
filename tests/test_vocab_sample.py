"""Offline tests for the vocab-sweep sample builder (#251).

Everything here runs with injected fixtures — a fake S3 client, fake parquet openers, no
`boto3`, no `pyarrow`, no `fsspec`, no credentials, no network. That is the point: the module
is above the seam and its heavy dependencies are lazy, so the test suite must exercise the
filter/tagging/fetch logic on a host with none of them installed. Mirrors
`tests/test_stack_v2.py`.
"""

from __future__ import annotations

import gzip
import io
import json
import threading

import pytest

from src.data.vocab_sample import (
    DEFAULT_MIXTURE, FetchStats, SliceSpec, _manifest_matches, build_sample, fetch_contents,
    iter_essential_web, iter_stack_v2_metadata, metadata_ok, scale_mixture)


class _FakeBody:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)

    def read(self) -> bytes:
        return self._buf.read()


class _FakeS3Client:
    """Serves canned gzip blobs; an unregistered blob_id raises, like a real missing object."""

    def __init__(self, blobs: dict):
        self.blobs = blobs
        self.calls = []

    def get_object(self, Bucket: str, Key: str):
        blob_id = Key.split("/")[-1]
        self.calls.append(blob_id)
        if blob_id not in self.blobs:
            raise KeyError(f"no such blob {blob_id}")
        return {"Body": _FakeBody(self.blobs[blob_id])}


class _GatedS3Client(_FakeS3Client):
    """Like `_FakeS3Client`, but every fetch after the first parks until released.

    A canned blob decompresses in microseconds, so against the plain fake a pool worker
    outruns anything the main thread wants to observe mid-chunk. Parking makes "the rest of
    the chunk has not been fetched yet" a fact rather than a race. The park happens *before*
    the call is recorded, so `calls` is exact at the suspension point.
    """

    def __init__(self, blobs: dict, release: threading.Event):
        super().__init__(blobs)
        self._release = release
        self._first_served = False

    def get_object(self, Bucket: str, Key: str):
        if self._first_served:
            # Bounded, so a mistake here fails the test rather than hanging the pool. Not an
            # `assert`: `python -O` would strip the wait itself and resurrect the race.
            self._release.wait(timeout=10)
        self._first_served = True
        return super().get_object(Bucket=Bucket, Key=Key)


class _FakeColumn:
    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return list(self._values)


class _FakeBatch:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return [dict(r) for r in self._rows]

    def column(self, name):
        return _FakeColumn([r[name] for r in self._rows])


class _FakeParquetFile:
    """Just enough of `pyarrow.parquet.ParquetFile` for the readers: `iter_batches`."""

    def __init__(self, rows, batch_size=2):
        self._rows = rows
        self._batch_size = batch_size

    def iter_batches(self, batch_size=None, columns=None):
        n = batch_size or self._batch_size
        for i in range(0, len(self._rows), n):
            yield _FakeBatch(self._rows[i:i + n])


def _row(blob_id="b0", *, licenses=("mit",), length=4096, vendor=False, generated=False):
    if isinstance(licenses, tuple):
        licenses = list(licenses)
    return {"blob_id": blob_id, "src_encoding": "UTF-8",
            "detected_licenses": licenses,
            "length_bytes": length, "is_vendor": vendor, "is_generated": generated,
            "path": "src/a.ts", "repo_name": "acme/app"}


# --- metadata pre-filter (the thing that saves two thirds of the S3 spend) ------------
def test_metadata_filter_keeps_permissive_source_files():
    assert metadata_ok(_row())


def test_metadata_filter_rejects_empty_license():
    # ~67% of Stack-v2 rows carry an empty `detected_licenses`, and the corpus's permissive
    # gate rejects an empty license for code — so these must never reach S3.
    assert not metadata_ok(_row(licenses=()))
    assert not metadata_ok(_row(licenses=None))


def test_metadata_filter_rejects_non_permissive_license():
    assert not metadata_ok(_row(licenses=("gpl-3.0",)))


def test_metadata_filter_accepts_scalar_license_string():
    assert metadata_ok(_row(licenses="mit"))


def test_metadata_filter_rejects_vendored_and_generated():
    # Minified bundles / codegen would teach garbage merges and distort bytes/token upward.
    assert not metadata_ok(_row(vendor=True))
    assert not metadata_ok(_row(generated=True))


@pytest.mark.parametrize("length", [0, 1023, 131073, None])
def test_metadata_filter_rejects_out_of_range_sizes(length):
    assert not metadata_ok(_row(length=length))


def test_metadata_filter_requires_a_blob_id():
    row = _row()
    row["blob_id"] = None
    assert not metadata_ok(row)


# --- parquet readers -----------------------------------------------------------------
def test_iter_essential_web_stops_at_target_bytes():
    docs = [{"text": "x" * 100} for _ in range(50)]
    opened = []

    def opener(path):
        opened.append(path)
        return _FakeParquetFile(docs, batch_size=5)

    out = list(iter_essential_web(250, files=["f0", "f1"], open_parquet=opener))
    assert sum(len(t.encode()) for t in out) >= 250
    assert len(out) == 3           # stops as soon as the target is crossed
    assert opened == ["f0"]        # never opens a file it does not need


def test_iter_essential_web_skips_empty_text():
    rows = [{"text": ""}, {"text": None}, {"text": "hello"}]
    out = list(iter_essential_web(1, files=["f0"],
                                  open_parquet=lambda p: _FakeParquetFile(rows)))
    assert out == ["hello"]


def test_iter_stack_v2_metadata_filters_and_tags_source_file():
    rows = [_row("keep1"), _row("drop", licenses=()), _row("keep2", vendor=False)]
    out = list(iter_stack_v2_metadata("TypeScript", files=["p0"],
                                      open_parquet=lambda p: _FakeParquetFile(rows)))
    assert [r["blob_id"] for r in out] == ["keep1", "keep2"]
    assert all(r["_parquet"] == "p0" for r in out)   # recorded in the manifest for auditability


def test_iter_stack_v2_metadata_uses_the_lister_when_no_files_given():
    seen = {}

    def lister(directory):
        seen["dir"] = directory
        return ["shard-a"]

    out = list(iter_stack_v2_metadata(
        "Python", list_parquet=lister,
        open_parquet=lambda p: _FakeParquetFile([_row("k")])))
    assert seen["dir"].endswith("/data/Python")
    assert [r["blob_id"] for r in out] == ["k"]


# --- threaded fetch ------------------------------------------------------------------
def test_fetch_contents_returns_documents_in_submission_order():
    texts = {f"b{i}": gzip.compress((f"doc{i}" + "y" * 100).encode()) for i in range(10)}
    client = _FakeS3Client(texts)
    rows = [_row(f"b{i}") for i in range(10)]
    out = list(fetch_contents(rows, 10_000, s3_client=client, threads=4, chunk=3))
    assert [t[:4] for t in out] == [f"doc{i}" for i in range(10)]


def test_fetch_contents_stops_at_target_and_reports_stats():
    texts = {f"b{i}": gzip.compress(b"z" * 100) for i in range(20)}
    stats = FetchStats()
    rows = [_row(f"b{i}") for i in range(20)]
    out = list(fetch_contents(rows, 250, s3_client=_FakeS3Client(texts), threads=4,
                              stats=stats, chunk=2))
    assert len(out) == 3
    assert stats.fetched == 3 and stats.bytes >= 250


def test_fetch_contents_counts_failures_without_aborting_the_run():
    # A handful of unreadable blobs must not lose a 15-minute ingest.
    texts = {"b0": gzip.compress(b"a" * 50), "b2": gzip.compress(b"c" * 50)}
    stats = FetchStats()
    rows = [_row("b0"), _row("b1"), _row("b2")]
    out = list(fetch_contents(rows, 10_000, s3_client=_FakeS3Client(texts), threads=2,
                              stats=stats, chunk=8))
    assert len(out) == 2
    assert stats.failed == 1


def test_fetch_contents_is_a_no_op_for_a_zero_target():
    client = _FakeS3Client({})
    assert list(fetch_contents([_row()], 0, s3_client=client)) == []
    assert client.calls == []


def test_fetch_contents_yields_the_first_result_without_draining_the_chunk():
    """With the old `pool.map`, nothing came back until every future in the chunk was done.

    Submitting individual futures means the first result is available immediately, so a
    target crossed mid-chunk never pays for the remainder. Gating the client is what makes
    that observable: the pool's single worker parks on `b1`, so `calls` at the suspension
    point is exactly what has been paid for. Under `pool.map` this fails (after the gated
    client's 10s timeout) instead of passing.

    The `f.cancel()` on the unstarted remainder is deliberately *not* asserted here — whether
    a queued future is cancelled before the worker picks it up is thread scheduling, not
    behaviour, and asserting it is what made this test flaky on loaded CI runners (#281).
    """
    texts = {f"b{i}": gzip.compress(b"z" * 300) for i in range(20)}
    release = threading.Event()
    client = _GatedS3Client(texts, release)
    rows = [_row(f"b{i}") for i in range(20)]
    gen = fetch_contents(rows, 100, s3_client=client, threads=1, chunk=20)
    try:
        first = next(gen)
        assert first is not None          # the first doc alone crosses the 100-byte target
        assert client.calls == ["b0"]     # ...and the rest of the chunk was never fetched
    finally:
        release.set()                     # unpark the worker so the pool can join
        gen.close()


# --- mixture + manifest --------------------------------------------------------------
def test_default_mixture_covers_prose_and_the_code_languages():
    langs = {s.lang for s in DEFAULT_MIXTURE}
    assert "en" in langs
    assert {"typescript", "python", "javascript", "java", "go", "rust", "cpp",
            "markdown"} <= langs


def test_scale_mixture_keeps_every_language():
    scaled = scale_mixture(DEFAULT_MIXTURE, 0.5)
    assert [s.lang for s in scaled] == [s.lang for s in DEFAULT_MIXTURE]
    assert all(b.target_bytes == a.target_bytes // 2
               for a, b in zip(DEFAULT_MIXTURE, scaled))


def test_scale_mixture_rejects_a_non_positive_scale():
    with pytest.raises(ValueError):
        scale_mixture(DEFAULT_MIXTURE, 0)


def test_manifest_matches_only_the_same_langs_and_targets():
    mixture = [SliceSpec("en", "essential_web", 100)]
    assert _manifest_matches(
        {"slices": {"en": {"kind": "essential_web", "source": "", "target_bytes": 100}}},
        mixture)
    assert not _manifest_matches(
        {"slices": {"en": {"kind": "essential_web", "source": "", "target_bytes": 200}}},
        mixture)
    assert not _manifest_matches({"slices": {}}, mixture)


def test_manifest_matches_rejects_a_different_kind_at_the_same_target_bytes():
    # A stack_v2 slice re-tagged "en" at the same byte budget as a cached essential_web
    # slice must NOT be treated as a cache hit -- same lang/target, different source data.
    mixture = [SliceSpec("en", "stack_v2", 100, "Python")]
    cached = {"slices": {"en": {"kind": "essential_web", "source": "", "target_bytes": 100}}}
    assert not _manifest_matches(cached, mixture)


def test_manifest_matches_rejects_a_different_stack_v2_source_at_the_same_target_bytes():
    # Same lang tag, same kind, same byte budget, but a different Stack-v2 language
    # directory -- reusing this cache would silently swap the corpus content.
    mixture = [SliceSpec("code", "stack_v2", 100, "Python")]
    cached = {"slices": {"code": {"kind": "stack_v2", "source": "JavaScript",
                                  "target_bytes": 100}}}
    assert not _manifest_matches(cached, mixture)


# --- build_sample (end to end, offline) ----------------------------------------------
def test_build_sample_writes_tagged_jsonl_and_a_manifest(tmp_path, monkeypatch):
    import src.data.vocab_sample as vs

    prose = [{"text": "prose " * 30} for _ in range(5)]
    code_rows = [_row(f"b{i}") for i in range(5)]
    blobs = {f"b{i}": gzip.compress(("const x = %d;\n" % i * 20).encode()) for i in range(5)}

    monkeypatch.setattr(vs, "_hf_parquet",
                        lambda path: _FakeParquetFile(prose if "essential" in path
                                                      else code_rows))
    monkeypatch.setattr(vs, "_list_parquet", lambda d: ["shard"])

    mixture = [SliceSpec("en", "essential_web", 200),
               SliceSpec("typescript", "stack_v2", 200, "TypeScript")]
    out = build_sample(tmp_path, mixture=mixture, threads=2,
                       s3_client=_FakeS3Client(blobs), log=lambda m: None)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["lang"] for r in rows} == {"en", "typescript"}
    assert all(r["text"] for r in rows)

    manifest = json.loads((tmp_path / "sample" / "manifest.json").read_text())
    assert set(manifest["slices"]) == {"en", "typescript"}
    assert manifest["total_bytes"] > 0
    assert manifest["slices"]["typescript"]["parquet_files"] == ["shard"]


def test_build_sample_reuses_a_cached_sample_without_touching_the_network(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    (sample_dir / "sample.jsonl").write_text('{"text": "x", "lang": "en"}\n')
    (sample_dir / "manifest.json").write_text(json.dumps(
        {"slices": {"en": {"kind": "essential_web", "source": "", "target_bytes": 200}},
         "total_bytes": 1, "total_docs": 1}))

    # No parquet opener patched and no S3 client: any network work would raise.
    out = build_sample(tmp_path, mixture=[SliceSpec("en", "essential_web", 200)],
                       log=lambda m: None)
    assert out.read_text().strip().endswith('"lang": "en"}')


def test_build_sample_requires_an_s3_client_for_code_slices(tmp_path):
    with pytest.raises(RuntimeError, match="S3 client"):
        build_sample(tmp_path, mixture=[SliceSpec("typescript", "stack_v2", 10, "TypeScript")],
                     log=lambda m: None)
