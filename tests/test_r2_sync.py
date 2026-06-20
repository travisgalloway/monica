"""Object-storage sync (#80): mirror a locally-built artifact tree to/from an fsspec backend.

Exercised against `memory://` and `file://` so the round-trip (and the distill-corpus `--push`
wiring) is fully covered offline, with no network or R2 credentials. The same code path resolves
`s3://` to R2 on a real host — only the backend differs.
"""

import subprocess
import sys

import pytest

pytest.importorskip("fsspec")

from src.data.r2_sync import download_dir, r2_endpoint, upload_dir


def _make_tree(root):
    (root / "corpus" / "tokenized" / "qwen25-1k").mkdir(parents=True)
    (root / "corpus" / "manifest.json").write_text('{"tokenizer": "qwen25"}')
    (root / "corpus" / "tokenized" / "qwen25-1k" / "part-00000.bin").write_bytes(b"\x01\x02\x03\x04")
    (root / "corpus" / "tokenized" / "qwen25-1k" / "part-00000.bounds").write_bytes(b"\x01\x00")
    return {
        "corpus/manifest.json": b'{"tokenizer": "qwen25"}',
        "corpus/tokenized/qwen25-1k/part-00000.bin": b"\x01\x02\x03\x04",
        "corpus/tokenized/qwen25-1k/part-00000.bounds": b"\x01\x00",
    }


def test_upload_then_download_roundtrip_memory(tmp_path):
    # Build a tree locally, push it to an in-memory store, pull it back elsewhere: byte-identical.
    src = tmp_path / "src"
    src.mkdir()
    expected = _make_tree(src)

    dst_uri = "memory://poc-distill"
    written = upload_dir(src, dst_uri)
    assert len(written) == len(expected)

    out = tmp_path / "out"
    got = download_dir(dst_uri, out)
    assert len(got) == len(expected)
    for rel, payload in expected.items():
        assert (out / rel).read_bytes() == payload


def test_upload_preserves_relative_tree_to_local_dir(tmp_path):
    # file:// destination: the relative structure under out-root is mirrored exactly.
    src = tmp_path / "src"
    src.mkdir()
    expected = _make_tree(src)
    dst = tmp_path / "remote"

    upload_dir(src, str(dst))
    for rel, payload in expected.items():
        assert (dst / rel).read_bytes() == payload


def test_upload_dir_rejects_missing_source(tmp_path):
    with pytest.raises(NotADirectoryError):
        upload_dir(tmp_path / "nope", "memory://x")


def test_r2_endpoint_reads_env(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("R2_ENDPOINT", raising=False)
    assert r2_endpoint() is None
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://acc.r2.cloudflarestorage.com")
    assert r2_endpoint() == "https://acc.r2.cloudflarestorage.com"


def test_distill_corpus_push_to_local_uri(tmp_path):
    # The `--push` wiring mirrors the built corpus to the destination prefix (file:// here).
    pytest.importorskip("pyarrow")
    out_root = tmp_path / "pd"
    push_dst = tmp_path / "remote"
    r = subprocess.run(
        [sys.executable, "-m", "src.data.distill_corpus", "--source", "dummy",
         "--max-docs", "200", "--byte-fallback", "--tokenizer", "qwen25",
         "--seq-len", "1024", "--out-root", str(out_root), "--push", str(push_dst)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "pushed" in r.stdout
    # The mirrored tree carries the corpus + tokenized manifests under the same layout.
    assert (push_dst / "corpus" / "manifest.json").exists()
    assert (push_dst / "corpus" / "tokenized" / "qwen25-1k" / "manifest.json").exists()
    assert (push_dst / "corpus" / "tokenized" / "qwen25-1k" / "part-00000.bin").exists()
