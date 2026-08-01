"""Differential parity test for the pure-Swift Parquet reader (#247,
swift/Sources/MonicaTokenizer/Parquet/): `monica-tokenize pack` fed a Parquet shard directly
must produce byte-identical shard output to the same corpus fed through the existing
`cleaned.jsonl` path. `monica-tokenize` has no "dump documents" subcommand (adding one is scope
creep), so the assertion is strictly stronger than a text comparison: identical tokenizer,
identical `pack` output (`part-*.bin` / `part-*.bounds` / `manifest.json`, byte-for-byte) from
two different input formats of the same underlying text.

`src.data.corpus.iter_shard_texts` is the oracle for "what text does this Parquet hold" on both
sides of the comparison — this file adds no second Python Parquet reader.

Coverage note (the #247 plan's open question #5): pyarrow buffers dictionary indices to
row-group end, so "multiple data pages under dictionary encoding" could not be produced — the
page-walking loop's multi-page arm is exercised only via PLAIN (no-dictionary) encoding here
(`test_parity_plain_multiple_data_pages`). That is honest coverage of the loop, not a claim that
the dictionary+multipage cell was generated.

Skips cleanly if `monica-tokenize` isn't built (`cd swift && swift build`) or pyarrow is absent.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from src.data.corpus import Record, _table_from_records, iter_shard_texts, write_shards

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_monica_tokenize() -> Path | None:
    env = os.environ.get("MONICA_TOKENIZE")
    if env and Path(env).exists():
        return Path(env)
    swift_dir = REPO_ROOT / "swift"
    for mode in ("release", "debug"):
        cand = swift_dir / ".build" / mode / "monica-tokenize"
        if cand.exists():
            return cand
    return None


@pytest.fixture(scope="module")
def monica_tokenize() -> Path:
    binary = _find_monica_tokenize()
    if binary is None:
        pytest.skip("monica-tokenize not built (cd swift && swift build)")
    return binary


def _run(binary: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(binary), *args], capture_output=True, text=True)


def _assert_dirs_byte_identical(a: Path, b: Path) -> None:
    a_files = sorted(p.name for p in a.iterdir())
    b_files = sorted(p.name for p in b.iterdir())
    assert a_files == b_files, f"{a} vs {b}: {a_files} != {b_files}"
    for name in a_files:
        a_bytes = (a / name).read_bytes()
        b_bytes = (b / name).read_bytes()
        assert a_bytes == b_bytes, f"{name} differs between {a} and {b}"


def _run_parity_case(monica_tokenize: Path, tmp_path: Path, parquet_target: Path,
                     seq_len: int = 1) -> None:
    """Build the JSONL oracle from `iter_shard_texts(parquet_target)`, train a tokenizer on it,
    then `pack` both the Parquet target and the oracle JSONL and assert identical output.

    `seq_len` defaults to 1: `Packing.pack` only emits *complete* sequences of `seqLen` tokens,
    dropping the trailing partial one — at the default `--seq-len 8192` every one of these small
    test corpora would pack down to zero complete sequences (and zero `n_documents`), and the
    parity assertion below would pass vacuously. `seq_len=1` means every token forms its own
    complete sequence, so nothing is ever dropped and `n_documents == expected_docs` is a real
    assertion; the byte-identical-output assertion holds for any seq_len (A and B always tokenize
    the same underlying text, so they drop the same tail either way).
    """
    texts = list(iter_shard_texts(parquet_target))
    oracle = tmp_path / "oracle.jsonl"
    with open(oracle, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")   # None -> "text": null, verbatim

    tok = tmp_path / "tok.json"
    r_train = _run(monica_tokenize, "train", "--in", str(oracle), "--out", str(tok),
                   "--vocab-size", "512")
    assert r_train.returncode == 0, r_train.stderr

    out_parquet = tmp_path / "out-parquet"
    out_jsonl = tmp_path / "out-jsonl"
    r_a = _run(monica_tokenize, "pack", "--tokenizer", str(tok), "--seq-len", str(seq_len),
              "--in", str(parquet_target), "--out", str(out_parquet))
    assert r_a.returncode == 0, r_a.stderr
    r_b = _run(monica_tokenize, "pack", "--tokenizer", str(tok), "--seq-len", str(seq_len),
              "--in", str(oracle), "--out", str(out_jsonl))
    assert r_b.returncode == 0, r_b.stderr

    manifest = json.loads((out_parquet / "manifest.json").read_text())
    expected_docs = len([t for t in texts if t is not None])
    assert manifest["n_documents"] == expected_docs

    _assert_dirs_byte_identical(out_parquet, out_jsonl)


# --------------------------------------------------------------------------------------- #
# Matrix — each case below is an independent probe of one corner of the encoding surface
# (see the table in the #247 plan). All shards here are written compression="snappy" (or
# "none") deliberately: this is the contract `write_shards`'s new `compression=` argument
# exists to serve — zstd (the storage default) is covered separately, as the negative case.
# --------------------------------------------------------------------------------------- #

def test_parity_snappy_dictionary_single_row_group(monica_tokenize, tmp_path):
    recs = [Record(text=f"function f{i}(x) {{ return x + {i}; }}", source="s") for i in range(20)]
    d = tmp_path / "pq"
    write_shards(recs, d, compression="snappy")
    _run_parity_case(monica_tokenize, tmp_path, d)


def test_parity_uncompressed_dictionary(monica_tokenize, tmp_path):
    recs = [Record(text=f"const x{i} = {i};", source="s") for i in range(15)]
    d = tmp_path / "pq"
    write_shards(recs, d, compression="none")
    _run_parity_case(monica_tokenize, tmp_path, d)


def test_parity_plain_multiple_data_pages(monica_tokenize, tmp_path):
    """use_dictionary=False + a small data_page_size forces several PLAIN DATA_PAGE pages in
    one row group — verified via pyarrow's ParquetFile metadata when this fixture family was
    probed (see swift/Fixtures/make-parquet-fixtures.py for the same recipe applied to the
    Swift-side fixture)."""
    import pyarrow.parquet as pq

    recs = [Record(text=f"doc {i:03d} " + ("z" * 3000), source="s") for i in range(60)]
    p = tmp_path / "multipage.parquet"
    pq.write_table(_table_from_records(recs), p, compression="snappy",
                   use_dictionary=False, data_page_size=4096)
    _run_parity_case(monica_tokenize, tmp_path, p)


def test_parity_nulls_multiple_row_groups(monica_tokenize, tmp_path):
    """A hand-built table (not `write_shards`, which never emits a null `text`) so nulls are
    actually exercised, with `row_group_size=7` over 40 rows (-> 6 row groups), each carrying
    its own dictionary page — proves the "reset dictionary per row group" requirement."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    texts = ["a", None, "", "ccc"] * 10
    n = len(texts)
    table = pa.table({
        "text": texts,
        "source": ["s"] * n,
        "lang": ["en"] * n,
        "license": ["unknown"] * n,
        "meta": ["{}"] * n,
    })
    p = tmp_path / "nulls.parquet"
    pq.write_table(table, p, compression="snappy", row_group_size=7)
    _run_parity_case(monica_tokenize, tmp_path, p)


def test_parity_multi_file_shard_directory(monica_tokenize, tmp_path):
    """`shard_size_mb=0` rolls a new shard per record -> many `part-NNNNN.parquet`, which
    proves the sorted-path determinism `readDocs`'s directory branch relies on for `pack`."""
    recs = [Record(text=f"line {i}\n", source="s") for i in range(10)]
    d = tmp_path / "pq"
    write_shards(recs, d, shard_size_mb=0, compression="snappy")
    assert len(list(d.glob("*.parquet"))) > 1
    _run_parity_case(monica_tokenize, tmp_path, d)


def test_parity_unicode_and_large_docs(monica_tokenize, tmp_path):
    recs = [
        Record(text="emoji 🚀 and CJK 数字 text", source="s"),
        Record(text="x" * 20000, source="s"),
        Record(text="plain ascii doc", source="s"),
    ]
    d = tmp_path / "pq"
    write_shards(recs, d, compression="snappy")
    _run_parity_case(monica_tokenize, tmp_path, d)


# --------------------------------------------------------------------------------------- #
# Negative path
# --------------------------------------------------------------------------------------- #

def test_zstd_shard_rejected_with_actionable_error(monica_tokenize, tmp_path):
    """`write_shards`'s default codec (zstd) is exactly what the pure-Swift reader refuses —
    `pack` must exit non-zero with an error naming both the offending codec and the fix."""
    recs = [Record(text="hello world", source="s")]
    d = tmp_path / "pq"
    write_shards(recs, d, compression="zstd")

    oracle = tmp_path / "oracle.jsonl"
    with open(oracle, "w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "hello world"}) + "\n")
    tok = tmp_path / "tok.json"
    r_train = _run(monica_tokenize, "train", "--in", str(oracle), "--out", str(tok),
                   "--vocab-size", "512")
    assert r_train.returncode == 0, r_train.stderr

    out = tmp_path / "out"
    r_pack = _run(monica_tokenize, "pack", "--tokenizer", str(tok), "--in", str(d), "--out", str(out))
    assert r_pack.returncode != 0
    assert "ZSTD" in r_pack.stderr
    assert "snappy" in r_pack.stderr
