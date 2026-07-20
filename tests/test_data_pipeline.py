"""End-to-end data pipeline test on synthetic ids (runs anywhere, numpy only).

Exercises pack -> split -> loader and asserts the two invariants that matter most:
  * the loader yields contiguous (input, target) windows (target = input shifted +1)
  * the validation shard does not overlap the training stream
"""

import subprocess
import sys

import numpy as np

from src.data.download import _normalize_doc
from src.data.pack import pack_ids, open_packed, packed_n_bytes, set_packed_n_bytes
from src.data.split import split_packed
from src.data.loader import PackedLoader
from src.data.tokenize import ByteTokenizer, tokenize_texts, _capped


def test_pack_roundtrip(tmp_path):
    ids = np.arange(1000, dtype=np.uint16)
    out = tmp_path / "packed.bin"
    n = pack_ids(ids, out)
    assert n == 1000
    back = open_packed(out)
    assert np.array_equal(np.asarray(back), ids)


def test_split_is_disjoint(tmp_path):
    ids = np.arange(2000, dtype=np.uint16)
    packed = tmp_path / "packed.bin"
    pack_ids(ids, packed)
    train_p, val_p = split_packed(packed, tmp_path / "split", val_tokens=200)
    train = np.asarray(open_packed(train_p))
    val = np.asarray(open_packed(val_p))
    # contiguous tail split -> disjoint and complete
    assert train.size == 1800 and val.size == 200
    assert set(train.tolist()).isdisjoint(val.tolist())
    assert np.array_equal(np.concatenate([train, val]), ids)


def test_normalize_doc_collapses_internal_newlines():
    # Internal newlines/tabs/runs must collapse so each doc stays a single line --
    # tokenize.py appends EOS per line, so a multi-line doc would inject stray EOS.
    out = _normalize_doc("  a\nb\t c\r\n\nd  ")
    assert out == "a b c d"
    assert "\n" not in out


def test_tokenize_streams_to_bin(tmp_path):
    # The .bin path streams straight through pack_ids: it writes the packed file AND
    # its meta sidecar, so split/open_packed can consume it without a separate pack.
    tok = ByteTokenizer()
    texts = ["abc", "de"]  # byte ids; ByteTokenizer has no eos -> no separators
    out = tmp_path / "packed.bin"
    n = pack_ids(_capped(tokenize_texts(texts, tok), None), out)
    expected = list(b"abc") + list(b"de")
    assert n == len(expected)
    assert (tmp_path / "packed.meta.json").exists()
    assert np.array_equal(np.asarray(open_packed(out)), np.asarray(expected, np.uint16))


def test_tokenize_counts_bytes(tmp_path):
    # stats["n_bytes"] accumulates the UTF-8 byte length of each doc as it is consumed.
    # It's only fully populated once the (lazy) generator has been drained by pack_ids,
    # so n_bytes is patched into the sidecar afterward via set_packed_n_bytes (#192) --
    # passing stats["n_bytes"] as a pack_ids(..., n_bytes=) kwarg would capture 0, since
    # Python evaluates call arguments before pack_ids's body runs the generator.
    tok = ByteTokenizer()
    texts = ["abc", "de"]
    out = tmp_path / "packed.bin"
    stats = {"n_bytes": 0}
    n = pack_ids(_capped(tokenize_texts(texts, tok, stats=stats), None), out)
    set_packed_n_bytes(out, stats["n_bytes"])
    assert stats["n_bytes"] == 5
    assert n == 5
    assert packed_n_bytes(out) == 5


def test_split_propagates_n_bytes(tmp_path):
    ids = np.arange(2000, dtype=np.uint16)
    packed = tmp_path / "packed.bin"
    pack_ids(ids, packed, n_bytes=4000)
    train_p, val_p = split_packed(packed, tmp_path / "split", val_tokens=200)
    assert packed_n_bytes(val_p) == round(4000 * 200 / 2000) == 400
    assert packed_n_bytes(train_p) == 3600

    # Legacy packed file (no n_bytes) -> both split metas carry no n_bytes either.
    legacy = tmp_path / "legacy.bin"
    pack_ids(ids, legacy)
    train_p2, val_p2 = split_packed(legacy, tmp_path / "split_legacy", val_tokens=200)
    assert packed_n_bytes(train_p2) is None
    assert packed_n_bytes(val_p2) is None


def test_capped_truncates_stream():
    assert list(_capped(iter(range(100)), 5)) == [0, 1, 2, 3, 4]
    assert list(_capped(iter(range(3)), 10)) == [0, 1, 2]  # cap above length
    assert list(_capped(iter(range(10)), None)) == list(range(10))  # pass-through


def test_nonpositive_max_tokens_rejected(tmp_path):
    # A value <= 0 would silently yield 0 tokens (enumerate starts at 0); the CLI must
    # reject it at parse time rather than produce a surprising "successful" empty run.
    inp = tmp_path / "in.txt"
    inp.write_text("abc\n")
    cmd = [sys.executable, "-m", "src.data.tokenize", "--in", str(inp),
           "--out", str(tmp_path / "o.bin"), "--byte-fallback", "--max-tokens", "-1"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "must be positive" in res.stderr


def test_loader_contiguous_windows(tmp_path):
    ids = np.arange(1, 1001, dtype=np.uint16)  # avoid 0 so shift is obvious
    packed = tmp_path / "packed.bin"
    pack_ids(ids, packed)
    loader = PackedLoader(packed, seq_len=9, batch_size=4, shuffle=False, seed=0)
    inputs, targets = next(iter(loader.epoch()))
    assert inputs.shape == (4, 9) and targets.shape == (4, 9)
    # target is input shifted by one token within each contiguous window
    assert np.array_equal(targets[:, :-1], inputs[:, 1:])
    # first window is the very start of the stream
    assert np.array_equal(inputs[0], ids[0:9])
    assert np.array_equal(targets[0], ids[1:10])
