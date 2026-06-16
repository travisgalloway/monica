"""Dtype-aware token packing (#90): uint16 (POC) stays the default; uint32 unlocks the
Qwen2.5 vocab (151,646). Pure numpy + stdlib, no backend.
"""

import json

import numpy as np
import pytest

from src.data.pack import (DTYPE, open_packed, pack_ids, packed_dtype, packing_dtype_for,
                          typecode_for)
from src.data.loader import PackedLoader
from src.data.shard import open_shard, pack_sequences
from src.model.blocks import MambaConfig, load_config


# --- dtype selection -----------------------------------------------------------------
def test_packing_dtype_for():
    assert packing_dtype_for(256) == np.dtype(np.uint16)
    assert packing_dtype_for(50280) == np.dtype(np.uint16)      # OLMo POC
    assert packing_dtype_for(65535) == np.dtype(np.uint16)
    assert packing_dtype_for(65536) == np.dtype(np.uint32)      # at the ceiling -> uint32
    assert packing_dtype_for(151646) == np.dtype(np.uint32)     # Qwen2.5
    assert typecode_for(np.uint16) == "H" and typecode_for(np.uint32) == "I"
    with pytest.raises(ValueError):
        typecode_for(np.int64)                                  # clear error, not KeyError


def test_pack_ids_rejects_float_array(tmp_path):
    with pytest.raises(ValueError):
        pack_ids(np.array([0.0, 1.0, 2.0]), tmp_path / "f.bin")   # float ids would truncate


# --- pack_ids round-trips ------------------------------------------------------------
def test_pack_ids_uint16_default_unchanged(tmp_path):
    p = tmp_path / "u16.bin"
    n = pack_ids([1, 2, 3, 65535], p)                           # default dtype
    arr = open_packed(p)
    assert n == 4 and arr.dtype == np.uint16 and list(arr) == [1, 2, 3, 65535]
    assert json.loads((tmp_path / "u16.meta.json").read_text())["dtype"] == "uint16"
    assert packed_dtype(p) == np.dtype(np.uint16)


def test_pack_ids_uint32_roundtrip(tmp_path):
    p = tmp_path / "u32.bin"
    ids = [1, 2, 70000, 151645, 3]                              # 70000/151645 > uint16 max
    n = pack_ids(ids, p, dtype=np.uint32)
    arr = open_packed(p)                                        # reads dtype from sidecar
    assert n == 5 and arr.dtype == np.uint32 and list(arr) == ids
    assert packed_dtype(p) == np.dtype(np.uint32)


def test_pack_ids_rejects_out_of_range(tmp_path):
    with pytest.raises(ValueError):
        pack_ids([1, 70000], tmp_path / "bad.bin", dtype=np.uint16)        # > uint16 max
    with pytest.raises(ValueError):
        pack_ids(np.array([1, 70000]), tmp_path / "bad2.bin", dtype=np.uint16)


def test_pack_ids_ndarray_uint32(tmp_path):
    p = tmp_path / "arr.bin"
    pack_ids(np.array([0, 151645], dtype=np.int64), p, dtype=np.uint32)
    assert list(open_packed(p)) == [0, 151645]


def test_loader_reads_uint32(tmp_path):
    p = tmp_path / "u32.bin"
    pack_ids(list(range(70000, 70000 + 64)), p, dtype=np.uint32)
    loader = PackedLoader(p, seq_len=4, batch_size=2, shuffle=False)
    inputs, _ = next(iter(loader.epoch()))
    assert inputs.dtype == np.int64 and int(inputs.max()) >= 70000   # uint32 ids survive


def test_legacy_file_without_dtype_meta_defaults_uint16(tmp_path):
    # A meta.json missing "dtype" (an old artifact) is read back as uint16.
    p = tmp_path / "legacy.bin"
    np.array([1, 2, 3], dtype=np.uint16).tofile(p)
    (tmp_path / "legacy.meta.json").write_text(json.dumps({"n_tokens": 3}))
    assert packed_dtype(p) == np.dtype(np.uint16)
    assert list(open_packed(p)) == [1, 2, 3]


# --- shard path ----------------------------------------------------------------------
def test_pack_sequences_uint32_roundtrip(tmp_path):
    docs = [[1, 2, 70000, 151645], [5, 6, 7, 8]]
    m = pack_sequences(docs, tmp_path, seq_len=4, dtype=np.uint32, tokenizer="qwen25")
    assert m["dtype"] == "uint32"
    toks, _ = open_shard(tmp_path, m["shards"][0]["name"])      # dtype read from manifest
    assert toks.dtype == np.uint32 and list(toks) == [1, 2, 70000, 151645, 5, 6, 7, 8]


def test_pack_sequences_uint32_allows_large_pad_id(tmp_path):
    # pad_id beyond uint16 is fine under uint32 (would raise under the default uint16).
    m = pack_sequences([[1, 2, 3]], tmp_path, seq_len=2, chunk_align=2, pad_id=70000,
                       dtype=np.uint32)
    toks, _ = open_shard(tmp_path, m["shards"][0]["name"])
    assert 70000 in list(toks)


# --- config gate ---------------------------------------------------------------------
def test_mambaconfig_packing_dtype_and_validate():
    base = dict(d_model=64, n_layers=2, head_dim=16)
    assert MambaConfig(vocab_size=50280, **base).packing_dtype == "uint16"
    big = MambaConfig(vocab_size=151646, **base)
    assert big.packing_dtype == "uint32"
    big.validate()                                              # no longer raises (#90)
    MambaConfig(vocab_size=1 << 32, **base).validate()         # max id 2**32-1 still fits
    with pytest.raises(ValueError):
        MambaConfig(vocab_size=(1 << 32) + 1, **base).validate()   # over the uint32 capacity


def test_student_1b_config_validates():
    cfg = load_config("config/student-1b.yaml")
    cfg.validate()
    assert cfg.vocab_size == 151646 and cfg.packing_dtype == "uint32"
