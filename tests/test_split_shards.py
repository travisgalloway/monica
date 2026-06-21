"""shard -> train/val split bridge (#80): turn the scale/datatrove tokenized shards
(`part-*.bin` + manifest) into the `train.bin`/`val.bin` the trainer's `PackedLoader` reads,
with no re-tokenize. Portable: numpy + stdlib.
"""

import numpy as np
import pytest

from src.data.loader import PackedLoader
from src.data.pack import open_packed, packed_dtype
from src.data.shard import pack_sequences
from src.data.split import split_shards


def _build_shards(tmp_path):
    # ~600k uint32 tokens over several docs -> multiple shards at shard_size_mb=1 (so the split
    # exercises whole-train, straddle, and whole-val shard branches).
    docs = [list(range(1, 1001)) for _ in range(600)]
    shard_dir = tmp_path / "shards"
    man = pack_sequences(iter(docs), shard_dir, seq_len=100, shard_size_mb=1, dtype=np.uint32)
    return shard_dir, man


def test_split_shards_disjoint_train_val(tmp_path):
    shard_dir, man = _build_shards(tmp_path)
    assert len(man["shards"]) > 1 and man["dtype"] == "uint32"
    total = man["n_tokens"]
    val_tokens = 50_000

    train_path, val_path = split_shards(shard_dir, tmp_path / "split", val_tokens)

    # Token counts add up and the dtype is preserved.
    assert packed_dtype(train_path) == np.dtype(np.uint32)
    train, val = open_packed(train_path), open_packed(val_path)
    assert val.shape[0] == val_tokens
    assert train.shape[0] == total - val_tokens     # disjoint: val is the contiguous tail
    # The trainer's loader consumes the split.
    loader = PackedLoader(train_path, seq_len=100, batch_size=4, vocab_size=1001)
    x, y = next(iter(loader.epoch()))
    assert x.shape == (4, 100) and y.shape == (4, 100)


def test_split_shards_rejects_oversized_val(tmp_path):
    shard_dir, man = _build_shards(tmp_path)
    with pytest.raises(ValueError):
        split_shards(shard_dir, tmp_path / "split", man["n_tokens"])


def test_poc_qwen_config_loads():
    from src.model.blocks import load_config
    cfg = load_config("config/poc-qwen.yaml")
    assert cfg.vocab_size == 151646 and cfg.seq_len == 1024 and cfg.tie_embeddings is True
