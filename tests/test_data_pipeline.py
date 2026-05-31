"""End-to-end data pipeline test on synthetic ids (runs anywhere, numpy only).

Exercises pack -> split -> loader and asserts the two invariants that matter most:
  * the loader yields contiguous (input, target) windows (target = input shifted +1)
  * the validation shard does not overlap the training stream
"""

import numpy as np

from src.data.pack import pack_ids, open_packed
from src.data.split import split_packed
from src.data.loader import PackedLoader


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
