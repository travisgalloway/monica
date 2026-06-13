"""DPOLoader (portable): the 6-tuple shape, independent per-side right-padding with
mask=0 on pad, vocab guard, and the loop-compatible surface."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.dpo_loader import DPOLoader


def _rec(c_ids, c_mask, r_ids, r_mask):
    return {"chosen_input_ids": c_ids[:-1], "chosen_target_ids": c_ids[1:],
            "chosen_mask": c_mask,
            "rejected_input_ids": r_ids[:-1], "rejected_target_ids": r_ids[1:],
            "rejected_mask": r_mask}


def _write(tmp_path, records):
    p = tmp_path / "dpo.jsonl"
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def test_yields_six_tuple_padded_per_side(tmp_path):
    recs = [
        _rec([1, 2, 3], [1, 1], [1, 2, 3, 4, 5], [1, 1, 1, 1]),       # c len2, r len4
        _rec([1, 2, 3, 4, 5], [1, 1, 1, 1], [1, 2, 3], [1, 1]),      # c len4, r len2
    ]
    p = _write(tmp_path, recs)
    loader = DPOLoader(p, seq_len=8, batch_size=2, shuffle=False, pad_id=0)
    c_in, c_tgt, c_mask, r_in, r_tgt, r_mask = next(iter(loader.epoch()))
    # Chosen padded to its max (4), rejected to its max (4), independently.
    assert c_in.shape == c_tgt.shape == c_mask.shape == (2, 4)
    assert r_in.shape == r_tgt.shape == r_mask.shape == (2, 4)
    # Pad positions carry mask 0; mask dtype float32, ids int64.
    assert c_mask[0].tolist() == [1, 1, 0, 0]
    assert c_mask.dtype == np.float32 and c_in.dtype == np.int64


def test_exposes_loop_surface(tmp_path):
    p = _write(tmp_path, [_rec([1, 2, 3], [1, 1], [1, 2, 3], [1, 1])])
    loader = DPOLoader(p, seq_len=8, batch_size=1)
    assert loader.batch_size == 1 and loader.seq_len == 8 and len(loader) == 1


def test_vocab_guard_raises(tmp_path):
    p = _write(tmp_path, [_rec([1, 2, 3], [1, 1], [1, 99, 3], [1, 1])])
    loader = DPOLoader(p, 8, 1, vocab_size=32)
    with pytest.raises(ValueError):
        next(iter(loader.epoch()))


def test_shuffle_seed_deterministic(tmp_path):
    recs = [_rec([i, i + 1, i + 2], [1, 1], [i, i + 1], [1]) for i in range(1, 9)]
    p = _write(tmp_path, recs)
    a = [b[0].tolist() for b in DPOLoader(p, 8, 2, seed=3).epoch()]
    b = [b[0].tolist() for b in DPOLoader(p, 8, 2, seed=3).epoch()]
    assert a == b
