"""SFTLoader (portable): right-padding shapes, mask=0 on pad, vocab guard, determinism,
and the PackedLoader-compatible surface the training loop reads."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.sft_loader import SFTLoader


def _write(tmp_path, records):
    p = tmp_path / "sft.jsonl"
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


def _rec(ids, mask):
    return {"input_ids": ids[:-1], "target_ids": ids[1:], "loss_mask": mask}


def test_right_pads_to_batch_max_with_zero_mask_on_pad(tmp_path):
    recs = [
        _rec([1, 2, 3], [1, 1]),          # length 2
        _rec([1, 2, 3, 4, 5], [1, 1, 1, 1]),  # length 4
    ]
    p = _write(tmp_path, recs)
    loader = SFTLoader(p, seq_len=8, batch_size=2, shuffle=False, pad_id=0)
    inputs, targets, mask = next(iter(loader.epoch()))
    assert inputs.shape == targets.shape == mask.shape == (2, 4)
    # Shorter example padded on the right; its pad positions carry mask 0.
    assert mask[0].tolist() == [1, 1, 0, 0]
    assert inputs[0].tolist() == [1, 2, 0, 0]
    assert mask.dtype == np.float32 and inputs.dtype == np.int64


def test_exposes_loop_surface(tmp_path):
    p = _write(tmp_path, [_rec([1, 2, 3], [1, 1])])
    loader = SFTLoader(p, seq_len=8, batch_size=1)
    assert loader.batch_size == 1 and loader.seq_len == 8
    assert len(loader) == 1


def test_shuffle_is_seed_deterministic(tmp_path):
    recs = [_rec([i, i + 1, i + 2], [1, 1]) for i in range(1, 9)]
    p = _write(tmp_path, recs)
    a = [b[0].tolist() for b in SFTLoader(p, 8, 2, seed=7).epoch()]
    b = [b[0].tolist() for b in SFTLoader(p, 8, 2, seed=7).epoch()]
    c = [b[0].tolist() for b in SFTLoader(p, 8, 2, seed=9).epoch()]
    assert a == b and a != c


def test_drop_last_controls_partial_batch(tmp_path):
    recs = [_rec([i, i + 1], [1]) for i in range(1, 6)]  # 5 records
    p = _write(tmp_path, recs)
    assert len(list(SFTLoader(p, 8, 2, drop_last=True).epoch())) == 2
    assert len(list(SFTLoader(p, 8, 2, drop_last=False).epoch())) == 3


def test_vocab_guard_raises(tmp_path):
    p = _write(tmp_path, [_rec([1, 99, 3], [1, 1])])
    loader = SFTLoader(p, 8, 1, vocab_size=32)
    with pytest.raises(ValueError):
        next(iter(loader.epoch()))


def test_empty_file_raises(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError):
        SFTLoader(p, 8, 1)
