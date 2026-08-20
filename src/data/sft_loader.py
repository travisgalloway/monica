"""Stream right-padded SFT batches from a JSONL record file (portable, backend-free).

The on-disk format is JSONL of int lists (`src/data/sft_data.py`): one
`{input_ids, target_ids, loss_mask}` per line. Unlike the packed-uint16 pretraining
format, SFT examples are variable-length with per-token loss masks and per-example
boundaries — JSONL keeps that structure, stays streamable/inspectable, and is
line-addressable. The sets are tiny (~10k), so loading all records into memory is fine.

Yields `(inputs, targets, mask)` numpy batches right-padded to the batch's longest
example; padding positions get `mask = 0` so they never contribute to the loss. The
loader mirrors `PackedLoader`'s surface (`.epoch(reseed)`, `.batch_size`, `.seq_len`,
`__len__`) so it drops straight into `train.loop.train`; the injected SFT `train_step`
unpacks the 3-tuple.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np


_REQUIRED_KEYS = ("input_ids", "target_ids", "loss_mask")


def load_sft_records(jsonl_path: Path) -> List[dict]:
    """Read one JSONL record file, failing with the path and line number rather than a bare
    `json.JSONDecodeError` (#306).

    Also rejects a structurally invalid record — a missing key, a non-list value, a non-numeric
    element, or three lists whose lengths disagree — with the same `path:line` context rather
    than a bare `TypeError`/`ValueError` from deep inside collation. A length-mismatched record
    does not crash on its own: `_collate` right-pads each field independently, so a short
    `loss_mask` silently shifts the mask off its tokens and the run trains on the prompt. That is
    exactly the silent-wrong-mask failure the #306 masking criterion exists to prevent, so it is
    caught at load.
    """
    path = Path(jsonl_path)
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed SFT record: {exc}") from exc
            if not isinstance(rec, dict) or any(k not in rec for k in _REQUIRED_KEYS):
                missing = [k for k in _REQUIRED_KEYS
                           if not isinstance(rec, dict) or k not in rec]
                raise ValueError(f"{path}:{lineno}: SFT record is missing {missing} "
                                 f"(expected {list(_REQUIRED_KEYS)})")
            not_list = [k for k in _REQUIRED_KEYS if not isinstance(rec[k], list)]
            if not_list:
                got = {k: type(rec[k]).__name__ for k in not_list}
                raise ValueError(f"{path}:{lineno}: SFT record field(s) {not_list} must be "
                                 f"lists, got {got}")
            non_numeric = [k for k in _REQUIRED_KEYS
                           if any(not isinstance(x, (int, float)) for x in rec[k])]
            if non_numeric:
                raise ValueError(f"{path}:{lineno}: SFT record field(s) {non_numeric} contain "
                                 f"a non-numeric element — ids/mask must be numbers.")
            lengths = {k: len(rec[k]) for k in _REQUIRED_KEYS}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"{path}:{lineno}: SFT record field lengths disagree "
                                 f"({lengths}) — the loss mask would not line up with its tokens.")
            records.append(rec)
    if not records:
        raise ValueError(f"no SFT records in {path}")
    return records


class SFTLoader:
    def __init__(self, jsonl_path: Path, seq_len: int, batch_size: int, *,
                 pad_id: int = 0, shuffle: bool = True, seed: int = 0,
                 drop_last: bool = True, vocab_size: Optional[int] = None,
                 records: Optional[List[dict]] = None):
        """`records` (#306) supplies an already-loaded, already-validated record list — the
        `src.data.sft_corpus` resolver hands over its train/val split in memory instead of
        writing temp files. `jsonl_path` is then only a label for error messages."""
        self.path = Path(jsonl_path)
        self.seq_len = seq_len           # loop.train reads this for tokens/step
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        if records is not None:
            self.records = list(records)
            if not self.records:
                raise ValueError(f"no SFT records supplied for {self.path}")
        else:
            self.records = load_sft_records(self.path)

    def __len__(self) -> int:
        n = len(self.records)
        full = n // self.batch_size
        return full if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def epoch(self, reseed: Optional[int] = None, skip_batches: int = 0
              ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield `(inputs, targets, mask)` for one pass; shuffles per epoch if enabled.

        `skip_batches` drops that many leading batches (resume fast-forward); the
        shuffle still runs first so the order and rng state are unchanged.
        """
        if reseed is not None:
            self.rng = np.random.default_rng(reseed)
        order = np.arange(len(self.records))
        if self.shuffle:
            self.rng.shuffle(order)
        if skip_batches:
            order = order[skip_batches * self.batch_size:]

        batch: List[dict] = []
        for idx in order:
            batch.append(self.records[int(idx)])
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._collate(batch)

    def _collate(self, batch: List[dict]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lmax = max(len(r["input_ids"]) for r in batch)
        b = len(batch)
        inputs = np.full((b, lmax), self.pad_id, dtype=np.int64)
        targets = np.full((b, lmax), self.pad_id, dtype=np.int64)
        mask = np.zeros((b, lmax), dtype=np.float32)
        for i, r in enumerate(batch):
            n = len(r["input_ids"])
            inputs[i, :n] = r["input_ids"]
            targets[i, :n] = r["target_ids"]
            mask[i, :n] = r["loss_mask"]
        if self.vocab_size is not None:
            top = int(max(inputs.max(), targets.max()))
            if top >= self.vocab_size:
                raise ValueError(
                    f"token id {top} >= vocab_size {self.vocab_size} in {self.path}. "
                    "SFT data does not match the model config — stale or wrong-tokenizer "
                    "records.")
        return inputs, targets, mask
