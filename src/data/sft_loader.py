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


class SFTLoader:
    def __init__(self, jsonl_path: Path, seq_len: int, batch_size: int, *,
                 pad_id: int = 0, shuffle: bool = True, seed: int = 0,
                 drop_last: bool = True, vocab_size: Optional[int] = None):
        self.path = Path(jsonl_path)
        self.seq_len = seq_len           # loop.train reads this for tokens/step
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        with open(self.path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if not self.records:
            raise ValueError(f"no SFT records in {self.path}")

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
