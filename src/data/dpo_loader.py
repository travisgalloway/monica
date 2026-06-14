"""Stream right-padded DPO preference batches from a JSONL record file (portable).

Records hold a chosen and a rejected sequence, each with a response mask (see
`src/data/dpo_data.py`). Each side is padded independently to that side's longest example
in the batch; padding positions get `mask = 0` so they never contribute to the
sequence log-prob. Yields the 6-tuple

    (chosen_inputs, chosen_targets, chosen_mask,
     rejected_inputs, rejected_targets, rejected_mask)

which the injected DPO `train_step` unpacks. Mirrors `SFTLoader`'s surface
(`.epoch(reseed)`, `.batch_size`, `.seq_len`, `__len__`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np


class DPOLoader:
    def __init__(self, jsonl_path: Path, seq_len: int, batch_size: int, *,
                 pad_id: int = 0, shuffle: bool = True, seed: int = 0,
                 drop_last: bool = True, vocab_size: Optional[int] = None):
        self.path = Path(jsonl_path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        with open(self.path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if not self.records:
            raise ValueError(f"no DPO records in {self.path}")

    def __len__(self) -> int:
        n = len(self.records)
        full = n // self.batch_size
        return full if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def epoch(self, reseed: Optional[int] = None) -> Iterator[tuple[np.ndarray, ...]]:
        if reseed is not None:
            self.rng = np.random.default_rng(reseed)
        order = np.arange(len(self.records))
        if self.shuffle:
            self.rng.shuffle(order)

        batch: List[dict] = []
        for idx in order:
            batch.append(self.records[int(idx)])
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._collate(batch)

    def _pad_side(self, batch: List[dict], prefix: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lmax = max(len(r[f"{prefix}_input_ids"]) for r in batch)
        b = len(batch)
        inputs = np.full((b, lmax), self.pad_id, dtype=np.int64)
        targets = np.full((b, lmax), self.pad_id, dtype=np.int64)
        mask = np.zeros((b, lmax), dtype=np.float32)
        for i, r in enumerate(batch):
            n = len(r[f"{prefix}_input_ids"])
            inputs[i, :n] = r[f"{prefix}_input_ids"]
            targets[i, :n] = r[f"{prefix}_target_ids"]
            mask[i, :n] = r[f"{prefix}_mask"]
        if self.vocab_size is not None:
            top = int(max(inputs.max(), targets.max()))
            if top >= self.vocab_size:
                raise ValueError(
                    f"token id {top} >= vocab_size {self.vocab_size} in {self.path}. "
                    "DPO data does not match the model config — stale or wrong-tokenizer "
                    "records.")
        return inputs, targets, mask

    def _collate(self, batch: List[dict]) -> tuple[np.ndarray, ...]:
        return (*self._pad_side(batch, "chosen"), *self._pad_side(batch, "rejected"))
