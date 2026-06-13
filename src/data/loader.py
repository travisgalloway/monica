"""Stream contiguous fixed-length chunks from a packed uint16 file.

Design: mmap + a chunk index, shuffled at the chunk level. No per-step parsing —
a slow loader bottlenecks a small model. Each item is a contiguous `seq_len + 1`
window so the training loop can form (input, target) by a one-token shift.

This module is backend-free: it yields numpy arrays. The backend converts them to
its own array type inside `forward`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .pack import open_packed


class PackedLoader:
    def __init__(self, packed_path: Path, seq_len: int, batch_size: int,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = True,
                 vocab_size: Optional[int] = None):
        self.path = packed_path
        self.data = open_packed(packed_path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)

        # Non-overlapping chunks of length seq_len+1 (extra token = shift target).
        self.stride = seq_len + 1
        self.n_chunks = (self.data.shape[0]) // self.stride
        if self.n_chunks == 0:
            raise ValueError("packed file too small for one chunk")

    def _chunk(self, idx: int) -> np.ndarray:
        start = idx * self.stride
        return np.asarray(self.data[start: start + self.stride], dtype=np.int64)

    def __len__(self) -> int:
        full = self.n_chunks // self.batch_size
        return full if self.drop_last else (self.n_chunks + self.batch_size - 1) // self.batch_size

    def epoch(self, reseed: Optional[int] = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (inputs, targets), each (batch, seq_len), for one pass over the data."""
        if reseed is not None:
            self.rng = np.random.default_rng(reseed)
        order = np.arange(self.n_chunks)
        if self.shuffle:
            self.rng.shuffle(order)

        batch = []
        for idx in order:
            batch.append(self._chunk(int(idx)))
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._collate(batch)

    def _collate(self, batch: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        arr = np.stack(batch)              # (B, seq_len+1)
        if self.vocab_size is not None:
            top = int(arr.max())
            if top >= self.vocab_size:
                raise ValueError(
                    f"token id {top} >= vocab_size {self.vocab_size} in {self.path}. "
                    "The packed data does not match the model config — likely a stale or "
                    "clobbered data file (e.g. a real-corpus run overwrote a toy/test path)."
                )
        return arr[:, :-1], arr[:, 1:]      # inputs, targets
