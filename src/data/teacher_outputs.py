"""Cached teacher top-k outputs (#94): the frozen teacher's per-token top-k logits, the
dominant precompute of the distillation program.

The teacher forward over the corpus depends only on (teacher, corpus), never on the
student — so it is computed ONCE (`scripts/precompute_teacher.py`) and every student trial
reads it back with zero teacher inference (`DistillLoader`). This module owns the on-disk
format and the loader; it is **portable** (numpy only, no `mlx`/`torch`) like `loader.py`.

Alignment is positional against the flat packed `train.bin`/`val.bin` the distill loader
reads (`split.py`'s output, NOT the multi-shard corpus). `PackedLoader` cuts non-overlapping
`seq_len+1` chunks; the teacher top-k is taken at the `seq_len` INPUT positions of each chunk
(unshifted), so teacher row `t` is the teacher's prediction of `targets[t]` — exactly what
the distill `_kl_topk` (#100) matches the student against. Per packed file we store
`n_chunks * seq_len` rows in on-disk chunk order, so chunk `c` -> rows
`[c*seq_len : c*seq_len+seq_len]`. Sharing `seq_len`/`n_chunks`/seed with `PackedLoader`
makes `DistillLoader`'s shuffle + `skip_batches` byte-identical, so training resume stays exact.

On-disk layout under a teacher-outputs dir (one set per split):

    teacher-<split>.topk_vals   float16 (n_rows, k)   raw C-order .tofile dump (mmap-friendly)
    teacher-<split>.topk_idx    uint32  (n_rows, k)
    teacher-<split>.meta.json   {split, k, n_rows, n_chunks, seq_len, vals_dtype, idx_dtype,
                                 vocab_size, src_packed, src_n_tokens}
    manifest.json               run-level summary (k, dtypes, splits, teacher, corpus ref)

Footprint is `k*(2+4)` bytes/token (fp16 vals + uint32 idx): k=50 ~= 600 GB at 2B tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

import numpy as np

from .pack import open_packed

#: On-disk dtypes. idx is uint32 (Qwen3 effective vocab 151669 overflows uint16); vals are
#: fp16 — the KL is a softmax over the k-support after `/T`, so fp16 logit precision is ample.
VALS_DTYPE = np.dtype(np.float16)
IDX_DTYPE = np.dtype(np.uint32)


def topk_outputs_paths(out_dir, split: str) -> dict:
    """The three file paths for one split's cached top-k (`teacher-<split>.*`)."""
    base = Path(out_dir) / f"teacher-{split}"
    return {"vals": base.with_suffix(".topk_vals"),
            "idx": base.with_suffix(".topk_idx"),
            "meta": base.with_suffix(".meta.json")}


def write_teacher_topk(out_dir, split: str, *, blocks: Iterable[Tuple[np.ndarray, np.ndarray]],
                       n_chunks: int, seq_len: int, vocab_size: int,
                       src_packed: str, src_n_tokens: Optional[int] = None) -> dict:
    """Stream per-batch top-k blocks to disk and write the split's `.meta.json`.

    `blocks` yields `(vals_block, idx_block)` in on-disk chunk order; each block is shaped
    `(..., k)` (e.g. `(batch, seq_len, k)`) and is flattened to `(rows, k)`, cast to fp16 /
    uint32, and appended. `k` is taken from the first block (the teacher may have clamped it
    to the vocab). Returns the meta dict. Streaming keeps memory bounded over a large corpus.
    """
    paths = topk_outputs_paths(out_dir, split)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_rows, k = 0, None
    with open(paths["vals"], "wb") as vf, open(paths["idx"], "wb") as xf:
        for vb, ib in blocks:
            vb = np.ascontiguousarray(np.asarray(vb).reshape(-1, np.asarray(vb).shape[-1]),
                                      dtype=VALS_DTYPE)
            ib = np.ascontiguousarray(np.asarray(ib).reshape(-1, np.asarray(ib).shape[-1]),
                                      dtype=IDX_DTYPE)
            if k is None:
                k = vb.shape[1]
            if vb.shape[1] != k or ib.shape[1] != k:
                raise ValueError(f"inconsistent k across blocks: {vb.shape[1]}/{ib.shape[1]} != {k}")
            if vb.shape[0] != ib.shape[0]:
                raise ValueError("vals/idx block row counts differ")
            vb.tofile(vf)
            ib.tofile(xf)
            n_rows += vb.shape[0]
    expected = n_chunks * seq_len
    if n_rows != expected:
        raise ValueError(f"wrote {n_rows} teacher rows, expected n_chunks*seq_len = {expected}")
    meta = {"split": split, "k": int(k or 0), "n_rows": int(n_rows), "n_chunks": int(n_chunks),
            "seq_len": int(seq_len), "vals_dtype": VALS_DTYPE.name, "idx_dtype": IDX_DTYPE.name,
            "vocab_size": int(vocab_size), "src_packed": str(src_packed),
            "src_n_tokens": (int(src_n_tokens) if src_n_tokens is not None else None)}
    paths["meta"].write_text(json.dumps(meta, indent=2))
    return meta


def read_teacher_meta(out_dir, split: str) -> dict:
    """Load one split's `.meta.json` (k, n_rows, n_chunks, seq_len, dtypes, ...)."""
    return json.loads(topk_outputs_paths(out_dir, split)["meta"].read_text())


def write_manifest(out_dir, *, k: int, seq_len: int, effective_vocab_size: int,
                   corpus_manifest: Optional[str], teacher: Optional[dict],
                   splits: Iterable[str]) -> dict:
    """Write the run-level `manifest.json` tying the per-split files to the teacher + corpus."""
    splits = list(splits)
    n_rows_total = sum(read_teacher_meta(out_dir, s)["n_rows"] for s in splits)
    manifest = {"k": int(k), "seq_len": int(seq_len),
                "vals_dtype": VALS_DTYPE.name, "idx_dtype": IDX_DTYPE.name,
                "effective_vocab_size": int(effective_vocab_size),
                "corpus_manifest": corpus_manifest, "teacher": teacher,
                "splits": splits, "n_rows_total": int(n_rows_total)}
    (Path(out_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class DistillLoader:
    """Stream `(inputs, targets, topk_vals, topk_idx)` for the logit-distill stage (#100).

    Mirrors `PackedLoader` exactly — same `stride`/`n_chunks`/shuffle/`skip_batches` math, so
    `epoch(reseed=s)` visits chunks in the SAME order as `PackedLoader(..., seed=s)` and resume
    fast-forwards identically. The extra outputs are the cached teacher top-k for each chunk's
    input positions: `topk_vals` `(B, seq_len, k)` fp32, `topk_idx` `(B, seq_len, k)` int64.
    """

    def __init__(self, packed_path: Path, topk_dir, split: str, seq_len: int, batch_size: int,
                 k: Optional[int] = None, shuffle: bool = True, seed: int = 0,
                 drop_last: bool = True, vocab_size: Optional[int] = None):
        self.path = packed_path
        self.data = open_packed(packed_path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)

        # Non-overlapping chunks of length seq_len+1 (extra token = shift target) — same as
        # PackedLoader, so the chunk index space (and thus the shuffle) is identical.
        self.stride = seq_len + 1
        self.n_chunks = (self.data.shape[0]) // self.stride
        if self.n_chunks == 0:
            raise ValueError("packed file too small for one chunk")

        meta = read_teacher_meta(topk_dir, split)
        if meta["seq_len"] != seq_len:
            raise ValueError(f"teacher-{split} seq_len {meta['seq_len']} != loader seq_len {seq_len}")
        if meta["n_chunks"] != self.n_chunks:
            raise ValueError(
                f"teacher-{split} n_chunks {meta['n_chunks']} != packed n_chunks {self.n_chunks} "
                f"({self.path}) — teacher outputs are not aligned to this packed file")
        self.k_stored = int(meta["k"])
        self.k = int(k) if k is not None else self.k_stored
        if self.k > self.k_stored:
            raise ValueError(f"requested k={self.k} exceeds stored k={self.k_stored}")
        n_rows = int(meta["n_rows"])
        paths = topk_outputs_paths(topk_dir, split)
        self.vals = np.memmap(paths["vals"], dtype=np.dtype(meta["vals_dtype"]), mode="r",
                              shape=(n_rows, self.k_stored))
        self.idx = np.memmap(paths["idx"], dtype=np.dtype(meta["idx_dtype"]), mode="r",
                             shape=(n_rows, self.k_stored))

    def _chunk(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start = idx * self.stride
        chunk = np.asarray(self.data[start: start + self.stride], dtype=np.int64)
        r0 = idx * self.seq_len
        vals = np.asarray(self.vals[r0: r0 + self.seq_len, : self.k], dtype=np.float32)
        tidx = np.asarray(self.idx[r0: r0 + self.seq_len, : self.k], dtype=np.int64)
        return chunk[:-1], chunk[1:], vals, tidx

    def __len__(self) -> int:
        full = self.n_chunks // self.batch_size
        return full if self.drop_last else (self.n_chunks + self.batch_size - 1) // self.batch_size

    def epoch(self, reseed: Optional[int] = None,
              skip_batches: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield `(inputs, targets, topk_vals, topk_idx)` for one pass. Shuffle + `skip_batches`
        semantics are byte-identical to `PackedLoader.epoch` (so resume order matches)."""
        if reseed is not None:
            self.rng = np.random.default_rng(reseed)
        order = np.arange(self.n_chunks)
        if self.shuffle:
            self.rng.shuffle(order)
        if skip_batches:
            order = order[skip_batches * self.batch_size:]

        batch = []
        for idx in order:
            batch.append(self._chunk(int(idx)))
            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._collate(batch)

    def _collate(self, batch: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        inputs = np.stack([b[0] for b in batch])      # (B, seq_len)
        targets = np.stack([b[1] for b in batch])
        vals = np.stack([b[2] for b in batch])        # (B, seq_len, k)
        idx = np.stack([b[3] for b in batch])
        if self.vocab_size is not None:
            # Check inputs AND targets (targets are inputs shifted by one, so a bad/overflow id
            # in the last position appears only in targets) — mirrors PackedLoader, which checks
            # the max over the full seq_len+1 chunk before the input/target split.
            top = int(max(int(inputs.max()), int(targets.max())))
            if top >= self.vocab_size:
                raise ValueError(
                    f"token id {top} >= vocab_size {self.vocab_size} in {self.path}. "
                    "The packed data does not match the model config — likely a stale or "
                    "clobbered data file.")
        return inputs, targets, vals, idx
