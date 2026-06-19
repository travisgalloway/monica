"""Stage 6b: Mamba-aware pack + shard (#74).

Pack per-document token streams into fixed-length sequences (default **8192** — linear
memory makes long sequences affordable) and write **few large shards** (high-hundreds-of-MB
to low-GB) to keep R2 Class-A op counts down. Alongside each token shard we write a
**document-boundary sidecar** (a uint8 flag, 1 at each doc's first token) so the loader and
#68 can **reset the SSM state at boundaries** instead of letting recurrent state bleed
across packed docs.

ABOVE THE SEAM — numpy + stdlib only. Token shards are the same flat format as ``pack.py``
(uint16 for the POC vocab, uint32 for the Qwen2.5 distillation vocab, #90 — the dtype is
recorded in the manifest, so ``PackedLoader``/``open_shard`` read them back correctly); the
`.bounds` sidecar is the new artifact a boundary-aware loader consumes.

Layout per output dir:
    part-00000.bin     uint16/uint32 tokens, length = n_sequences * seq_len
    part-00000.bounds  uint8 doc-start flags, same length
    manifest.json      {seq_len, dtype, tokenizer, shards:[{name,n_sequences,n_tokens}], ...}
"""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .pack import DTYPE, typecode_for


def pack_sequences(token_docs: Iterable[Sequence[int]], out_dir, *, seq_len: int = 8192,
                   shard_size_mb: int = 512, prefix: str = "part", tokenizer: str = "",
                   chunk_align: int | None = None, pad_id: int = 0, dtype=DTYPE) -> dict:
    """Pack per-document token lists into fixed `seq_len` sequences across few large shards,
    writing token `.bin` + doc-boundary `.bounds` sidecars + a `manifest.json`. The final
    partial sequence (< seq_len) is dropped. Returns the manifest dict.

    `token_docs` yields one id list per document (see `tokenize.tokenize_docs`); each doc's
    first token is flagged 1 in `.bounds`, the rest 0.

    `chunk_align` (set it to the model's `chunk_size`) pads each document up to a multiple of
    that length with `pad_id`, so every document **starts on a chunk boundary** — the
    requirement for the SSM's packing-aware boundary reset (#68). `seq_len` should be a
    multiple of `chunk_align`.

    `dtype` is the packed token dtype: uint16 (POC default) or uint32 (Qwen2.5 vocab, #90).
    The shards are the same flat format as `pack.py`; the manifest records the dtype."""
    dtype = np.dtype(dtype)
    hi = int(np.iinfo(dtype).max)
    if chunk_align is not None:
        if chunk_align <= 0:
            raise ValueError(f"chunk_align must be positive, got {chunk_align}")
        if seq_len % chunk_align:
            raise ValueError(
                f"seq_len {seq_len} must be a multiple of chunk_align {chunk_align}")
    if not 0 <= pad_id <= hi:
        raise ValueError(f"pad_id {pad_id} out of {dtype.name} range [0, {hi}]")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shard budget rounded down to a whole number of sequences (>= 1 sequence).
    bytes_per_token = dtype.itemsize
    budget = max(seq_len, (shard_size_mb * (1 << 20)) // bytes_per_token)
    budget -= budget % seq_len

    tok_buf = array(typecode_for(dtype))   # uint16 'H' / uint32 'I' — overflow-checked
    bnd_buf = bytearray()
    shards: List[dict] = []
    state = {"idx": 0, "docs": 0, "seqs": 0, "tokens": 0}

    def emit(n_tokens: int) -> None:
        name = f"{prefix}-{state['idx']:05d}"
        toks = np.frombuffer(tok_buf, dtype=dtype, count=n_tokens)
        toks.tofile(out_dir / f"{name}.bin")
        del toks   # release the exported buffer before resizing tok_buf below
        (out_dir / f"{name}.bounds").write_bytes(bytes(bnd_buf[:n_tokens]))
        n_seq = n_tokens // seq_len
        shards.append({"name": name, "n_sequences": n_seq, "n_tokens": n_tokens})
        state["idx"] += 1
        state["seqs"] += n_seq
        state["tokens"] += n_tokens
        # Count docs by the doc-starts actually emitted, so n_documents matches the
        # .bounds sidecar and excludes starts that land in the dropped final partial.
        state["docs"] += int(sum(bnd_buf[:n_tokens]))
        del tok_buf[:n_tokens]
        del bnd_buf[:n_tokens]

    for doc in token_docs:
        ids = list(doc)
        if not ids:
            continue
        if chunk_align is not None:               # pad doc so the NEXT doc is chunk-aligned
            rem = len(ids) % chunk_align
            if rem:
                ids = ids + [pad_id] * (chunk_align - rem)
        tok_buf.extend(ids)                       # OverflowError if any id exceeds the dtype
        bnd_buf.append(1)
        bnd_buf.extend(b"\x00" * (len(ids) - 1))
        while len(tok_buf) >= budget:
            emit(budget)
    # Flush remaining COMPLETE sequences; drop the final partial.
    full = (len(tok_buf) // seq_len) * seq_len
    if full:
        emit(full)

    manifest = {"seq_len": seq_len, "dtype": dtype.name, "tokenizer": tokenizer,
                "n_documents": state["docs"], "n_sequences": state["seqs"],
                "n_tokens": state["tokens"], "shards": shards}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def pack_atomic(token_docs: Iterable[Sequence[int]], out_dir, *, seq_len: int = 8192,
                shard_size_mb: int = 512, prefix: str = "part", tokenizer: str = "",
                chunk_align: int | None = None, pad_id: int = 0, dtype=DTYPE) -> dict:
    """Pack per-document token lists into fixed `seq_len` sequences such that **no document spans a
    sequence boundary** — the atomic-packing guarantee #96 needs for reasoning traces.

    Unlike `pack_sequences` (which concatenates docs into a flat stream, so a doc can straddle a
    `seq_len` edge and be split across two training rows), this greedily bin-packs: a document that
    will not fit in the current sequence's remaining space pads that sequence to its end with
    `pad_id` and starts the document at the head of the next sequence. Documents whose
    (chunk-aligned) length exceeds `seq_len` cannot be packed atomically and are **dropped**
    (counted in `n_dropped_overlength`); the final partial sequence is padded out and kept (no
    document is lost to a dropped tail). `chunk_align` still pads each document to a chunk multiple
    so it starts on a chunk boundary for the SSM reset (#68); writes the same
    `.bin`/`.bounds`/`manifest.json` artifacts as `pack_sequences`."""
    dtype = np.dtype(dtype)
    hi = int(np.iinfo(dtype).max)
    if chunk_align is not None:
        if chunk_align <= 0:
            raise ValueError(f"chunk_align must be positive, got {chunk_align}")
        if seq_len % chunk_align:
            raise ValueError(
                f"seq_len {seq_len} must be a multiple of chunk_align {chunk_align}")
    if not 0 <= pad_id <= hi:
        raise ValueError(f"pad_id {pad_id} out of {dtype.name} range [0, {hi}]")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bytes_per_token = dtype.itemsize
    budget = max(seq_len, (shard_size_mb * (1 << 20)) // bytes_per_token)
    budget -= budget % seq_len

    tok_buf = array(typecode_for(dtype))
    bnd_buf = bytearray()
    cur: List[int] = []                 # the in-progress sequence (< seq_len until flushed)
    cur_bnd: List[int] = []
    shards: List[dict] = []
    state = {"idx": 0, "docs": 0, "seqs": 0, "tokens": 0, "dropped": 0}

    def flush_seq() -> None:
        """Pad the in-progress sequence out to `seq_len` and append it to the shard buffers."""
        nonlocal cur, cur_bnd
        if not cur:
            return
        pad = seq_len - len(cur)
        tok_buf.extend(cur + [pad_id] * pad)         # OverflowError if any id exceeds the dtype
        bnd_buf.extend(bytes(cur_bnd) + b"\x00" * pad)
        cur, cur_bnd = [], []

    def emit(n_tokens: int) -> None:
        name = f"{prefix}-{state['idx']:05d}"
        toks = np.frombuffer(tok_buf, dtype=dtype, count=n_tokens)
        toks.tofile(out_dir / f"{name}.bin")
        del toks
        (out_dir / f"{name}.bounds").write_bytes(bytes(bnd_buf[:n_tokens]))
        n_seq = n_tokens // seq_len
        shards.append({"name": name, "n_sequences": n_seq, "n_tokens": n_tokens})
        state["idx"] += 1
        state["seqs"] += n_seq
        state["tokens"] += n_tokens
        state["docs"] += int(sum(bnd_buf[:n_tokens]))
        del tok_buf[:n_tokens]
        del bnd_buf[:n_tokens]

    for doc in token_docs:
        ids = list(doc)
        if not ids:
            continue
        if chunk_align is not None:
            rem = len(ids) % chunk_align
            if rem:
                ids = ids + [pad_id] * (chunk_align - rem)
        if len(ids) > seq_len:                       # cannot fit atomically -> drop
            state["dropped"] += 1
            continue
        if len(cur) + len(ids) > seq_len:            # won't fit -> pad current seq, start a new one
            flush_seq()
        cur.extend(ids)
        cur_bnd.append(1)
        cur_bnd.extend([0] * (len(ids) - 1))
        while len(tok_buf) >= budget:
            emit(budget)
    flush_seq()                                      # pad + keep the final partial sequence
    while len(tok_buf) >= budget:
        emit(budget)
    if len(tok_buf):                                 # remaining whole sequences (< one shard)
        emit(len(tok_buf))

    manifest = {"seq_len": seq_len, "dtype": dtype.name, "tokenizer": tokenizer,
                "n_documents": state["docs"], "n_sequences": state["seqs"],
                "n_tokens": state["tokens"], "n_dropped_overlength": state["dropped"],
                "shards": shards}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def open_shard(out_dir, name: str) -> Tuple[np.memmap, np.memmap]:
    """Memory-map a shard's (tokens, boundaries uint8) pair, read-only. Token dtype comes
    from the manifest (uint16 / uint32; fallback uint16 for legacy shards)."""
    out_dir = Path(out_dir)
    manifest_path = Path(out_dir) / "manifest.json"
    dtype = np.dtype(DTYPE)
    if manifest_path.exists():
        dtype = np.dtype(json.loads(manifest_path.read_text()).get("dtype", "uint16"))
    toks = np.memmap(out_dir / f"{name}.bin", dtype=dtype, mode="r")
    bnds = np.memmap(out_dir / f"{name}.bounds", dtype=np.uint8, mode="r")
    return toks, bnds


def read_manifest(out_dir) -> dict:
    return json.loads((Path(out_dir) / "manifest.json").read_text())


def doc_start_offsets(bounds: Sequence[int]) -> List[int]:
    """Global token offsets where a document begins (where bounds == 1)."""
    return [int(i) for i in np.nonzero(np.asarray(bounds))[0]]


def segment_ids(bounds: Sequence[int]) -> np.ndarray:
    """Per-position document id from the doc-start `.bounds` flags: `cumsum(bounds) - 1`.

    This is the `seg_ids` the boundary-aware forward (#68) consumes — feed a sequence's
    slice of it alongside the tokens so SSM/attention state resets at each document."""
    return np.cumsum(np.asarray(bounds, dtype=np.int64)) - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="cleaned corpus shards (dir / .parquet) to tokenize + pack")
    ap.add_argument("--out", type=Path, required=True, help="output dir for token shards")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--shard-size-mb", type=int, default=512)
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--tokenizer", choices=("qwen25", "starcoder2", "olmo"),
                    default="qwen25")
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()

    from .corpus import iter_shard_texts
    from .pack import packing_dtype_for
    from .tokenize import (ByteTokenizer, load_olmo_tokenizer, load_qwen25_tokenizer,
                           load_starcoder2_tokenizer, tokenize_docs)

    if args.byte_fallback:
        tok = ByteTokenizer()
    elif args.tokenizer == "olmo":
        tok = load_olmo_tokenizer(args.model_id)
    elif args.tokenizer == "starcoder2":
        tok = load_starcoder2_tokenizer(args.model_id)
    else:
        tok = load_qwen25_tokenizer(args.model_id)

    tok_label = "byte" if args.byte_fallback else getattr(tok, "name_or_path", args.tokenizer)
    dtype = packing_dtype_for(tok.vocab_size)          # uint16 (POC) / uint32 (Qwen2.5)
    docs = tokenize_docs(iter_shard_texts(args.inp), tok)
    manifest = pack_sequences(docs, args.out, seq_len=args.seq_len,
                              shard_size_mb=args.shard_size_mb, tokenizer=tok_label, dtype=dtype)
    print(f"packed {manifest['n_sequences']} seq x {args.seq_len} "
          f"({manifest['n_tokens']} tokens, {len(manifest['shards'])} shard(s)) -> {args.out}")


if __name__ == "__main__":
    main()
