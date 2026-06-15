"""Stage 6b: Mamba-aware pack + shard (#74).

Pack per-document token streams into fixed-length sequences (default **8192** — linear
memory makes long sequences affordable) and write **few large shards** (high-hundreds-of-MB
to low-GB) to keep R2 Class-A op counts down. Alongside each token shard we write a
**document-boundary sidecar** (a uint8 flag, 1 at each doc's first token) so the loader and
#68 can **reset the SSM state at boundaries** instead of letting recurrent state bleed
across packed docs.

ABOVE THE SEAM — numpy + stdlib only. Token shards are the same flat uint16 format as
``pack.py`` (so the existing ``PackedLoader`` reads them unchanged); the `.bounds` sidecar
is the new, optional artifact a boundary-aware loader consumes. The storage URI is an
fsspec seam (`file://` now, `s3://` at #80) — same code path.

Layout per output dir:
    part-00000.bin     uint16 tokens, length = n_sequences * seq_len
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

DTYPE = np.uint16


def pack_sequences(token_docs: Iterable[Sequence[int]], out_dir, *, seq_len: int = 8192,
                   shard_size_mb: int = 512, prefix: str = "part",
                   tokenizer: str = "") -> dict:
    """Pack per-document token lists into fixed `seq_len` sequences across few large shards,
    writing token `.bin` + doc-boundary `.bounds` sidecars + a `manifest.json`. The final
    partial sequence (< seq_len) is dropped. Returns the manifest dict.

    `token_docs` yields one id list per document (see `tokenize.tokenize_docs`); each doc's
    first token is flagged 1 in `.bounds`, the rest 0."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shard budget rounded down to a whole number of sequences (>= 1 sequence).
    budget = max(seq_len, (shard_size_mb * (1 << 20)) // 2)   # 2 bytes/token (uint16)
    budget -= budget % seq_len

    tok_buf = array("H")          # uint16 — raises OverflowError on out-of-range ids
    bnd_buf = bytearray()
    shards: List[dict] = []
    state = {"idx": 0, "docs": 0, "seqs": 0, "tokens": 0}

    def emit(n_tokens: int) -> None:
        name = f"{prefix}-{state['idx']:05d}"
        toks = np.frombuffer(tok_buf, dtype=DTYPE, count=n_tokens)
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
        tok_buf.extend(ids)                       # OverflowError if any id not in [0, 65535]
        bnd_buf.append(1)
        bnd_buf.extend(b"\x00" * (len(ids) - 1))
        while len(tok_buf) >= budget:
            emit(budget)
    # Flush remaining COMPLETE sequences; drop the final partial.
    full = (len(tok_buf) // seq_len) * seq_len
    if full:
        emit(full)

    manifest = {"seq_len": seq_len, "dtype": "uint16", "tokenizer": tokenizer,
                "n_documents": state["docs"], "n_sequences": state["seqs"],
                "n_tokens": state["tokens"], "shards": shards}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def open_shard(out_dir, name: str) -> Tuple[np.memmap, np.memmap]:
    """Memory-map a shard's (tokens uint16, boundaries uint8) pair, read-only."""
    out_dir = Path(out_dir)
    toks = np.memmap(out_dir / f"{name}.bin", dtype=DTYPE, mode="r")
    bnds = np.memmap(out_dir / f"{name}.bounds", dtype=np.uint8, mode="r")
    return toks, bnds


def read_manifest(out_dir) -> dict:
    return json.loads((Path(out_dir) / "manifest.json").read_text())


def doc_start_offsets(bounds: Sequence[int]) -> List[int]:
    """Global token offsets where a document begins (where bounds == 1)."""
    return [int(i) for i in np.nonzero(np.asarray(bounds))[0]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="cleaned corpus shards (dir / .parquet) to tokenize + pack")
    ap.add_argument("--out", type=Path, required=True, help="output dir for token shards")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--shard-size-mb", type=int, default=512)
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--tokenizer", choices=("starcoder2", "olmo"), default="starcoder2")
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()

    from .corpus import iter_shard_texts
    from .tokenize import (ByteTokenizer, load_olmo_tokenizer,
                           load_starcoder2_tokenizer, tokenize_docs)

    if args.byte_fallback:
        tok = ByteTokenizer()
    elif args.tokenizer == "olmo":
        tok = load_olmo_tokenizer(args.model_id)
    else:
        tok = load_starcoder2_tokenizer(args.model_id)

    tok_label = "byte" if args.byte_fallback else getattr(tok, "name_or_path", args.tokenizer)
    docs = tokenize_docs(iter_shard_texts(args.inp), tok)
    manifest = pack_sequences(docs, args.out, seq_len=args.seq_len,
                              shard_size_mb=args.shard_size_mb, tokenizer=tok_label)
    print(f"packed {manifest['n_sequences']} seq x {args.seq_len} "
          f"({manifest['n_tokens']} tokens, {len(manifest['shards'])} shard(s)) -> {args.out}")


if __name__ == "__main__":
    main()
