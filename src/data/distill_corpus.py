"""Build the FROZEN distillation corpus (#92): the cleaned + Qwen2.5-tokenized,
uint32-packed token corpus every student trial consumes.

This is a thin **orchestrator** over the existing data stages — it adds no new cleaning or
packing logic, only the `poc-distill/` prefixed layout and a corpus-level manifest tying the
two artifacts together:

    <out_root>/corpus/
        cleaned/    part-*.parquet          # durable, re-mixable text (corpus.build_corpus)
        tokenized/  qwen25-8k/              # training shards (shard.pack_sequences)
            part-*.bin     uint32 tokens (Qwen2.5 vocab 151,646 -> uint32, #90)
            part-*.bounds  uint8 doc-start flags  (SSM state reset, #68)
            manifest.json  {seq_len, dtype, tokenizer, n_tokens, ...}
        manifest.json       # the two-stage summary (the artifact #94 freezes against)

The corpus is **precomputed once** (docs/design/10-distillation.md): the teacher logit
precompute (#94) and every student layout sweep (#98/#100) read it unchanged. The student
manifests (`config/manifests/student-1b-*.yaml`) already name
`poc-distill/corpus/tokenized/qwen25-8k`; `tokenized_subdir` produces exactly that name.

ABOVE THE SEAM — no `mlx`/`torch`. Heavy data deps (pyarrow via `corpus`, `transformers` via
the tokenizer loaders) are imported LAZILY, so importing this module stays cheap and the seam
guard (tests/test_import_guard.py) needs nothing extra.

CLI (mirrors download/tokenize/pack/corpus):
    # real run (needs the HF Qwen2.5 tokenizer): a few-B-token slice
    python -m src.data.distill_corpus --source text --in data/raw/slice.txt --tokenizer qwen25
    # offline smoke (no network/tokenizer; byte fallback):
    python -m src.data.distill_corpus --source dummy --byte-fallback --out-root /tmp/pd
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from . import corpus
from . import storage
from .corpus import Record, ingest_dummy, ingest_text_file, iter_shard_texts

#: Default local root for the distillation-corpus prefix (the `poc-distill` class of the #97
#: three-class layout, under the default `data/` base).
DEFAULT_OUT_ROOT = storage.class_root("data", storage.POC_DISTILL)

#: Tokenized-folder name-pin (`<tokenizer>-<seqlen_k>`). Canonical definition lives in `storage`
#: (#97); kept here as an alias because the student sweep manifests + tests reference this name.
tokenized_subdir = storage.tokenized_dir_name


def _load_tokenizer(tokenizer: str, model_id: str | None, byte_fallback: bool):
    """Resolve the tokenizer, reusing the `tokenize.py` loaders. `byte_fallback` is the
    offline-only path (vocab 256 -> uint16); a real run uses `qwen25` (vocab 151,646 -> uint32)."""
    from .tokenize import (ByteTokenizer, load_olmo_tokenizer, load_qwen25_tokenizer,
                           load_starcoder2_tokenizer)

    if byte_fallback:
        return ByteTokenizer()
    loaders = {"qwen25": load_qwen25_tokenizer, "olmo": load_olmo_tokenizer,
               "starcoder2": load_starcoder2_tokenizer}
    return loaders[tokenizer](model_id)


def build_distill_corpus(records: Iterable[Record], out_root, *, tokenizer: str = "qwen25",
                         model_id: str | None = None, seq_len: int = 8192,
                         byte_fallback: bool = False, chunk_align: int | None = None,
                         shard_size_mb: int = 512, clean_shard_size_mb: int = 128,
                         **clean_kwargs) -> dict:
    """Build the distillation corpus end to end: clean text shards, then Qwen2.5-tokenize +
    uint32-pack into fixed `seq_len` sequences with doc-boundary sidecars, under `out_root`.

    Stage 1 reuses `corpus.build_corpus` (normalize -> filter/dedup -> Parquet text shards);
    `clean_kwargs` are forwarded to it (e.g. `quality=True`, `dedup="minhash"`). Stage 2 reuses
    `shard.pack_sequences`; the packed dtype follows the tokenizer vocab via `packing_dtype_for`
    (uint16 for the byte/OLMo path, uint32 for Qwen2.5). Returns the corpus-level manifest dict.
    """
    from .pack import packing_dtype_for
    from .shard import pack_sequences
    from .tokenize import tokenize_docs

    out_root = Path(out_root)
    corpus_root = out_root / "corpus"
    cleaned_dir = storage.corpus_cleaned_dir(out_root)
    tokenized_dir = storage.corpus_tokenized_dir(out_root, tokenizer, seq_len)

    # Stage 1 — cleaned, re-mixable text shards (durable artifact).
    cleaned_shards = corpus.build_corpus(records, cleaned_dir,
                                         shard_size_mb=clean_shard_size_mb, **clean_kwargs)

    # Stage 2 — tokenize + pack the cleaned text. Pass the short tokenizer label so the pack
    # manifest records `tokenizer: qwen25` (the acceptance criterion), not the HF repo path.
    tok = _load_tokenizer(tokenizer, model_id, byte_fallback)
    dtype = packing_dtype_for(tok.vocab_size)          # uint16 (byte/OLMo) / uint32 (Qwen2.5)
    docs = tokenize_docs(iter_shard_texts(cleaned_dir), tok)
    pack_manifest = pack_sequences(docs, tokenized_dir, seq_len=seq_len,
                                   shard_size_mb=shard_size_mb, tokenizer=tokenizer,
                                   chunk_align=chunk_align, dtype=dtype)

    # Corpus-level manifest — the two-stage summary the teacher precompute (#94) freezes against.
    manifest = {
        "tokenizer": tokenizer,
        "model_id": getattr(tok, "name_or_path", None) if not byte_fallback else None,
        "byte_fallback": byte_fallback,
        "seq_len": seq_len,
        "dtype": pack_manifest["dtype"],
        "n_tokens": pack_manifest["n_tokens"],
        "n_documents": pack_manifest["n_documents"],
        "n_sequences": pack_manifest["n_sequences"],
        "n_cleaned_shards": len(cleaned_shards),
        "cleaned_dir": str(cleaned_dir),
        "tokenized_dir": str(tokenized_dir),
    }
    (corpus_root).mkdir(parents=True, exist_ok=True)
    (corpus_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("dummy", "text"), default="text",
                    help="dummy: synthetic offline docs; text: a UTF-8 text file (--in)")
    ap.add_argument("--in", dest="inp", default=None, help="input text file (--source text)")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                    help="root for the poc-distill prefix (writes <root>/corpus/...)")
    ap.add_argument("--source-name", default=None,
                    help="value for the schema `source` field (default: source type / filename)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--license", default="unknown")
    ap.add_argument("--max-docs", type=int, default=1000, help="--source dummy: #synthetic docs")
    # Tokenize + pack
    ap.add_argument("--tokenizer", choices=("qwen25", "olmo", "starcoder2"), default="qwen25",
                    help="HF tokenizer; the packed dtype is uint16/uint32 per its vocab")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--chunk-align", type=int, default=None,
                    help="pad each doc to a multiple of this (the model chunk_size) so docs "
                         "start on a chunk boundary for the SSM reset (#68)")
    ap.add_argument("--shard-size-mb", type=int, default=512, help="tokenized shard roll size")
    ap.add_argument("--clean-shard-size-mb", type=int, default=128, help="cleaned shard roll size")
    # Stage-3/4 cleaning passthrough (default-off keeps the local gate path unchanged)
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--quality", action="store_true", help="Stage-3 text-quality heuristics")
    ap.add_argument("--dedup", choices=("exact", "minhash"), default=None,
                    help="Stage-4 dedup: exact-hash, or exact+MinHash-LSH near-dedup")
    args = ap.parse_args()

    if args.source == "dummy":
        records = ingest_dummy(args.max_docs, source=args.source_name or "dummy")
    else:
        if not args.inp:
            ap.error("--in is required for --source text")
        name = args.source_name or Path(args.inp).stem
        records = ingest_text_file(args.inp, source=name, lang=args.lang, license=args.license)

    manifest = build_distill_corpus(
        records, args.out_root, tokenizer=args.tokenizer, model_id=args.model_id,
        seq_len=args.seq_len, byte_fallback=args.byte_fallback, chunk_align=args.chunk_align,
        shard_size_mb=args.shard_size_mb, clean_shard_size_mb=args.clean_shard_size_mb,
        min_chars=args.min_chars, quality=args.quality, dedup=args.dedup)
    print(f"distill corpus: {manifest['n_tokens']} tokens "
          f"({manifest['dtype']}, {manifest['n_documents']} docs, "
          f"{manifest['n_sequences']} seq x {manifest['seq_len']}, "
          f"tokenizer={manifest['tokenizer']}) -> {manifest['tokenized_dir']}")


if __name__ == "__main__":
    main()
