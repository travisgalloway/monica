"""Stream the cleaned jsonl.gz corpus (local or R2) into build_distill_corpus.

The cleaned shards are already normalized/filtered/deduped text, so this skips the
datatrove re-clean and feeds iter_jsonl_texts -> Record -> build_distill_corpus
(clean[no-op]->qwen3 tokenize->uint32 pack), writing the poc-distill/corpus layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.corpus import Record, iter_jsonl_texts
from src.data.distill_corpus import build_distill_corpus


def _records(uri: str, source: str, lang: str = "en", license: str = "odc-by"):
    for text in iter_jsonl_texts(uri):
        yield Record(text=text, source=source, lang=lang, license=license)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="cleaned jsonl(.gz) dir or file; local path or s3://...")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--source-name", default="reserve-pretrain")
    ap.add_argument("--tokenizer", default="qwen3",
                    choices=("qwen3", "qwen25", "olmo", "starcoder2"))
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--shard-size-mb", type=int, default=512)
    ap.add_argument("--clean-shard-size-mb", type=int, default=512)
    ap.add_argument("--byte-fallback", action="store_true")
    ap.add_argument("--lang", default="en",
                    help="BCP-47 language tag written into each Record (default: en)")
    ap.add_argument("--license", default="odc-by",
                    help="SPDX license written into each Record (default: odc-by)")
    args = ap.parse_args()

    m = build_distill_corpus(
        _records(args.inp, args.source_name, lang=args.lang, license=args.license),
        args.out_root,
        tokenizer=args.tokenizer, model_id=args.model_id, seq_len=args.seq_len,
        byte_fallback=args.byte_fallback, shard_size_mb=args.shard_size_mb,
        clean_shard_size_mb=args.clean_shard_size_mb)
    print(f"distill corpus: {m['n_tokens']} tokens ({m['dtype']}, {m['n_documents']} docs, "
          f"{m['n_sequences']} seq x {m['seq_len']}, tok={m['tokenizer']}) -> {m['tokenized_dir']}")


if __name__ == "__main__":
    main()
