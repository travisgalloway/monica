"""Build the shared **instruct SFT corpus** under the Qwen chat template (#95).

Phase-1 shared artifact (docs/design/11-post-training.md): general instruction->response pairs
rendered under Qwen **ChatML** (`src/data/chat_template.py`) with the assistant spans masked for
loss, written as the two-artifact `shared/sft/` layout the student manifests reference:

    <out_root>/sft/
        cleaned/instruct/records.jsonl              # tokenizer-agnostic {messages,source,license}
        tokenized/qwen3-8k/instruct.jsonl           # response-masked {input_ids,target_ids,loss_mask}
        tokenized/qwen3-8k/manifest.json            # tokenizer, template, chat_eos, counts

The cleaned rows are durable + re-tokenizable; the tokenized records drop straight into the M9
`SFTLoader` / `make_sft_train_step` the instruct SFT layer (#101) reuses. The chat EOS is
`<|im_end|>`, trained as the stop token and kept identical to serving (the invariant in
`chat_template`).

ABOVE THE SEAM — no `mlx`/`torch`. Reuses the clean-license loaders in `sft_sources` and the
message-normalizer in `sft_data`; `datasets` is imported lazily by those loaders. Offline path:
the checked-in `handauthored` set + `--byte-fallback`.

CLI (mirrors sft_sources):
    # offline smoke:
    python -m src.data.instruct_sft --sources handauthored --byte-fallback --out-root /tmp/shared
    # real run (HF Qwen3 tokenizer + datasets):
    python -m src.data.instruct_sft --sources oasst1 flan handauthored --tokenizer qwen3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from . import chat_template, storage
from .sft_data import _messages_of


def build_chat_sft_records(rows: Iterable[dict], tokenizer, *, max_seq_len: Optional[int] = 8192,
                           stats: Optional[dict] = None) -> Iterator[dict]:
    """Yield `{input_ids, target_ids, loss_mask}` records from chat rows, masked to the assistant
    turns under the Qwen chat template. Mirrors `sft_data.build_sft_records` but renders ChatML and
    appends **no** extra EOS — `<|im_end|>` is already the last token of each assistant span and is
    trained as the stop token. Rows whose tokenized length exceeds `max_seq_len` are dropped
    (truncating a response would teach the model to stop mid-answer)."""
    vocab = getattr(tokenizer, "vocab_size", None)
    for rec in rows:
        messages = _messages_of(rec)
        out = _record_from_messages(messages, tokenizer, max_seq_len, vocab)
        if out is None:
            if stats is not None:
                stats["skipped"] = stats.get("skipped", 0) + 1
            continue
        if stats is not None:
            stats["kept"] = stats.get("kept", 0) + 1
        yield out


def _record_from_messages(messages: List[dict], tokenizer, max_seq_len: Optional[int],
                          vocab: Optional[int]) -> Optional[dict]:
    """One masked record, or None to skip (no assistant content / over length)."""
    if not messages or messages[-1]["role"] != "assistant":
        return None
    seq, spans = chat_template.response_spans(messages, tokenizer)
    if not spans:
        return None
    if max_seq_len is not None and len(seq) - 1 > max_seq_len:
        return None
    if vocab is not None and seq and max(seq) >= vocab:
        raise ValueError(
            f"token id {max(seq)} >= tokenizer vocab_size {vocab} — wrong tokenizer?")

    # mask[j] == 1 iff target_ids[j] (= seq[j+1]) is an assistant-span token (content + <|im_end|>).
    mask = [0] * (len(seq) - 1)
    for s, e in spans:
        for j in range(max(0, s - 1), min(e - 1, len(mask))):
            mask[j] = 1
    if not any(mask):
        return None
    return {"input_ids": seq[:-1], "target_ids": seq[1:], "loss_mask": mask}


def _write_cleaned(rows: Iterable[dict], path: Path) -> List[dict]:
    """Write the tokenizer-agnostic `{messages, source, license}` rows as JSONL; return them
    (materialized so they can be re-iterated for tokenization)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kept: List[dict] = []
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in rows:
            messages = _messages_of(rec)
            if not messages or messages[-1]["role"] != "assistant":
                continue
            row = {"messages": messages, "source": rec.get("source", "unknown"),
                   "license": rec.get("license", "unknown")}
            f.write(json.dumps(row) + "\n")
            kept.append(row)
    return kept


def _load_tokenizer(tokenizer: str, model_id: Optional[str], byte_fallback: bool):
    from .tokenize import (ByteTokenizer, load_olmo_tokenizer, load_qwen3_tokenizer,
                           load_qwen25_tokenizer, load_starcoder2_tokenizer)
    if byte_fallback:
        return ByteTokenizer()
    loaders = {"qwen3": load_qwen3_tokenizer, "qwen25": load_qwen25_tokenizer,
               "olmo": load_olmo_tokenizer, "starcoder2": load_starcoder2_tokenizer}
    return loaders[tokenizer](model_id)


def build_instruct_sft(rows: Iterable[dict], out_root, *, tokenizer: str = "qwen3",
                       model_id: Optional[str] = None, seq_len: int = 8192,
                       byte_fallback: bool = False, max_seq_len: int = 8192) -> dict:
    """Build the shared instruct SFT corpus end to end: write cleaned chat rows, then tokenize +
    response-mask them under the Qwen chat template into the `shared/sft/tokenized/<tok>-<k>` prefix
    with a manifest. Returns the manifest dict."""
    out_root = Path(out_root)
    cleaned_path = storage.sft_cleaned_dir(out_root, "instruct") / "records.jsonl"
    tok_dir = storage.sft_tokenized_dir(out_root, tokenizer, seq_len)
    tokenized_path = tok_dir / "instruct.jsonl"

    cleaned_rows = _write_cleaned(rows, cleaned_path)

    tok = _load_tokenizer(tokenizer, model_id, byte_fallback)
    tok_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    n_tokens = 0
    sources: dict = {}
    with open(tokenized_path, "w", encoding="utf-8", newline="\n") as f:
        for row, out in _records_with_source(cleaned_rows, tok, max_seq_len, stats):
            f.write(json.dumps(out) + "\n")
            n_tokens += len(out["input_ids"])
            sources[row["source"]] = sources.get(row["source"], 0) + 1

    manifest = {
        "tokenizer": tokenizer,
        "model_id": getattr(tok, "name_or_path", None) if not byte_fallback else None,
        "byte_fallback": byte_fallback,
        "template": "qwen-chatml",
        "chat_eos": chat_template.CHAT_EOS,
        "seq_len": seq_len,
        "max_seq_len": max_seq_len,
        "n_records": stats.get("kept", 0),
        "n_skipped": stats.get("skipped", 0),
        "n_tokens": n_tokens,
        "sources": sources,
        "cleaned_path": str(cleaned_path),
        "tokenized_path": str(tokenized_path),
    }
    (tok_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _records_with_source(cleaned_rows: List[dict], tok, max_seq_len: int, stats: dict):
    """Yield `(cleaned_row, masked_record)` for rows that survive masking — pairs the kept record
    with its source row so per-source counts stay accurate."""
    vocab = getattr(tok, "vocab_size", None)
    for row in cleaned_rows:
        out = _record_from_messages(row["messages"], tok, max_seq_len, vocab)
        if out is None:
            stats["skipped"] = stats.get("skipped", 0) + 1
            continue
        stats["kept"] = stats.get("kept", 0) + 1
        yield row, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from .sft_sources import iter_clean_sft, _LOADERS
    ap.add_argument("--sources", nargs="+", default=["handauthored"],
                    choices=tuple(_LOADERS), help="clean-license SFT sources to include")
    ap.add_argument("--out-root", type=Path, default=Path("data/shared"),
                    help="root for the shared prefix (writes <root>/sft/...)")
    ap.add_argument("--tokenizer", choices=("qwen3", "qwen25", "olmo", "starcoder2"),
                    default="qwen3")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--seq-len", type=int, default=8192, help="tokenized-prefix dir width (qwen3-8k)")
    ap.add_argument("--max-seq-len", type=int, default=8192,
                    help="drop examples longer than this (truncation would teach mid-answer stops)")
    ap.add_argument("--max-per-source", type=int, default=None)
    args = ap.parse_args()

    rows = iter_clean_sft(args.sources, max_per_source=args.max_per_source)
    manifest = build_instruct_sft(rows, args.out_root, tokenizer=args.tokenizer,
                                  model_id=args.model_id, seq_len=args.seq_len,
                                  byte_fallback=args.byte_fallback, max_seq_len=args.max_seq_len)
    print(f"instruct sft: {manifest['n_records']} records ({manifest['n_skipped']} skipped, "
          f"{manifest['n_tokens']} tokens, template={manifest['template']}, "
          f"chat_eos={manifest['chat_eos']}, sources={manifest['sources']}) "
          f"-> {manifest['tokenized_path']}")


if __name__ == "__main__":
    main()
