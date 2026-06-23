"""Build the shared **reasoning-trace SFT corpus** (#96) — the thinking layer's data.

`<think>`/`<answer>`-formatted traces (`src/data/reasoning_traces.py`) rendered under the Qwen
chat template, written as the `shared/sft/` layout, in two complementary forms:

    <out_root>/sft/
        cleaned/reasoning-traces/records.jsonl        # tokenizer-agnostic {messages,source,license}
        tokenized/qwen3-8k/
            reasoning.jsonl                           # masked {input_ids,target_ids,loss_mask} (SFTLoader)
            reasoning-packed/                         # the atomic 8K packing (the #96 deliverable)
                part-*.bin   uint32 trace tokens
                part-*.bounds  doc-start flags (SSM reset, #68)
                manifest.json
            reasoning-manifest.json                   # summary across both forms

The **masked JSONL** is the trainable artifact (drop-in for the M9 `SFTLoader` /
`make_sft_train_step`); the **packed** form is the "long 8K packing with document boundaries
marked for reset" #96 asks to store: each trace is one document, padded to a `chunk_align`
multiple so it starts on a chunk boundary and **no trace spans a sequence boundary** — verifiable
via `n_documents == kept traces` and chunk-aligned `.bounds`. Over-length traces (> `seq_len`) are
**dropped, never truncated** (truncating teaches mid-reasoning stops).

Unlike the instruct corpus, reasoning content is **not** whitespace-collapsed — the `<think>`
trace keeps its internal newlines (SFT records store token ids, so newlines are encoded, not read
back as document boundaries). It therefore masks the already-clean `reasoning_traces` messages
directly via `chat_template.response_spans` (the same assistant-span masking as #95), computing the
spans once and reusing them for both the masked record and the packed document.

ABOVE THE SEAM — no `mlx`/`torch`. Reuses `chat_template`, `instruct_sft`, and `shard`; `datasets`
is lazy inside the loaders. Offline path: the checked-in handauthored traces + `--byte-fallback`.

CLI:
    python -m src.data.reasoning_sft --sources handauthored --byte-fallback --out-root /tmp/shared
    python -m src.data.reasoning_sft --sources mot --tokenizer qwen3          # real run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

from . import chat_template, storage


def _valid_rows(rows: Iterable[dict]) -> List[dict]:
    """Keep `{messages, source, license}` rows ending in a non-empty assistant turn (no whitespace
    collapse — reasoning traces keep their newlines)."""
    out: List[dict] = []
    for rec in rows:
        messages = rec.get("messages")
        if not messages or messages[-1].get("role") != "assistant" \
                or not (messages[-1].get("content") or "").strip():
            continue
        out.append({"messages": [dict(m) for m in messages],
                    "source": rec.get("source", "unknown"),
                    "license": rec.get("license", "unknown")})
    return out


def _load_tokenizer(tokenizer: str, model_id: Optional[str], byte_fallback: bool):
    from .tokenize import (ByteTokenizer, load_olmo_tokenizer, load_qwen3_tokenizer,
                           load_qwen25_tokenizer, load_starcoder2_tokenizer)
    if byte_fallback:
        return ByteTokenizer()
    loaders = {"qwen3": load_qwen3_tokenizer, "qwen25": load_qwen25_tokenizer,
               "olmo": load_olmo_tokenizer, "starcoder2": load_starcoder2_tokenizer}
    return loaders[tokenizer](model_id)


def build_reasoning_sft(rows: Iterable[dict], out_root, *, tokenizer: str = "qwen3",
                        model_id: Optional[str] = None, seq_len: int = 8192,
                        chunk_align: int = 64, byte_fallback: bool = False) -> dict:
    """Build the reasoning-trace SFT corpus: cleaned rows, masked JSONL records, and the atomic
    chunk-aligned packed `.bin`/`.bounds` artifact. Returns the summary manifest dict."""
    from .pack import packing_dtype_for
    from .shard import pack_atomic

    out_root = Path(out_root)
    cleaned_path = storage.sft_cleaned_dir(out_root, "reasoning-traces") / "records.jsonl"
    tok_dir = storage.sft_tokenized_dir(out_root, tokenizer, seq_len)
    masked_path = tok_dir / "reasoning.jsonl"
    packed_dir = tok_dir / "reasoning-packed"

    valid = _valid_rows(rows)

    # Cleaned (tokenizer-agnostic) — newlines preserved.
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cleaned_path, "w", encoding="utf-8", newline="\n") as f:
        for row in valid:
            f.write(json.dumps(row) + "\n")

    tok = _load_tokenizer(tokenizer, model_id, byte_fallback)
    dtype = packing_dtype_for(tok.vocab_size)          # uint16 (byte) / uint32 (Qwen3)
    vocab = getattr(tok, "vocab_size", None)

    # Masked JSONL records + the per-trace token streams for atomic packing (kept in lockstep, so
    # the packed documents are exactly the traces that survived masking + the length cap).
    tok_dir.mkdir(parents=True, exist_ok=True)
    n_records = n_tokens = 0
    n_overlength = 0
    sources: dict = {}
    trace_id_lists: List[List[int]] = []
    with open(masked_path, "w", encoding="utf-8", newline="\n") as f:
        for row in valid:
            # Tokenize + find assistant spans once; reuse for the masked record AND the packed doc.
            full_ids, spans = chat_template.response_spans(row["messages"], tok)
            if not spans:
                continue
            # A trace must fit in one sequence to be packed atomically. Use the SAME chunk-aligned
            # length the packer will use, so the masked and packed sets are identical (dropped, not
            # split, when over length — truncating would teach mid-reasoning stops).
            padded_len = -(-len(full_ids) // chunk_align) * chunk_align
            if padded_len > seq_len:
                n_overlength += 1
                continue
            if vocab is not None and full_ids and max(full_ids) >= vocab:
                raise ValueError(
                    f"token id {max(full_ids)} >= tokenizer vocab_size {vocab} — wrong tokenizer?")
            mask = [0] * (len(full_ids) - 1)
            for s, e in spans:
                for j in range(max(0, s - 1), min(e - 1, len(mask))):
                    mask[j] = 1
            if not any(mask):
                continue
            rec = {"input_ids": full_ids[:-1], "target_ids": full_ids[1:], "loss_mask": mask}
            f.write(json.dumps(rec) + "\n")
            n_records += 1
            n_tokens += len(rec["input_ids"])
            sources[row["source"]] = sources.get(row["source"], 0) + 1
            # One document per trace = the full rendered ChatML ids (atomic unit for packing).
            trace_id_lists.append(full_ids)

    # Atomic packed artifact: each trace a chunk-aligned document that fits in one sequence, so
    # none spans a sequence boundary (every kept trace was pre-filtered to padded_len <= seq_len,
    # so pack_atomic drops nothing -> n_documents == n_masked_records).
    pack_manifest = pack_atomic(trace_id_lists, packed_dir, seq_len=seq_len,
                                chunk_align=chunk_align, tokenizer=tokenizer, dtype=dtype)

    manifest = {
        "tokenizer": tokenizer,
        "model_id": getattr(tok, "name_or_path", None) if not byte_fallback else None,
        "byte_fallback": byte_fallback,
        "template": "qwen-chatml",
        "format": "think-answer",
        "chat_eos": chat_template.CHAT_EOS,
        "seq_len": seq_len,
        "chunk_align": chunk_align,
        "n_traces": len(valid),
        "n_masked_records": n_records,
        "n_overlength_dropped": n_overlength,
        "n_tokens": n_tokens,
        "packed_n_documents": pack_manifest["n_documents"],
        "packed_n_sequences": pack_manifest["n_sequences"],
        "sources": sources,
        "cleaned_path": str(cleaned_path),
        "masked_path": str(masked_path),
        "packed_dir": str(packed_dir),
    }
    (tok_dir / "reasoning-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from .reasoning_traces import _LOADERS, iter_reasoning_traces
    ap.add_argument("--sources", nargs="+", default=["handauthored"],
                    choices=tuple(_LOADERS), help="reasoning-trace sources (handauthored / mot)")
    ap.add_argument("--out-root", type=Path, default=Path("data/shared"))
    ap.add_argument("--tokenizer", choices=("qwen3", "qwen25", "olmo", "starcoder2"),
                    default="qwen3")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--seq-len", type=int, default=8192,
                    help="atomic-packing sequence length; traces longer than this are dropped")
    ap.add_argument("--chunk-align", type=int, default=64,
                    help="pad each trace to this multiple (the model chunk_size) so it starts on a "
                         "chunk boundary for the SSM reset (#68)")
    ap.add_argument("--max-per-source", type=int, default=None)
    args = ap.parse_args()

    rows = iter_reasoning_traces(args.sources, max_per_source=args.max_per_source)
    m = build_reasoning_sft(rows, args.out_root, tokenizer=args.tokenizer, model_id=args.model_id,
                            seq_len=args.seq_len, chunk_align=args.chunk_align,
                            byte_fallback=args.byte_fallback)
    print(f"reasoning sft: {m['n_masked_records']} masked records "
          f"({m['n_overlength_dropped']} over-length dropped, {m['n_tokens']} tokens), "
          f"packed {m['packed_n_documents']} traces atomically x{m['seq_len']} "
          f"(chunk_align={m['chunk_align']}, sources={m['sources']}) -> {m['masked_path']}")


if __name__ == "__main__":
    main()
