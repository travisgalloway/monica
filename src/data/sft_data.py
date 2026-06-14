"""Build SFT training records from a chat dataset (portable, above the seam).

Supervised fine-tuning trains only on the *assistant* tokens: the model should learn to
produce responses, not to re-predict the prompt. Each example becomes a per-token
`(input_ids, target_ids, loss_mask)` record where `loss_mask` is 1 only on positions
whose target is an assistant-response token (and the terminal EOS, so the model learns to
stop) and 0 on prompt/system/user tokens. The assistant token ranges come from
`instruct_format.response_spans`, so the render and the mask can never drift.

Records are written one-per-line as JSON int lists (see `src/data/sft_loader.py` for the
rationale): tiny dataset, streamable, line-addressable, inspectable. The text extractor
`build_sft_records` is a pure function over an injected record iterable — unit-testable
without any network or `datasets` dependency, mirroring `download.iter_instruct_texts`.

The default source is `HuggingFaceH4/no_robots` (~10k high-quality human-written, CC-BY).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .download import _normalize_doc
from .instruct_format import response_spans


def _messages_of(rec: dict) -> List[dict]:
    """Normalize a dataset record into `[{role, content}, ...]` with collapsed (newline-
    free) content. Accepts the `messages` chat field (no_robots / ultrafeedback) or a
    Dolly-style `instruction`/`response`/`context` triple."""
    msgs = rec.get("messages")
    if isinstance(msgs, (list, tuple)):
        out = []
        for m in msgs:
            role = m.get("role")
            content = _normalize_doc(m.get("content") or "")
            if role in ("system", "user", "assistant") and content:
                out.append({"role": role, "content": content})
        return out
    # Dolly fallback.
    instr = _normalize_doc(rec.get("instruction") or "")
    resp = _normalize_doc(rec.get("response") or "")
    ctx = _normalize_doc(rec.get("context") or "")
    if not instr or not resp:
        return []
    user = f"{instr} {ctx}".strip() if ctx else instr
    return [{"role": "user", "content": user}, {"role": "assistant", "content": resp}]


def _record_from_messages(messages: List[dict], tokenizer,
                          max_seq_len: Optional[int]) -> Optional[dict]:
    """One `(input_ids, target_ids, loss_mask)` record, or None to skip.

    Skips examples with no assistant content, or whose length exceeds `max_seq_len`
    (truncating a response would teach the model to stop mid-answer — drop instead).
    """
    if not messages or messages[-1]["role"] != "assistant":
        return None
    full_ids, spans = response_spans(messages, tokenizer)
    if not spans:
        return None

    eos = getattr(tokenizer, "eos_token_id", None)
    seq = list(full_ids) + ([int(eos)] if eos is not None else [])
    if max_seq_len is not None and len(seq) - 1 > max_seq_len:
        return None

    vocab = getattr(tokenizer, "vocab_size", None)
    if vocab is not None and seq and max(seq) >= vocab:
        raise ValueError(
            f"token id {max(seq)} >= tokenizer vocab_size {vocab} — wrong tokenizer?")

    # mask[j] == 1 iff target_ids[j] (= seq[j+1]) is a response token. For span [s,e),
    # the content tokens seq[s..e-1] are predicted from target indices [s-1, e-1). The
    # final assistant turn additionally trains the EOS prediction (one position past its
    # content) so the model learns to stop.
    mask = [0] * (len(seq) - 1)
    for k, (s, e) in enumerate(spans):
        last = k == len(spans) - 1
        hi = e if (last and eos is not None) else e - 1
        for j in range(max(0, s - 1), min(hi, len(mask))):
            mask[j] = 1
    if not any(mask):
        return None
    return {"input_ids": seq[:-1], "target_ids": seq[1:], "loss_mask": mask}


def build_sft_records(records: Iterable[dict], tokenizer, *,
                      max_seq_len: Optional[int] = None,
                      stats: Optional[dict] = None) -> Iterator[dict]:
    """Yield SFT `(input_ids, target_ids, loss_mask)` records from chat-dataset rows.

    `stats` (if given) accumulates `kept` / `skipped` counts for a run summary.
    """
    for rec in records:
        out = _record_from_messages(_messages_of(rec), tokenizer, max_seq_len)
        if out is None:
            if stats is not None:
                stats["skipped"] = stats.get("skipped", 0) + 1
            continue
        if stats is not None:
            stats["kept"] = stats.get("kept", 0) + 1
        yield out


def write_sft_jsonl(records: Iterable[dict], tokenizer, out: Path, *,
                    max_seq_len: Optional[int] = None) -> int:
    """Write SFT records to a JSONL file; return the number kept."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for rec in build_sft_records(records, tokenizer, max_seq_len=max_seq_len,
                                     stats=stats):
            f.write(json.dumps(rec) + "\n")
    kept, skipped = stats.get("kept", 0), stats.get("skipped", 0)
    print(f"wrote {kept} SFT records ({skipped} skipped) -> {out}")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="HuggingFaceH4/no_robots")
    ap.add_argument("--split", default="train", help="HF split (e.g. train / test)")
    ap.add_argument("--out", type=Path, required=True, help="output .jsonl path")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--byte-fallback", action="store_true",
                    help="offline ByteTokenizer (toy only; not OLMo-compatible)")
    args = ap.parse_args()

    from .tokenize import ByteTokenizer, load_olmo_tokenizer
    tok = ByteTokenizer() if args.byte_fallback else load_olmo_tokenizer()

    from datasets import load_dataset  # lazy (optional `data` extra)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.max_examples is not None:
        ds = ds.take(args.max_examples)
    write_sft_jsonl(ds, tok, args.out, max_seq_len=args.max_seq_len)


if __name__ == "__main__":
    main()
