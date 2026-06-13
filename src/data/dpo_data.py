"""Build DPO preference records from a binarized preference dataset (portable).

Each example is a (prompt, chosen, rejected) triple. We render the prompt+chosen and
prompt+rejected as two SFT-style sequences (reusing `sft_data._record_from_messages`, so
the chat format and response masking are identical to SFT) and store both with their
response masks. The DPO step sums the response-token log-probs of each side under the
policy and the frozen reference.

Records are JSONL, one per line:
  {chosen_input_ids, chosen_target_ids, chosen_mask,
   rejected_input_ids, rejected_target_ids, rejected_mask}

The default source is `HuggingFaceH4/ultrafeedback_binarized`, whose `chosen` / `rejected`
fields are message lists `[{user prompt}, {assistant response}]`. The extractor
`build_dpo_records` is a pure function over an injected iterable — unit-testable offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from .download import _normalize_doc
from .sft_data import _record_from_messages


def _last_assistant(messages: object) -> str:
    """The final assistant turn's (normalized) content from a message list, or ""."""
    if not isinstance(messages, (list, tuple)):
        return ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return _normalize_doc(m.get("content") or "")
    return ""


def _extract(rec: dict) -> tuple[str, str, str]:
    """(prompt, chosen_response, rejected_response) from a binarized-preference row."""
    prompt = _normalize_doc(rec.get("prompt") or "")
    return prompt, _last_assistant(rec.get("chosen")), _last_assistant(rec.get("rejected"))


def _side_record(prompt: str, response: str, tokenizer,
                 max_seq_len: Optional[int]) -> Optional[dict]:
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    return _record_from_messages(messages, tokenizer, max_seq_len)


def build_dpo_records(records: Iterable[dict], tokenizer, *,
                      max_seq_len: Optional[int] = None,
                      stats: Optional[dict] = None) -> Iterator[dict]:
    """Yield DPO preference records (chosen + rejected, each response-masked).

    Skips rows missing a prompt / either response, or where either side exceeds
    `max_seq_len`. `stats` (if given) accumulates `kept` / `skipped` counts.
    """
    for rec in records:
        prompt, chosen, rejected = _extract(rec)
        c = _side_record(prompt, chosen, tokenizer, max_seq_len) if (prompt and chosen) else None
        r = _side_record(prompt, rejected, tokenizer, max_seq_len) if (prompt and rejected) else None
        if c is None or r is None:
            if stats is not None:
                stats["skipped"] = stats.get("skipped", 0) + 1
            continue
        if stats is not None:
            stats["kept"] = stats.get("kept", 0) + 1
        yield {
            "chosen_input_ids": c["input_ids"], "chosen_target_ids": c["target_ids"],
            "chosen_mask": c["loss_mask"],
            "rejected_input_ids": r["input_ids"], "rejected_target_ids": r["target_ids"],
            "rejected_mask": r["loss_mask"],
        }


def write_dpo_jsonl(records: Iterable[dict], tokenizer, out: Path, *,
                    max_seq_len: Optional[int] = None) -> int:
    """Write DPO records to a JSONL file; return the number kept."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for rec in build_dpo_records(records, tokenizer, max_seq_len=max_seq_len,
                                     stats=stats):
            f.write(json.dumps(rec) + "\n")
    kept, skipped = stats.get("kept", 0), stats.get("skipped", 0)
    print(f"wrote {kept} DPO records ({skipped} skipped) -> {out}")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    ap.add_argument("--split", default="train_prefs", help="HF split")
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
    write_dpo_jsonl(ds, tok, args.out, max_seq_len=args.max_seq_len)


if __name__ == "__main__":
    main()
