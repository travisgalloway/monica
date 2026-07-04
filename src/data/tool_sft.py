"""Build the shared **tool-use SFT corpus** (#102) — the tool-calling skill's data.

`tool_sources.py`-formatted tool-use rows (`src/data/tool_sources.py`) rendered under the
Qwen chat template, written as the `shared/sft/` layout:

    <out_root>/sft/
        cleaned/tool/records.jsonl           # tokenizer-agnostic {messages,source,license}
        tokenized/<tok>-<k>/
            tool.jsonl                       # masked {input_ids,target_ids,loss_mask} (SFTLoader)
            tool-manifest.json               # summary (named to avoid colliding with manifest.json)

Tool-use is ordinary multi-turn chat masked by `chat_template.response_spans`:
- tool calls = assistant content (`<tool_call>{json}</tool_call>`)
- tool results = user turn (`<tool_response>{json}</tool_response>`)
- available tools (incl. distractors) = system turn (`<tools>[...]</tools>`)
No new chat roles are introduced.

Unlike the instruct corpus, tool-call content is **not** whitespace-collapsed — JSON args
must survive verbatim. It therefore does NOT route through `sft_data._messages_of` /
`instruct_sft._write_cleaned`, which whitespace-collapse content. Instead it uses a local
verbatim `_valid_rows` writer + an INLINED mask loop (the identical approach from
`reasoning_sft.py:121-127`). This is the same divergence the reasoning corpus established.

Over-length examples are dropped, never truncated (truncating a call teaches malformed calls).
Abstention rows (assistant turn has no `<tool_call>`) are counted separately in the manifest.

ABOVE THE SEAM — no `mlx`/`torch`. Reuses `chat_template` and `storage`; `datasets` is
lazy inside the loaders. Offline path: handauthored tool records + `--byte-fallback`.

CLI:
    python -m src.data.tool_sft --sources handauthored --byte-fallback --out-root /tmp/shared-tool
    python -m src.data.tool_sft --sources xlam toolace when2call --tokenizer qwen3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from . import chat_template, storage
from .instruct_sft import _effective_vocab_size
from .tool_sources import (TOOL_CALL_OPEN, TOOL_CALL_CLOSE, TOOLS_OPEN, TOOLS_CLOSE,
                           validate_call_against_tools)


def _valid_rows(rows: Iterable[dict]) -> List[dict]:
    """Keep {messages, source, license} rows ending in a non-empty assistant turn (verbatim — no
    whitespace collapse; tool-call JSON keeps its braces/quotes/newlines). Identical in shape to
    reasoning_sft._valid_rows."""
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
    # identical to reasoning_sft._load_tokenizer
    from .tokenize import (ByteTokenizer, load_olmo_tokenizer, load_qwen3_tokenizer,
                           load_qwen25_tokenizer, load_starcoder2_tokenizer)
    if byte_fallback:
        return ByteTokenizer()
    loaders = {"qwen3": load_qwen3_tokenizer, "qwen25": load_qwen25_tokenizer,
               "olmo": load_olmo_tokenizer, "starcoder2": load_starcoder2_tokenizer}
    return loaders[tokenizer](model_id)


def _is_abstention(messages: List[dict]) -> bool:
    """A row is abstention if its final assistant turn contains no <tool_call> block."""
    return TOOL_CALL_OPEN not in (messages[-1].get("content") or "")


def _row_tools(messages: List[dict]) -> List[dict]:
    """The row's declared tools: parsed out of the (first) system turn's
    `<tools>[...]</tools>` block. Empty list if absent/malformed."""
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content") or ""
            start = content.find(TOOLS_OPEN)
            end = content.find(TOOLS_CLOSE)
            if start != -1 and end != -1 and end > start:
                body = content[start + len(TOOLS_OPEN):end].strip()
                try:
                    tools = json.loads(body)
                    return [t for t in tools if isinstance(t, dict)]
                except (json.JSONDecodeError, ValueError):
                    return []
            break
    return []


def _iter_calls(messages: List[dict]) -> Iterator[dict]:
    """Yield every parsed `<tool_call>{json}</tool_call>` block across assistant
    turns (each a dict with at least a "name" key). Malformed JSON blocks are
    skipped, never raised — shared by `_has_distractors` and the schema-validation
    gate in `build_tool_sft`."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        pos = 0
        while True:
            s = content.find(TOOL_CALL_OPEN, pos)
            if s == -1:
                break
            e = content.find(TOOL_CALL_CLOSE, s)
            if e == -1:
                break
            block = content[s + len(TOOL_CALL_OPEN):e].strip()
            try:
                call = json.loads(block)
                if isinstance(call, dict) and "name" in call:
                    yield call
            except (json.JSONDecodeError, ValueError):
                pass
            pos = e + len(TOOL_CALL_CLOSE)


def _has_distractors(messages: List[dict]) -> bool:
    """Heuristic audit flag: the system turn lists more tools than the row's calls reference."""
    listed_names = {t.get("name", "") for t in _row_tools(messages)}
    listed_names.discard("")
    called_names = {c["name"] for c in _iter_calls(messages)}
    return len(listed_names) > len(called_names)


def _row_schema_valid(messages: List[dict]) -> bool:
    """True iff every call in the row's assistant turns validates against the row's
    declared `<tools>` list (see `tool_sources.validate_call_against_tools`).
    Abstention rows (no calls at all) are vacuously valid."""
    tools = _row_tools(messages)
    return all(validate_call_against_tools(c, tools) for c in _iter_calls(messages))


def build_tool_sft(rows: Iterable[dict], out_root, *, tokenizer: str = "qwen3",
                   model_id: Optional[str] = None, seq_len: int = 8192,
                   byte_fallback: bool = False, max_seq_len: int = 8192) -> dict:
    """Build the shared tool-use SFT corpus end to end: verbatim cleaned rows, then tokenize +
    response-mask under Qwen ChatML into shared/sft/tokenized/<tok>-<k>/tool.jsonl + tool-manifest.json.
    Returns the manifest dict. Masking is INLINED (the reasoning_sft precedent), NOT via
    instruct_sft._record_from_messages, to keep JSON content verbatim."""
    out_root = Path(out_root)
    cleaned_path = storage.sft_cleaned_dir(out_root, "tool") / "records.jsonl"
    tok_dir = storage.sft_tokenized_dir(out_root, tokenizer, seq_len)
    tokenized_path = tok_dir / "tool.jsonl"

    valid = _valid_rows(rows)

    n_schema_invalid = 0
    schema_ok: List[dict] = []
    for row in valid:
        if _row_schema_valid(row["messages"]):
            schema_ok.append(row)
        else:
            n_schema_invalid += 1
    valid = schema_ok

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cleaned_path, "w", encoding="utf-8", newline="\n") as f:
        for row in valid:
            f.write(json.dumps(row) + "\n")

    tok = _load_tokenizer(tokenizer, model_id, byte_fallback)
    vocab = _effective_vocab_size(tok)
    tok_dir.mkdir(parents=True, exist_ok=True)

    n_records = n_tokens = n_skipped = n_abstention = n_with_distractors = 0
    sources: dict = {}
    with open(tokenized_path, "w", encoding="utf-8", newline="\n") as f:
        for row in valid:
            full_ids, spans = chat_template.response_spans(row["messages"], tok)
            if not spans:
                n_skipped += 1
                continue
            if max_seq_len is not None and len(full_ids) - 1 > max_seq_len:
                n_skipped += 1
                continue
            if vocab is not None and full_ids and max(full_ids) >= vocab:
                raise ValueError(
                    f"token id {max(full_ids)} >= tokenizer vocab_size {vocab} — wrong tokenizer?")
            mask = [0] * (len(full_ids) - 1)
            for s, e in spans:
                for j in range(max(0, s - 1), min(e - 1, len(mask))):
                    mask[j] = 1
            if not any(mask):
                n_skipped += 1
                continue
            rec = {"input_ids": full_ids[:-1], "target_ids": full_ids[1:], "loss_mask": mask}
            f.write(json.dumps(rec) + "\n")
            n_records += 1
            n_tokens += len(rec["input_ids"])
            sources[row["source"]] = sources.get(row["source"], 0) + 1
            if _is_abstention(row["messages"]):
                n_abstention += 1
            if _has_distractors(row["messages"]):
                n_with_distractors += 1

    manifest = {
        "tokenizer": tokenizer,
        "model_id": getattr(tok, "name_or_path", None) if not byte_fallback else None,
        "byte_fallback": byte_fallback,
        "template": "qwen-chatml",
        "format": "qwen-tool-call",
        "chat_eos": chat_template.CHAT_EOS,
        "seq_len": seq_len,
        "max_seq_len": max_seq_len,
        "n_records": n_records,
        "n_skipped": n_skipped,
        "n_schema_invalid": n_schema_invalid,
        "n_tokens": n_tokens,
        "n_abstention": n_abstention,
        "n_with_distractors": n_with_distractors,
        "sources": sources,
        "cleaned_path": str(cleaned_path),
        "tokenized_path": str(tokenized_path),
    }
    (tok_dir / "tool-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from .tool_sources import _LOADERS, iter_tool_sft
    ap.add_argument("--sources", nargs="+", default=["handauthored"], choices=tuple(_LOADERS),
                    help="tool-use sources (BFCL is eval-only, not here)")
    ap.add_argument("--out-root", type=Path, default=Path("data/shared"))
    ap.add_argument("--tokenizer", choices=("qwen3", "qwen25", "olmo", "starcoder2"), default="qwen3")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--max-seq-len", type=int, default=8192,
                    help="drop examples longer than this (truncating a call teaches malformed calls)")
    ap.add_argument("--max-per-source", type=int, default=None)
    args = ap.parse_args()

    rows = iter_tool_sft(args.sources, max_per_source=args.max_per_source)
    m = build_tool_sft(rows, args.out_root, tokenizer=args.tokenizer, model_id=args.model_id,
                       seq_len=args.seq_len, byte_fallback=args.byte_fallback,
                       max_seq_len=args.max_seq_len)
    print(f"tool sft: {m['n_records']} records ({m['n_skipped']} skipped, "
          f"{m['n_schema_invalid']} schema-invalid, {m['n_tokens']} tokens, "
          f"{m['n_abstention']} abstention, {m['n_with_distractors']} with distractors, "
          f"sources={m['sources']}) -> {m['tokenized_path']}")


if __name__ == "__main__":
    main()
