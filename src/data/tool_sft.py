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
from typing import Iterable, List, Optional

from . import chat_template, storage
from .instruct_sft import _effective_vocab_size
from .tool_sources import TOOL_CALL_OPEN, TOOLS_OPEN, TOOLS_CLOSE


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


def _has_distractors(messages: List[dict]) -> bool:
    """Heuristic audit flag: the system turn lists more tools than the row's calls reference.
    Count <tool_call> blocks across assistant turns vs tool entries in the system <tools> list."""
    # Extract tool names from the system turn's <tools>[...] block
    system_tools: List[str] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content") or ""
            start = content.find(TOOLS_OPEN)
            end = content.find(TOOLS_CLOSE)
            if start != -1 and end != -1 and end > start:
                body = content[start + len(TOOLS_OPEN):end].strip()
                try:
                    tool_list = json.loads(body)
                    system_tools = [t.get("name", "") for t in tool_list if isinstance(t, dict)]
                except (json.JSONDecodeError, ValueError):
                    pass
            break

    # Collect tool names actually called across all assistant turns
    called_names: set = set()
    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content") or ""
            pos = 0
            while True:
                s = content.find(TOOL_CALL_OPEN, pos)
                if s == -1:
                    break
                e = content.find("</tool_call>", s)
                if e == -1:
                    break
                block = content[s + len(TOOL_CALL_OPEN):e].strip()
                try:
                    call = json.loads(block)
                    if isinstance(call, dict) and "name" in call:
                        called_names.add(call["name"])
                except (json.JSONDecodeError, ValueError):
                    pass
                pos = e + len("</tool_call>")

    # Has distractors if there are more listed tools than called tools
    listed_names = set(n for n in system_tools if n)
    return len(listed_names) > len(called_names)


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
    print(f"tool sft: {m['n_records']} records ({m['n_skipped']} skipped, {m['n_tokens']} tokens, "
          f"{m['n_abstention']} abstention, {m['n_with_distractors']} with distractors, "
          f"sources={m['sources']}) -> {m['tokenized_path']}")


if __name__ == "__main__":
    main()
