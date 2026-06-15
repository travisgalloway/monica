"""Clean-license SFT sources (#76).

The M9 SFT machinery (`sft_data.build_sft_records`, `scripts/sft.py`) already handles
multi-turn masking via `instruct_format.response_spans`; #76 is the **clean-license data
sourcing** layer on top (docs/design/08-corpus-pipeline.md line 115): OASST1 (multi-turn,
Apache-2.0), Dolly (CC-BY-SA, flagged), FLAN, plus a small hand-authored in-format set. The
licensing reframe is the load-bearing part — no commercial-model-distilled responses.

Each source yields chat rows in the shape `build_sft_records` consumes — a `{messages:
[{role, content}, ...]}` dict (or a Dolly `instruction`/`response` triple) — tagged with
`source` and `license`. The reconstruction/mapping logic is pure and tested; the HF `load_*`
helpers are thin lazy wrappers (network, optional `data` extra) around it.

ABOVE THE SEAM — stdlib only; `datasets` imported lazily inside the loaders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

# Source -> license, for the clean-license accounting (CC-BY-SA flagged share-alike).
SOURCE_LICENSES = {"oasst1": "apache-2.0", "flan": "apache-2.0",
                   "dolly": "cc-by-sa-3.0", "handauthored": "cc0"}


def _tag(messages: List[dict], source: str) -> dict:
    return {"messages": messages, "source": source,
            "license": SOURCE_LICENSES.get(source, "unknown")}


# --------------------------------------------------------------------------- #
# OASST1 — reconstruct multi-turn conversation threads from the message tree
# --------------------------------------------------------------------------- #
def build_oasst1_threads(rows: Iterable[dict], lang: Optional[str] = "en",
                         ) -> Iterator[dict]:
    """Walk the OASST1 message tree: for each assistant node, emit the root->node path as a
    multi-turn `{messages}` example (so every conversational prefix ending in an assistant
    turn becomes a training example). `prompter`->user, `assistant`->assistant. If `lang`
    is set, every node on the path must match it OR have no language tag (missing tags are
    permitted). Rows: {message_id, parent_id, text, role, lang}."""
    by_id = {r.get("message_id"): r for r in rows}
    for r in by_id.values():
        if r.get("role") != "assistant":
            continue
        path: List[dict] = []
        cur: Optional[dict] = r
        ok = True
        while cur is not None:
            if lang is not None and cur.get("lang") not in (None, lang):
                ok = False
                break
            path.append(cur)
            pid = cur.get("parent_id")
            cur = by_id.get(pid) if pid else None
        if not ok:
            continue
        path.reverse()
        messages: List[dict] = []
        for n in path:
            content = (n.get("text") or "").strip()
            if not content:
                ok = False
                break
            role = "user" if n.get("role") == "prompter" else "assistant"
            messages.append({"role": role, "content": content})
        if ok and messages and messages[0]["role"] == "user" \
                and messages[-1]["role"] == "assistant":
            yield _tag(messages, "oasst1")


def load_oasst1(split: str = "train", lang: Optional[str] = "en",
                max_examples: Optional[int] = None) -> Iterator[dict]:
    """Load OASST1 from the Hub and reconstruct multi-turn threads (lazy `datasets`). The
    split is materialized, not streamed — thread reconstruction needs the whole message
    tree (random access by message_id)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    rows = load_dataset("OpenAssistant/oasst1", split=split)
    out = build_oasst1_threads(rows, lang=lang)
    for i, rec in enumerate(out):
        if max_examples is not None and i >= max_examples:
            break
        yield rec


# --------------------------------------------------------------------------- #
# FLAN — single-turn instruction rows (inputs -> targets)
# --------------------------------------------------------------------------- #
def flan_to_messages(row: dict) -> Optional[dict]:
    """A FLAN `{inputs, targets}` row -> a single-turn `{messages}` example (or None)."""
    user = (row.get("inputs") or "").strip()
    resp = (row.get("targets") or "").strip()
    if not user or not resp:
        return None
    return _tag([{"role": "user", "content": user},
                 {"role": "assistant", "content": resp}], "flan")


def load_flan_slice(split: str = "train", max_examples: Optional[int] = None,
                    config: str = "default") -> Iterator[dict]:
    """Stream a FLAN slice from the Hub (lazy `datasets`)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("Muennighoff/flan", config, split=split, streaming=True)
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        rec = flan_to_messages(row)
        if rec is not None:
            yield rec


# --------------------------------------------------------------------------- #
# Dolly — reuse the existing instruction/response shape (flagged CC-BY-SA)
# --------------------------------------------------------------------------- #
def load_dolly(max_examples: Optional[int] = None) -> Iterator[dict]:
    """Stream Dolly-15k as instruction/response rows, license-tagged (lazy `datasets`)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        yield {"instruction": row.get("instruction"), "response": row.get("response"),
               "context": row.get("context"), "source": "dolly",
               "license": SOURCE_LICENSES["dolly"]}


# --------------------------------------------------------------------------- #
# Hand-authored in-format set (checked in, CC0) — guarantees clean format coverage
# --------------------------------------------------------------------------- #
HANDAUTHORED: List[List[dict]] = [
    [{"role": "user", "content": "What is the capital of France?"},
     {"role": "assistant", "content": "The capital of France is Paris."}],
    [{"role": "user", "content": "Write a haiku about autumn."},
     {"role": "assistant",
      "content": "Crisp leaves drift downward, amber light fades into dusk, frost waits at the edge."}],
    [{"role": "user", "content": "Define recursion in one sentence."},
     {"role": "assistant",
      "content": "Recursion is when a function solves a problem by calling itself on smaller instances until it reaches a base case."},
     {"role": "user", "content": "Give a tiny example."},
     {"role": "assistant",
      "content": "factorial(n) returns 1 if n == 0, else n * factorial(n - 1)."}],
    [{"role": "user", "content": "Convert 2 kilometers to meters."},
     {"role": "assistant", "content": "2 kilometers is 2000 meters."}],
]


def handauthored_records() -> Iterator[dict]:
    """The checked-in hand-authored multi-turn examples (always available, offline)."""
    for messages in HANDAUTHORED:
        yield _tag([dict(m) for m in messages], "handauthored")


# --------------------------------------------------------------------------- #
# Aggregator + CLI
# --------------------------------------------------------------------------- #
_LOADERS = {
    "oasst1": lambda n: load_oasst1(max_examples=n),
    "flan": lambda n: load_flan_slice(max_examples=n),
    "dolly": lambda n: load_dolly(max_examples=n),
    "handauthored": lambda n: handauthored_records(),
}


def iter_clean_sft(sources: Iterable[str], max_per_source: Optional[int] = None,
                   ) -> Iterator[dict]:
    """Concatenate chat rows from the named clean-license sources."""
    for name in sources:
        if name not in _LOADERS:
            raise ValueError(f"unknown SFT source {name!r} (have {sorted(_LOADERS)})")
        yield from _LOADERS[name](max_per_source)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["handauthored"],
                    choices=tuple(_LOADERS), help="clean-license SFT sources to include")
    ap.add_argument("--out", type=Path, required=True, help="output SFT JSONL")
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=1024,
                    help="drop examples longer than this (bounds SFTLoader batch padding)")
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()

    from .sft_data import write_sft_jsonl
    from .tokenize import ByteTokenizer, load_olmo_tokenizer

    tok = ByteTokenizer() if args.byte_fallback else load_olmo_tokenizer(args.model_id)
    rows = iter_clean_sft(args.sources, max_per_source=args.max_per_source)
    write_sft_jsonl(rows, tok, args.out, max_seq_len=args.max_seq_len)


if __name__ == "__main__":
    main()
