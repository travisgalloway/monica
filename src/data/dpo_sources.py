"""Clean-license DPO preference sources (#77).

The M9 DPO machinery (`dpo_math`, `make_dpo_train_step`, `scripts/dpo.py`) is complete; #77
adds the **clean-preference sourcing** on top (docs/design/08-corpus-pipeline.md line 117):
OASST1 rankings, SHP, and **on-policy self-generated** pairs (inherently clean). HH-RLHF is
excluded despite its MIT license — its responses are commercial-model output.

Each source yields a `{prompt, chosen, rejected}` row in the shape `dpo_data.build_dpo_records`
consumes (`chosen`/`rejected` are `[{role: assistant, content}]` message lists), tagged with
`source` + `license`. The reconstruction/pairing logic is pure and tested; the HF `load_*`
helpers are thin lazy wrappers, and on-policy pairs come from `scripts/gen_onpolicy_prefs.py`
via `pairs_from_scored`.

ABOVE THE SEAM — stdlib only; `datasets` imported lazily inside the loaders.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

# SHP: human Reddit preferences (not model-distilled) — clean for our purpose.
SOURCE_LICENSES = {"oasst1": "apache-2.0", "shp": "open", "onpolicy": "cc0"}


def _pref(prompt: str, chosen: str, rejected: str, source: str) -> dict:
    return {"prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
            "source": source, "license": SOURCE_LICENSES.get(source, "unknown")}


# --------------------------------------------------------------------------- #
# OASST1 — rank sibling assistant replies into chosen/rejected pairs
# --------------------------------------------------------------------------- #
def build_oasst1_prefs(rows: Iterable[dict], lang: Optional[str] = "en") -> Iterator[dict]:
    """For each prompter node with >=2 ranked assistant children, pair the best-ranked
    (chosen) against the worst-ranked (rejected). `rank` 0 is best. Rows: {message_id,
    parent_id, text, role, rank, lang}."""
    children: dict = defaultdict(list)
    prompters: list = []
    for r in rows:                                   # single pass (streaming-friendly)
        children[r.get("parent_id")].append(r)
        if r.get("role") == "prompter":
            prompters.append(r)
    for r in prompters:
        if lang is not None and r.get("lang") not in (None, lang):
            continue
        kids = [c for c in children.get(r.get("message_id"), [])
                if c.get("role") == "assistant" and c.get("rank") is not None
                and (lang is None or c.get("lang") in (None, lang))]
        if len(kids) < 2:
            continue
        kids.sort(key=lambda c: c["rank"])
        best, worst = kids[0], kids[-1]
        if best.get("rank") == worst.get("rank"):
            continue
        prompt = (r.get("text") or "").strip()
        bc, wc = (best.get("text") or "").strip(), (worst.get("text") or "").strip()
        if prompt and bc and wc:
            yield _pref(prompt, bc, wc, "oasst1")


def load_oasst1_prefs(split: str = "train", lang: Optional[str] = "en",
                      max_examples: Optional[int] = None) -> Iterator[dict]:
    """Load OASST1 and pair ranked sibling replies (lazy `datasets`; materializes split)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    rows = load_dataset("OpenAssistant/oasst1", split=split)
    for i, rec in enumerate(build_oasst1_prefs(rows, lang=lang)):
        if max_examples is not None and i >= max_examples:
            break
        yield rec


# --------------------------------------------------------------------------- #
# SHP — Stanford Human Preferences (history + two refs + a label)
# --------------------------------------------------------------------------- #
def shp_to_pref(row: dict) -> Optional[dict]:
    """A SHP row -> a `{prompt, chosen, rejected}` pref (or None). `labels==1` means
    `human_ref_A` is preferred."""
    prompt = (row.get("history") or "").strip()
    a, b = (row.get("human_ref_A") or "").strip(), (row.get("human_ref_B") or "").strip()
    label = row.get("labels")
    if not (prompt and a and b) or label not in (0, 1):
        return None                                  # reject malformed / non-binary labels
    chosen, rejected = (a, b) if label == 1 else (b, a)
    return _pref(prompt, chosen, rejected, "shp")


def load_shp_slice(split: str = "train", max_examples: Optional[int] = None) -> Iterator[dict]:
    """Stream SHP from the Hub (lazy `datasets`)."""
    from datasets import load_dataset  # pragma: no cover - network/optional extra

    ds = load_dataset("stanfordnlp/SHP", split=split, streaming=True)
    n = 0
    for row in ds:
        if max_examples is not None and n >= max_examples:
            break
        rec = shp_to_pref(row)
        if rec is not None:
            n += 1
            yield rec


# --------------------------------------------------------------------------- #
# On-policy self-generated pairs (the inherently-clean source)
# --------------------------------------------------------------------------- #
def pairs_from_scored(prompt: str, scored: Sequence[Tuple[str, float]],
                      source: str = "onpolicy") -> Optional[dict]:
    """Build a pref from K scored on-policy samples: highest-scored is chosen, lowest is
    rejected. Returns None for an empty prompt, fewer than 2 non-empty candidates, equal
    best/worst scores, or identical chosen/rejected text (all degenerate). The scorer (a
    verifier/reward/heuristic) lives in the caller (gen_onpolicy_prefs.py)."""
    if not prompt or not prompt.strip():
        return None
    cands = [(str(resp).strip(), float(s)) for resp, s in scored if str(resp).strip()]
    if len(cands) < 2:
        return None
    best, worst = max(cands, key=lambda c: c[1]), min(cands, key=lambda c: c[1])
    if best[1] == worst[1] or best[0] == worst[0]:
        return None
    return _pref(prompt, best[0], worst[0], source)


# --------------------------------------------------------------------------- #
# Aggregator + CLI
# --------------------------------------------------------------------------- #
_LOADERS = {
    "oasst1": lambda n: load_oasst1_prefs(max_examples=n),
    "shp": lambda n: load_shp_slice(max_examples=n),
}


def iter_clean_dpo(sources: Iterable[str], max_per_source: Optional[int] = None,
                   ) -> Iterator[dict]:
    """Concatenate preference rows from the named clean-license sources (HF-backed)."""
    for name in sources:
        if name not in _LOADERS:
            raise ValueError(f"unknown DPO source {name!r} (have {sorted(_LOADERS)}; "
                             "on-policy pairs come from scripts/gen_onpolicy_prefs.py)")
        yield from _LOADERS[name](max_per_source)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["oasst1"], choices=tuple(_LOADERS))
    ap.add_argument("--out", type=Path, required=True, help="output DPO JSONL")
    ap.add_argument("--max-per-source", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()

    from .dpo_data import write_dpo_jsonl
    from .tokenize import ByteTokenizer, load_olmo_tokenizer

    tok = ByteTokenizer() if args.byte_fallback else load_olmo_tokenizer(args.model_id)
    rows = iter_clean_dpo(args.sources, max_per_source=args.max_per_source)
    write_dpo_jsonl(rows, tok, args.out, max_seq_len=args.max_seq_len)


if __name__ == "__main__":
    main()
