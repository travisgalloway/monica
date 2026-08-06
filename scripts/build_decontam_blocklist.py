#!/usr/bin/env python3
"""Build the eval decontamination blocklist + its provenance manifest (#221, T8).

The consumer already exists: `src/data/corpus.py --decontam-file` /
`scripts/build_corpus.py` feed a plain UTF-8 text file — **one benchmark text per line** —
into `src.data.dedup.Decontaminator.from_texts`, which n-grams on `text.lower().split()`
(13-gram and 7-gram). So whitespace *shape* inside a record does not matter, but line
integrity does: embedded newlines are collapsed to single spaces, or one record would
become several and its cross-line n-grams would be lost.

    .venv/bin/python scripts/build_decontam_blocklist.py --out eval_sets/decontam/blocklist.txt

Output is **sorted and de-duplicated**, so the file is byte-reproducible: running this twice
produces identical bytes, which is what makes "decontam blocklist wired" checkable rather
than merely asserted. A sibling `blocklist.manifest.json` records the per-source line counts,
each external set's revision pin (or `null`), and the blocklist's sha256 —
`scripts/eval_code_suite.py` echoes that hash into its results JSON.

Network: **off by default**. External suites contribute their checked-in *synthetic*
fixtures unless `--allow-network` is passed, and a live pull still requires a pinned
revision (see `src/eval/external_sets.py`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The largest n-gram `Decontaminator` uses. A text shorter than this yields no n-gram of
#: that size, so it can never match — keeping it would only inflate the file.
DEFAULT_MIN_WORDS = 13

#: Built-in (non-external) sources: name -> (path, field list).
_BUILTIN: Dict[str, Tuple[Path, Tuple[str, ...]]] = {
    "ts_error_injection": (REPO_ROOT / "eval_sets/ts_error_injection/eval.jsonl",
                           ("prompt", "gold_completion")),
    "humaneval_ts": (REPO_ROOT / "eval_sets/humaneval_ts/humaneval_ts.jsonl",
                     ("prompt", "tests")),
    "code_recall": (REPO_ROOT / "eval_sets/code_recall/fixture_repo.jsonl", ("text",)),
    "code_needle": (REPO_ROOT / "eval_sets/code_needle/haystack.jsonl", ("text",)),
}


def flatten(text: Optional[str]) -> str:
    """One line, single-spaced. `Decontaminator` splits on whitespace, so collapsing is
    lossless for matching — and it is what keeps one record on one line."""
    if not text:
        return ""
    return " ".join(str(text).split())


def all_source_names() -> List[str]:
    from src.eval.external_sets import EXTERNAL_SETS

    return sorted(_BUILTIN) + [f"external:{n}" for n in sorted(EXTERNAL_SETS)]


def _texts_for(name: str, *, allow_network: bool) -> Tuple[List[str], dict]:
    """Return `(texts, provenance)` for one source name."""
    from src.eval.code_suite import read_jsonl

    if name.startswith("external:"):
        from src.eval.external_sets import get_external_set, load_external

        set_name = name[len("external:"):]
        spec = get_external_set(set_name)
        fixture_only = not allow_network
        rows = load_external(set_name, fixture_only=fixture_only)
        texts = []
        for row in rows:
            for field in ("prompt", "suffix", "answer"):
                texts.append(row.get(field))
            ctx = (row.get("meta") or {}).get("crossfile_context")
            if ctx:
                texts.append(ctx)
        return [t for t in texts if t], {
            "source": spec.hf_repo if not fixture_only else str(
                spec.fixture.relative_to(REPO_ROOT)),
            "n_records": len(rows),
            "revision": spec.revision,
            "fixture_only": fixture_only,
        }

    if name not in _BUILTIN:
        raise SystemExit(f"unknown source {name!r}; known: {all_source_names()}")
    path, fields = _BUILTIN[name]
    if not path.exists():
        raise SystemExit(f"source {name!r}: {path} does not exist — a missing eval set must "
                         "not silently shrink the blocklist")
    rows = read_jsonl(path)
    texts = [row.get(f) for row in rows for f in fields]
    return [t for t in texts if t], {
        "source": str(path.relative_to(REPO_ROOT)),
        "n_records": len(rows),
        "revision": None,
        "fixture_only": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=None,
                    help="comma-separated source names (default: every checked-in set). "
                         "External suites are named 'external:<name>'.")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "eval_sets/decontam/blocklist.txt")
    ap.add_argument("--allow-network", action="store_true",
                    help="let external suites pull live rows instead of their checked-in "
                         "synthetic fixtures (still requires a pinned revision)")
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
                    help=f"skip texts shorter than this many words (default "
                         f"{DEFAULT_MIN_WORDS}, the largest n-gram size — a shorter text can "
                         "never match)")
    ap.add_argument("--list-sets", action="store_true", help="print the source names and exit")
    args = ap.parse_args()

    if args.list_sets:
        for name in all_source_names():
            print(name)
        return 0

    from src.eval.code_suite import sha256_file

    names = ([s.strip() for s in args.sets.split(",") if s.strip()]
             if args.sets else all_source_names())

    lines: set[str] = set()
    provenance: Dict[str, dict] = {}
    for name in names:
        texts, prov = _texts_for(name, allow_network=args.allow_network)
        kept = 0
        for text in texts:
            flat = flatten(text)
            if len(flat.split()) < args.min_words:
                continue
            lines.add(flat)
            kept += 1
        prov["n_lines"] = kept
        provenance[name] = prov
        print(f"  {name:<34} {prov['n_records']:>5} records -> {kept:>5} lines")

    if not lines:
        raise SystemExit(
            "the blocklist is empty. An empty decontamination file silently disables "
            "decontamination — that is a failure, not a valid result. Check --sets and "
            f"--min-words {args.min_words}.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for line in sorted(lines):
            f.write(line + "\n")

    manifest = {
        "blocklist": str(args.out.name),
        "sha256": sha256_file(args.out),
        "n_lines": len(lines),
        "min_words": args.min_words,
        "allow_network": args.allow_network,
        "sources": provenance,
    }
    manifest_path = args.out.with_name(args.out.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote {len(lines)} lines -> {args.out}")
    print(f"sha256 {manifest['sha256']}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
