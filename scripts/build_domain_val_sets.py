#!/usr/bin/env python3
"""Build one held-out, packed val set PER DOMAIN from a cleaned corpus (#221, T5).

This is the tagging step that does not otherwise exist. Packed shards carry **no**
domain/language field: `src/data/shard.py` writes `manifest.json = {seq_len, dtype,
tokenizer, n_documents, n_sequences, n_tokens, shards}` with `.bin` + `.bounds` sidecars
only, and `swift/Sources/MonicaTokenizer/Packing.swift` writes the identical shape. The
per-record `source` / `lang` / `meta` survive only as far as the **cleaned Parquet** stage
(`src/data/corpus.py`), which is what this script reads. `src/data/split.py` additionally
records no `n_bytes`, so `val_bpb` is silently omitted on the shard path — hence purpose-
built val sets that always carry `n_bytes`.

    .venv/bin/python -m src.data.corpus --source dummy --out /tmp/cleaned --max-docs 200
    .venv/bin/python scripts/build_domain_val_sets.py \\
        --in /tmp/cleaned --group-by source --out /tmp/domains \\
        --tokenizer byte --val-docs 20 --seed 0

Output layout:

    <out>/<domain>/val.bin          packed token file (uint16 or uint32 by max id)
    <out>/<domain>/val.meta.json    {dtype, n_tokens, n_bytes}   <- n_bytes ALWAYS written
    <out>/domains.json              {config, domains: {...}, dropped: {...}}

`src/eval/domain_bpb.py` consumes `domains.json` and **raises** for any domain missing
`n_bytes`, so a val set built any other way will be rejected rather than silently reported
as fine.

What this measurement is and is not
-----------------------------------
This yields BPB over a held-out sample *drawn from each domain*, which is not the same as
"BPB per domain over the actual training mix". The alternative — a `.domains` uint8 sidecar
written next to `.bounds` by both packers — would let per-domain BPB be read off the single
training corpus, but it changes the data pipeline and needs a re-pack. Out of scope for an
eval issue; filed as the follow-up in `docs/design/13-code-model-moe.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str) -> str:
    """Filesystem-safe directory name for a domain value. Collisions are impossible to rule
    out in general, so the original value is always recorded in `domains.json`."""
    out = _SAFE.sub("_", str(value)).strip("_")
    return out or "unknown"


def _stable_seed(seed: int, domain: str) -> np.random.Generator:
    """A per-domain generator that depends only on (seed, domain name).

    Deriving each domain's shuffle from a stable hash — rather than consuming one shared
    stream — means adding or removing a domain does not perturb the held-out selection of
    every other domain, so a rebuild stays comparable to the previous one.
    """
    digest = hashlib.sha256(f"{seed}:{domain}".encode("utf-8")).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _read_records(uri) -> Iterator[dict]:
    """Yield `{text, source, lang, license, meta}` dicts from cleaned Parquet shards, or
    from JSON-lines carrying the same keys (the datatrove clean-pass shape)."""
    path = Path(str(uri))
    jsonl = sorted(path.glob("*.jsonl")) if path.is_dir() else (
        [path] if path.suffix == ".jsonl" else [])
    if jsonl:
        for f in jsonl:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        yield {"text": row.get("text", ""), "source": row.get("source", ""),
                               "lang": row.get("lang", ""), "license": row.get("license", ""),
                               "meta": row.get("meta") or {}}
        return

    from src.data.corpus import read_shards

    for rec in read_shards(uri):
        yield {"text": rec.text, "source": rec.source, "lang": rec.lang,
               "license": rec.license, "meta": rec.meta}


def domain_of(record: dict, group_by: str) -> Optional[str]:
    """The domain value for a record, or None when the field is absent/empty.

    A record with no domain value is **dropped and counted**, never bucketed into a
    catch-all — an "other" bucket silently mixes languages, which is the exact thing a
    per-domain BPB report exists to separate.
    """
    if group_by.startswith("meta:"):
        value = (record.get("meta") or {}).get(group_by[len("meta:"):])
    else:
        value = record.get(group_by)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _encoder(args):
    from src.eval.code_suite import make_byte_encoder, make_swift_encoder

    if args.tokenizer == "byte":
        return make_byte_encoder(), "byte(vocab=256)"
    if not args.tokenizer_json:
        raise SystemExit("--tokenizer swift requires --tokenizer-json <tokenizer.json>")
    return (make_swift_encoder(args.tokenizer_json, binary=args.tokenizer_bin),
            f"swift:{args.tokenizer_json}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True,
                    help="cleaned corpus URI (Parquet shard dir via src.data.corpus.read_shards, "
                         "or a JSONL dir/file with the same keys)")
    ap.add_argument("--group-by", default="lang",
                    help="'lang', 'source', 'license', or 'meta:<key>' (e.g. meta:lang for "
                         "stack-v2's per-file language)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-docs", type=int, default=None,
                    help="held-out documents per domain")
    ap.add_argument("--val-bytes", type=int, default=None,
                    help="held-out UTF-8 bytes per domain (stops at the first doc that "
                         "reaches the budget)")
    ap.add_argument("--min-docs", type=int, default=1,
                    help="drop domains with fewer documents than this; dropped domains are "
                         "RECORDED in domains.json, never silently discarded")
    ap.add_argument("--tokenizer", choices=("byte", "swift"), default="byte")
    ap.add_argument("--tokenizer-json", default=None, help="required for --tokenizer swift")
    ap.add_argument("--tokenizer-bin", default="monica-tokenize")
    ap.add_argument("--eos-id", type=int, default=None,
                    help="token id appended after each document (the Swift packer's "
                         "convention). Omitted by default — the byte tokenizer's 256-id "
                         "space has no reserved sentinel.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.val_docs is None and args.val_bytes is None:
        raise SystemExit("pass --val-docs and/or --val-bytes (an unbounded held-out set is "
                         "not a val set)")

    encode, tokenizer_id = _encoder(args)

    from src.data.pack import pack_ids, packing_dtype_for

    # Pass 1 — group document texts by domain. Held-out sets are small by construction, so
    # the texts fit in memory; this keeps the selection a pure function of (seed, domain).
    by_domain: Dict[str, List[str]] = {}
    n_records = 0
    n_untagged = 0
    for rec in _read_records(args.inp):
        n_records += 1
        domain = domain_of(rec, args.group_by)
        if domain is None:
            n_untagged += 1
            continue
        by_domain.setdefault(domain, []).append(rec["text"])

    if not by_domain:
        raise SystemExit(
            f"no records carried a '{args.group_by}' value in {args.inp} — nothing to group. "
            "An empty domain index would make every downstream BPB report vacuous.")

    args.out.mkdir(parents=True, exist_ok=True)
    domains: Dict[str, dict] = {}
    dropped: Dict[str, int] = {}

    for domain in sorted(by_domain):
        texts = by_domain[domain]
        if len(texts) < args.min_docs:
            dropped[domain] = len(texts)
            continue

        rng = _stable_seed(args.seed, domain)
        order = [int(i) for i in rng.permutation(len(texts))]

        selected: List[str] = []
        n_bytes = 0
        for idx in order:
            if args.val_docs is not None and len(selected) >= args.val_docs:
                break
            if args.val_bytes is not None and n_bytes >= args.val_bytes:
                break
            selected.append(texts[idx])
            n_bytes += len(texts[idx].encode("utf-8"))

        ids: List[int] = []
        for text in selected:
            ids.extend(int(t) for t in encode(text))
            if args.eos_id is not None:
                ids.append(int(args.eos_id))
        if not ids:
            dropped[domain] = len(texts)
            continue

        arr = np.asarray(ids, dtype=np.int64)
        dtype = packing_dtype_for(int(arr.max()) + 1)
        packed = args.out / _safe_name(domain) / "val.bin"
        n_tokens = pack_ids(arr, packed, dtype=dtype, n_bytes=n_bytes)

        domains[domain] = {
            "packed": str(packed.relative_to(args.out)),
            "group_value": domain,
            "n_docs": len(selected),
            "n_docs_available": len(texts),
            "n_tokens": int(n_tokens),
            "n_bytes": int(n_bytes),
            "dtype": dtype.name,
        }
        print(f"  {domain:<24} {len(selected):>5} docs  {n_tokens:>9} tok  {n_bytes:>10} B "
              f"-> {packed}")

    if not domains:
        raise SystemExit(
            f"every domain fell below --min-docs {args.min_docs} — no val set was written. "
            "An empty index is a failure, not an empty-but-valid result.")

    index = {
        "config": {
            "in": str(args.inp), "group_by": args.group_by, "out": str(args.out),
            "val_docs": args.val_docs, "val_bytes": args.val_bytes,
            "min_docs": args.min_docs, "tokenizer": tokenizer_id, "eos_id": args.eos_id,
            "seed": args.seed,
        },
        "n_records_read": n_records,
        "n_records_untagged": n_untagged,
        "domains": domains,
        "dropped_domains": dropped,
    }
    index_path = args.out / "domains.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(domains)} domains ({len(dropped)} dropped, {n_untagged} untagged "
          f"records) -> {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
