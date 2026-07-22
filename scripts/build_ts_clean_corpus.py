#!/usr/bin/env python3
"""Build the M12 TS "LSP-clean" corpus: Stack v2 TypeScript -> dedup -> prettier -> the
LSP-clean filter -> cleaned JSONL (#193). Tokenize + pack is now the native Swift
`monica-tokenize pack` step (`swift/`, #191/M13), which consumes this stage's `cleaned.jsonl`.

WHY THIS EXISTS. Locked composability decision #3 (`docs/design/13-code-model-moe.md`):
train the code model only on TypeScript `tsc --noEmit` already accepts with **zero**
diagnostics, so M12's inference-time LSP feedback (the #199/#226-#230 SSI axis) corrects
distribution shift at generation time rather than fighting a training prior that was full
of tsc-flagged code. #199's Phase-0 gate is CLOSED with a positive lift (logit-level
hard-ban raised diagnostic-clean rate 0.312 -> 0.688), so the pipeline this script drives
is validated machinery, not a speculative bet.

This issue's acceptance is a **sample** end-to-end run -> a `cleaned.jsonl` + a manifest
recording the LSP-clean filter rate -- NOT the full ~2-3B-token build (that spend is a
separate, user-driven decision once #199 unblocks it). The uint16 packing is verified
separately by the Swift `monica-tokenize` toolchain (swift/, its self-check + the
Python-reads-Swift-shards smoke).

THE FIVE STAGES:
  1. Stream Stack v2 TypeScript metadata (`bigcode/the-stack-v2-dedup`), resolve each
     file's content from Software Heritage's S3 bucket (`src.data.stack_v2`), and keep only
     permissively-licensed files (`src.data.filters.license_ok`).
  2. Cross-doc MinHash-LSH near-dedup (`src.data.dedup.near_dedup`). At the full ~2-3B-token
     scale this in-process engine is swapped for datatrove's `run_minhash_dedup` -- same
     semantics, distributed.
  3. Normalize formatting with a pinned `prettier` (`src.lsp.prettier`) -- gracefully
     skipped (pass-through, unformatted) on a host with no local `npm install`.
  4. THE LSP-CLEAN FILTER (`src.data.ts_clean.tsc_clean`): keep only files a pinned `tsc`
     accepts with zero diagnostics (ignoring the module-resolution family so import-bearing
     real-world files aren't all marked dirty -- the load-bearing lesson from
     `scripts/build_clean_prefix_set.py`). Gracefully skipped (pass-through, unfiltered,
     filter rate recorded as `null`) on a host with no local `tsc`.
  5. Write the surviving cleaned files as `{"text": ...}` JSONL (`cleaned.jsonl`). The
     native Swift `monica-tokenize pack` then tokenizes + packs this into fixed-length
     uint16 shards (same `src/data/shard.py` layout the training loop reads).

Stages 2-5's logic lives in `run_pipeline` below (importable, not locked inside `main`), so
`tests/test_build_ts_clean_corpus.py` can drive the whole chain OFFLINE with a stub tsc
runner and prettier off -- the thing that makes this pipeline's acceptance verifiable in CI
with no SWH/AWS credentials.

RUN RECIPES.

Offline dry-run (no creds, no toolchain -- exercises Stages 2-5 + the manifest against a
tiny local JSONL of `{"text": ...}` rows):

    .venv/bin/python scripts/build_ts_clean_corpus.py \\
        --from-jsonl /tmp/sample.jsonl \\
        --out /tmp/ts-clean-sample
    # then tokenize + pack natively:
    ( cd swift && swift run monica-tokenize pack \\
        --tokenizer /tmp/tokenizer.json \\
        --in /tmp/ts-clean-sample/cleaned.jsonl --out /tmp/ts-clean-shards )

True sample end-to-end (needs AWS creds for Software Heritage S3 + the `stack-v2` extra,
and a local prettier/tsc toolchain for Stages 3-4):

    pip install -e ".[stack-v2]"
    ( cd eval_sets/ts_error_injection && npm install )
    set -a; . ./.env; set +a          # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    .venv/bin/python scripts/build_ts_clean_corpus.py --limit 50 --out /tmp/ts-clean-swh
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.corpus import Record  # noqa: E402
from src.data.filters import license_ok, normalize_license  # noqa: E402
from src.data.stack_v2 import iter_stack_v2_ts, resolve_swh_s3  # noqa: E402
from src.lsp.prettier import PrettierRunner, resolve_prettier  # noqa: E402
from src.lsp.tsc import SET_DIR, TscRunner, resolve_tsc  # noqa: E402


# --------------------------------------------------------------------------- #
# Stage 1 (offline variant): a local JSONL of {"text"|"content": ...} rows
# --------------------------------------------------------------------------- #
def iter_jsonl_records(path) -> Iterator[Record]:
    """Offline Stage-1 source: one JSON object per line, `{"text": ...}` or
    `{"content": ...}` (plus optional `license`/`path`). This is the `--from-jsonl`
    path -- it bypasses Stack v2 / Software Heritage S3 entirely, so Stages 2-5 (and this
    script's CI coverage) never need SWH/AWS credentials."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text") or row.get("content")
            if not text:
                continue
            yield Record(text=text, source="stack-v2-jsonl", lang="typescript",
                        license=normalize_license(row.get("license", "")),
                        meta={"is_code": True, "path": row.get("path")})


# --------------------------------------------------------------------------- #
# Stages 2-5 -- composable, importable (no argparse/CLI state), tested offline
# --------------------------------------------------------------------------- #
def run_pipeline(records: Iterable[Record], out_dir, *, seq_len: int = 1024,
                 threshold: float = 0.8, prettier_runner: Optional[PrettierRunner] = None,
                 tsc_runner: Optional[TscRunner] = None, ignore_module_resolution: bool = True,
                 dedup_stats=None, clean_stats=None) -> dict:
    """Stage 2 (near-dedup) -> Stage 3 (prettier) -> Stage 4 (LSP-clean filter) ->
    Stage 5 (write cleaned text as JSONL). Writes `<out_dir>/cleaned.jsonl` and
    `<out_dir>/manifest.json`, and returns the manifest dict.

    Tokenize + pack is NO LONGER done here. The code tokenizer (#191) is now the native
    Swift `monica-tokenize` toolchain (`swift/`, cross-platform), which reads the
    `cleaned.jsonl` this stage emits and writes the uint16 `.bin`/`.bounds`/`manifest.json`
    training shards in the same `src/data/shard.py` layout the training loop consumes:

        monica-tokenize pack --tokenizer <tokenizer.json> --in <out_dir>/cleaned.jsonl \\
                             --out <shards_dir> --seq-len <N>

    `prettier_runner`/`tsc_runner` being `None` means "skip that stage, pass records
    through unchanged" -- the graceful-degrade behavior for a host missing the optional
    toolchain (mirrors `resolve_prettier`/`resolve_tsc` returning `None`). The caller
    (`main`) is responsible for logging *why* a stage was skipped; this function only
    records *whether* it ran, in the manifest's `prettier_applied`/`tsc_clean_applied`.
    """
    from src.data.dedup import DedupStats, near_dedup
    from src.data.ts_clean import CleanRateStats, tsc_clean

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup_stats = dedup_stats if dedup_stats is not None else DedupStats()
    clean_stats = clean_stats if clean_stats is not None else CleanRateStats()

    source_records = list(records)
    n_source = len(source_records)

    # Stage 2: cross-doc MinHash-LSH near-dedup.
    deduped = list(near_dedup(source_records, threshold=threshold, stats=dedup_stats))

    # Stage 3: prettier normalization (pass-through, unchanged, when skipped).
    if prettier_runner is not None:
        formatted = [Record(text=prettier_runner.format(r.text), source=r.source, lang=r.lang,
                            license=r.license, meta=r.meta) for r in deduped]
    else:
        formatted = deduped

    # Stage 4: the LSP-clean filter (pass-through, unfiltered, when skipped -- the
    # acceptance-critical clean-rate below is then honestly recorded as unavailable
    # rather than a fabricated 100%).
    if tsc_runner is not None:
        cleaned = list(tsc_clean(formatted, tsc_runner=tsc_runner,
                                 ignore_module_resolution=ignore_module_resolution,
                                 stats=clean_stats))
    else:
        cleaned = formatted

    # Stage 5: write the cleaned corpus as {"text": ...} JSONL for the Swift packer.
    cleaned_path = out_dir / "cleaned.jsonl"
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps({"text": r.text}) + "\n")

    manifest = {
        "stage_counts": {
            "n_source": n_source,
            "n_after_dedup": len(deduped),
            "n_after_prettier": len(formatted),
            "n_after_clean": len(cleaned),
        },
        "dedup": dedup_stats.as_dict(),
        "prettier_applied": prettier_runner is not None,
        "tsc_clean_applied": tsc_runner is not None,
        # The acceptance-critical LSP-clean filter rate -- null when Stage 4 was skipped
        # (no tsc toolchain), never a stand-in number.
        "clean_rate": clean_stats.as_dict() if tsc_runner is not None else None,
        "n_cleaned_docs": len(cleaned),
        "cleaned_jsonl": cleaned_path.name,
        "seq_len": seq_len,
        # Tokenize + pack is now the native Swift step (see the docstring); this pipeline
        # stops at cleaned text.
        "pack_note": "tokenize+pack with `monica-tokenize pack` (native Swift, #191/M13)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir for cleaned.jsonl + manifest.json (feed cleaned.jsonl "
                    "to `monica-tokenize pack` for the uint16 training shards)")
    ap.add_argument("--limit", type=int, default=-1,
                    help="cap the number of Stage-1 source records (-1: no cap)")
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="recorded in the manifest as a hint for the Swift pack step")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="MinHash-LSH near-dedup Jaccard threshold")
    ap.add_argument("--no-prettier", action="store_true", help="skip Stage 3 (prettier)")
    ap.add_argument("--dataset", default="bigcode/the-stack-v2-dedup")
    ap.add_argument("--config", default="TypeScript")
    ap.add_argument("--from-jsonl", type=Path, default=None,
                    help="offline source of {text}/{content} rows, bypassing Stack v2 / "
                         "Software Heritage S3 entirely -- runs Stages 2-5 end to end with "
                         "no SWH/AWS credentials (the CI path)")
    args = ap.parse_args()

    # --- Stage 1: source records + the permissive-license gate. ---
    if args.from_jsonl is not None:
        print(f"Stage 1: reading offline records from {args.from_jsonl} "
             "(bypassing Stack v2 / SWH S3)")
        records: Iterable[Record] = iter_jsonl_records(args.from_jsonl)
        if args.limit >= 0:
            records = itertools.islice(records, args.limit)
        records = [r for r in records if license_ok(r)]
    else:
        s3_client = resolve_swh_s3()
        if s3_client is None:
            print("Stage 1 SKIPPED: no Software Heritage S3 client resolvable (set "
                 "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY and `pip install -e \".[stack-v2]\"`) "
                 "and no --from-jsonl given -- nothing to build.")
            return 0
        print(f"Stage 1: streaming {args.dataset} ({args.config}) from Stack v2 / SWH "
             f"(limit={args.limit})")
        records = list(iter_stack_v2_ts(limit=args.limit, s3_client=s3_client,
                                        dataset=args.dataset, config=args.config))
    print(f"  -> {len(records)} permissively-licensed source record(s)")
    if not records:
        print("no source records -- nothing to build.")
        return 0

    # --- Stage 3 toolchain resolution. ---
    prettier_runner = None
    if args.no_prettier:
        print("Stage 3 SKIPPED: --no-prettier")
    else:
        prettier_argv = resolve_prettier()
        if prettier_argv is not None:
            prettier_runner = PrettierRunner(prettier_argv)
        else:
            print(f"Stage 3 SKIPPED: no prettier toolchain resolvable (run `npm install` in "
                 f"{SET_DIR}); passing records through unformatted.")

    # --- Stage 4 toolchain resolution. ---
    tsc_runner = None
    tsc_argv = resolve_tsc()
    if tsc_argv is not None:
        tsc_runner = TscRunner(tsc_argv)
    else:
        print(f"Stage 4 SKIPPED: no node/tsc toolchain resolvable (run `npm install` in "
             f"{SET_DIR}); the LSP-clean filter rate will be recorded as null.")

    out_dir = args.out
    try:
        manifest = run_pipeline(records, out_dir, seq_len=args.seq_len,
                                threshold=args.threshold,
                                prettier_runner=prettier_runner, tsc_runner=tsc_runner)
    finally:
        if tsc_runner is not None:
            tsc_runner.close()

    print(f"cleaned {manifest['n_cleaned_docs']} doc(s) -> {out_dir}/{manifest['cleaned_jsonl']}")
    print(f"  stage counts: {manifest['stage_counts']}")
    print(f"  LSP-clean filter rate: {manifest['clean_rate']}")
    print(f"  next: monica-tokenize pack --tokenizer <tokenizer.json> "
         f"--in {out_dir}/{manifest['cleaned_jsonl']} --out <shards> --seq-len {args.seq_len}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
