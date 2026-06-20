"""Build the scale corpus with datatrove (#80) — the pod/cluster driver.

Runs the staged datatrove pipeline (`src/data/datatrove_pipeline.py`): ingest a source, apply the
project filters (reusing `src/data/filters.py` semantics), write cleaned text shards, then
optionally run cross-source MinHash dedup. Output is a `storage.py` class prefix — a local dir for
the Mac/py3.11 smoke, `s3://monica-training/...` on a RunPod CPU pod. The cleaned shards then feed
the existing `python -m src.data.shard` tokenize->uint32 step (trainer format unchanged).

MUST run in the py3.11 datatrove venv (`.venv-dt`), not the main py3.14 env:

    .venv-dt/bin/python scripts/build_corpus.py --source fineweb-edu --limit 500 \
        --out data/reserve-pretrain --executor local --quality --scrub
    # pod:
    .venv-dt/bin/python scripts/build_corpus.py --source fineweb-edu \
        --out s3://monica-training/reserve-pretrain --executor slurm --tasks 200 \
        --quality --license-filter --scrub --dedup
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("fineweb-edu",), default="fineweb-edu",
                    help="corpus source (more readers added incrementally; #70/#71)")
    ap.add_argument("--out", required=True,
                    help="output class prefix (local dir or s3://monica-training/<class>); "
                         "writes <out>/cleaned and, with --dedup, <out>/dedup")
    ap.add_argument("--limit", type=int, default=-1, help="max docs to read (-1 = no cap)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--executor", choices=("local", "slurm"), default="local")
    ap.add_argument("--tasks", type=int, default=1, help="parallel tasks (shards)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent local workers")
    ap.add_argument("--logging-dir", default=None, help="datatrove logs/stats (default <out>/logs)")
    # Stage-3 filters (opt-in, mirror src/data/corpus.py / build_corpus flags)
    ap.add_argument("--quality", action="store_true", help="text-quality heuristics")
    ap.add_argument("--license-filter", action="store_true", help="permissive-only gate on code")
    ap.add_argument("--drop-minified", action="store_true")
    ap.add_argument("--drop-autogen", action="store_true")
    ap.add_argument("--scrub", action="store_true", help="redact secrets/PII")
    ap.add_argument("--decontam-file", default=None,
                    help="text file of eval-benchmark lines to strip (13/7-gram overlap)")
    # Stage-4 dedup
    ap.add_argument("--dedup", action="store_true", help="run cross-source MinHash dedup after clean")
    args = ap.parse_args()

    from src.data import datatrove_pipeline as dt

    logging_dir = args.logging_dir or f"{str(args.out).rstrip('/')}/logs"

    decon = None
    if args.decontam_file:
        from src.data.dedup import Decontaminator
        with open(args.decontam_file, encoding="utf-8") as f:
            decon = Decontaminator.from_texts(f)

    if args.source == "fineweb-edu":
        reader = dt.fineweb_edu_reader(limit=args.limit, split=args.split)

    pipeline = dt.clean_pipeline(
        reader, args.out, quality=args.quality, license_filter=args.license_filter,
        drop_minified=args.drop_minified, drop_autogen=args.drop_autogen, scrub=args.scrub,
        decontaminator=decon)
    executor = dt.make_executor(pipeline, f"{logging_dir}/clean", kind=args.executor,
                                tasks=args.tasks, workers=args.workers)
    executor.run()
    print(f"clean pass complete -> {str(args.out).rstrip('/')}/cleaned")

    if args.dedup:
        dt.run_minhash_dedup(f"{str(args.out).rstrip('/')}/cleaned", f"{str(args.out).rstrip('/')}/dedup",
                             kind=args.executor, tasks=args.tasks, workers=args.workers,
                             logging_dir=f"{logging_dir}/dedup")
        print(f"dedup complete -> {str(args.out).rstrip('/')}/dedup/deduplicated")

    # A streaming HF reader truncated by --limit leaves a non-daemon prefetch thread alive, which
    # hangs the interpreter at teardown after all shards/markers are already flushed. Force a clean
    # exit ONLY in that case; a full (unbounded) run drains the reader and exits normally, so it
    # keeps normal teardown (finally blocks, atexit, FS cleanup).
    if args.limit and args.limit > 0:
        import os
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
