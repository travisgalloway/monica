"""Build the scale corpus with datatrove (#80, #252) — the pod/cluster driver.

Runs the staged datatrove pipeline (`src/data/datatrove_pipeline.py`): ingest a source, apply the
project filters (reusing `src/data/filters.py` semantics), write cleaned text shards, then
optionally run cross-source MinHash dedup. The cleaned shards then feed the native Swift
`monica-tokenize pack` step (or `src/data/shard.py`) to emit packed uint16 shards
(`.bin` + `.bounds` + `manifest.json`) readable directly by `src/data/loader.py`.

MUST run in the py3.11 datatrove venv (`.venv-dt`), not the main py3.14 env:

    # Local sample shard generation & verification (#252 Phase 1):
    .venv-dt/bin/python scripts/build_corpus.py --source sample \
        --out data/mhm_sample_clean --pack --shards-out data/mhm_sample_shards \
        --quality --license-filter --scrub --decontam --val-tokens 1024

    # Pod/cluster production run (#252 Phase 2):
    .venv-dt/bin/python scripts/build_corpus.py --source fineweb-edu \
        --out s3://monica-training/reserve-pretrain --executor slurm --tasks 200 \
        --quality --license-filter --scrub --decontam --dedup
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def find_monica_tokenize(explicit: str | Path | None = None) -> Path | None:
    """Locate the monica-tokenize binary across known build directories."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    env = os.environ.get("MONICA_TOKENIZE")
    if env and Path(env).exists():
        return Path(env)
    for root in (REPO_ROOT, Path("/Users/travisgalloway/github/monica")):
        swift_dir = root / "swift"
        for mode in ("release", "debug"):
            for cand in (
                swift_dir / ".build" / mode / "monica-tokenize",
                swift_dir / ".build" / "arm64-apple-macosx" / mode / "monica-tokenize",
            ):
                if cand.exists():
                    return cand
    return None


def find_default_tokenizer(explicit: str | Path | None = None) -> Path | None:
    """Locate default MHM tokenizer vocabulary (preferring 32k/49k MHM vocabs)."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    candidates = [
        REPO_ROOT / "data" / "tokenizer" / "vocab-32768.json",
        Path.home() / "monica-data" / "vocab-sweep-251-half" / "tok" / "vocab-32768.json",
        Path.home() / "monica-data" / "vocab-sweep-251" / "tok" / "vocab-32768.json",
        Path.home() / "monica-data" / "vocab-sweep-251-half" / "tok" / "vocab-49152.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_default_sample(explicit: str | Path | None = None) -> Path | None:
    """Locate sample corpus file."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    candidates = [
        Path.home() / "monica-data" / "vocab-sweep-251-half" / "sample" / "sample.jsonl",
        Path.home() / "monica-data" / "vocab-sweep-251" / "sample" / "sample.jsonl",
        REPO_ROOT / "data" / "sample" / "sample.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def iter_cleaned_docs(cleaned_dir: Path | str) -> Iterator[dict]:
    """Iterate over cleaned JSONL/GZ files emitted by Datatrove."""
    cleaned_dir = Path(cleaned_dir)
    files = sorted(glob.glob(f"{cleaned_dir}/**/*.jsonl*", recursive=True))
    for fp in files:
        if fp.endswith(".gz"):
            with gzip.open(fp, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif fp.endswith(".jsonl"):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def consolidate_cleaned(cleaned_dir: Path | str, out_file: Path | str) -> Tuple[int, Dict[str, Any], int]:
    """Consolidate cleaned records to a single cleaned.jsonl and compute per-language mix."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    total_bytes = 0
    lang_counts: Dict[str, int] = {}
    lang_bytes: Dict[str, int] = {}

    with open(out_file, "w", encoding="utf-8") as f:
        for doc in iter_cleaned_docs(cleaned_dir):
            text = doc.get("text", "")
            if not text:
                continue
            f.write(json.dumps({"text": text}) + "\n")
            n_docs += 1
            n_b = len(text.encode("utf-8"))
            total_bytes += n_b
            md = doc.get("metadata") or {}
            lang = doc.get("lang") or md.get("lang") or "unknown"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            lang_bytes[lang] = lang_bytes.get(lang, 0) + n_b

    lang_mix = {}
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        b = lang_bytes[lang]
        lang_mix[lang] = {
            "docs": count,
            "bytes": b,
            "pct_docs": round(100.0 * count / n_docs, 2) if n_docs else 0.0,
            "pct_bytes": round(100.0 * b / total_bytes, 2) if total_bytes else 0.0,
        }
    return n_docs, lang_mix, total_bytes


def pack_cleaned_shards(cleaned_jsonl: Path, shards_out: Path, tokenizer_path: Path,
                        seq_len: int = 8192, shard_size_mb: int = 512,
                        tokenize_bin: Path | None = None,
                        chunk_align: int | None = None) -> dict:
    """Pack cleaned.jsonl into uint16 .bin + .bounds + manifest.json shards."""
    shards_out = Path(shards_out)
    shards_out.mkdir(parents=True, exist_ok=True)
    if tokenize_bin is not None:
        cmd = [
            str(tokenize_bin), "pack",
            "--tokenizer", str(tokenizer_path),
            "--in", str(cleaned_jsonl),
            "--out", str(shards_out),
            "--seq-len", str(seq_len),
            "--shard-size-mb", str(shard_size_mb),
        ]
        if chunk_align:
            cmd.extend(["--chunk-align", str(chunk_align)])
        subprocess.run(cmd, check=True)
    else:
        # Fallback to Python pack_sequences
        from src.data.shard import pack_sequences
        from src.data.tokenize import ByteTokenizer
        tok = ByteTokenizer()
        docs = []
        with open(cleaned_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line).get("text", ""))
        tokenized = [tok.encode(d) for d in docs if d]
        pack_sequences(tokenized, shards_out, seq_len=seq_len,
                       shard_size_mb=shard_size_mb, tokenizer="code")

    manifest_path = shards_out / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"pack step failed to emit {manifest_path}")
    return json.loads(manifest_path.read_text())


def run_corpus_pipeline(
    source: str = "sample",
    out_dir: Path | str = "data/mhm_sample_clean",
    *,
    from_jsonl: Path | str | None = None,
    limit: int = -1,
    split: str = "train",
    executor_kind: str = "local",
    tasks: int = 1,
    workers: int = 1,
    logging_dir: Path | str | None = None,
    quality: bool = True,
    license_filter: bool = True,
    drop_minified: bool = False,
    drop_autogen: bool = False,
    scrub: bool = True,
    decontam: bool = True,
    decontam_file: Path | str | None = None,
    dedup: bool = False,
    pack: bool = False,
    tokenizer: Path | str | None = None,
    shards_out: Path | str | None = None,
    seq_len: int = 8192,
    shard_size_mb: int = 512,
    val_tokens: int | None = None,
    tokenize_bin: Path | str | None = None,
) -> dict:
    """End-to-end driver: clean -> optional dedup -> consolidate -> pack -> split."""
    from src.data import datatrove_pipeline as dt

    out_dir = Path(out_dir)
    logging_dir = Path(logging_dir or (out_dir / "logs"))

    # 1. Resolve source reader
    sample_path = None
    if source == "fineweb-edu":
        reader = dt.fineweb_edu_reader(limit=limit, split=split)
    elif source in ("sample", "jsonl"):
        sample_path = find_default_sample(from_jsonl)
        if sample_path is None or not Path(sample_path).exists():
            raise FileNotFoundError(
                f"Sample corpus not found (given: {from_jsonl}). Provide --from-jsonl <path> "
                "or run scripts/vocab_sweep.py --sample-only to populate monica-data/."
            )
        reader = dt.jsonl_reader(str(sample_path), limit=limit)
    else:
        raise ValueError(f"Unknown source {source!r}")

    # 2. Decontamination setup
    decon = None
    decontam_path = None
    if decontam or decontam_file:
        decontam_path = Path(decontam_file) if decontam_file else (REPO_ROOT / "eval_sets/decontam/blocklist.txt")
        if decontam_path.exists():
            from src.data.dedup import Decontaminator
            with open(decontam_path, encoding="utf-8") as f:
                decon = Decontaminator.from_texts(f)

    # 3. Clean pipeline
    pipeline = dt.clean_pipeline(
        reader, str(out_dir), quality=quality, license_filter=license_filter,
        drop_minified=drop_minified, drop_autogen=drop_autogen, scrub=scrub,
        decontaminator=decon)
    executor = dt.make_executor(pipeline, str(logging_dir / "clean"), kind=executor_kind,
                                tasks=tasks, workers=workers)
    executor.run()
    cleaned_dir = out_dir / "cleaned"
    print(f"clean pass complete -> {cleaned_dir}")

    # 4. Optional MinHash dedup
    text_dir = cleaned_dir
    if dedup:
        dedup_dir = out_dir / "dedup"
        dt.run_minhash_dedup(str(cleaned_dir), str(dedup_dir),
                             kind=executor_kind, tasks=tasks, workers=workers,
                             logging_dir=str(logging_dir / "dedup"))
        text_dir = dedup_dir / "deduplicated"
        print(f"dedup complete -> {text_dir}")

    # 5. Consolidate into cleaned.jsonl and calculate stats
    cleaned_jsonl = out_dir / "cleaned.jsonl"
    n_cleaned, lang_mix, total_bytes = consolidate_cleaned(text_dir, cleaned_jsonl)

    # Extract source count from stats if possible
    n_source = n_cleaned
    stats_file = logging_dir / "clean" / "stats.json"
    if stats_file.exists():
        try:
            stats_data = json.loads(stats_file.read_text())
            for step in stats_data:
                if "READER" in step.get("name", ""):
                    doc_stats = step.get("stats", {}).get("documents", {})
                    if isinstance(doc_stats, dict) and "total" in doc_stats:
                        n_source = int(doc_stats["total"])
                    elif "documents" in step.get("stats", {}):
                        n_source = int(step["stats"]["documents"])
                    break
        except Exception:
            pass
    if limit > 0 and n_source > limit:
        n_source = limit

    n_dropped = max(0, n_source - n_cleaned)
    filter_rate = {
        "n_source": n_source,
        "n_cleaned": n_cleaned,
        "n_dropped": n_dropped,
        "drop_rate": round(n_dropped / n_source, 4) if n_source > 0 else 0.0,
    }

    decontam_info = {
        "applied": decon is not None,
        "blocklist": str(decontam_path.relative_to(REPO_ROOT)) if decontam_path and decontam_path.is_relative_to(REPO_ROOT) else str(decontam_path) if decontam_path else None,
        "eval_sets": ["eval_sets/humaneval_ts", "eval_sets/ts_error_injection"] if decon is not None else [],
    }

    result = {
        "source": source,
        "cleaned_jsonl": str(cleaned_jsonl),
        "n_cleaned": n_cleaned,
        "total_bytes": total_bytes,
        "language_mix": lang_mix,
        "filter_rate": filter_rate,
        "decontamination": decontam_info,
    }

    # 6. Packing
    if pack:
        shards_dir = Path(shards_out or (out_dir / "shards"))
        tok_bin = find_monica_tokenize(tokenize_bin)
        tok_file = find_default_tokenizer(tokenizer)
        if tok_bin is not None and tok_file is None:
            raise FileNotFoundError(
                f"Tokenizer JSON file not found (given: {tokenizer}). Provide --tokenizer <path>."
            )

        manifest = pack_cleaned_shards(
            cleaned_jsonl, shards_dir, tok_file if tok_file else Path("dummy"),
            seq_len=seq_len, shard_size_mb=shard_size_mb,
            tokenize_bin=tok_bin,
        )

        # Enrich manifest with language mix, filter rate, and decontamination provenance
        manifest["language_mix"] = lang_mix
        manifest["filter_rate"] = filter_rate
        manifest["decontamination"] = decontam_info
        manifest["total_raw_bytes"] = total_bytes
        if tok_file:
            manifest["tokenizer_file"] = tok_file.name

        (shards_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        result["shards_dir"] = str(shards_dir)
        result["manifest"] = manifest

        # 7. Validation split
        if val_tokens:
            from src.data.split import split_shards
            split_dir = shards_dir / "split"
            tr_path, va_path = split_shards(shards_dir, split_dir, val_tokens=val_tokens)
            result["split"] = {
                "train_bin": str(tr_path),
                "val_bin": str(va_path),
                "val_tokens": val_tokens,
            }
            print(f"validation split ({val_tokens} tokens) -> {split_dir}")

        print(f"packed {manifest.get('n_sequences', 0)} seq x {seq_len} "
              f"({manifest.get('n_tokens', 0)} tokens) -> {shards_dir}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("fineweb-edu", "jsonl", "sample"), default="fineweb-edu",
                    help="corpus source (fineweb-edu, local jsonl, or sample mixture; #70/#252)")
    ap.add_argument("--from-jsonl", default=None,
                    help="path to input JSONL file or directory (for --source jsonl or sample)")
    ap.add_argument("--out", required=True,
                    help="output directory prefix (writes <out>/cleaned, <out>/cleaned.jsonl)")
    ap.add_argument("--limit", type=int, default=-1, help="max docs to read (-1 = no cap)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--executor", choices=("local", "slurm"), default="local")
    ap.add_argument("--tasks", type=int, default=1, help="parallel tasks (shards)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent local workers")
    ap.add_argument("--logging-dir", default=None, help="datatrove logs/stats (default <out>/logs)")
    # Stage-3 filters
    ap.add_argument("--quality", action="store_true", help="text-quality heuristics")
    ap.add_argument("--license-filter", action="store_true", help="permissive-only gate on code")
    ap.add_argument("--drop-minified", action="store_true")
    ap.add_argument("--drop-autogen", action="store_true")
    ap.add_argument("--scrub", action="store_true", help="redact secrets/PII")
    ap.add_argument("--decontam", action="store_true",
                    help="filter docs overlapping eval benchmarks (default blocklist: eval_sets/decontam/blocklist.txt)")
    ap.add_argument("--decontam-file", default=None,
                    help="custom text file of eval-benchmark lines to strip (13/7-gram overlap)")
    # Stage-4 dedup
    ap.add_argument("--dedup", action="store_true", help="run cross-source MinHash dedup after clean")
    # Stage-5 tokenization & packing
    ap.add_argument("--pack", action="store_true",
                    help="tokenize and pack cleaned docs into .bin + .bounds + manifest.json shards")
    ap.add_argument("--tokenizer", default=None,
                    help="path to tokenizer.json / vocab-32768.json for monica-tokenize pack")
    ap.add_argument("--shards-out", default=None,
                    help="output directory for packed shards (default: <out>/shards)")
    ap.add_argument("--seq-len", type=int, default=8192,
                    help="packed sequence length (default: 8192)")
    ap.add_argument("--shard-size-mb", type=int, default=512,
                    help="packed shard size budget in MB (default: 512)")
    ap.add_argument("--val-tokens", type=int, default=None,
                    help="carve out a validation split with split_shards")
    ap.add_argument("--tokenize-bin", default=None,
                    help="path to monica-tokenize binary")
    args = ap.parse_args()

    run_corpus_pipeline(
        source=args.source,
        out_dir=args.out,
        from_jsonl=args.from_jsonl,
        limit=args.limit,
        split=args.split,
        executor_kind=args.executor,
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=args.logging_dir,
        quality=args.quality,
        license_filter=args.license_filter,
        drop_minified=args.drop_minified,
        drop_autogen=args.drop_autogen,
        scrub=args.scrub,
        decontam=args.decontam or (args.decontam_file is not None),
        decontam_file=args.decontam_file,
        dedup=args.dedup,
        pack=args.pack,
        tokenizer=args.tokenizer,
        shards_out=args.shards_out,
        seq_len=args.seq_len,
        shard_size_mb=args.shard_size_mb,
        val_tokens=args.val_tokens,
        tokenize_bin=args.tokenize_bin,
    )

    # A streaming HF reader truncated by --limit leaves a non-daemon prefetch thread alive, which
    # hangs the interpreter at teardown after all shards/markers are already flushed. Force a clean
    # exit ONLY in that case.
    if args.source == "fineweb-edu" and args.limit and args.limit > 0:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
