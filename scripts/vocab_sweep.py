#!/usr/bin/env python3
"""Sweep the code-BPE vocab size on a multilingual sample and print the bytes/token table (#251).

    .venv/bin/python scripts/vocab_sweep.py [--work-dir ~/monica-data/vocab-sweep-251]
        [--vocab-sizes 16384,32768,49152] [--threads 64] [--mixture-scale 1.0]
        [--rebuild-sample] [--no-truncate] [--sample-only] [--tokenize-bin PATH]

Flow: build (or reuse) the tagged sample -> train the BPE **once** at the largest size ->
derive the smaller sizes by truncating `merges` -> `monica-tokenize stats` per size ->
bytes/token table (overall + per language) + the sanity checks.

**Why one train and truncate.** `Trainer.train` is greedy with no lookahead, so a k-vocab
run's merge list is exactly the first k-262 merges of any larger run (262 = 6 specials + 256
base bytes) — verified empirically on a 2.46 MB corpus and now guarded by `monica-selfcheck`.
Training once is ~2.5x cheaper than three independent runs on the dominant cost.
`--no-truncate` trains each size independently and is kept as the audit path.

The sample lives outside the repo (default `~/monica-data/vocab-sweep-251/`) and is cached:
a re-run costs zero network time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.vocab_sample import (  # noqa: E402
    DEFAULT_MIXTURE, build_sample, resolve_s3, scale_mixture)

#: 6 special tokens + the 256 base bytes precede the merges in the id space.
BASE_OFFSET = 262
DEFAULT_WORK_DIR = Path.home() / "monica-data" / "vocab-sweep-251"
REPO_ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_tokenize_bin(explicit: str | None) -> Path:
    """The **release** binary — every timing this sweep is budgeted against was measured with
    it, and the debug build is several times slower. Built on demand if absent."""
    if explicit:
        return Path(explicit)
    swift_dir = REPO_ROOT / "swift"
    binary = swift_dir / ".build" / "release" / "monica-tokenize"
    if not binary.exists():
        log("building monica-tokenize (release)...")
        subprocess.run(["swift", "build", "-c", "release"], cwd=swift_dir, check=True)
    if not binary.exists():
        raise SystemExit(f"monica-tokenize not found at {binary}")
    return binary


def train_tokenizer(binary: Path, corpus: Path, out: Path, vocab: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"training vocab {vocab} on {corpus} ...")
    t0 = time.time()
    subprocess.run([str(binary), "train", "--in", str(corpus), "--out", str(out),
                    "--vocab-size", str(vocab)], check=True)
    log(f"  trained {vocab} in {time.time() - t0:.0f}s -> {out}")


def truncate_tokenizer(src: Path, dst: Path, vocab: int) -> None:
    """Derive a smaller tokenizer by keeping the first `vocab - 262` merges.

    Fails loudly if the source run produced fewer merges than requested: that means the
    trainer hit its `pairCounts.isEmpty` break because the sample had too few unique
    pre-tokens, and the large-vocab number would be an artifact of sample size rather than a
    measurement of compression.
    """
    fmt = json.loads(src.read_text())
    want = vocab - BASE_OFFSET
    have = len(fmt["merges"])
    if have < want:
        raise SystemExit(
            f"cannot derive vocab {vocab}: {src.name} has only {have} merges "
            f"({have + BASE_OFFSET} tokens), fewer than the {want} requested. The sample "
            f"has too few unique pre-tokens — rebuild it larger rather than reporting this.")
    fmt["merges"] = fmt["merges"][:want]
    dst.write_text(json.dumps(fmt))
    log(f"  truncated -> {dst.name} ({want} merges)")


def run_stats(binary: Path, tokenizer: Path, corpus: Path) -> Dict:
    log(f"stats: {tokenizer.name} ...")
    t0 = time.time()
    res = subprocess.run([str(binary), "stats", "--tokenizer", str(tokenizer),
                          "--in", str(corpus), "--json"],
                         check=True, capture_output=True, text=True)
    log(f"  stats {tokenizer.name} in {time.time() - t0:.0f}s")
    return json.loads(res.stdout)


def format_table(stats: Dict[int, Dict], sizes: List[int]) -> str:
    """Markdown bytes/token table: rows = languages then `overall`, columns = vocab sizes."""
    base = sizes[0]
    langs = sorted(stats[base]["by_lang"].keys())
    cols = "".join(f" {s} |" for s in sizes) + "".join(f" Δ vs {base} |" for s in sizes[1:])
    lines = [f"| Language | bytes |{cols}",
             "|---|---:|" + "---:|" * (2 * len(sizes) - 1)]

    def row(name: str, entries: Dict[int, Dict]) -> str:
        b = entries[base]["bytes"]
        cells = [f"{entries[s]['bytes_per_token']:.4f}" for s in sizes]
        best = entries[base]["bytes_per_token"]
        deltas = [f"{(entries[s]['bytes_per_token'] / best - 1) * 100:+.2f}%" for s in sizes[1:]]
        return f"| {name} | {b / 1e6:.2f} MB | " + " | ".join(cells + deltas) + " |"

    for lang in langs:
        lines.append(row(lang, {s: stats[s]["by_lang"][lang] for s in sizes}))
    lines.append(row("**overall**", {s: stats[s]["overall"] for s in sizes}))
    return "\n".join(lines)


def sanity_checks(stats: Dict[int, Dict], tokenizers: Dict[int, Path],
                  sizes: List[int]) -> List[str]:
    """The Step-8 invariants, reported as PASS/FAIL lines. A failure here means a real bug in
    the truncation or in `stats`, not a surprising measurement."""
    out: List[str] = []
    base = sizes[0]

    specials = {s: json.loads(tokenizers[s].read_text())["special_tokens"] for s in sizes}
    ok = all(specials[s] == specials[base] for s in sizes)
    out.append(f"{'PASS' if ok else 'FAIL'}: special-token layout identical across vocab sizes")

    ok = all(stats[s]["overall"]["pretokens"] == stats[base]["overall"]["pretokens"]
             for s in sizes)
    ok = ok and all(stats[s]["by_lang"][lang]["pretokens"]
                    == stats[base]["by_lang"][lang]["pretokens"]
                    for s in sizes for lang in stats[base]["by_lang"])
    out.append(f"{'PASS' if ok else 'FAIL'}: pre-token counts identical across vocab sizes "
               f"({stats[base]['overall']['pretokens']} overall)")

    worst = max(stats[s]["overall"]["max_token_id"] for s in sizes)
    out.append(f"{'PASS' if worst < 65536 else 'FAIL'}: max token id {worst} < 65536 "
               f"(uint16 packing)")

    violations = []
    for lang in list(stats[base]["by_lang"]) + ["__overall__"]:
        def bpt(s: int) -> float:
            d = stats[s]["overall"] if lang == "__overall__" else stats[s]["by_lang"][lang]
            return d["bytes_per_token"]
        for a, b in zip(sizes, sizes[1:]):
            if bpt(b) < bpt(a) - 1e-9:
                violations.append(f"{lang}: {a}->{b} {bpt(a):.4f}->{bpt(b):.4f}")
    out.append(("PASS" if not violations else "FAIL") +
               ": bytes/token is monotonic non-decreasing in vocab size"
               + ("" if not violations else " — " + "; ".join(violations)))
    return out


def decide(stats: Dict[int, Dict], sizes: List[int]) -> Tuple[str, int]:
    """Apply the pre-registered decision rule (written down before the numbers were read).

    1. Reject a candidate more than 3% worse than the best on ANY single language.
    2. Among survivors, take the SMALLEST vocab within 2% of the best overall bytes/token —
       vocab is not free: with `d_model 768` and tied embeddings each extra token costs 768
       params twice over (embedding + output head).
    3. Tie-break on TypeScript (>3% better on the TS row wins) — the program is TS-first.
    """
    langs = list(stats[sizes[0]]["by_lang"])
    lines = ["**Rule 1 — per-language anti-starvation (reject >3% worse than best on any language):**"]
    survivors = []
    for s in sizes:
        worst_gap = 0.0
        worst_lang = ""
        for lang in langs:
            best = max(stats[t]["by_lang"][lang]["bytes_per_token"] for t in sizes)
            gap = (best - stats[s]["by_lang"][lang]["bytes_per_token"]) / best
            if gap > worst_gap:
                worst_gap, worst_lang = gap, lang
        keep = worst_gap <= 0.03
        if keep:
            survivors.append(s)
        lines.append(f"  {s}: worst per-language gap {worst_gap * 100:.2f}% "
                     f"({worst_lang or 'n/a'}) -> {'keep' if keep else 'REJECT'}")

    best_overall = max(stats[s]["overall"]["bytes_per_token"] for s in sizes)
    lines.append("**Rule 2 — smallest survivor within 2% of the best overall bytes/token:**")
    within = []
    for s in survivors:
        gap = (best_overall - stats[s]["overall"]["bytes_per_token"]) / best_overall
        ok = gap <= 0.02
        if ok:
            within.append(s)
        lines.append(f"  {s}: overall {stats[s]['overall']['bytes_per_token']:.4f} B/tok, "
                     f"{gap * 100:.2f}% off best -> {'within 2%' if ok else 'outside 2%'}")
    pool = within or survivors or sizes
    winner = min(pool)

    lines.append("**Rule 3 — TypeScript tie-break (>3% better on the TS row):**")
    ts_best = max(stats[s]["by_lang"]["typescript"]["bytes_per_token"] for s in pool)
    bumped = [s for s in pool if s > winner
              and (ts_best - stats[winner]["by_lang"]["typescript"]["bytes_per_token"]) / ts_best > 0.03
              and stats[s]["by_lang"]["typescript"]["bytes_per_token"] >= ts_best * 0.999]
    if bumped:
        lines.append(f"  {winner} is >3% off the best TS row -> promote to {min(bumped)}")
        winner = min(bumped)
    else:
        lines.append(f"  no candidate is >3% better than {winner} on TypeScript -> no change")

    lines.append(f"\n**WINNER: {winner}** "
                 f"(overall {stats[winner]['overall']['bytes_per_token']:.4f} bytes/token)")
    return "\n".join(lines), winner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    ap.add_argument("--vocab-sizes", default="16384,32768,49152")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--mixture-scale", type=float, default=1.0,
                    help="scale every slice's byte target (keeps all languages)")
    ap.add_argument("--rebuild-sample", action="store_true")
    ap.add_argument("--no-truncate", action="store_true",
                    help="train each vocab size independently (audit path; much slower)")
    ap.add_argument("--sample-only", action="store_true",
                    help="build the sample and stop (the network-bound half)")
    ap.add_argument("--tokenize-bin", default=None)
    args = ap.parse_args()

    sizes = sorted(int(x) for x in args.vocab_sizes.split(","))
    work_dir = args.work_dir.expanduser()

    mixture = scale_mixture(DEFAULT_MIXTURE, args.mixture_scale)
    s3 = resolve_s3()
    if s3 is None:
        log("warning: no boto3/S3 client — only a cached sample can be used")
    log(f"work dir: {work_dir}")
    corpus = build_sample(work_dir, mixture=mixture, threads=args.threads, s3_client=s3,
                          rebuild=args.rebuild_sample, log=log)
    if args.sample_only:
        return 0

    binary = resolve_tokenize_bin(args.tokenize_bin)
    tok_dir = work_dir / "tok"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizers: Dict[int, Path] = {s: tok_dir / f"vocab-{s}.json" for s in sizes}

    if args.no_truncate:
        for s in sizes:
            train_tokenizer(binary, corpus, tokenizers[s], s)
    else:
        biggest = sizes[-1]
        train_tokenizer(binary, corpus, tokenizers[biggest], biggest)
        for s in sizes[:-1]:
            truncate_tokenizer(tokenizers[biggest], tokenizers[s], s)

    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[int, Dict] = {}
    for s in sizes:
        stats[s] = run_stats(binary, tokenizers[s], corpus)
        (results_dir / f"stats-{s}.json").write_text(json.dumps(stats[s], indent=2) + "\n")

    table = format_table(stats, sizes)
    checks = sanity_checks(stats, tokenizers, sizes)
    rule, winner = decide(stats, sizes)

    body = "\n\n".join(["## bytes/token", table, "## sanity checks",
                        "\n".join(f"- {c}" for c in checks), "## decision rule", rule])
    (results_dir / "table.md").write_text(body + "\n")
    print("\n" + body + "\n")
    log(f"wrote {results_dir / 'table.md'}")
    return 0 if all(c.startswith("PASS") for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
