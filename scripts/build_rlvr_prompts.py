"""#230 -- build disjoint train/val RLVR prompt JSONLs from the checked-in TS
eval sets (`eval_sets/ts_error_injection/eval.jsonl` + `clean_prefixes.jsonl`,
optionally `eval_sets/humaneval_ts/humaneval_ts.jsonl`), stratified by
`error_class` so every error family lands on both sides, with a reproducible
split manifest recording the contamination boundary. Every input is already
checked into the repo (third-party / hand-authored eval data, none of it
Claude-generated) -- this script writes nothing new, it only re-slices.

Deliberately keeps the `{"answer": ...}` field name from the source eval sets
rather than the issue's `{prompt, reference}` rename -- `scripts/rlvr.py`
reads `prob.get("answer", "")` and Phase 1's LSP reward ignores it entirely,
so renaming would break `--reward math` compatibility for no gain (recorded
as a deliberate deviation in #230's plan).

  python scripts/build_rlvr_prompts.py --out-dir data/rlvr_ts --seed 0

CONTAMINATION NOTE: RLVR-training on rows drawn from the #194 eval set means
any LATER `scripts/eval_lsp_harness.py` run over the FULL `eval.jsonl` is
contaminated and not comparable to the published #199 numbers (clean-rate
0.887->0.962, pass@1 0.503). Report only on the held-out val split emitted
here, or on a source the training split never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_PATH = _REPO_ROOT / "eval_sets" / "ts_error_injection" / "eval.jsonl"
DEFAULT_CLEAN_PREFIXES_PATH = _REPO_ROOT / "eval_sets" / "ts_error_injection" / "clean_prefixes.jsonl"
DEFAULT_HUMANEVAL_PATH = _REPO_ROOT / "eval_sets" / "humaneval_ts" / "humaneval_ts.jsonl"

CONTAMINATION_NOTE = (
    "RLVR-training on rows drawn from the #194 eval set means any later "
    "scripts/eval_lsp_harness.py run over the FULL eval.jsonl is contaminated and "
    "not comparable to the published #199 numbers (clean-rate 0.887->0.962, "
    "pass@1 0.503). Report only on the held-out val split, or on a source the "
    "training split never touched."
)


def _load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_prompt_records(*, eval_path: Path = DEFAULT_EVAL_PATH,
                        clean_prefixes_path: Path = DEFAULT_CLEAN_PREFIXES_PATH,
                        humaneval_path: Path = DEFAULT_HUMANEVAL_PATH,
                        include_humaneval: bool = False) -> List[dict]:
    """Load and normalize every source into `{"id", "prompt", "answer",
    "error_class"}` (`answer` = the source's `gold_completion`). Raises on any
    duplicate `id` across sources -- the three sources use disjoint id
    prefixes by construction (`member-access-*`/`clean-prefix-*`/
    `HumanEval_*`), so a collision means an input changed underneath this
    script and must not be silently merged."""
    sources = [eval_path, clean_prefixes_path]
    if include_humaneval:
        sources.append(humaneval_path)

    out: List[dict] = []
    seen_ids: set = set()
    for path in sources:
        for r in _load_jsonl(path):
            rid = r["id"]
            if rid in seen_ids:
                raise ValueError(f"duplicate id {rid!r} across input sources "
                                 f"(last seen while reading {path})")
            seen_ids.add(rid)
            out.append({
                "id": rid,
                "prompt": r["prompt"],
                "answer": r.get("gold_completion", ""),
                "error_class": r["error_class"],
            })
    return out


def _stable_frac(seed: int, key: str) -> float:
    """`sha256(f"{seed}:{key}")` mapped to `[0, 1)` -- deliberately NOT
    Python's salted `hash()`, which is per-interpreter and not reproducible
    across runs (the same gotcha `src/eval/ssi_contract.py:split_by_repo`
    already documents for the #225 M3 repo split)."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_records(records: Sequence[dict], *, seed: int,
                  val_fraction: float = 0.2) -> Tuple[List[dict], List[dict]]:
    """Stratified stable-hash split: every `error_class` present in `records`
    contributes to BOTH sides. A pure per-record `frac < val_fraction`
    threshold, applied globally, could by chance strand an entire small class
    (`clean_control` is 12 rows in the base eval set) on one side; stratifying
    per-class and taking `round(n * val_fraction)` (at least 1 once a class has
    more than one member) rules that out while the ranking itself still comes
    from the stable per-id hash, so the split stays reproducible."""
    by_class: Dict[str, List[dict]] = {}
    for r in records:
        by_class.setdefault(r["error_class"], []).append(r)

    train: List[dict] = []
    val: List[dict] = []
    for recs in by_class.values():
        ranked = sorted(recs, key=lambda r: (_stable_frac(seed, r["id"]), r["id"]))
        n = len(ranked)
        n_val = min(n, max(1, round(n * val_fraction))) if n > 1 else 0
        val.extend(ranked[:n_val])
        train.extend(ranked[n_val:])
    return train, val


def _ids_sha256(records: Sequence[dict]) -> str:
    ids = sorted(r["id"] for r in records)
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def build_manifest(train: Sequence[dict], val: Sequence[dict], *, seed: int,
                   val_fraction: float, sources: Sequence[str]) -> dict:
    """Byte-reproducible manifest, following
    `src/eval/ssi_contract.py:split_manifest`'s provenance-manifest pattern:
    counts, per-class counts, and a sha256 over each side's sorted id list, so
    a run's claimed split can be CHECKED, not merely trusted."""
    classes = sorted({r["error_class"] for r in list(train) + list(val)})
    per_class = {
        cls: {
            "train": sum(1 for r in train if r["error_class"] == cls),
            "val": sum(1 for r in val if r["error_class"] == cls),
        }
        for cls in classes
    }
    manifest = {
        "seed": seed,
        "val_fraction": val_fraction,
        "sources": list(sources),
        "n_train": len(train),
        "n_val": len(val),
        "per_class_counts": per_class,
        "train_ids_sha256": _ids_sha256(train),
        "val_ids_sha256": _ids_sha256(val),
        "contamination_note": CONTAMINATION_NOTE,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("data/rlvr_ts"),
                    help="directory for train.jsonl/val.jsonl/rlvr_split_manifest.json; "
                         "override individual paths with --out-train/--out-val/--out-manifest")
    ap.add_argument("--out-train", type=Path, default=None)
    ap.add_argument("--out-val", type=Path, default=None)
    ap.add_argument("--out-manifest", type=Path, default=None)
    ap.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    ap.add_argument("--clean-prefixes-path", type=Path, default=DEFAULT_CLEAN_PREFIXES_PATH)
    ap.add_argument("--humaneval-path", type=Path, default=DEFAULT_HUMANEVAL_PATH)
    ap.add_argument("--include-humaneval", action="store_true",
                    help="also draw clean_control prompts from eval_sets/humaneval_ts "
                         "(159 rows; off by default to keep the base split small)")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_train = args.out_train or (args.out_dir / "train.jsonl")
    out_val = args.out_val or (args.out_dir / "val.jsonl")
    out_manifest = args.out_manifest or (args.out_dir / "rlvr_split_manifest.json")

    records = load_prompt_records(eval_path=args.eval_path,
                                  clean_prefixes_path=args.clean_prefixes_path,
                                  humaneval_path=args.humaneval_path,
                                  include_humaneval=args.include_humaneval)
    train, val = split_records(records, seed=args.seed, val_fraction=args.val_fraction)

    sources = [str(args.eval_path), str(args.clean_prefixes_path)]
    if args.include_humaneval:
        sources.append(str(args.humaneval_path))
    manifest = build_manifest(train, val, seed=args.seed, val_fraction=args.val_fraction,
                              sources=sources)

    _write_jsonl(out_train, train)
    _write_jsonl(out_val, val)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {len(train)} train / {len(val)} val rows "
         f"({len(train) + len(val)} total, seed={args.seed})")
    print(f"  train    -> {out_train}")
    print(f"  val      -> {out_val}")
    print(f"  manifest -> {out_manifest}")
    print()
    print("CONTAMINATION NOTE:")
    print(f"  {CONTAMINATION_NOTE}")


if __name__ == "__main__":
    main()
