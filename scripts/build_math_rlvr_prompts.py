"""#103 -- build disjoint train/val RLVR math prompt JSONLs from the
checked-in GSM8K and MATH eval sets (`eval_sets/math/gsm8k.jsonl` and `math.jsonl`),
stratified by dataset source with reproducible SHA-256 split manifests.

Every prompt is verifiable with ground-truth numeric or LaTeX boxed solutions
for `src.train.verifiers.MathVerifier` and `scripts/rlvr.py --reward math`.

Usage:
    python scripts/build_math_rlvr_prompts.py --out-dir eval_sets/math --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GSM8K_PATH = _REPO_ROOT / "eval_sets" / "math" / "gsm8k.jsonl"
DEFAULT_MATH_PATH = _REPO_ROOT / "eval_sets" / "math" / "math.jsonl"


def _load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_math_records(*, gsm8k_path: Path = DEFAULT_GSM8K_PATH,
                      math_path: Path = DEFAULT_MATH_PATH) -> List[dict]:
    """Load and normalize math prompt records from GSM8K and MATH sources."""
    sources = [gsm8k_path, math_path]
    out: List[dict] = []
    seen_ids: set = set()
    for path in sources:
        for r in _load_jsonl(path):
            rid = r["id"]
            if rid in seen_ids:
                raise ValueError(f"duplicate id {rid!r} across math sources (in {path})")
            seen_ids.add(rid)
            out.append({
                "id": rid,
                "prompt": r["prompt"],
                "answer": r["answer"],
                "error_class": r.get("error_class", "math"),
                "domain": "math",
            })
    return out


def _stable_frac(seed: int, key: str) -> float:
    """Byte-reproducible sha256(f"{seed}:{key}") mapped to [0, 1)."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_records(records: Sequence[dict], *, seed: int,
                  val_fraction: float = 0.2) -> Tuple[List[dict], List[dict]]:
    """Stratified stable-hash split: every error_class/subset contributes to both sides."""
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
    """Provenance manifest recording counts and SHA-256 digests."""
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
    ap.add_argument("--out-dir", type=Path, default=Path("eval_sets/math"),
                    help="directory for train.jsonl / val.jsonl / manifest.json")
    ap.add_argument("--out-train", type=Path, default=None)
    ap.add_argument("--out-val", type=Path, default=None)
    ap.add_argument("--out-manifest", type=Path, default=None)
    ap.add_argument("--gsm8k-path", type=Path, default=DEFAULT_GSM8K_PATH)
    ap.add_argument("--math-path", type=Path, default=DEFAULT_MATH_PATH)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_train = args.out_train or (args.out_dir / "train.jsonl")
    out_val = args.out_val or (args.out_dir / "val.jsonl")
    out_manifest = args.out_manifest or (args.out_dir / "manifest.json")

    records = load_math_records(gsm8k_path=args.gsm8k_path, math_path=args.math_path)
    train, val = split_records(records, seed=args.seed, val_fraction=args.val_fraction)

    sources = [str(args.gsm8k_path.resolve().relative_to(_REPO_ROOT)), str(args.math_path.resolve().relative_to(_REPO_ROOT))]
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


if __name__ == "__main__":
    main()
