"""Estimate training wall-clock for various param sizes on reference hardware.

Answers "how long would training a model of size X take on hardware Y?" using the
6·N·D FLOPs model and a small hardware registry (M1 Pro — calibrated from the
measured poc 99 s/step anchor; single H100 and 8×H100 — peak × assumed MFU). All
numbers are planning estimates; see `src/model/train_time.py` for the assumptions.

  python scripts/train_time.py                            # default ladder, all 3 machines
  python scripts/train_time.py --tokens 3B                # fixed budget (m1pro col ≈ 26 d)
  python scripts/train_time.py --hours 24 --params 200M   # tokens trainable in 24h (M1 ≈ 72M)
  python scripts/train_time.py --params 1B,7B --hardware h100,8xh100
  python scripts/train_time.py --config config/student-1b.yaml
  python scripts/train_time.py --mfu 0.5 --scaling 0.9    # H100 sensitivity
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Allow `python scripts/train_time.py` from the repo root without installation.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.blocks import load_config  # noqa: E402
from src.model.train_time import (  # noqa: E402
    DEFAULT_MFU,
    DEFAULT_SCALING,
    default_registry,
    default_sizes,
    format_count,
    format_report,
    format_trainable_report,
    parse_count,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None,
                    help="estimate a single config YAML (exact param count) instead of the ladder")
    ap.add_argument("--config-dir", type=Path, default=Path("config"),
                    help="directory holding the family YAMLs (default: config/)")
    ap.add_argument("--params", type=str, default=None,
                    help="comma-separated param ladder with K/M/B suffixes, e.g. 270M,3B,7B")
    ap.add_argument("--tokens", type=str, default=None,
                    help="fixed token budget for every size (e.g. 3B); default is Chinchilla 20×params")
    ap.add_argument("--hours", type=float, default=None,
                    help="inverse mode: show TOKENS trainable in this many hours instead of time-to-train")
    ap.add_argument("--hardware", type=str, default=None,
                    help="comma-separated subset/order of hardware (default: m1pro,h100,8xh100)")
    ap.add_argument("--mfu", type=float, default=DEFAULT_MFU,
                    help=f"assumed H100 model-FLOPs utilization (default: {DEFAULT_MFU})")
    ap.add_argument("--scaling", type=float, default=DEFAULT_SCALING,
                    help=f"assumed 8×H100 scaling efficiency (default: {DEFAULT_SCALING})")
    args = ap.parse_args()

    # Which model sizes.
    if args.config is not None:
        cfg = load_config(args.config)
        sizes = [(args.config.stem, cfg.num_parameters())]
    elif args.params is not None:
        sizes = [(label.strip(), parse_count(label)) for label in args.params.split(",") if label.strip()]
        if not sizes:
            ap.error("--params resolved to no sizes (check for stray commas/whitespace)")
        # Re-label with the canonical compact form so 0.27B and 270M read the same.
        sizes = [(format_count(p), p) for _, p in sizes]
    else:
        sizes = default_sizes(args.config_dir)

    # Which hardware.
    registry = default_registry(mfu=args.mfu, scaling=args.scaling)
    if args.hardware is not None:
        names = [n.strip() for n in args.hardware.split(",") if n.strip()]
        if not names:
            ap.error("--hardware resolved to no entries (check for stray commas/whitespace)")
        unknown = [n for n in names if n not in registry]
        if unknown:
            ap.error(f"unknown hardware {unknown}; choose from {list(registry)}")
        hardware = [registry[n] for n in names]
    else:
        hardware = list(registry.values())

    if args.hours is not None:
        print(format_trainable_report(sizes, hardware, args.hours * 3600.0))
    else:
        fixed_tokens = parse_count(args.tokens) if args.tokens is not None else None
        print(format_report(sizes, hardware, fixed_tokens=fixed_tokens))


if __name__ == "__main__":
    main()
