"""Print a manifest-driven student sweep table (#98).

Portable — imports no backend. Loads a set of sibling student manifests
(`config/manifests/*.yaml`), verifies they share one frozen teacher signal, and prints the
per-trial param/memory + layout table (attention fraction, layer placement, state size — the
three swept architecture variables). See `docs/design/10-distillation.md`.

  python scripts/sweep.py                                   # all of config/manifests/
  python scripts/sweep.py config/manifests/student-1b-attn-lo.yaml ...   # explicit set
  python scripts/sweep.py --manifest-dir config/manifests   # an explicit directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Allow `python scripts/sweep.py` from the repo root without installation.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.sweep import format_sweep_table, load_sweep, load_sweep_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="*", type=Path,
                    help="manifest YAMLs to sweep (default: all of --manifest-dir)")
    ap.add_argument("--manifest-dir", type=Path, default=Path("config/manifests"),
                    help="directory of sibling manifests (default: config/manifests)")
    args = ap.parse_args()

    sweep = load_sweep(args.manifests) if args.manifests else load_sweep_dir(args.manifest_dir)
    print(format_sweep_table(sweep))


if __name__ == "__main__":
    main()
