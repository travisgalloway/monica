"""Print the Mamba config-family sizing table (params + memory per tier).

Portable — imports no backend. Loads the 100M -> 1B -> 2B -> 4B ladder from
`config/*.yaml` and prints the table that backs the #65 epic's GPU/RAM sizing.

  python scripts/model_size.py                 # the whole family (poc,1b,2b,4b)
  python scripts/model_size.py --config config/1b.yaml   # one config
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Allow `python scripts/model_size.py` from the repo root without installation.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.blocks import load_config  # noqa: E402
from src.model.sizing import format_family_table, load_family  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None,
                    help="size a single config YAML instead of the whole family")
    ap.add_argument("--config-dir", type=Path, default=Path("config"),
                    help="directory holding the family YAMLs (default: config/)")
    args = ap.parse_args()

    if args.config is not None:
        cfg = load_config(args.config)
        configs = [(args.config.stem, cfg)]
    else:
        configs = load_family(args.config_dir)

    print(format_family_table(configs))


if __name__ == "__main__":
    main()
