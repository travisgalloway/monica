"""The checked-in `monica-bench --config` sidecars are not stale (#170).

`swift/engine/Benchmarks/configs/*.json` are `MambaConfig.to_dict()` exports
(`scripts/export_bench_config.py`) that let `monica-bench` run poc-scale, random-init
timing/memory sweeps without checking in a ~500 MB safetensors artifact. Like the Swift
parity fixtures (`tests/test_parity_fixture_export.py`), that makes them a artifact that
can silently drift from `config/*.yaml` — so re-derive each one into a tmpdir and assert
it matches the checked-in file exactly.

Portable: `export_bench_config` only touches `MambaConfig`/`load_config` (no MLX/torch),
so this runs on every CI job, not just the mlx-gated ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.export_bench_config import export_bench_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "swift" / "engine" / "Benchmarks" / "configs"

# (config yaml, checked-in sidecar) pairs — every file under CONFIGS_DIR must appear
# here, enforced by test_no_untracked_sidecars_in_configs_dir below.
PAIRS = [
    ("config/poc.yaml", CONFIGS_DIR / "poc.config.json"),
    ("config/poc-small.yaml", CONFIGS_DIR / "poc-small.config.json"),
]


def test_checked_in_sidecars_match_todays_yaml(tmp_path):
    for config_path, checked_in in PAIRS:
        assert checked_in.is_file(), f"missing checked-in sidecar {checked_in}"
        out = tmp_path / checked_in.name
        export_bench_config(config_path, str(out))
        fresh = json.loads(out.read_text())
        expected = json.loads(checked_in.read_text())
        assert fresh == expected, (
            f"{checked_in} drifted from {config_path} — re-run "
            f"`.venv/bin/python scripts/export_bench_config.py --config {config_path} "
            f"--out {checked_in}` and commit the result."
        )


def test_no_untracked_sidecars_in_configs_dir():
    """Every *.config.json under CONFIGS_DIR must be listed in PAIRS above, so a new
    config added to the directory without a test pairing cannot go unverified."""
    tracked = {p.name for _, p in PAIRS}
    on_disk = {p.name for p in CONFIGS_DIR.glob("*.config.json")}
    assert on_disk == tracked, (
        f"swift/engine/Benchmarks/configs/ has files not covered by PAIRS: "
        f"{on_disk - tracked}, or PAIRS lists files that no longer exist: "
        f"{tracked - on_disk}"
    )
