"""MLX safety net for the closed-form param count (skipped where mlx is absent).

`MambaConfig.num_parameters()` is a hand-derived formula; this asserts it equals
the EXACT number of elements in the built model's portable state dict, so the two
can never silently drift. Covers toy (cheap) and poc (the real ~127M anchor).
"""

from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.mark.parametrize("name", ["toy", "poc", "toy-hybrid", "toy-moe"])
def test_num_parameters_matches_built_model(name):
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    model = MLXMambaModel(cfg)
    actual = sum(int(v.size) for v in model._portable_state_dict().values())
    assert cfg.num_parameters() == actual, (
        f"{name}: formula {cfg.num_parameters()} != built model {actual}"
    )
