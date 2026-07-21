"""Portable taxonomy test for `is_muon_param` (#237) — no backend import required.

Pins the exact routing table from the plan/issue: 2D hidden weight matrices (Mamba
in/out/x_proj, attention qkv/o_proj, MoE expert gate/up/down) go to Muon; everything else
(embedding, LM head, router, dt_proj, and all 1D/3D params) stays on AdamW.

Belt-and-suspenders: when mlx is importable, also build the real toy-hybrid + toy-moe MLX
models and assert their actual flattened param names classify identically, guarding against
name drift between this hardcoded table and the real modules.
"""

import pytest

from src.model.blocks import is_muon_param

MUON_TRUE = [
    ("layers.0.in_proj.weight", 2),
    ("layers.0.out_proj.weight", 2),
    ("layers.0.x_proj.weight", 2),
    ("layers.1.qkv_proj.weight", 2),
    ("layers.1.o_proj.weight", 2),
    ("layers.1.experts.0.gate.weight", 2),
    ("layers.1.experts.0.up.weight", 2),
    ("layers.1.experts.0.down.weight", 2),
]

MUON_FALSE = [
    ("embedding.weight", 2),
    ("lm_head.weight", 2),
    ("layers.1.router.weight", 2),
    ("layers.0.dt_proj.weight", 2),
    ("layers.0.dt_proj.bias", 1),
    ("layers.0.A_log", 1),
    ("layers.0.D", 1),
    ("layers.0.norm.weight", 1),
    ("norm_f.weight", 1),
    ("layers.0.conv.weight", 3),
    ("layers.0.conv.bias", 1),
    ("layers.0.in_proj.bias", 1),
    ("layers.1.qkv_proj.bias", 1),
]


@pytest.mark.parametrize("name,ndim", MUON_TRUE)
def test_muon_routes_hidden_matrices(name, ndim):
    assert is_muon_param(name, ndim) is True


@pytest.mark.parametrize("name,ndim", MUON_FALSE)
def test_adamw_routes_everything_else(name, ndim):
    assert is_muon_param(name, ndim) is False


def test_ndim_gate_alone_excludes_non_2d():
    # Even a name that would otherwise pass (no exact/suffix exclusion) must be ndim==2.
    assert is_muon_param("layers.0.in_proj.weight", 1) is False
    assert is_muon_param("layers.0.in_proj.weight", 3) is False


def test_real_mlx_models_classify_identically():
    """Guard against name drift: build the real toy-hybrid + toy-moe MLX models and check
    every flattened param name against the hardcoded table's exclusion rules directly
    (rather than re-deriving the table), i.e. re-run `is_muon_param` and sanity-check the
    counts are non-trivial on both sides of the partition."""
    pytest.importorskip("mlx")
    from mlx.utils import tree_flatten
    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel

    for cfg_path in ("config/toy-hybrid.yaml", "config/toy-moe.yaml"):
        cfg = load_config(cfg_path)
        model = MLXMambaModel(cfg)
        leaves = tree_flatten(model.parameters())
        assert leaves, f"{cfg_path} produced no parameters"

        muon_names = [name for name, v in leaves if is_muon_param(name, v.ndim)]
        adam_names = [name for name, v in leaves if not is_muon_param(name, v.ndim)]

        assert muon_names, f"{cfg_path}: expected at least one Muon-routed param"
        assert adam_names, f"{cfg_path}: expected at least one AdamW-routed param"
        # Excluded-by-name params must never appear in the Muon set.
        for name in muon_names:
            assert not name.endswith((".router.weight", ".dt_proj.weight"))
            assert name not in ("embedding.weight", "lm_head.weight")
        # Every param actually excluded by ndim must be 1D or 3D+, never 2D.
        for name, v in leaves:
            if name in adam_names and (
                name.endswith((".router.weight", ".dt_proj.weight"))
                or name in ("embedding.weight", "lm_head.weight")
            ):
                continue
            if name in adam_names:
                assert v.ndim != 2, f"{cfg_path}: {name} is 2D but routed to AdamW unexpectedly"
