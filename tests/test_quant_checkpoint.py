"""Portable tests (#168): the `quant` sidecar block round-trips through
`save_weights`/`load_config_sidecar`/`load_quant_sidecar`, its absence leaves the
sidecar byte-identical to before #168, and `quant_targets`/`quantize_portable_state_dict`
pick the right tensors and produce a loadable format. No backend required."""

import json

import numpy as np
import pytest

from src.eval.quantize import (
    dequantize_portable_state_dict,
    mlx_affine_dequantize,
    pack_uint32_codes,
    quant_targets,
    quantize_portable_state_dict,
    unpack_uint32_codes,
)
from src.train.checkpoint import load_config_sidecar, load_quant_sidecar, save_weights


class _FakeConfig:
    """Minimal stand-in with the `to_dict()` shape `save_weights` expects, independent
    of the real `MambaConfig` so this stays a portable-only test."""

    def to_dict(self):
        return {"d_model": 64, "n_layers": 2, "vocab_size": 128}


# --------------------------------------------------------------------------- #
# quant_targets
# --------------------------------------------------------------------------- #
def _toy_moe_state_dict():
    rng = np.random.default_rng(0)
    return {
        "embedding.weight": rng.standard_normal((128, 64)).astype(np.float32),
        "layers.0.mamba.in_proj.weight": rng.standard_normal((256, 64)).astype(np.float32),
        "layers.0.mamba.dt_proj.weight": rng.standard_normal((8, 4)).astype(np.float32),
        "layers.0.mamba.conv1d.weight": rng.standard_normal((256, 4, 1)).astype(np.float32),  # 3-D
        "layers.0.mamba.norm.weight": np.ones((64,), np.float32),                              # 1-D
        "layers.0.mamba.ssm.A_log": np.zeros((8,), np.float32),                                # 1-D
        "layers.1.moe.router.weight": rng.standard_normal((4, 64)).astype(np.float32),
        "layers.1.moe.experts.0.up.weight": rng.standard_normal((128, 64)).astype(np.float32),
        "moe_route_bias.1": np.zeros((4,), np.float32),                                        # non-param
    }


def test_quant_targets_excludes_conv_1d_dt_proj_and_router():
    sd = _toy_moe_state_dict()
    targets = quant_targets(sd, group_size=64, bits=8)
    assert "embedding" in targets
    assert "layers.0.mamba.in_proj" in targets
    assert "layers.1.moe.experts.0.up" in targets
    assert "layers.0.mamba.dt_proj" not in targets       # excluded by name
    assert "layers.1.moe.router" not in targets           # excluded by name
    assert "layers.0.mamba.conv1d" not in targets          # 3-D
    assert "layers.0.mamba.norm" not in targets             # 1-D
    assert "layers.0.mamba.ssm" not in targets               # 1-D (A_log)
    assert all(b == 8 for b in targets.values())


def test_quant_targets_excludes_non_divisible_last_dim():
    sd = {"layers.0.x.weight": np.zeros((16, 100), np.float32)}  # 100 % 64 != 0
    assert quant_targets(sd, group_size=64, bits=8) == {}
    assert quant_targets(sd, group_size=50, bits=8) == {"layers.0.x": 8}


def test_quant_targets_head_bits_overrides_embedding_only():
    sd = _toy_moe_state_dict()
    targets = quant_targets(sd, group_size=64, bits=4, head_bits=8)
    assert targets["embedding"] == 8
    assert targets["layers.0.mamba.in_proj"] == 4


# --------------------------------------------------------------------------- #
# quantize_portable_state_dict / dequantize_portable_state_dict
# --------------------------------------------------------------------------- #
def test_quantized_state_dict_has_exactly_weight_scales_biases_per_target():
    sd = _toy_moe_state_dict()
    targets = quant_targets(sd, group_size=64, bits=8)
    qsd, quant_block = quantize_portable_state_dict(sd, targets, group_size=64)

    for path in targets:
        assert f"{path}.weight" in qsd
        assert f"{path}.scales" in qsd
        assert f"{path}.biases" in qsd
        assert qsd[f"{path}.weight"].dtype == np.uint32

    # untouched tensors pass through byte-identical
    for name in ("layers.0.mamba.dt_proj.weight", "layers.0.mamba.norm.weight",
                 "layers.0.mamba.ssm.A_log", "moe_route_bias.1"):
        assert np.array_equal(qsd[name], sd[name])

    assert quant_block == {"mode": "affine", "group_size": 64, "targets": targets}


def test_dequantize_portable_state_dict_reconstructs_original_up_to_quant_error():
    sd = _toy_moe_state_dict()
    targets = quant_targets(sd, group_size=64, bits=8)
    qsd, quant_block = quantize_portable_state_dict(sd, targets, group_size=64)
    deq = dequantize_portable_state_dict(qsd, quant_block)

    # Every original key is present, with original shapes, and no leftover
    # scales/biases keys leaked into the fake-quant reference dict.
    assert set(deq) == set(sd)
    for name in targets:
        key = f"{name}.weight"
        assert deq[key].shape == sd[key].shape
        # int8 group-wise affine on a random normal tensor should be a close, not exact,
        # reconstruction — this also implicitly checks we didn't accidentally return the
        # packed uint32 codes unconverted.
        assert deq[key].dtype == np.float32
        err = np.abs(deq[key] - sd[key]).max() / (np.abs(sd[key]).std() + 1e-8)
        assert err < 1.0
    for name in ("layers.0.mamba.dt_proj.weight", "moe_route_bias.1"):
        assert np.array_equal(deq[name], sd[name])


def test_pack_unpack_roundtrip_is_exact():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 16, size=(5, 64)).astype(np.uint32)
    packed = pack_uint32_codes(codes, bits=4)
    recovered = unpack_uint32_codes(packed, bits=4, cols=64)
    assert np.array_equal(codes, recovered)


# --------------------------------------------------------------------------- #
# sidecar round-trip
# --------------------------------------------------------------------------- #
def test_quant_sidecar_absent_leaves_config_sidecar_unaffected(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_weights({"a": np.zeros((2, 2), np.float32)}, str(path), config=_FakeConfig())
    sidecar = json.loads((tmp_path / "weights.safetensors.config.json").read_text())
    assert "quant" not in sidecar
    assert load_quant_sidecar(str(path)) is None


def test_quant_sidecar_round_trips(tmp_path, capsys):
    path = tmp_path / "weights.safetensors"
    quant_block = {"mode": "affine", "group_size": 64, "targets": {"embedding": 8}}
    save_weights({"a": np.zeros((2, 2), np.float32)}, str(path), config=_FakeConfig(),
                 quant=quant_block)

    loaded_quant = load_quant_sidecar(str(path))
    assert loaded_quant == quant_block

    # load_config_sidecar must silently drop "quant" — no "dropping fields unknown"
    # note printed for the expected #168 key.
    cfg = load_config_sidecar(str(path))
    assert cfg is not None
    captured = capsys.readouterr()
    assert "dropping fields unknown" not in captured.out
