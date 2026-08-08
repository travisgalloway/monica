"""Tests for sparse upcycle (#214): `src/train/upcycle.py` (portable) plus the
`check_weight_keys` / `load_config_sidecar` checkpoint helpers it depends on, plus a
focused test of `scripts/train.py`'s `--resume` > `--init` precedence.

Portable tests need no backend. The step-0 exactness tests (the #214 acceptance
criterion — "upcycled init matches the dense forward at step 0") are backend-guarded:
one on MLX, one on CUDA/torch (CPU-only torch is fine; no GPU needed).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.pack import pack_ids
from src.model.blocks import MambaConfig, load_config
from src.train.checkpoint import (
    check_weight_keys, load_config_sidecar, load_weights_dict, save_weights,
)
from src.train.upcycle import (
    UpcycleError, _expected_keys, check_upcycle_compatible, upcycle_dense_to_moe,
    upcycle_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _dense_cfg(**over):
    base = dict(d_model=32, n_layers=2, head_dim=8, d_state=8, vocab_size=64, seq_len=16,
                precision="fp32", moe_every=2, n_experts=1, top_k=1, moe_d_ff=8)
    base.update(over)
    return MambaConfig(**base)


def _fine_cfg(**over):
    base = dict(d_model=32, n_layers=2, head_dim=8, d_state=8, vocab_size=64, seq_len=16,
                precision="fp32", moe_every=2, n_experts=6, top_k=2, moe_d_ff=8)
    base.update(over)
    return MambaConfig(**base)


def _fake_weights(cfg, seed=0):
    """A synthetic-but-shape-correct portable weight dict for `cfg`, without needing a
    backend — built straight off `_expected_keys`."""
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal(shp).astype(np.float32)
            for k, shp in _expected_keys(cfg).items()}


try:
    import mlx.core as mx
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False
requires_mlx = pytest.mark.skipif(not HAVE_MLX, reason="requires mlx (Apple Silicon)")


# --------------------------------------------------------------------------- #
# check_upcycle_compatible
# --------------------------------------------------------------------------- #
def test_check_upcycle_compatible_accepts_dense_fine_pair():
    check_upcycle_compatible(_dense_cfg(), _fine_cfg())     # no raise


def test_check_upcycle_compatible_accepts_the_real_toy_fixtures():
    src_cfg = load_config(str(REPO_ROOT / "config/toy-moe-dense.yaml"))
    dst_cfg = load_config(str(REPO_ROOT / "config/toy-moe-fine.yaml"))
    check_upcycle_compatible(src_cfg, dst_cfg)              # no raise


def test_check_upcycle_compatible_rejects_already_moe_source():
    with pytest.raises(UpcycleError, match="degenerate n_experts=1"):
        check_upcycle_compatible(_fine_cfg(), _fine_cfg())  # src has n_experts=6


def test_check_upcycle_compatible_rejects_d_model_mismatch_names_223():
    dst = _fine_cfg(d_model=64)
    with pytest.raises(UpcycleError, match="#223") as exc:
        check_upcycle_compatible(_dense_cfg(), dst)
    assert "d_model" in str(exc.value)
    assert "widen" in str(exc.value) or "Net2Net" in str(exc.value)


def test_check_upcycle_compatible_reports_multiple_mismatches_together():
    dst = _fine_cfg(d_state=4, vocab_size=128)
    with pytest.raises(UpcycleError) as exc:
        check_upcycle_compatible(_dense_cfg(), dst)
    msg = str(exc.value)
    # BOTH mismatches must be visible in ONE raise, not the first one only.
    assert "d_state" in msg
    assert "vocab_size" in msg


def test_check_upcycle_compatible_rejects_moe_layer_index_mismatch():
    dst = _fine_cfg(moe_every=1)                            # different interleave
    with pytest.raises(UpcycleError, match="moe_every|moe_layer_indices"):
        check_upcycle_compatible(_dense_cfg(), dst)


# --------------------------------------------------------------------------- #
# upcycle_dense_to_moe: the transform itself (portable, synthetic weights)
# --------------------------------------------------------------------------- #
def test_upcycle_output_key_set_and_shapes_match_dst():
    src_cfg, dst_cfg = _dense_cfg(), _fine_cfg()
    weights = _fake_weights(src_cfg, seed=1)
    out = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=2)
    expected = _expected_keys(dst_cfg)
    assert set(out) == set(expected)
    for k, shp in expected.items():
        assert tuple(out[k].shape) == shp, k


def test_upcycle_experts_are_independent_copies_of_source():
    src_cfg, dst_cfg = _dense_cfg(), _fine_cfg()
    weights = _fake_weights(src_cfg, seed=1)
    out = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=2)
    src_gate = weights["layers.1.experts.0.gate.weight"]
    for j in range(dst_cfg.n_experts):
        assert np.array_equal(out[f"layers.1.experts.{j}.gate.weight"], src_gate)
    # Mutating one expert's array must not affect any other (independent .copy()s, not
    # the same array aliased n_experts times).
    out["layers.1.experts.0.gate.weight"][0, 0] += 1000.0
    assert not np.array_equal(out["layers.1.experts.0.gate.weight"],
                              out["layers.1.experts.1.gate.weight"])
    for j in range(2, dst_cfg.n_experts):
        assert np.array_equal(out[f"layers.1.experts.{j}.gate.weight"], src_gate)


def test_upcycle_router_shape_and_seed_reproducible():
    src_cfg, dst_cfg = _dense_cfg(), _fine_cfg()
    weights = _fake_weights(src_cfg, seed=1)
    router_key = "layers.1.router.weight"

    out1 = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=42)
    assert out1[router_key].shape == (dst_cfg.n_experts, dst_cfg.d_model)

    out2 = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=42)
    assert np.array_equal(out1[router_key], out2[router_key])       # same seed -> identical

    out3 = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=43)
    assert not np.array_equal(out1[router_key], out3[router_key])   # different seed -> different


def test_upcycle_drops_moe_route_bias_keys():
    src_cfg, dst_cfg = _dense_cfg(), _fine_cfg()
    weights = _fake_weights(src_cfg, seed=1)
    weights["moe_route_bias.1"] = np.zeros((1,), dtype=np.float32)   # E=1 -> length-1 bias
    out = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=2)
    assert not any(k.startswith("moe_route_bias.") for k in out)


def test_upcycle_total_size_matches_dst_num_parameters():
    src_cfg, dst_cfg = _dense_cfg(), _fine_cfg()
    weights = _fake_weights(src_cfg, seed=1)
    out = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=2)
    assert sum(v.size for v in out.values()) == dst_cfg.num_parameters()


def test_upcycle_shared_expert_zero_down_default():
    src_cfg = _dense_cfg()
    dst_cfg = _fine_cfg(n_shared_experts=1)
    weights = _fake_weights(src_cfg, seed=1)
    out = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=2)
    down = out["layers.1.shared_experts.0.down.weight"]
    assert np.array_equal(down, np.zeros_like(down))
    gate = out["layers.1.shared_experts.0.gate.weight"]
    assert not np.all(gate == 0.0)                        # fresh random, not degenerate


def test_upcycle_shared_expert_forbid_raises_on_new_expert():
    src_cfg = _dense_cfg()
    dst_cfg = _fine_cfg(n_shared_experts=1)
    weights = _fake_weights(src_cfg, seed=1)
    with pytest.raises(UpcycleError, match="forbid"):
        upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=2, shared_expert_init="forbid")


def test_upcycle_rejects_unknown_shared_expert_init():
    src_cfg, dst_cfg = _dense_cfg(), _fine_cfg()
    weights = _fake_weights(src_cfg, seed=1)
    with pytest.raises(UpcycleError, match="shared_expert_init"):
        upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=2, shared_expert_init="bogus")


def test_upcycle_raises_on_missing_source_expert_key():
    src_cfg, dst_cfg = _dense_cfg(), _fine_cfg()
    weights = _fake_weights(src_cfg, seed=1)
    del weights["layers.1.experts.0.gate.weight"]
    with pytest.raises(UpcycleError, match="experts.0.gate.weight"):
        upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=2)


def test_upcycle_manifest_contents():
    src_cfg, dst_cfg = _dense_cfg(), _fine_cfg()
    m = upcycle_manifest(src="a.safetensors", src_sha256="deadbeef", seed=7,
                         router_init_scale=1.0, shared_expert_init="zero_down",
                         src_cfg=src_cfg, dst_cfg=dst_cfg)
    assert m["src"] == "a.safetensors"
    assert m["src_sha256"] == "deadbeef"
    assert m["seed"] == 7
    assert m["n_experts_src"] == 1
    assert m["n_experts_dst"] == 6
    assert m["moe_layers"] == [1]


# --------------------------------------------------------------------------- #
# _expected_keys cross-checked against a real backend model (portable mirror must not
# silently drift from the real thing)
# --------------------------------------------------------------------------- #
@requires_mlx
def test_expected_keys_matches_real_mlx_model():
    from src.model.mlx_backend import MLXMambaModel
    cfg = _fine_cfg(n_shared_experts=1, attn_every=None)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    real = {k: tuple(v.shape) for k, v in model._portable_state_dict().items()}
    assert real == _expected_keys(cfg)


# --------------------------------------------------------------------------- #
# check_weight_keys
# --------------------------------------------------------------------------- #
def test_check_weight_keys_passes_on_exact_match():
    expected = {"a": (2, 3), "b": (4,)}
    weights = {"a": np.zeros((2, 3)), "b": np.zeros((4,))}
    check_weight_keys(weights, expected, where="unit-test")   # no raise


def test_check_weight_keys_reports_missing_unexpected_mismatched_together():
    expected = {"a": (2, 3), "b": (4,), "c": (1,)}
    weights = {"a": np.zeros((2, 3)), "b": np.zeros((5,)), "d": np.zeros((1,))}
    with pytest.raises(ValueError) as exc:
        check_weight_keys(weights, expected, where="unit-test")
    msg = str(exc.value)
    assert "missing" in msg and "c" in msg
    assert "unexpected" in msg and "d" in msg
    assert "mis-shaped" in msg and "b" in msg


def test_check_weight_keys_ignores_moe_route_bias():
    expected = {"a": (2,)}
    weights = {"a": np.zeros((2,)), "moe_route_bias.0": np.zeros((1,))}
    check_weight_keys(weights, expected, where="unit-test")   # no raise: bias not checked


# --------------------------------------------------------------------------- #
# load_config_sidecar
# --------------------------------------------------------------------------- #
def test_load_config_sidecar_round_trip(tmp_path):
    cfg = _dense_cfg()
    path = tmp_path / "weights.safetensors"
    save_weights({"a": np.zeros((2,), dtype=np.float32)}, str(path), config=cfg)
    loaded = load_config_sidecar(str(path))
    assert loaded is not None
    assert loaded.d_model == cfg.d_model
    assert loaded.n_experts == cfg.n_experts
    assert loaded.moe_d_ff == cfg.moe_d_ff


def test_load_config_sidecar_returns_none_when_absent(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_weights({"a": np.zeros((2,), dtype=np.float32)}, str(path))   # no config=
    assert load_config_sidecar(str(path)) is None


def test_load_config_sidecar_drops_unknown_fields(tmp_path, capsys):
    cfg = _dense_cfg()
    d = cfg.to_dict()
    d["totally_unknown_field_from_the_future"] = 123
    (tmp_path / "weights.safetensors.config.json").write_text(json.dumps(d))
    loaded = load_config_sidecar(str(tmp_path / "weights.safetensors"))
    assert loaded is not None
    assert not hasattr(loaded, "totally_unknown_field_from_the_future")
    assert "dropping fields unknown" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Step-0 exactness (#214's stated acceptance criterion) — real backend models, the
# checked-in toy-moe-dense/fine.yaml fixture pair.
# --------------------------------------------------------------------------- #
@requires_mlx
def test_mlx_step0_exactness_and_zero_router_grad():
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from src.model.mlx_backend import MLXMambaModel

    src_cfg = load_config(str(REPO_ROOT / "config/toy-moe-dense.yaml"))
    dst_cfg = load_config(str(REPO_ROOT / "config/toy-moe-fine.yaml"))

    mx.random.seed(0)
    dense_model = MLXMambaModel(src_cfg)
    tokens = np.arange(2 * src_cfg.seq_len).reshape(2, src_cfg.seq_len) % src_cfg.vocab_size
    dense_logits = np.array(dense_model.forward(mx.array(tokens)))

    with tempfile.TemporaryDirectory() as tmp:
        wpath = Path(tmp) / "dense.safetensors"
        dense_model.save(str(wpath))
        weights = load_weights_dict(str(wpath))
        upcycled = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=214)

    mx.random.seed(0)
    fine_model = MLXMambaModel(dst_cfg)
    fine_model._load_portable(upcycled)
    fine_logits = np.array(fine_model.forward(mx.array(tokens)))

    max_diff = float(np.abs(dense_logits - fine_logits).max())
    print(f"[test_mlx_step0_exactness] max|logits_dense - logits_upcycled| = {max_diff:.3e}")
    assert max_diff < 1e-4

    # router.weight.grad == 0 at step 0: gate weights always sum to 1 no matter what the
    # router says, and every expert computes the IDENTICAL function (copied from the same
    # source expert) -- so the combined output (and hence the loss) does not depend on the
    # router's value at all. Expected, documented behavior, not a bug.
    def loss_fn(model, x):
        return model.forward(x).astype(mx.float32).sum()

    value_and_grad = nn.value_and_grad(fine_model, loss_fn)
    _, grads = value_and_grad(fine_model, mx.array(tokens))
    flat = dict(tree_flatten(grads))
    router_grad_keys = [k for k in flat if k.endswith("router.weight")]
    assert router_grad_keys
    for k in router_grad_keys:
        g = np.array(flat[k])
        assert np.abs(g).max() < 1e-5, f"{k}: max|grad|={np.abs(g).max():.3e}"


@requires_mlx
def test_mlx_step0_exactness_with_shared_expert():
    from src.model.mlx_backend import MLXMambaModel

    src_cfg = load_config(str(REPO_ROOT / "config/toy-moe-dense.yaml"))
    dst_cfg = MambaConfig(**{**load_config(str(REPO_ROOT / "config/toy-moe-fine.yaml")).to_dict(),
                             "n_shared_experts": 1})
    dst_cfg.validate()

    mx.random.seed(0)
    dense_model = MLXMambaModel(src_cfg)
    tokens = np.arange(2 * src_cfg.seq_len).reshape(2, src_cfg.seq_len) % src_cfg.vocab_size
    dense_logits = np.array(dense_model.forward(mx.array(tokens)))

    with tempfile.TemporaryDirectory() as tmp:
        wpath = Path(tmp) / "dense.safetensors"
        dense_model.save(str(wpath))
        weights = load_weights_dict(str(wpath))
        upcycled = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=214)

    mx.random.seed(0)
    fine_model = MLXMambaModel(dst_cfg)
    fine_model._load_portable(upcycled)
    fine_logits = np.array(fine_model.forward(mx.array(tokens)))

    max_diff = float(np.abs(dense_logits - fine_logits).max())
    print(f"[test_mlx_step0_exactness_with_shared_expert] max|diff| = {max_diff:.3e}")
    assert max_diff < 1e-4


def test_cuda_step0_exactness_and_zero_router_grad():
    torch = pytest.importorskip("torch")
    from src.model.cuda_backend import CUDAMambaModel

    src_cfg = load_config(str(REPO_ROOT / "config/toy-moe-dense.yaml"))
    dst_cfg = load_config(str(REPO_ROOT / "config/toy-moe-fine.yaml"))

    torch.manual_seed(0)
    dense_model = CUDAMambaModel(src_cfg)
    tokens = np.arange(2 * src_cfg.seq_len).reshape(2, src_cfg.seq_len) % src_cfg.vocab_size
    with torch.no_grad():
        dense_logits = dense_model.forward(tokens).detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as tmp:
        wpath = Path(tmp) / "dense.safetensors"
        dense_model.save(str(wpath))
        weights = load_weights_dict(str(wpath))
        upcycled = upcycle_dense_to_moe(weights, src_cfg, dst_cfg, seed=214)

    torch.manual_seed(0)
    fine_model = CUDAMambaModel(dst_cfg)
    fine_model._load_portable(upcycled)
    with torch.no_grad():
        fine_logits = fine_model.forward(tokens).detach().cpu().numpy()

    max_diff = float(np.abs(dense_logits - fine_logits).max())
    print(f"[test_cuda_step0_exactness] max|logits_dense - logits_upcycled| = {max_diff:.3e}")
    assert max_diff < 1e-4

    from src.model.cuda_backend import MoEBlock
    block = next(l for l in fine_model.layers if isinstance(l, MoEBlock))
    loss = fine_model.forward(tokens).float().sum()
    loss.backward()
    g = block.router.weight.grad
    assert g is not None
    assert float(g.abs().max()) < 1e-5


# --------------------------------------------------------------------------- #
# --resume beats --init on scripts/train.py (focused test of that branch)
# --------------------------------------------------------------------------- #
def _load_train_module():
    spec = importlib.util.spec_from_file_location("_scripts_train_for_upcycle_test",
                                                   REPO_ROOT / "scripts/train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


@requires_mlx
def test_resume_beats_init(tmp_path, monkeypatch, capsys):
    tiny = dict(d_model=16, n_layers=1, head_dim=4, d_state=4, expand=2, d_conv=2,
               dt_rank="auto", vocab_size=32, seq_len=8, tie_embeddings=True,
               precision="fp32", chunk_size=None, grad_checkpoint=False,
               dt_min=0.001, dt_max=0.1, dt_init_floor=0.0001)
    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(tiny))

    data_dir = tmp_path / "data"
    pack_ids(np.arange(300, dtype=np.uint16) % 32, data_dir / "train.bin", dtype=np.uint16)
    pack_ids(np.arange(100, dtype=np.uint16) % 32, data_dir / "val.bin", dtype=np.uint16)
    out_dir = tmp_path / "out"

    mod = _load_train_module()
    common = ["--config", str(cfg_path), "--data", str(data_dir), "--out", str(out_dir),
              "--batch-size", "2", "--grad-accum", "1", "--ckpt-every", "1",
              "--eval-every", "1000000", "--log-every", "1000000", "--backend", "mlx"]

    monkeypatch.setattr(sys, "argv", ["train.py", *common, "--total-steps", "1"])
    mod.main()
    capsys.readouterr()   # discard first run's output

    bogus_init = tmp_path / "does-not-exist.safetensors"
    monkeypatch.setattr(sys, "argv",
                        ["train.py", *common, "--total-steps", "2",
                         "--init", str(bogus_init)])
    mod.main()            # must NOT raise: a real --resume checkpoint exists under out_dir,
                           # so --resume wins and --init (pointing at a NONEXISTENT file) is
                           # never read.
    out = capsys.readouterr().out
    assert "[init] IGNORED" in out
    assert "[resume] from step 1" in out
