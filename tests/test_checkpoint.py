"""Unit tests for portable weight save/load and the within-backend resume bundle.

Backend-free: uses numpy weight dicts and dummy optimizer (de)serializers, so it
runs anywhere. Guards the resume bundle against a realistic NumPy RNG state
(nested dict + np.ndarray + np scalars), which a shallow JSON conversion breaks.
"""

import json

import numpy as np
import pytest

from src.train.checkpoint import (
    save_weights, load_weights_dict, save_resume, load_resume, _jsonable,
)


def test_weights_roundtrip(tmp_path):
    path = tmp_path / "w.safetensors"
    state = {
        "embedding.weight": np.random.randn(8, 4).astype(np.float32),
        "layers.0.norm.weight": np.ones(4, dtype=np.float32),
    }
    save_weights(state, str(path))
    loaded = load_weights_dict(str(path))
    assert set(loaded) == set(state)
    for k in state:
        assert np.array_equal(loaded[k], state[k])


def test_save_weights_writes_config_sidecar(tmp_path):
    path = tmp_path / "w.safetensors"

    class Cfg:
        def to_dict(self):
            return {"d_model": 64, "n_layers": 2}

    save_weights({"x": np.zeros(2, dtype=np.float32)}, str(path), config=Cfg())
    sidecar = json.loads((tmp_path / "w.safetensors.config.json").read_text())
    assert sidecar == {"d_model": 64, "n_layers": 2}


def _dummy_opt_io():
    def serializer(p):
        np.save(p + ".npy", np.arange(3))

    def deserializer(p):
        return np.load(p + ".npy")

    return serializer, deserializer


def test_resume_bundle_roundtrips_nested_state(tmp_path):
    """A nested NumPy structure (the worst case _jsonable handles) survives exactly."""
    bundle = tmp_path / "resume"
    nested = np.random.default_rng(0).bit_generator.state  # nested dict + np ndarray/scalars
    ser, deser = _dummy_opt_io()

    save_resume(str(bundle), step=42, loss_scale_state=nested,
                optimizer_serializer=ser)
    meta = load_resume(str(bundle), optimizer_deserializer=deser)

    assert meta["step"] == 42
    assert np.array_equal(meta["optimizer"], np.arange(3))
    # The restored structure equals the jsonable projection of the original — a real
    # content check, not just "is serializable".
    assert meta["loss_scale_state"] == _jsonable(nested)


def test_resume_bundle_roundtrips_loss_scale_state(tmp_path):
    """The actual payload: a DynamicLossScaler.state_dict() restores its values."""
    bundle = tmp_path / "resume"
    state = {"scale": 4096.0, "good_steps": 137}
    ser, deser = _dummy_opt_io()

    save_resume(str(bundle), step=7, loss_scale_state=state, optimizer_serializer=ser)
    meta = load_resume(str(bundle), optimizer_deserializer=deser)

    assert meta["loss_scale_state"] == state


def test_load_resume_refuses_bundle_without_commit_marker(tmp_path):
    """An interrupted checkpoint (no COMMIT) is rejected, not silently resumed."""
    bundle = tmp_path / "resume"
    ser, deser = _dummy_opt_io()
    save_resume(str(bundle), step=5, loss_scale_state=None, optimizer_serializer=ser)

    (bundle / "COMMIT").unlink()  # simulate a mid-write interruption

    with pytest.raises(RuntimeError, match="COMMIT"):
        load_resume(str(bundle), optimizer_deserializer=deser)


def test_save_resume_leaves_no_temp_files(tmp_path):
    """Atomic writes clean up after themselves — no stray .tmp/.swap on success."""
    bundle = tmp_path / "resume"
    ser, _ = _dummy_opt_io()
    save_resume(str(bundle), step=1, loss_scale_state=None, optimizer_serializer=ser)

    leftovers = [p.name for p in bundle.iterdir()
                 if p.name.endswith((".tmp", ".swap")) or p.name.startswith(".tmp-")]
    assert leftovers == [], f"unexpected temp files left behind: {leftovers}"
