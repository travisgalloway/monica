"""Unit tests for portable weight save/load and the within-backend resume bundle.

Backend-free: uses numpy weight dicts and dummy optimizer (de)serializers, so it
runs anywhere. Guards the resume bundle against a realistic NumPy RNG state
(nested dict + np.ndarray + np scalars), which a shallow JSON conversion breaks.
"""

import json

import numpy as np

from src.train.checkpoint import (
    save_weights, load_weights_dict, save_resume, load_resume,
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


def test_resume_bundle_with_realistic_rng_state(tmp_path):
    bundle = tmp_path / "resume"
    rng_state = np.random.default_rng(0).bit_generator.state  # nested dict + np types

    saved = {}

    def opt_serializer(p):
        saved["path"] = p
        # stand in for a backend optimizer dump
        np.save(p + ".npy", np.arange(3))

    def opt_deserializer(p):
        return np.load(p + ".npy")

    # Must not raise on nested RNG state.
    save_resume(str(bundle), step=42, rng_state=rng_state,
                optimizer_serializer=opt_serializer)

    meta = load_resume(str(bundle), optimizer_deserializer=opt_deserializer)
    assert meta["step"] == 42
    assert np.array_equal(meta["optimizer"], np.arange(3))
    # rng_state survived JSON serialization as a plain structure.
    assert json.dumps(meta["rng_state"])  # serializable
