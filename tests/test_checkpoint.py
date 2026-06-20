"""Unit tests for portable weight save/load and the double-buffered CheckpointStore.

Backend-free: uses numpy weight dicts and dummy weight/optimizer (de)serializers, so it
runs anywhere. The resume metadata persists `step` + the fp16 `loss_scale_state` (NOT an
RNG state — the data order on resume is reconstructed deterministically); these tests
guard `_jsonable` against a realistic nested NumPy structure (nested dict + np.ndarray +
np scalars — the worst case it handles), which a shallow JSON conversion breaks, plus the
crash-safety contract (the previous checkpoint survives a write interrupted mid-flight).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.train.checkpoint import (
    save_weights, load_weights_dict, CheckpointStore, _jsonable,
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


def _dummy_io(weights_tag="w0", opt_value=3):
    """Dummy weight + optimizer (de)serializers writing recognizable content. `seen`
    captures what `load` deserialized so tests can assert an exact round-trip."""
    seen = {}

    def w_ser(p):
        Path(p).write_text(weights_tag)

    def o_ser(p):
        np.save(p + ".npy", np.arange(opt_value))

    def w_deser(p):
        seen["weights"] = Path(p).read_text()

    def o_deser(p):
        return np.load(p + ".npy")

    return seen, w_ser, o_ser, w_deser, o_deser


def test_store_roundtrips_step_weights_and_nested_state(tmp_path):
    """A full checkpoint (weights + optimizer + nested loss-scale state) round-trips."""
    store = CheckpointStore(str(tmp_path / "resume"))
    nested = np.random.default_rng(0).bit_generator.state  # nested dict + np ndarray/scalars
    seen, w_ser, o_ser, w_deser, o_deser = _dummy_io(weights_tag="hello")

    slot = store.save(step=42, loss_scale_state=nested,
                      weights_serializer=w_ser, optimizer_serializer=o_ser)
    assert slot == "slot-a"
    meta = store.load(weights_deserializer=w_deser, optimizer_deserializer=o_deser)

    assert meta["step"] == 42 and meta["slot"] == "slot-a"
    assert seen["weights"] == "hello"
    assert np.array_equal(meta["optimizer"], np.arange(3))
    # Restored structure equals the jsonable projection — a real content check.
    assert meta["loss_scale_state"] == _jsonable(nested)


def test_store_roundtrips_loss_scale_state(tmp_path):
    store = CheckpointStore(str(tmp_path / "resume"))
    state = {"scale": 4096.0, "good_steps": 137}
    _, w_ser, o_ser, w_deser, o_deser = _dummy_io()
    store.save(step=7, loss_scale_state=state, weights_serializer=w_ser,
               optimizer_serializer=o_ser)
    meta = store.load(weights_deserializer=w_deser, optimizer_deserializer=o_deser)
    assert meta["loss_scale_state"] == state


def test_store_alternates_slots(tmp_path):
    """Successive checkpoints ping-pong between the two slots."""
    store = CheckpointStore(str(tmp_path / "resume"))
    _, w_ser, o_ser, _, _ = _dummy_io()
    slots = [store.save(step=i, loss_scale_state=None, weights_serializer=w_ser,
                        optimizer_serializer=o_ser) for i in range(3)]
    assert slots == ["slot-a", "slot-b", "slot-a"]


def test_store_load_without_checkpoint_raises(tmp_path):
    store = CheckpointStore(str(tmp_path / "resume"))
    assert not store.has_checkpoint()
    _, _, _, w_deser, o_deser = _dummy_io()
    with pytest.raises(RuntimeError, match="no committed checkpoint"):
        store.load(weights_deserializer=w_deser, optimizer_deserializer=o_deser)


def test_store_crash_mid_write_preserves_previous_checkpoint(tmp_path):
    """The core double-buffering guarantee: a checkpoint interrupted mid-write leaves
    the PREVIOUS committed checkpoint fully intact and loadable."""
    store = CheckpointStore(str(tmp_path / "resume"))
    seen, w_ser, o_ser, w_deser, o_deser = _dummy_io(weights_tag="good-step-10")
    store.save(step=10, loss_scale_state=None, weights_serializer=w_ser,
               optimizer_serializer=o_ser)

    # Second checkpoint dies mid-write (serializer raises) — LATEST is never flipped.
    def boom(p):
        Path(p).write_text("half-written")
        raise RuntimeError("killed mid-checkpoint")

    with pytest.raises(RuntimeError, match="killed mid-checkpoint"):
        store.save(step=20, loss_scale_state=None, weights_serializer=boom,
                   optimizer_serializer=o_ser)

    # The live checkpoint is still step 10, with its original weights.
    meta = store.load(weights_deserializer=w_deser, optimizer_deserializer=o_deser)
    assert meta["step"] == 10 and meta["slot"] == "slot-a"
    assert seen["weights"] == "good-step-10"


def test_store_save_leaves_no_temp_files(tmp_path):
    store = CheckpointStore(str(tmp_path / "resume"))
    _, w_ser, o_ser, _, _ = _dummy_io()
    store.save(step=1, loss_scale_state=None, weights_serializer=w_ser,
               optimizer_serializer=o_ser)
    stray = [p.name for p in (tmp_path / "resume" / "slot-a").iterdir()
             if p.name.endswith((".tmp", ".swap")) or p.name.startswith(".tmp-")]
    assert stray == [], f"unexpected temp files: {stray}"
