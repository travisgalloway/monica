"""Unit tests for portable weight save/load and the double-buffered CheckpointStore.

Backend-free: uses numpy weight dicts and dummy weight/optimizer (de)serializers, so it
runs anywhere. The resume metadata persists `step`, the fp16 `loss_scale_state`, and —
since #216 — the dataloader position (`data_state`). These tests guard `_jsonable`
against a realistic nested NumPy structure (nested dict + np.ndarray + np scalars — the
worst case it handles), which a shallow JSON conversion breaks; the crash-safety contract
(the previous checkpoint survives a write interrupted mid-flight); and that `data_state`
lives INSIDE the slot's existing meta file, so it is committed by the same atomic LATEST
flip as the weights rather than drifting out of sync with them.
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


def test_store_roundtrips_data_state(tmp_path):
    """#216: the dataloader position rides inside resume_meta.json, so it is committed by
    the same atomic LATEST flip as the weights and the step. It contains a nested numpy
    RNG state (the stream's tripwire), which only survives via `_jsonable`."""
    store = CheckpointStore(str(tmp_path / "resume"))
    data_state = {
        "version": 1, "stage_idx": 1, "micro_in_stage": 4, "global_micro": 22,
        "epoch_idx": 3, "batches_into_epoch": 2,
        "rng_state": np.random.default_rng(7).bit_generator.state,
        "stages_at_save": [[2048, 32, 100], [16384, 4, 50]],
        "fingerprint": {"seed": 0, "grad_accum": 4,
                        "stage_shapes": [[2048, 32], [16384, 4]]},
    }
    _, w_ser, o_ser, w_deser, o_deser = _dummy_io()
    store.save(step=150, loss_scale_state=None, weights_serializer=w_ser,
               optimizer_serializer=o_ser, data_state=data_state)
    meta = store.load(weights_deserializer=w_deser, optimizer_deserializer=o_deser)

    assert meta["data_state"] == _jsonable(data_state)
    assert meta["data_state"]["global_micro"] == 22
    assert meta["data_state"]["fingerprint"]["stage_shapes"] == [[2048, 32], [16384, 4]]


def test_store_without_data_state_reads_back_as_none(tmp_path):
    """Backward compatibility: a bundle saved with no dataloader position (SFT/DPO, or a
    pre-#216 run) must read back as None so callers fall back to reconstruction."""
    store = CheckpointStore(str(tmp_path / "resume"))
    _, w_ser, o_ser, w_deser, o_deser = _dummy_io()
    store.save(step=7, loss_scale_state=None, weights_serializer=w_ser,
               optimizer_serializer=o_ser)
    meta = store.load(weights_deserializer=w_deser, optimizer_deserializer=o_deser)
    assert meta.get("data_state") is None


def test_legacy_bundle_missing_the_key_entirely_reads_back_as_none(tmp_path):
    """A bundle written BEFORE #216 has no `data_state` key at all, not a null one."""
    store = CheckpointStore(str(tmp_path / "resume"))
    _, w_ser, o_ser, w_deser, o_deser = _dummy_io()
    store.save(step=7, loss_scale_state=None, weights_serializer=w_ser,
               optimizer_serializer=o_ser)
    slot_meta = tmp_path / "resume" / "slot-a" / "resume_meta.json"
    legacy = json.loads(slot_meta.read_text())
    del legacy["data_state"]
    slot_meta.write_text(json.dumps(legacy))

    meta = store.load(weights_deserializer=w_deser, optimizer_deserializer=o_deser)
    assert "data_state" not in meta and meta.get("data_state") is None
    assert meta["step"] == 7


def test_data_state_adds_no_file_to_the_slot(tmp_path):
    """It must live in the EXISTING meta file, never next to LATEST and never as a new
    slot entry — anything outside the slot would lose the commit protocol's atomicity."""
    store = CheckpointStore(str(tmp_path / "resume"))
    _, w_ser, o_ser, _, _ = _dummy_io()
    store.save(step=1, loss_scale_state=None, weights_serializer=w_ser,
               optimizer_serializer=o_ser)
    without = sorted(p.name for p in (tmp_path / "resume" / "slot-a").iterdir())

    store2 = CheckpointStore(str(tmp_path / "resume2"))
    store2.save(step=1, loss_scale_state=None, weights_serializer=w_ser,
                optimizer_serializer=o_ser, data_state={"version": 1, "x": 2})
    with_state = sorted(p.name for p in (tmp_path / "resume2" / "slot-a").iterdir())

    assert without == with_state
    assert sorted(p.name for p in (tmp_path / "resume2").iterdir()) == ["LATEST", "slot-a"]


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
