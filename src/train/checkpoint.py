"""Checkpointing: two DISTINCT concerns, deliberately not conflated.

1. Weights — PORTABLE format (safetensors). This is what lets an MLX checkpoint
   seed a CUDA run and lets a CUDA-trained model run on the Mac. Backend-agnostic:
   a flat dict of {param_name: numpy array} plus the config.

2. Optimizer state — needed ONLY for exact resume on the SAME backend after an
   interruption. It does NOT need to be cross-backend portable (MLX and PyTorch
   optimizer state differ internally; the migration trains fresh on CUDA anyway).
   Saved via a backend-provided serializer, scoped to within-backend resume.

`safetensors` is imported lazily so this module loads without the dependency.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


# --- durable (crash-safe) writes -------------------------------------------
# A long run (the poc is ~26 days) WILL be interrupted mid-checkpoint by
# preemption / OOM / a manual kill. A naive `write_text` or `save_file` leaves a
# truncated, half-written file that silently corrupts the only checkpoint. Every
# artifact below is therefore written to a temp file in the same directory,
# flushed + fsync'd, then `os.replace`d into place (atomic on POSIX): a reader
# only ever sees the complete old file or the complete new one, never a partial.
def _fsync_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    """fsync a directory so a rename/creation inside it is durable across power loss."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: str, data: bytes) -> None:
    path = str(path)
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".swap")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(d)   # make the rename itself durable, not just the file data
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_text(path: str, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


# --- portable weights -------------------------------------------------------
def save_weights(state_dict: Dict[str, np.ndarray], path: str,
                 config: Optional[Any] = None) -> None:
    """Save weights as safetensors + a `<path>.config.json` sidecar (atomically)."""
    from safetensors.numpy import save_file  # lazy

    path = str(path)
    tensors = {k: np.asarray(v) for k, v in state_dict.items()}
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".safetensors")
    os.close(fd)  # safetensors writes by path, not fd
    try:
        save_file(tensors, tmp)
        _fsync_path(tmp)
        os.replace(tmp, path)
        # Durable rename even when save_weights runs standalone (model.save outside
        # CheckpointStore), not only when a later CheckpointStore dir-fsync covers it.
        _fsync_dir(d)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if config is not None:
        cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        _atomic_write_text(path + ".config.json", json.dumps(cfg, indent=2))


def load_weights_dict(path: str) -> Dict[str, np.ndarray]:
    """Load the portable weight dict (numpy). Backends map it into their params."""
    from safetensors.numpy import load_file  # lazy

    return load_file(str(path))


def load_weights(model: Any, path: str) -> None:
    """Load portable weights into a backend model via its `_load_portable` hook."""
    weights = load_weights_dict(path)
    if not hasattr(model, "_load_portable"):
        raise NotImplementedError(
            "Backend must implement `_load_portable(dict)` to map portable weights."
        )
    model._load_portable(weights)


# --- within-backend resume: double-buffered, crash-safe ---------------------
class CheckpointStore:
    """Crash-safe, double-buffered checkpoint area (one conversation's worth of resume).

    A long run (the poc is ~26 days) WILL be interrupted mid-checkpoint by preemption /
    OOM / a manual kill. Atomic per-file writes stop *truncation*, but a single-slot
    layout still loses resumability when a crash lands between writing the new weights and
    the new optimizer (newer weights, older optimizer; or no marker at all). This store
    keeps TWO slots plus an atomically-flipped ``LATEST`` pointer:

      <root>/LATEST        text file naming the live slot ("slot-a" | "slot-b")
      <root>/slot-a/       weights.safetensors (+ .config.json), optimizer.state*, resume_meta.json
      <root>/slot-b/       (same)

    Each checkpoint is written **in full** to the INACTIVE slot (so the live one is never
    touched), every file is fsync'd, and only then is ``LATEST`` flipped in one atomic
    rename — the single commit point. A crash before the flip leaves ``LATEST`` pointing
    at the previous, fully-intact checkpoint; a crash during the flip leaves either the
    old or the new name, both naming a complete slot. The previous checkpoint always
    survives until the next one is durably committed.

    Each slot holds the FULL checkpoint: portable weights (the cross-backend bridge) AND
    the within-backend resume bundle (optimizer + step + fp16 loss-scale state). Note the
    loss-scale state is NOT an RNG state — the data order on resume is reconstructed
    deterministically from (seed, step, grad_accum) and the model has no train-time RNG.
    """

    _SLOTS = ("slot-a", "slot-b")

    def __init__(self, root: str):
        self.root = Path(root)

    # -- pointer ----------------------------------------------------------------
    def _latest_file(self) -> Path:
        return self.root / "LATEST"

    def latest_slot(self) -> Optional[str]:
        f = self._latest_file()
        if not f.exists():
            return None
        slot = f.read_text().strip()
        return slot if slot in self._SLOTS else None

    def has_checkpoint(self) -> bool:
        return self.latest_slot() is not None

    def _inactive_slot(self) -> str:
        return self._SLOTS[1] if self.latest_slot() == self._SLOTS[0] else self._SLOTS[0]

    # -- write/read -------------------------------------------------------------
    def save(self, *, step: int, loss_scale_state: Any,
             weights_serializer: Callable[[str], None],
             optimizer_serializer: Callable[[str], None]) -> str:
        """Write a full checkpoint to the inactive slot, then atomically make it live.

        `weights_serializer(path)` writes portable weights (e.g. `model.save`) and
        `optimizer_serializer(path)` the backend optimizer state — both supplied by the
        caller so this stays backend-free. Returns the slot that was committed.
        """
        slot = self._inactive_slot()
        slot_dir = self.root / slot
        # Start the inactive slot fresh — safe because LATEST still names the other slot,
        # so destroying a stale/partial inactive slot never touches the live checkpoint.
        if slot_dir.exists():
            shutil.rmtree(slot_dir)
        slot_dir.mkdir(parents=True, exist_ok=True)

        weights_serializer(str(slot_dir / "weights.safetensors"))   # + .config.json sidecar
        optimizer_serializer(str(slot_dir / "optimizer.state"))     # backend owns the extension
        meta = {"step": int(step), "loss_scale_state": _jsonable(loss_scale_state)}
        _atomic_write_text(slot_dir / "resume_meta.json", json.dumps(meta))

        # Durably flush every file's DATA (the backend optimizer dump may not fsync
        # itself, and we don't know its exact name), then the slot directory.
        for f in slot_dir.iterdir():
            _fsync_path(str(f))
        _fsync_dir(slot_dir)

        # Commit: flip LATEST (atomic rename), then fsync root so the flip persists.
        _atomic_write_text(self._latest_file(), slot)
        _fsync_dir(self.root)
        return slot

    def load(self, *, weights_deserializer: Callable[[str], None],
             optimizer_deserializer: Callable[[str], Any]) -> dict:
        """Load the live checkpoint into the model + optimizer. Returns
        {step, loss_scale_state, optimizer, slot}. Raises if no checkpoint is committed."""
        slot = self.latest_slot()
        if slot is None:
            raise RuntimeError(f"no committed checkpoint under {str(self.root)!r}")
        slot_dir = self.root / slot
        weights_deserializer(str(slot_dir / "weights.safetensors"))
        meta = json.loads((slot_dir / "resume_meta.json").read_text())
        meta["optimizer"] = optimizer_deserializer(str(slot_dir / "optimizer.state"))
        meta["slot"] = slot
        return meta

    def latest_weights_path(self) -> Optional[str]:
        """Path to the live slot's portable weights (for downstream eval/serving), or None."""
        slot = self.latest_slot()
        return str(self.root / slot / "weights.safetensors") if slot else None


def _jsonable(x: Any) -> Any:
    """Recursively convert RNG/optimizer state into JSON-serializable values.

    Real RNG states (e.g. ``np.random.default_rng(...).bit_generator.state``) are
    nested dicts containing np.ndarrays and NumPy scalar types, so a shallow
    conversion would raise inside ``json.dumps``.
    """
    if isinstance(x, np.ndarray):
        return {"__ndarray__": x.tolist(), "dtype": str(x.dtype)}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x
