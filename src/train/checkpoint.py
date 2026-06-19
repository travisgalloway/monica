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


# --- within-backend resume bundle ------------------------------------------
# The COMMIT marker is written LAST, after the optimizer + meta are durably on
# disk. Its presence is the bundle's "this is complete and consistent" flag:
# `load_resume` refuses any bundle without it, so a checkpoint interrupted
# mid-write is detected loudly rather than silently resumed half-formed.
_COMMIT = "COMMIT"


def save_resume(path: str, *, step: int, loss_scale_state: Any,
                optimizer_serializer: Callable[[str], None]) -> None:
    """Save a same-backend resume bundle: step, fp16 loss-scale state, optimizer state.

    Note `loss_scale_state` holds the dynamic fp16 loss scaler's state (or None for
    fp32/bf16) — NOT an RNG state. The data order on resume is reconstructed
    deterministically from (seed, step, grad_accum) by the training loop, and the
    model has no train-time RNG (no dropout; the only random op is weight init,
    overwritten by the loaded weights), so there is no RNG to persist.

    `optimizer_serializer(opt_path)` is supplied by the backend (e.g. MLX/torch),
    since optimizer state layout is backend-specific. Written via temp+rename so a
    partial optimizer dump never overwrites the previous good one.
    """
    path = str(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    commit = Path(path, _COMMIT)
    # Invalidate the bundle for the duration of the write. The backend optimizer
    # serializer owns its real filename(s) (it appends an extension), so we cannot
    # temp+rename it ourselves; instead we gate consistency on the COMMIT marker.
    # If we crash anywhere below before COMMIT is rewritten, `load_resume` refuses
    # this bundle rather than resuming a half-written optimizer + newer weights.
    # (Single-slot: a crashed checkpoint is not resumable — double-buffered slots
    # that preserve the previous checkpoint are a documented follow-up.)
    if commit.exists():
        commit.unlink()
    optimizer_serializer(str(Path(path, "optimizer.state")))
    meta = {"step": int(step), "loss_scale_state": _jsonable(loss_scale_state)}
    _atomic_write_text(Path(path, "resume_meta.json"), json.dumps(meta))
    _atomic_write_text(commit, json.dumps({"step": int(step)}))


def load_resume(path: str, optimizer_deserializer: Callable[[str], Any]) -> dict:
    """Restore the resume bundle. Returns {step, loss_scale_state, optimizer}.

    Refuses a bundle whose COMMIT marker is missing — that means the checkpoint
    was interrupted mid-write and is not safe to resume from.
    """
    path = str(path)
    if not Path(path, _COMMIT).exists():
        raise RuntimeError(
            f"resume bundle at {path!r} has no COMMIT marker — it was written "
            "partially (interrupted mid-checkpoint) and is unsafe to resume from."
        )
    meta = json.loads(Path(path, "resume_meta.json").read_text())
    meta["optimizer"] = optimizer_deserializer(str(Path(path, "optimizer.state")))
    return meta


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
