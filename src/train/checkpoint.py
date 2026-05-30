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
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


# --- portable weights -------------------------------------------------------
def save_weights(state_dict: Dict[str, np.ndarray], path: str,
                 config: Optional[Any] = None) -> None:
    """Save weights as safetensors + a `<path>.config.json` sidecar."""
    from safetensors.numpy import save_file  # lazy

    path = str(path)
    tensors = {k: np.asarray(v) for k, v in state_dict.items()}
    save_file(tensors, path)
    if config is not None:
        cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        Path(path + ".config.json").write_text(json.dumps(cfg, indent=2))


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
def save_resume(path: str, *, step: int, rng_state: Any,
                optimizer_serializer: Callable[[str], None]) -> None:
    """Save a same-backend resume bundle: training step, RNG state, optimizer state.

    `optimizer_serializer(opt_path)` is supplied by the backend (e.g. MLX/torch),
    since optimizer state layout is backend-specific.
    """
    path = str(path)
    Path(path).mkdir(parents=True, exist_ok=True)
    meta = {"step": int(step), "rng_state": _jsonable(rng_state)}
    Path(path, "resume_meta.json").write_text(json.dumps(meta))
    optimizer_serializer(str(Path(path, "optimizer.state")))


def load_resume(path: str, optimizer_deserializer: Callable[[str], Any]) -> dict:
    """Restore the resume bundle. Returns {step, rng_state, optimizer}."""
    path = str(path)
    meta = json.loads(Path(path, "resume_meta.json").read_text())
    meta["optimizer"] = optimizer_deserializer(str(Path(path, "optimizer.state")))
    return meta


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return {"__ndarray__": x.tolist(), "dtype": str(x.dtype)}
    return x
