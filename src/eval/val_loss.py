"""Tier-1 evaluation: held-out validation loss / perplexity.

This is the primary pipeline-health signal for the POC: a smoothly decreasing val
perplexity IS the success criterion (no external harness needed). The numeric core
(`cross_entropy`, `perplexity`) is pure numpy and testable anywhere; `evaluate`
orchestrates it over a loader using only `ModelInterface.forward`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..model.interface import ModelInterface
from ..data.loader import PackedLoader


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean token-level cross-entropy (nats). logits (..., V), targets (...,)."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets).reshape(-1)
    flat = logits.reshape(-1, logits.shape[-1])
    # log-softmax in a numerically stable way
    m = flat.max(axis=-1, keepdims=True)
    logZ = m[:, 0] + np.log(np.exp(flat - m).sum(axis=-1))
    chosen = flat[np.arange(flat.shape[0]), targets]
    return float(np.mean(logZ - chosen))


def perplexity(mean_ce_nats: float) -> float:
    return float(np.exp(mean_ce_nats))


def evaluate(model: ModelInterface, loader: PackedLoader,
             max_batches: Optional[int] = None, to_numpy=np.asarray) -> dict:
    """Run `forward` over held-out batches; return {val_loss, val_perplexity}.

    `to_numpy` converts backend logits to numpy (identity by default; on MLX pass
    a converter). Backend-free otherwise.
    """
    total, n = 0.0, 0
    for i, (inputs, targets) in enumerate(loader.epoch()):
        if max_batches is not None and i >= max_batches:
            break
        logits = to_numpy(model.forward(inputs))
        total += cross_entropy(logits, targets)
        n += 1
    mean_ce = total / max(1, n)
    return {"val_loss": mean_ce, "val_perplexity": perplexity(mean_ce)}
