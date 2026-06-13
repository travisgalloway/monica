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


def masked_cross_entropy(logits: np.ndarray, targets: np.ndarray,
                         mask: np.ndarray) -> float:
    """Response-token mean cross-entropy (nats): sum(mask * per-token CE) / sum(mask).

    The portable reference for the SFT masked-CE objective and masked val perplexity.
    Positions with mask 0 (prompt + padding) drop out, so their target ids are never
    read for the loss value (they still index the logits, so pass in-range pad ids).
    Returns 0.0 for an all-zero mask (an all-padding batch contributes nothing).
    """
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    flat = logits.reshape(-1, logits.shape[-1])
    m = flat.max(axis=-1, keepdims=True)
    logZ = m[:, 0] + np.log(np.exp(flat - m).sum(axis=-1))
    chosen = flat[np.arange(flat.shape[0]), targets]
    tok_ce = logZ - chosen
    denom = mask.sum()
    if denom == 0:
        return 0.0
    return float((tok_ce * mask).sum() / denom)


def perplexity(mean_ce_nats: float) -> float:
    return float(np.exp(mean_ce_nats))


def evaluate(model: ModelInterface, loader: PackedLoader,
             max_batches: Optional[int] = None, to_numpy=np.asarray) -> dict:
    """Run `forward` over held-out batches; return {val_loss, val_perplexity}.

    `to_numpy` converts backend logits to numpy (identity by default; on MLX pass
    a converter). Backend-free otherwise.
    """
    # Weight each batch's mean CE by its token count so a smaller final batch
    # (drop_last=False) does not bias the result.
    total_ce, total_tokens = 0.0, 0
    for i, (inputs, targets) in enumerate(loader.epoch()):
        if max_batches is not None and i >= max_batches:
            break
        logits = to_numpy(model.forward(inputs))
        n_tokens = int(np.asarray(targets).size)
        total_ce += cross_entropy(logits, targets) * n_tokens
        total_tokens += n_tokens
    mean_ce = total_ce / max(1, total_tokens)
    return {"val_loss": mean_ce, "val_perplexity": perplexity(mean_ce)}


def evaluate_masked(model: ModelInterface, loader,
                    max_batches: Optional[int] = None, to_numpy=np.asarray) -> dict:
    """Masked held-out loss for SFT: perplexity over *response* tokens only.

    `loader` yields `(inputs, targets, mask)` (an `SFTLoader`). Each batch's masked CE is
    weighted by its response-token count so partial final batches do not bias the mean.
    Returns {val_loss, val_perplexity}.
    """
    total_ce, total_tokens = 0.0, 0.0
    for i, (inputs, targets, mask) in enumerate(loader.epoch()):
        if max_batches is not None and i >= max_batches:
            break
        n_tokens = float(np.asarray(mask).sum())
        if n_tokens == 0:
            continue
        logits = to_numpy(model.forward(inputs))
        total_ce += masked_cross_entropy(logits, targets, mask) * n_tokens
        total_tokens += n_tokens
    mean_ce = total_ce / max(1.0, total_tokens)
    return {"val_loss": mean_ce, "val_perplexity": perplexity(mean_ce)}
