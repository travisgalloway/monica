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

# Bound a single stable-softmax temporary — `blk` and the `exp(blk - m)` it spawns are
# each (chunk, V) in float64 — to this many bytes, so eval memory scales with vocab, not
# batch size. The cap keeps small/test vocabs (where the budget far exceeds the row count)
# in a single block, so their result stays bit-identical to the full-array float64 reduction.
_CE_CHUNK_BYTES = 256 * 1024 * 1024  # ~256 MiB per float64 (chunk, V) temporary
_CE_CHUNK_CAP = 4096                 # upper bound on rows/chunk


def _ce_sum_chunked(flat: np.ndarray, targets: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> tuple[float, float]:
    """Sum of per-token cross-entropy (optionally `weights`-weighted) over `flat`
    (rows = tokens, cols = vocab), plus the total weight. Computed in float64 over
    row-chunks so the stable-softmax temporaries are bounded to (chunk, V) instead of
    the full (B*T, V): at the Qwen3 vocab (151,669) a single eval batch's logits are
    ~40 GB in float64, which made eval thrash/OOM on the host. `chunk` shrinks with V so
    each (chunk, V) float64 temporary stays under `_CE_CHUNK_BYTES` regardless of vocab
    (~221 rows at the Qwen3 vocab), capped at `_CE_CHUNK_CAP`. For chunk >= n (small /
    test vocabs, where the byte budget exceeds the row count) the whole array is one block,
    so the result is bit-identical to the full-array float64 computation and the numeric
    contract with `masked_cross_entropy` is preserved.
    """
    n = flat.shape[0]
    vocab = flat.shape[1]
    chunk = min(_CE_CHUNK_CAP, max(1, _CE_CHUNK_BYTES // (vocab * 8)))
    total, wsum = 0.0, 0.0
    for s in range(0, n, chunk):
        blk = np.asarray(flat[s:s + chunk], dtype=np.float64)
        tgt = targets[s:s + chunk]
        # log-softmax in a numerically stable way
        m = blk.max(axis=-1, keepdims=True)
        logZ = m[:, 0] + np.log(np.exp(blk - m).sum(axis=-1))
        tok_ce = logZ - blk[np.arange(blk.shape[0]), tgt]
        if weights is None:
            total += float(tok_ce.sum())
            wsum += tok_ce.shape[0]
        else:
            w = weights[s:s + chunk]
            total += float((tok_ce * w).sum())
            wsum += float(w.sum())
    return total, wsum


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean token-level cross-entropy (nats). logits (..., V), targets (...,)."""
    flat = np.asarray(logits)
    flat = flat.reshape(-1, flat.shape[-1])
    targets = np.asarray(targets).reshape(-1)
    total, n = _ce_sum_chunked(flat, targets)
    return total / n


def masked_cross_entropy(logits: np.ndarray, targets: np.ndarray,
                         mask: np.ndarray) -> float:
    """Response-token mean cross-entropy (nats): sum(mask * per-token CE) / sum(mask).

    The portable reference for the SFT masked-CE objective and masked val perplexity.
    Positions with mask 0 (prompt + padding) drop out, so their target ids are never
    read for the loss value (they still index the logits, so pass in-range pad ids).
    Returns 0.0 for an all-zero mask (an all-padding batch contributes nothing).
    """
    flat = np.asarray(logits)
    flat = flat.reshape(-1, flat.shape[-1])
    targets = np.asarray(targets).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    total, denom = _ce_sum_chunked(flat, targets, weights=mask)
    if denom == 0:
        return 0.0
    return total / denom


def perplexity(mean_ce_nats: float) -> float:
    return float(np.exp(mean_ce_nats))


def bits_per_byte(total_ce_nats: float, n_bytes: float) -> float:
    """Tokenizer-invariant bits-per-byte (#192): total cross-entropy (nats) over a token
    span, converted to bits (÷ ln2) and normalized by the UTF-8 byte length of that span.
    BPB = total_ce_nats / (ln2 * n_bytes)."""
    return float(total_ce_nats / (np.log(2.0) * n_bytes))


def evaluate(model: ModelInterface, loader: PackedLoader,
             max_batches: Optional[int] = None, to_numpy=np.asarray) -> dict:
    """Run `forward` over held-out batches; return {val_loss, val_perplexity}, plus
    `val_bpb` (#192, tokenizer-invariant) when `loader` exposes corpus byte/token totals
    (`.n_bytes`/`.n_tokens`) — omitted otherwise (legacy artifacts, no byte count).

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
    if total_tokens == 0:
        # Otherwise mean_ce=0 -> perplexity=1.0, a false "perfect model" that silently
        # masks a misconfigured eval (empty/missing val split, wrong path).
        raise ValueError("evaluate(): no tokens evaluated — val loader is empty")
    mean_ce = total_ce / total_tokens
    result = {"val_loss": mean_ce, "val_perplexity": perplexity(mean_ce)}
    n_bytes = getattr(loader, "n_bytes", None)
    n_tokens_total = getattr(loader, "n_tokens", None)
    if n_bytes and n_tokens_total:                 # both present and non-zero
        bytes_per_token = n_bytes / n_tokens_total
        effective_bytes = total_tokens * bytes_per_token
        result["val_bpb"] = bits_per_byte(total_ce, effective_bytes)
    return result


def evaluate_masked(model: ModelInterface, loader,
                    max_batches: Optional[int] = None, to_numpy=np.asarray) -> dict:
    """Masked held-out loss for SFT: perplexity over *response* tokens only.

    `loader` yields `(inputs, targets, mask)` (an `SFTLoader`). Each batch's masked CE is
    weighted by its response-token count so partial final batches do not bias the mean.
    Returns {val_loss, val_perplexity}, plus `val_bpb` (#192) when `loader` exposes
    `.n_bytes`/`.n_tokens` — SFT loaders don't carry byte counts today, so this is the
    common case (omitted); kept for the day a byte-aware loader is passed.
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
    if total_tokens == 0:
        # No response tokens at all -> a false perplexity=1.0; fail loudly instead.
        raise ValueError("evaluate_masked(): no response tokens evaluated — "
                         "val loader is empty or fully masked")
    mean_ce = total_ce / total_tokens
    result = {"val_loss": mean_ce, "val_perplexity": perplexity(mean_ce)}
    n_bytes = getattr(loader, "n_bytes", None)
    n_tokens_total = getattr(loader, "n_tokens", None)
    if n_bytes and n_tokens_total:
        bytes_per_token = n_bytes / n_tokens_total
        effective_bytes = total_tokens * bytes_per_token
        result["val_bpb"] = bits_per_byte(total_ce, effective_bytes)
    return result
