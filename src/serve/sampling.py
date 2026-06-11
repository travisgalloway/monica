"""Token sampling over a logits vector (portable, numpy only).

The generation core hands raw logits here and gets back one token id. Greedy decoding
(`temperature == 0`) is the lm-eval `do_sample=False` default and is fully
deterministic; sampling applies temperature, then optional top-k and top-p (nucleus)
truncation, then draws from the renormalized distribution using a caller-supplied
`numpy.random.Generator` (so a seed makes runs reproducible).

Above the seam: pure numpy, no backend. The caller converts backend logits to numpy
at the boundary (as `src/eval/val_loss.py` does), so this never touches mlx/torch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Return one token id sampled from a 1-D logits vector.

    `temperature == 0` is greedy (argmax). Otherwise: scale by 1/temperature, restrict
    to the top-`k` logits and/or the smallest set whose probability mass reaches
    `top_p`, renormalize, and draw.
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature == 0:
        return int(logits.argmax())
    if temperature < 0:
        raise ValueError(f"temperature {temperature} must be >= 0")

    logits = logits / temperature

    # top-k: keep only the k highest logits.
    if top_k is not None and 0 < top_k < logits.size:
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)

    probs = _softmax(logits)

    # top-p (nucleus): keep the smallest prefix of the sorted mass reaching top_p.
    if top_p is not None and 0.0 < top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        # Keep everything up to and including the token that crosses top_p.
        keep = cumulative <= top_p
        keep[0] = True  # always keep the most probable token
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()

    rng = rng or np.random.default_rng()
    return int(rng.choice(probs.size, p=probs))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over a 1-D vector (handles -inf masking)."""
    m = np.max(logits)
    exp = np.exp(logits - m)
    return exp / exp.sum()
