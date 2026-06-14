"""Token sampling over a logits vector (portable, numpy only).

The generation core hands raw logits here and gets back one token id. Greedy decoding
(`temperature == 0`) is the lm-eval `do_sample=False` default and is fully
deterministic; sampling applies temperature, then optional top-k and top-p (nucleus)
truncation, then draws from the renormalized distribution using a caller-supplied
`numpy.random.Generator` (so a seed makes runs reproducible).

Repetition control (`repetition_penalty`, `no_repeat_ngram_size`) is applied to the raw
logits *before* temperature, given the caller-supplied running context
(`previous_tokens`). It is stateless — this module holds no decode history — so the
generation loop passes the accumulated context each step. The defaults (penalty 1.0, no
ngram limit) are an exact no-op, so existing callers are unaffected.

Above the seam: pure numpy, no backend. The caller converts backend logits to numpy
at the boundary (as `src/eval/val_loss.py` does), so this never touches mlx/torch.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def sample(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    previous_tokens: Optional[Sequence[int]] = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: Optional[int] = None,
) -> int:
    """Return one token id sampled from a 1-D logits vector.

    `temperature == 0` is greedy (argmax). Otherwise: scale by 1/temperature, restrict
    to the top-`k` logits and/or the smallest set whose probability mass reaches
    `top_p`, renormalize, and draw.

    Repetition control (applied to the raw logits, before temperature, so it shapes both
    greedy and sampled draws):
      * `repetition_penalty > 1.0` — CTRL-style (Keskar et al.): for every id already in
        `previous_tokens`, a positive logit is divided by the penalty and a negative one
        multiplied, pushing seen tokens down. `1.0` is a no-op.
      * `no_repeat_ngram_size = n` — hard-ban (logit -> -inf) any token that would
        complete an n-gram already present in `previous_tokens` (the standard HF rule).
    Both require `previous_tokens`; without it they are skipped.
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)

    if previous_tokens is not None and len(previous_tokens) and (
            repetition_penalty != 1.0 or no_repeat_ngram_size):
        logits = logits.copy()  # own the buffer before in-place penalty edits
        prev = np.asarray(previous_tokens, dtype=np.int64).reshape(-1)
        if repetition_penalty != 1.0:
            if repetition_penalty < 1.0:
                raise ValueError(
                    f"repetition_penalty {repetition_penalty} must be >= 1.0 "
                    "(1.0 = off; CTRL-style penalty only suppresses, never boosts)")
            seen = np.unique(prev)
            seen = seen[(seen >= 0) & (seen < logits.size)]
            v = logits[seen]
            logits[seen] = np.where(v > 0, v / repetition_penalty,
                                    v * repetition_penalty)
        if no_repeat_ngram_size:
            banned = _banned_ngram_tokens(prev, int(no_repeat_ngram_size),
                                          logits.size)
            if banned.size:
                logits[banned] = -np.inf

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
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        # Standard nucleus: keep the smallest prefix whose mass reaches top_p,
        # *including* the token that crosses the threshold. A token is dropped only
        # once the mass strictly BEFORE it has already reached top_p, so the crossing
        # token survives (unlike a plain `cumulative <= top_p`, which drops it).
        keep = (cumulative - sorted_probs) < top_p
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


def _banned_ngram_tokens(prev: np.ndarray, n: int, vocab_size: int) -> np.ndarray:
    """Token ids that would complete a repeated `n`-gram in the context `prev`.

    The trailing `n-1` tokens of `prev` form the active prefix; wherever that same
    `(n-1)`-gram occurred earlier in `prev`, the token that followed it is banned (the
    standard no-repeat-ngram rule). `n == 1` bans every previously-seen token. The
    prefix at the very tail is not counted as a match against itself (it has no
    successor yet). Out-of-vocab ids are dropped.
    """
    if n < 1:
        return np.empty(0, dtype=np.int64)
    p = [int(t) for t in prev.tolist()]
    n1 = n - 1
    if len(p) < n1:
        return np.empty(0, dtype=np.int64)
    prefix = tuple(p[len(p) - n1:]) if n1 else ()
    banned = [p[i + n1] for i in range(len(p) - n1)
              if tuple(p[i:i + n1]) == prefix]
    ids = np.unique(np.asarray(banned, dtype=np.int64)) if banned else \
        np.empty(0, dtype=np.int64)
    return ids[(ids >= 0) & (ids < vocab_size)]
