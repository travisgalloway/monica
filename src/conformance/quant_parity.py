"""Quantized-vs-fp quality gate (#168): top-1 agreement + mean KL + max logit drift of
a quantized model against its fp reference, with bits-dependent thresholds. This is the
STATISTICAL gate #168's acceptance criteria ask for — it never uses `allclose` (a
quantized model's logits are expected to differ elementwise from the fp reference; the
question is whether the quantized model's *decisions* still agree, not whether its
numbers do).

Portable — numpy only, no backend import (see `tests/test_import_guard.py`). Mirrors
`forward_step_parity.py`'s documented convention: `check_quant_parity` RETURNS the
verdict and does NOT raise; the caller's `assert result["ok"]` is the real gate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# int8 is expected to be nearly lossless; int4 tolerates real but bounded drift.
QUANT_THRESHOLDS = {
    8: {"top1": 0.99, "kl": 1e-3},
    4: {"top1": 0.95, "kl": 5e-2},
}


def _log_softmax_f64(logits: np.ndarray) -> np.ndarray:
    """Log-softmax over the last axis in float64 — KL over a wide vocab is numerically
    thin in fp32 (the individual probabilities can be tiny), so this promotes before the
    exp/log rather than after."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def check_quant_parity(fp_logits: np.ndarray, q_logits: np.ndarray, bits: int,
                       thresholds: Optional[dict] = None) -> dict:
    """Compare a quantized model's logits against the fp reference's.

    `fp_logits`/`q_logits`: `(..., vocab)`, any leading shape (e.g. `(B, T, V)`) as long
    as they match — flattened to `(N, V)` internally so top-1 agreement is measured over
    every position, not just the last one.

    `thresholds`, if given, overrides `QUANT_THRESHOLDS[bits]` (`{"top1": ..., "kl":
    ...}`); a caller comparing at a bit-width not in the default table must supply one.

    Returns `{top1_agreement, mean_kl, max_abs_drift, ok}` — does NOT raise.
    """
    fp = np.asarray(fp_logits, dtype=np.float64)
    q = np.asarray(q_logits, dtype=np.float64)
    if fp.shape != q.shape:
        return {"top1_agreement": 0.0, "mean_kl": float("inf"),
                "max_abs_drift": float("inf"), "ok": False,
                "error": f"shape mismatch: fp {fp.shape} vs q {q.shape}"}

    vocab = fp.shape[-1]
    fp_flat = fp.reshape(-1, vocab)
    q_flat = q.reshape(-1, vocab)

    fp_top1 = fp_flat.argmax(axis=-1)
    q_top1 = q_flat.argmax(axis=-1)
    top1_agreement = float(np.mean(fp_top1 == q_top1))

    fp_logp = _log_softmax_f64(fp_flat)
    q_logp = _log_softmax_f64(q_flat)
    # KL(fp || q) — the fp distribution is the reference, q is being measured against it.
    fp_p = np.exp(fp_logp)
    kl = np.sum(fp_p * (fp_logp - q_logp), axis=-1)
    mean_kl = float(np.mean(kl))

    max_abs_drift = float(np.abs(fp_flat - q_flat).max())

    thr = thresholds if thresholds is not None else QUANT_THRESHOLDS.get(bits)
    if thr is None:
        return {"top1_agreement": top1_agreement, "mean_kl": mean_kl,
                "max_abs_drift": max_abs_drift, "ok": False,
                "error": f"no thresholds for bits={bits} and none supplied"}

    ok = bool(top1_agreement >= thr["top1"] and mean_kl <= thr["kl"])
    return {"top1_agreement": top1_agreement, "mean_kl": mean_kl,
            "max_abs_drift": max_abs_drift, "ok": ok}
