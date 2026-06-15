"""Associative-recall probe data (#67) — the test that the attention fraction works.

Pure SSMs lag Transformers on in-context retrieval (recall a value bound to a key
seen earlier) — and the gap widens with the number of key-value pairs, because an
SSM's fixed-width recurrent state is a capacity bottleneck while attention can look
back at any position. This generates the standard **multi-query associative recall
(MQAR)** task used to measure exactly that: a context of (key, value) pairs followed
by a query section that re-presents some keys; the model must emit each queried key's
value. A pure-Mamba and a hybrid are trained on it (`scripts/retrieval_probe.py`) and
the hybrid's higher recall accuracy at enough pairs is the measurable signal.

ABOVE THE SEAM — pure numpy, no backend. Keys occupy token ids `[0, n_keys)` and
values `[n_keys, n_keys + n_values)` (disjoint → `vocab_size = n_keys + n_values`).

Sequence layout (length `2*n_pairs + 2*n_queries`):
    k1 v1 ... k_n v_n   q1 vq1  q2 vq2 ...        (context ++ query section)
Each query key `qj` re-presents one of the context keys; under next-token prediction
the position holding `qj` is supervised to predict its value `vqj` (mask=1 there,
0 elsewhere — context-value positions are random and carry no signal). Dense
supervision over many query positions trains fast and makes the capacity gap visible.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def vocab_size(n_keys: int, n_values: int) -> int:
    """Token-id space for the task (disjoint key/value ranges)."""
    return n_keys + n_values


def seq_len(n_pairs: int, n_queries: Optional[int] = None) -> int:
    """Sequence length produced by `make_recall_batch` for these settings."""
    n_queries = n_pairs if n_queries is None else n_queries
    return 2 * n_pairs + 2 * n_queries


def make_recall_batch(rng: np.random.Generator, batch_size: int, n_pairs: int,
                      n_keys: int, n_values: int,
                      n_queries: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A batch of MQAR sequences as an (inputs, targets, mask) 3-tuple.

    Matches the SFTLoader contract so it feeds straight into `make_sft_train_step`
    (masked cross-entropy over the supervised positions).
      * inputs  (B, L) int64 — context pairs then the query section.
      * targets (B, L) int64 — the queried value at each query-key position (else 0).
      * mask    (B, L) int64 — 1 at the supervised query-key positions, else 0.
    """
    if n_pairs > n_keys:
        raise ValueError(f"n_pairs={n_pairs} cannot exceed n_keys={n_keys} (distinct keys)")
    n_queries = n_pairs if n_queries is None else n_queries
    L = seq_len(n_pairs, n_queries)
    inputs = np.zeros((batch_size, L), dtype=np.int64)
    targets = np.zeros((batch_size, L), dtype=np.int64)
    mask = np.zeros((batch_size, L), dtype=np.int64)
    for b in range(batch_size):
        keys = rng.choice(n_keys, size=n_pairs, replace=False)          # distinct keys
        values = rng.integers(0, n_values, size=n_pairs) + n_keys       # value id range
        inputs[b, 0:2 * n_pairs:2] = keys
        inputs[b, 1:2 * n_pairs:2] = values
        # Query section: re-present a (possibly repeated) sample of keys to recall.
        qidx = rng.integers(0, n_pairs, size=n_queries)
        base = 2 * n_pairs
        for j, qi in enumerate(qidx):
            kp = base + 2 * j                                          # query-key position
            inputs[b, kp] = keys[qi]
            inputs[b, kp + 1] = values[qi]                            # teacher-forced value
            targets[b, kp] = values[qi]                              # predict value from key
            mask[b, kp] = 1
    return inputs, targets, mask


def recall_accuracy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of supervised (mask=1) positions whose argmax prediction is correct.

    `logits` (B, L, V); `targets`/`mask` (B, L). Pure numpy — usable from any host."""
    pred = np.asarray(logits).argmax(axis=-1)                          # (B, L)
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return float("nan")
    return float((pred[m] == np.asarray(targets)[m]).mean())
