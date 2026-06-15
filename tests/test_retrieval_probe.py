"""Retrieval probe (#67): task-generator correctness (portable) + a small end-to-end
train comparing pure-Mamba vs hybrid recall (MLX-gated).

The portable tests pin the MQAR data contract. The MLX test trains both architectures
briefly and checks the harness produces valid recall accuracies and that the hybrid
clears a recall bar pure SSMs struggle at when the number of pairs stresses their
fixed-width state — the signal the attention fraction is doing its job. (At toy scale
pure Mamba-2 is itself strong at recall; the headline pure-vs-hybrid comparison at
larger scale is what `scripts/retrieval_probe.py` runs.)
"""

import numpy as np
import pytest

from src.eval.retrieval_probe import (make_recall_batch, recall_accuracy, seq_len,
                                       vocab_size)


def test_vocab_and_seq_len():
    assert vocab_size(40, 24) == 64
    assert seq_len(n_pairs=8, n_queries=5) == 2 * 8 + 2 * 5


def test_recall_batch_shapes_and_ranges():
    rng = np.random.default_rng(0)
    B, n_pairs, n_keys, n_values, n_q = 4, 8, 16, 10, 5
    inputs, targets, mask = make_recall_batch(rng, B, n_pairs, n_keys, n_values, n_q)
    L = seq_len(n_pairs, n_q)
    assert inputs.shape == targets.shape == mask.shape == (B, L)
    # Exactly n_queries supervised positions per row.
    assert (mask.sum(axis=1) == n_q).all()
    # Keys live in [0, n_keys); values in [n_keys, n_keys+n_values).
    keys = inputs[:, 0:2 * n_pairs:2]
    vals = inputs[:, 1:2 * n_pairs:2]
    assert keys.max() < n_keys and keys.min() >= 0
    assert vals.min() >= n_keys and vals.max() < n_keys + n_values


def test_recall_batch_targets_consistent_with_context():
    # Each supervised position predicts the value bound to that key in the context.
    rng = np.random.default_rng(1)
    n_pairs, n_keys, n_values = 6, 12, 8
    inputs, targets, mask = make_recall_batch(rng, 16, n_pairs, n_keys, n_values, n_pairs)
    for b in range(inputs.shape[0]):
        ctx = {int(inputs[b, 2 * i]): int(inputs[b, 2 * i + 1]) for i in range(n_pairs)}
        for p in np.flatnonzero(mask[b]):
            assert targets[b, p] == ctx[int(inputs[b, p])]      # value matches the binding


def test_recall_batch_rejects_too_many_pairs():
    with pytest.raises(ValueError):
        make_recall_batch(np.random.default_rng(0), 2, n_pairs=10, n_keys=8, n_values=4)


def test_recall_accuracy_perfect_and_chance():
    rng = np.random.default_rng(2)
    _, targets, mask = make_recall_batch(rng, 8, 6, 12, 8, 6)
    V = 20
    # Perfect logits: argmax == target everywhere -> accuracy 1.0 over masked positions.
    perfect = np.full(targets.shape + (V,), -10.0)
    for idx in np.ndindex(targets.shape):
        perfect[idx + (targets[idx],)] = 10.0
    assert recall_accuracy(perfect, targets, mask) == 1.0
    # All-zero logits argmax to class 0; targets are >= n_keys so accuracy ~ 0.
    assert recall_accuracy(np.zeros(targets.shape + (V,)), targets, mask) == 0.0


# --- end-to-end probe: the hybrid measurably out-recalls pure Mamba (MLX, ~20s) ---
mx = pytest.importorskip("mlx.core")


def test_hybrid_outrecalls_pure_mamba():
    """At enough key-value pairs the SSM's fixed-width state saturates and pure Mamba
    stays near chance, while a hybrid with a couple of attention layers recalls almost
    perfectly. Fixed seed; the measured gap is ~0.85 (hybrid ~1.0 vs Mamba ~0.15) — we
    assert a comfortable margin. `scripts/retrieval_probe.py` runs the fuller sweep."""
    from scripts.retrieval_probe import run_probe

    r = run_probe(n_pairs=16, n_keys=32, n_values=24, steps=300, lr=2e-3,
                  batch_size=64, d_model=64, n_layers=4, attn_every=2, n_attn_heads=4,
                  seed=0, backend_name="mlx")
    assert 0.0 <= r["mamba_acc"] <= 1.0 and 0.0 <= r["hybrid_acc"] <= 1.0
    assert r["hybrid_acc"] > 0.7, r                    # the hybrid learns recall
    assert r["hybrid_acc"] - r["mamba_acc"] > 0.3, r   # measurable improvement vs pure Mamba
