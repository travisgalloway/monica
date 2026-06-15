"""Copying / retrieval / long-context probes (#79).

Pure numpy. Verifies the generators produce well-formed supervised positions and that an
oracle (one-hot on targets) scores 1.0 — the scorer/target/mask alignment is the contract.
"""

import numpy as np
import pytest

from src.eval.probes import (fewshot_vocab, format_probe_table, make_fewshot_copy_batch,
                            make_needle_batch, make_phonebook_batch, needle_vocab,
                            phonebook_vocab, probe_accuracy, run_probes)


def _oracle_logits(inputs, targets, vocab):
    """Logits whose argmax is the target id at every position (a perfect model)."""
    B, L = inputs.shape
    logits = np.zeros((B, L, vocab), dtype=np.float64)
    for b in range(B):
        for l in range(L):
            logits[b, l, targets[b, l]] = 1.0
    return logits


def test_needle_shapes_and_supervision():
    rng = np.random.default_rng(0)
    x, t, m = make_needle_batch(rng, 4, 64)
    assert x.shape == t.shape == m.shape == (4, 64)
    assert m.sum() == 4                                    # exactly one supervised pos/seq
    # default needle id ranges: filler [0,128), keys [128,192), values [192,256)
    for b in range(4):
        q = int(np.flatnonzero(m[b])[0])
        qkey = x[b, q]
        assert 128 <= qkey < 192                           # query is a key id
        earlier = np.flatnonzero(x[b, :q] == qkey)         # the planted needle key, earlier
        assert earlier.size >= 1
        p = int(earlier[0])
        assert x[b, p + 1] == t[b, q] >= 192               # planted value == supervised target
    assert probe_accuracy(_oracle_logits(x, t, needle_vocab()), t, m) == 1.0


def test_needle_rejects_tiny_context():
    with pytest.raises(ValueError):
        make_needle_batch(np.random.default_rng(0), 1, 3)


def test_phonebook_exact_multi_token_copy():
    rng = np.random.default_rng(1)
    x, t, m = make_phonebook_batch(rng, 3, n_entries=5, code_len=4)
    assert m.sum() == 3 * 4                                # code_len supervised per seq
    # every supervised target is a digit token, and the oracle copies the whole code
    assert (t[m.astype(bool)] >= 128).all()               # digit range starts at n_names
    assert probe_accuracy(_oracle_logits(x, t, phonebook_vocab()), t, m) == 1.0


def test_phonebook_rejects_too_many_entries():
    with pytest.raises(ValueError):
        make_phonebook_batch(np.random.default_rng(0), 1, n_entries=200, n_names=128)


def test_fewshot_recall():
    rng = np.random.default_rng(2)
    x, t, m = make_fewshot_copy_batch(rng, 4, n_shots=5)
    assert x.shape == (4, 2 * 5 + 2) and m.sum() == 4
    assert (t[m.astype(bool)] >= 64).all()                # answers in [n_questions, ..)
    assert probe_accuracy(_oracle_logits(x, t, fewshot_vocab()), t, m) == 1.0


def test_run_probes_structure_and_perfect_oracle():
    # A forward that perfectly recalls: returns one-hot logits at the *input* positions'
    # required targets is impossible without the targets, so use a chance model and just
    # assert the harness shape + value ranges.
    V = 512

    def chance_forward(x):
        return np.zeros((x.shape[0], x.shape[1], V))

    res = run_probes(chance_forward, batch_size=8, seed=3,
                     needle_lengths=(16, 64), phonebook_entries=(8,), n_shots=5)
    assert set(res) == {"needle", "phonebook", "fewshot"}
    assert set(res["needle"]) == {16, 64} and set(res["phonebook"]) == {8}
    for curve in res.values():
        for acc in curve.values():
            assert 0.0 <= acc <= 1.0
    assert "needle" in format_probe_table(res)
