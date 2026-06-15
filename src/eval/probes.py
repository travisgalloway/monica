"""Copying / retrieval / long-context probes (#79).

Pure SSMs lag Transformers on exact copying and in-context retrieval, and the gap widens
with context length / number of pairs (a fixed-width recurrent state is a capacity
bottleneck). These probes measure exactly that, so the **hybrid attention fraction** can be
tuned — raise it if retrieval lags, lower it for speed. First exercised at the 100M smoke
gate (#81), then re-run on each Phase-5 tier (#75).

Three probes, all framed as the standard capacity tasks (like the MQAR probe in
``retrieval_probe.py``) with **disjoint synthetic id ranges**, so each is an architecture
probe runnable on a fresh model and scored by an oracle:

* **needle-in-a-haystack** — one (key,value) needle buried in `context_len` filler tokens,
  re-queried at the end (sweep `context_len` for the long-context curve).
* **phonebook** — N (name, multi-digit code) entries; query a name and **exactly copy** its
  whole code (the code-copying analog; multi-token exact copy).
* **few-shot copy (5-shot MMLU-style)** — K labeled demos, then re-query one; recall its label.

ABOVE THE SEAM — pure numpy, no backend. Each returns an (inputs, targets, mask) 3-tuple
matching the SFTLoader contract, scored by ``recall_accuracy`` (re-exported as
``probe_accuracy``).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .retrieval_probe import recall_accuracy

#: Same scorer as MQAR — argmax accuracy over the supervised (mask=1) positions.
probe_accuracy = recall_accuracy

Batch = Tuple[np.ndarray, np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# Needle in a haystack
# --------------------------------------------------------------------------- #
def needle_vocab(n_filler: int = 128, n_keys: int = 64, n_values: int = 64) -> int:
    return n_filler + n_keys + n_values


def make_needle_batch(rng: np.random.Generator, batch_size: int, context_len: int, *,
                      n_filler: int = 128, n_keys: int = 64, n_values: int = 64) -> Batch:
    """One (key,value) needle planted at a random depth in `context_len` filler tokens,
    re-queried at the end. ids: filler [0,n_filler); keys [n_filler,..); values [..,..)."""
    if context_len < 4:
        raise ValueError("context_len must be >= 4")
    key_base, val_base = n_filler, n_filler + n_keys
    L = context_len
    inputs = rng.integers(0, n_filler, size=(batch_size, L)).astype(np.int64)   # haystack
    targets = np.zeros((batch_size, L), dtype=np.int64)
    mask = np.zeros((batch_size, L), dtype=np.int64)
    for b in range(batch_size):
        key = key_base + int(rng.integers(0, n_keys))
        val = val_base + int(rng.integers(0, n_values))
        p = int(rng.integers(0, L - 3))            # needle at p (key), p+1 (value)
        inputs[b, p], inputs[b, p + 1] = key, val
        q = L - 2                                   # query re-presents the key at the end
        inputs[b, q], inputs[b, q + 1] = key, val   # teacher-forced value at q+1
        targets[b, q], mask[b, q] = val, 1          # predict the value from the queried key
    return inputs, targets, mask


# --------------------------------------------------------------------------- #
# Phonebook (multi-token exact copy)
# --------------------------------------------------------------------------- #
def phonebook_vocab(n_names: int = 128, n_digits: int = 10) -> int:
    return n_names + n_digits


def make_phonebook_batch(rng: np.random.Generator, batch_size: int, n_entries: int, *,
                         n_names: int = 128, n_digits: int = 10, code_len: int = 4,
                         n_queries: int = 1) -> Batch:
    """`n_entries` (name, code) rows where code is `code_len` digit tokens; then query a
    name and exactly copy its whole code. ids: names [0,n_names); digits [n_names,..)."""
    if n_entries > n_names:
        raise ValueError(f"n_entries={n_entries} cannot exceed n_names={n_names}")
    digit_base = n_names
    entry_len = 1 + code_len
    L = n_entries * entry_len + n_queries * entry_len
    inputs = np.zeros((batch_size, L), dtype=np.int64)
    targets = np.zeros((batch_size, L), dtype=np.int64)
    mask = np.zeros((batch_size, L), dtype=np.int64)
    for b in range(batch_size):
        names = rng.choice(n_names, size=n_entries, replace=False)
        codes = rng.integers(0, n_digits, size=(n_entries, code_len)) + digit_base
        for i in range(n_entries):
            o = i * entry_len
            inputs[b, o] = names[i]
            inputs[b, o + 1:o + 1 + code_len] = codes[i]
        qsel = rng.integers(0, n_entries, size=n_queries)
        base = n_entries * entry_len
        for j, qi in enumerate(qsel):
            o = base + j * entry_len
            inputs[b, o] = names[qi]
            inputs[b, o + 1:o + 1 + code_len] = codes[qi]      # teacher-forced code
            # positions [name, d0..d_{code_len-2}] predict [d0..d_{code_len-1}] — exact copy
            for c in range(code_len):
                targets[b, o + c] = codes[qi][c]
                mask[b, o + c] = 1
    return inputs, targets, mask


# --------------------------------------------------------------------------- #
# Few-shot copy (5-shot MMLU-style in-context recall)
# --------------------------------------------------------------------------- #
def fewshot_vocab(n_questions: int = 64, n_answers: int = 4) -> int:
    return n_questions + n_answers


def make_fewshot_copy_batch(rng: np.random.Generator, batch_size: int, n_shots: int = 5, *,
                            n_questions: int = 64, n_answers: int = 4) -> Batch:
    """`n_shots` (question, answer) demos, then re-query one question; recall its answer.
    ids: questions [0,n_questions); answers [n_questions,..)."""
    if n_shots > n_questions:
        raise ValueError(f"n_shots={n_shots} cannot exceed n_questions={n_questions}")
    ans_base = n_questions
    L = 2 * n_shots + 2
    inputs = np.zeros((batch_size, L), dtype=np.int64)
    targets = np.zeros((batch_size, L), dtype=np.int64)
    mask = np.zeros((batch_size, L), dtype=np.int64)
    for b in range(batch_size):
        qs = rng.choice(n_questions, size=n_shots, replace=False)
        ans = rng.integers(0, n_answers, size=n_shots) + ans_base
        inputs[b, 0:2 * n_shots:2] = qs
        inputs[b, 1:2 * n_shots:2] = ans
        sel = int(rng.integers(0, n_shots))
        qp = 2 * n_shots
        inputs[b, qp], inputs[b, qp + 1] = qs[sel], ans[sel]   # teacher-forced answer
        targets[b, qp], mask[b, qp] = ans[sel], 1
    return inputs, targets, mask


# --------------------------------------------------------------------------- #
# Harness: run all probes, return per-setting accuracy curves
# --------------------------------------------------------------------------- #
def run_probes(forward: Callable[[np.ndarray], object], *, to_numpy=np.asarray,
               batch_size: int = 16, seed: int = 0,
               needle_lengths: Sequence[int] = (64, 256, 1024),
               phonebook_entries: Sequence[int] = (16, 64),
               n_shots: int = 5) -> Dict[str, Dict[int, float]]:
    """Run every probe through `forward` (a model.forward-style callable) and return
    ``{probe: {setting: accuracy}}``. Per-context-length needle curves let the attention
    fraction be tuned. The probe vocab (see the ``*_vocab`` helpers) must fit the model."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[int, float]] = {"needle": {}, "phonebook": {}, "fewshot": {}}
    for L in needle_lengths:
        x, t, m = make_needle_batch(rng, batch_size, L)
        out["needle"][L] = probe_accuracy(to_numpy(forward(x)), t, m)
    for n in phonebook_entries:
        x, t, m = make_phonebook_batch(rng, batch_size, n)
        out["phonebook"][n] = probe_accuracy(to_numpy(forward(x)), t, m)
    x, t, m = make_fewshot_copy_batch(rng, batch_size, n_shots)
    out["fewshot"][n_shots] = probe_accuracy(to_numpy(forward(x)), t, m)
    return out


def format_probe_table(results: Dict[str, Dict[int, float]]) -> str:
    """One line per probe/setting: ``needle  len=1024  acc=0.123``."""
    lines = []
    for probe, curve in results.items():
        for setting, acc in curve.items():
            lines.append(f"{probe:10s} setting={setting:<6d} acc={acc:.3f}")
    return "\n".join(lines)
