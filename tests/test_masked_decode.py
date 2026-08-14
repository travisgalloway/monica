"""Tests for `src/lsp/masked_decode.py`'s `generate_masked` — the #226 decode
loop. Binary-free — a tiny fixed-logit `FakeLM` stands in for a real backend.
"""

from __future__ import annotations

import numpy as np

from src.lsp.completion_mask import CompletionMasker
from src.lsp.harness import generate_baseline
from src.lsp.masked_decode import generate_masked


class _FakeLM:
    """Fixed logits every step, independent of what's been generated — enough to
    make two independently-constructed instances produce IDENTICAL sequences
    given the same rng seed and call order (the `masker=None` control test), and
    to exercise masking's effect on an otherwise-uniform preference (the
    confinement test)."""

    def __init__(self, alphabet, logits):
        self._alphabet = list(alphabet)
        self._logits = np.asarray(logits, dtype=np.float32)
        self.n_forward_tokens = 0
        self.n_forward_tokens_nocache = 0

    def encode(self, text):
        return [self._alphabet.index(c) for c in text]

    def decode(self, token_ids):
        return "".join(self._alphabet[i] for i in token_ids)

    def reset(self, context):
        self.n_forward_tokens += 1
        return self._logits.copy()

    def step(self, token_id):
        self.n_forward_tokens += 1
        return self._logits.copy()

    def rollback(self, n_tokens):
        pass


_ALPHA = ["n", "a", "m", "e", ";", "x", "y", ".", "z", "w"]
_LOGITS = [1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 1.0]  # id 5 ('x') dominates


class _FixedLabelSource:
    def __init__(self, labels):
        self.labels = labels
        self.n_queries = 0

    def query(self, path, text, anchor_offset):
        self.n_queries += 1
        return list(self.labels)


# --------------------------------------------------------------------------- #
# masker=None is byte-identical to generate_baseline
# --------------------------------------------------------------------------- #

def test_generate_masked_none_matches_generate_baseline():
    lm1 = _FakeLM(_ALPHA, _LOGITS)
    lm2 = _FakeLM(_ALPHA, _LOGITS)
    prompt = "y"
    r1 = generate_baseline(lm1, prompt, budget="block", block_size=6, temperature=0.9,
                           rng=np.random.default_rng(99))
    r2 = generate_masked(lm2, None, prompt, budget="block", block_size=6, temperature=0.9,
                         rng=np.random.default_rng(99))
    assert r1.completion == r2.completion
    assert r1.n_generated_tokens == r2.n_generated_tokens
    assert r1.checkpoints == r2.checkpoints
    # masker=None never touches the #226 counters
    assert r2.n_mask_steps == 0
    assert r2.n_completion_calls == 0


# --------------------------------------------------------------------------- #
# masker active: confinement + bypass
# --------------------------------------------------------------------------- #

def test_generate_masked_confines_output_to_the_label_despite_model_preference():
    # Unmasked, greedy always picks 'x' (id 5, dominant logit) -- but the label
    # "name" forces the model down a completely different path.
    lm = _FakeLM(_ALPHA, _LOGITS)
    source = _FixedLabelSource(["name"])
    masker = CompletionMasker(source, "f.ts", lm.decode, mask_scope="member")

    result = generate_masked(lm, masker, "u.", budget="block", block_size=5, temperature=0.0)

    assert result.completion == "name;"
    assert source.n_queries == 1              # one query for the whole span
    assert masker.n_completion_calls == 1
    assert masker.n_mask_bypass == 0
    assert result.n_completion_calls == 1
    assert result.n_mask_bypass == 0
    assert result.n_mask_steps == masker.n_mask_steps


def test_generate_masked_bypasses_and_counts_when_labels_are_empty():
    # The label source found nothing valid here (a real, legitimate LSP outcome)
    # -- the masker must degrade to unmasked rather than crash or feed sample()
    # an empty allowed_ids, and generation must continue.
    lm = _FakeLM(_ALPHA, _LOGITS)
    source = _FixedLabelSource([])
    masker = CompletionMasker(source, "f.ts", lm.decode, mask_scope="member")

    result = generate_masked(lm, masker, "u.", budget="block", block_size=3, temperature=0.0)

    assert source.n_queries == 1
    assert masker.n_mask_bypass >= 1
    assert len(result.completion) > 0
    # unmasked greedy always prefers 'x' -- confirms generation genuinely
    # continued unconstrained rather than stalling
    assert result.completion == "xxx"
