"""Tests for #226's constrained-decode seam: `sample(allowed_ids=...)`
(`src/serve/sampling.py`) and the prefix-constraint mechanism
(`src/serve/constrained.py`). Binary-free — pure numpy/stdlib.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.serve.constrained import (VocabTrie, allowed_extensions,
                                    build_vocab_table)
from src.serve.sampling import sample


# --------------------------------------------------------------------------- #
# sample(allowed_ids=...)
# --------------------------------------------------------------------------- #

def test_allowed_ids_none_is_byte_identical_to_today():
    logits = np.array([1.0, 5.0, 2.0, 8.0, 0.5], dtype=np.float32)
    a = sample(logits, temperature=0.7, rng=np.random.default_rng(42))
    b = sample(logits, temperature=0.7, rng=np.random.default_rng(42), allowed_ids=None)
    assert a == b


def test_allowed_ids_empty_raises_value_error():
    logits = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        sample(logits, allowed_ids=[])


def test_allowed_ids_forces_the_lowest_logit_token_greedy():
    logits = np.array([10.0, 9.0, 8.0, -100.0], dtype=np.float32)
    tok = sample(logits, temperature=0.0, allowed_ids=[3])
    assert tok == 3


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_allowed_ids_forces_the_lowest_logit_token_sampled(seed):
    logits = np.array([10.0, 9.0, 8.0, -100.0], dtype=np.float32)
    tok = sample(logits, temperature=1.0, rng=np.random.default_rng(seed), allowed_ids=[3])
    assert tok == 3


def test_out_of_range_allowed_ids_are_dropped():
    logits = np.array([10.0, 9.0, 8.0], dtype=np.float32)
    tok = sample(logits, temperature=0.0, allowed_ids=[1, 99, -5])
    assert tok == 1


def test_all_finite_fallback_stays_within_allowed_ids_when_repetition_bans_the_rest():
    # allowed_ids=[0, 1] is a valid, non-empty constraint set, but
    # no_repeat_ngram_size=1 (bans every previously-seen token) bans both of them
    # via previous_tokens=[0, 1] -- so every logit ends up -inf and the "all
    # non-finite" fallback fires. It must draw from {0, 1} only, never token 2
    # (which was never in allowed_ids), even though 2 has the highest raw logit.
    logits = np.array([1.0, 2.0, 100.0], dtype=np.float32)
    rng = np.random.default_rng(3)
    seen = set()
    for _ in range(50):
        tok = sample(logits, temperature=1.0, rng=rng, allowed_ids=[0, 1],
                     previous_tokens=[0, 1], no_repeat_ngram_size=1)
        seen.add(tok)
    assert seen <= {0, 1}


def test_duplicate_allowed_ids_do_not_bias_the_fallback_draw():
    # #285: the fallback draws uniformly over the DISTINCT allowed ids. Listing 7
    # three times must not make it 3x as likely as 9. Same fallback recipe as the
    # test above: no_repeat_ngram_size=1 with previous_tokens covering the whole
    # allowed set bans every remaining logit, so the all-non-finite fallback fires.
    logits = np.zeros(12, dtype=np.float32)
    rng = np.random.default_rng(11)
    draws = [sample(logits, temperature=1.0, rng=rng, allowed_ids=[7, 7, 7, 9],
                    previous_tokens=[7, 9], no_repeat_ngram_size=1)
             for _ in range(2000)]
    assert set(draws) == {7, 9}
    for tok in (7, 9):
        share = draws.count(tok) / len(draws)
        # Pre-fix 7 takes ~75%; post-fix ~50%. At n=2000, p=0.5 this band is ~4.5 sigma.
        assert 0.4 <= share <= 0.6, f"id {tok} drawn {share:.3f} of the time"


def test_all_duplicates_allowed_ids_draws_the_single_candidate():
    # Every entry is the same id -> exactly one candidate after the dedupe, so the
    # fallback returns it deterministically.
    logits = np.zeros(8, dtype=np.float32)
    rng = np.random.default_rng(5)
    for _ in range(50):
        tok = sample(logits, temperature=1.0, rng=rng, allowed_ids=[5, 5, 5],
                     previous_tokens=[5], no_repeat_ngram_size=1)
        assert tok == 5


def test_duplicate_out_of_range_allowed_ids_still_raise():
    # Dedupe must not weaken the all-out-of-range guard: {99, -1} is still empty
    # after the range filter, so this raises rather than narrowing silently.
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="no ids within"):
        sample(logits, temperature=0.0, allowed_ids=[99, 99, -1])


# --------------------------------------------------------------------------- #
# ordering: mask first, then repetition penalty, then temperature/top-k/top-p
# --------------------------------------------------------------------------- #

def test_repetition_penalty_cannot_resurrect_a_masked_token():
    # id 2 has the highest logit and is already "seen" -- repetition_penalty would
    # normally suppress it, but it's not in allowed_ids at all, so it must never
    # win regardless of penalty direction.
    logits = np.array([1.0, 2.0, 100.0], dtype=np.float32)
    tok = sample(logits, temperature=0.0, allowed_ids=[0, 1],
                 previous_tokens=[0], repetition_penalty=2.0)
    assert tok != 2
    assert tok in (0, 1)


def test_top_p_renormalizes_over_survivors_only():
    # Without masking, id 0 dominates the distribution so heavily that top_p=0.99
    # keeps only it. With id 0 masked out, the same top_p must be computed over
    # the SURVIVING mass (ids 1, 2), not the original full distribution.
    logits = np.array([100.0, 1.0, 1.0], dtype=np.float32)
    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(50):
        tok = sample(logits, temperature=1.0, top_p=0.99, rng=rng, allowed_ids=[1, 2])
        seen.add(tok)
    assert seen <= {1, 2}
    assert len(seen) == 2  # both survivors reachable -- top_p wasn't computed pre-mask


# --------------------------------------------------------------------------- #
# VocabTrie / allowed_extensions
# --------------------------------------------------------------------------- #

def _decode_factory(pieces):
    """A tiny fake `decode` for `build_vocab_table`: `pieces[i]` is the standalone
    text of vocab id `i` (independent of context — no anchor sensitivity needed
    for these tests)."""
    def decode(ids):
        return "".join(pieces[i] for i in ids)
    return decode


def test_vocab_trie_prefix_matches_multi_token_labels():
    # vocab: 0="na", 1="me", 2="s", 3="x"
    vocab = build_vocab_table(_decode_factory(["na", "me", "s", "x"]), 4, anchor_ids=())
    trie = VocabTrie(vocab)
    # "name" is spelled by tokens 0 ("na") then 1 ("me") -- prefix_matches("name")
    # should return every token whose STRING is a prefix of "name": "na" (0).
    assert trie.prefix_matches("name") == [0]
    assert trie.prefix_matches("me") == [1]
    assert trie.prefix_matches("zzz") == []


def test_allowed_extensions_returns_exactly_reachable_ids():
    # vocab: 0="na", 1="me", 2="mes", 3="s", 4="zzz"
    vocab = build_vocab_table(_decode_factory(["na", "me", "mes", "s", "zzz"]), 5, anchor_ids=())
    trie = VocabTrie(vocab)
    labels = ["name", "names"]
    # prefix="" -- remaining suffixes are "name" and "names"; reachable first
    # tokens are anything that's a prefix of either: "na" (0).
    allowed = allowed_extensions(trie, labels, "", exit_ids=[99])
    assert allowed == [0]

    # prefix="na" -- remaining suffixes "me" and "mes"; "me" (1) and "mes" (2)
    # are both prefixes of a remaining suffix.
    allowed = allowed_extensions(trie, labels, "na", exit_ids=[99])
    assert set(allowed) == {1, 2}

    # prefix="name" is a COMPLETE label ("name") but also a proper prefix of
    # "names" (remaining suffix "s", token 3) -- both continuation AND exit_ids
    # must be present.
    allowed = allowed_extensions(trie, labels, "name", exit_ids=[99])
    assert set(allowed) == {3, 99}


def test_exit_ids_absent_when_prefix_is_not_a_complete_label():
    vocab = build_vocab_table(_decode_factory(["na", "me"]), 2, anchor_ids=())
    trie = VocabTrie(vocab)
    allowed = allowed_extensions(trie, ["name"], "na", exit_ids=[99])
    assert 99 not in allowed


# --------------------------------------------------------------------------- #
# unprobeable ids
# --------------------------------------------------------------------------- #

def test_unprobeable_ids_excluded_from_vocab_table_and_never_allowed():
    # id 1 decodes to empty text; id 2 decodes to the replacement character --
    # both are unsound and must be recorded as None, never surfaced.
    vocab = build_vocab_table(_decode_factory(["na", "", "�"]), 3, anchor_ids=())
    assert vocab[0] == "na"
    assert vocab[1] is None
    assert vocab[2] is None

    trie = VocabTrie(vocab)
    allowed = allowed_extensions(trie, ["na"], "", exit_ids=[])
    assert 1 not in allowed
    assert 2 not in allowed
