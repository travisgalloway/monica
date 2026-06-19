"""Tests for self-speculative decoding (#52).

Portable: the prompt-lookup drafter and the greedy accept rule. MLX-guarded: the
batched `verify_block` matches sequential `step`, and the full speculative loop is
byte-identical to plain greedy decoding (the distribution-preserving guarantee).
"""

import numpy as np
import pytest

from src.serve.spec_decode import first_mismatch, propose

try:
    import mlx.core as mx
    HAVE_MLX = True
except ImportError:                     # portable drafter/accept tests must run without mlx
    mx = None
    HAVE_MLX = False

requires_mlx = pytest.mark.skipif(not HAVE_MLX, reason="requires mlx (Apple Silicon)")


# --------------------------------------------------------------------------- #
# Portable: drafter
# --------------------------------------------------------------------------- #
def test_propose_copies_after_recent_match():
    # tail [1, 2] recurred earlier, followed by [3, 4]
    ctx = [1, 2, 3, 4, 9, 9, 1, 2]
    assert propose(ctx, gamma=2, max_n=8) == [3, 4]


def test_propose_prefers_longer_pattern():
    # both [2] and [1, 2] recur; the longer match [1,2]->[3] should win
    ctx = [1, 2, 3, 7, 2, 5, 1, 2]
    assert propose(ctx, gamma=1, max_n=8) == [3]


def test_propose_limited_by_gamma_and_returns_empty_when_no_match():
    assert propose([1, 2, 3, 1, 2, 3], gamma=2, max_n=8) == [1, 2]  # tail [1,2,3] recurs at 0
    assert propose([5, 6, 7, 8], gamma=2, max_n=8) == []            # no tail recurs
    assert propose([1], gamma=2, max_n=8) == []                     # too short


def test_propose_respects_max_n_window():
    ctx = list(range(10)) + [3, 4]          # tail [3,4]; earlier [3,4] at index 3
    assert propose(ctx, gamma=2, max_n=8) == [5, 6]


# --------------------------------------------------------------------------- #
# Portable: accept rule
# --------------------------------------------------------------------------- #
def test_first_mismatch_counts_leading_agreement():
    assert first_mismatch([1, 2, 3], [1, 2, 3]) == 3   # all accepted
    assert first_mismatch([1, 2, 3], [1, 9, 3]) == 1   # mismatch at index 1
    assert first_mismatch([1, 2, 3], [9, 2, 3]) == 0   # immediate mismatch
    assert first_mismatch([], []) == 0


# --------------------------------------------------------------------------- #
# MLX-guarded: verifier + end-to-end equivalence
# (each test skips individually when mlx is absent, so the portable tests above
# still run on a non-Mac host — see the `requires_mlx` marker)
# --------------------------------------------------------------------------- #
def _toy_model():
    from src.model.blocks import MambaConfig
    from src.model.mlx_backend import MLXMambaModel
    cfg = MambaConfig(d_model=32, n_layers=2, head_dim=16, d_state=8,
                      vocab_size=32, seq_len=16, precision="fp32")
    mx.random.seed(0)
    return MLXMambaModel(cfg)


@requires_mlx
def test_verify_block_matches_sequential_step():
    model = _toy_model()
    tokens = [3, 7, 1, 4, 9]
    state = model.init_state(1)

    seq_logits = []
    h = state
    for t in tokens:
        logit, h = model.step(mx.array([t]), h)
        seq_logits.append(np.array(logit))

    block_logits, block_states = model.verify_block(tokens, state)
    for a, b in zip(seq_logits, block_logits):
        assert np.allclose(a, np.array(b), atol=1e-5)
    # final state from the block must equal the sequential final state
    final_seq = np.array(h[0][1])           # layer 0 ssm state
    final_blk = np.array(block_states[-1][0][1])
    assert np.allclose(final_seq, final_blk, atol=1e-5)


def _greedy_plain(model, prompt, max_new):
    state = model.init_state(1)
    logits = None
    for t in prompt:
        logits, state = model.step(mx.array([int(t)]), state)
    out = []
    for _ in range(max_new):
        x = int(mx.argmax(logits[0]).item())
        out.append(x)
        logits, state = model.step(mx.array([x]), state)
    return out


@requires_mlx
def test_speculative_decode_equals_plain_greedy():
    import scripts.spec_decode as sd
    model = _toy_model()
    # A structured (repeating) prompt so the prompt-lookup drafter finds matches.
    prompt = [1, 2, 3, 4, 1, 2, 3, 4, 5, 6]
    plain = _greedy_plain(model, prompt, max_new=40)
    spec, _, stats = sd.spec_decode(model, prompt, max_new=40, gamma=4, max_n=8, mx=mx)
    assert spec == plain                    # distribution-preserving (exact, greedy)
    assert stats["rounds"] >= 1


@requires_mlx
def test_speculative_decode_accepts_on_repetitive_text():
    import scripts.spec_decode as sd
    model = _toy_model()
    prompt = [7, 7, 7, 7, 7, 7, 7, 7]       # maximally predictable -> high acceptance
    spec, _, stats = sd.spec_decode(model, prompt, max_new=32, gamma=4, max_n=8, mx=mx)
    assert spec == _greedy_plain(model, prompt, max_new=32)
    assert stats["accept_rate"] > 0.0       # the drafter landed at least some tokens
