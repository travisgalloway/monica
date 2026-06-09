"""OLMES adapter tests (Milestone 6).

The numpy scoring core is exercised offline with a deterministic FakeModel — in
particular the loglikelihood token-indexing off-by-one (logits[i] predicts
token[i+1]), left-truncation, and the rolling disjoint windows. An MLX
cross-check (skipped where mlx is unavailable) verifies the core against a
naive independent computation on the real backend, and a light lm-eval
integration test (skipped where lm_eval is unavailable) exercises the
TemplateLM shell.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

# Optional backends imported at collection time (module level), matching the
# repo convention: tests/test_import_guard.py deletes mlx/torch from
# sys.modules mid-run, and re-importing a native extension afterwards aborts
# the process — so grab the references before the guard runs and never
# importorskip these inside a test body.
try:
    import mlx.core as mx
except ImportError:  # non-Apple-Silicon host
    mx = None
try:
    import lm_eval
except ImportError:  # eval extra not installed
    lm_eval = None

from src.eval.olmes_adapter import (
    disjoint_rolling_windows,
    make_lm_eval_adapter,
    score_continuation,
)

BIG = 10.0


class FakeModel:
    """Successor model: logits[b, t] = BIG * one_hot((input[b, t] + 1) % V).

    The greedy prediction after token t is exactly t+1 (mod V), so a continuation
    is is_greedy iff every token is its predecessor + 1 — which makes the correct
    off-by-one slice the only one that scores it as greedy.
    """

    def __init__(self, vocab_size=8, seq_len=64):
        self.config = SimpleNamespace(vocab_size=vocab_size, seq_len=seq_len)
        self.last_input = None

    def forward(self, token_batch):
        self.last_input = np.asarray(token_batch)
        succ = (self.last_input + 1) % self.config.vocab_size
        eye = np.eye(self.config.vocab_size, dtype=np.float32)
        return BIG * eye[succ]


def _lp_match(v):
    """log P of the peaked (matching) token: BIG - log(e^BIG + (V-1))."""
    return BIG - math.log(math.exp(BIG) + (v - 1))


def _lp_miss(v):
    """log P of a non-peaked token: 0 - log(e^BIG + (V-1))."""
    return -math.log(math.exp(BIG) + (v - 1))


def test_off_by_one_greedy_continuation():
    m = FakeModel()
    lp, greedy = score_continuation(m, [1, 2, 3], [4, 5], max_length=64)
    # Input must be whole[:-1]: the final continuation token is never fed.
    np.testing.assert_array_equal(m.last_input, [[1, 2, 3, 4]])
    assert greedy is True
    assert np.isclose(lp, 2 * _lp_match(8))


def test_off_by_one_non_greedy():
    m = FakeModel()
    lp, greedy = score_continuation(m, [1, 2, 3], [5, 5], max_length=64)
    assert greedy is False
    # Position 1 predicts 4 (target 5 misses); position 2 then predicts 6 (miss).
    assert np.isclose(lp, 2 * _lp_miss(8))


def test_single_token_continuation():
    m = FakeModel()
    lp, greedy = score_continuation(m, [6], [7], max_length=64)
    np.testing.assert_array_equal(m.last_input, [[6]])
    assert greedy is True
    assert np.isclose(lp, _lp_match(8))
    _, greedy = score_continuation(m, [7], [0], max_length=64)  # wraps mod 8
    assert greedy is True


def test_left_truncation_keeps_continuation():
    m = FakeModel()
    ctx = list(range(8, 18)) * 0 + [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]  # length 10
    lp, greedy = score_continuation(m, ctx, [2, 3], max_length=4)
    assert m.last_input.shape == (1, 4)
    # Identical to scoring with the pre-truncated context (last 3 ctx tokens).
    m2 = FakeModel()
    lp2, greedy2 = score_continuation(m2, ctx[-3:], [2, 3], max_length=4)
    assert (lp, greedy) == (lp2, greedy2)
    assert greedy is True


def test_continuation_fills_max_length():
    # len(cont) == max_length: exactly one context token must survive.
    m = FakeModel()
    lp, greedy = score_continuation(m, [1], [2, 3, 4, 5], max_length=4)
    np.testing.assert_array_equal(m.last_input, [[1, 2, 3, 4]])
    assert greedy is True
    assert np.isclose(lp, 4 * _lp_match(8))


def test_invalid_inputs_raise():
    m = FakeModel()
    with pytest.raises(ValueError):
        score_continuation(m, [], [1], max_length=4)
    with pytest.raises(ValueError):
        score_continuation(m, [1], [], max_length=4)
    with pytest.raises(ValueError):
        score_continuation(m, [1], [2, 3, 4, 5, 6], max_length=4)


def test_disjoint_rolling_windows():
    assert disjoint_rolling_windows(list(range(10)), prefix_token=99, max_length=4) == [
        ([99], [0, 1, 2, 3]),
        ([3], [4, 5, 6, 7]),
        ([5, 6, 7], [8, 9]),  # final short window: context fills to max_length+1
    ]
    # A doc that fits one window: rolling sum == direct prefix-conditioned score.
    m = FakeModel()
    windows = disjoint_rolling_windows([1, 2, 3], prefix_token=0, max_length=8)
    assert windows == [([0], [1, 2, 3])]
    lp_roll = sum(score_continuation(m, c, t, max_length=8)[0] for c, t in windows)
    lp_direct, _ = score_continuation(m, [0], [1, 2, 3], max_length=8)
    assert np.isclose(lp_roll, lp_direct)


@pytest.mark.skipif(mx is None, reason="mlx unavailable")
def test_score_continuation_mlx_cross_check():
    """Core vs a naive independent computation on the real MLX backend."""
    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel

    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    model = MLXMambaModel(cfg)
    np_to = lambda a: np.array(a)

    rng = np.random.default_rng(0)
    ctx = rng.integers(0, cfg.vocab_size, size=12).tolist()
    cont = rng.integers(0, cfg.vocab_size, size=5).tolist()
    lp, greedy = score_continuation(model, ctx, cont,
                                    max_length=cfg.seq_len, to_numpy=np_to)

    # Naive reference: full forward, log-softmax everywhere, explicit indexing.
    whole = ctx + cont
    logits = np_to(model.forward(np.asarray(whole[:-1])[None, :]))[0].astype(np.float64)
    m = logits.max(axis=-1, keepdims=True)
    lsm = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
    ref = sum(lsm[len(ctx) - 1 + j, whole[len(ctx) + j]] for j in range(len(cont)))
    ref_greedy = all(int(logits[len(ctx) - 1 + j].argmax()) == whole[len(ctx) + j]
                     for j in range(len(cont)))
    assert np.isclose(lp, ref, rtol=1e-5, atol=1e-5)
    assert greedy == ref_greedy


@pytest.mark.skipif(lm_eval is None, reason="lm_eval unavailable")
def test_lm_eval_adapter_integration():
    """TemplateLM shell over FakeModel + ByteTokenizer (skipped without lm_eval)."""
    from lm_eval.api.instance import Instance

    from src.data.tokenize import ByteTokenizer

    model = FakeModel(vocab_size=256, seq_len=32)
    lm = make_lm_eval_adapter(model, ByteTokenizer())
    assert lm.max_length == 32
    assert lm.eot_token_id == 0  # ByteTokenizer has no eos -> NUL byte

    reqs = [Instance(request_type="loglikelihood", doc={}, arguments=(c, t), idx=i)
            for i, (c, t) in enumerate([("ab", "c"), ("ab", " longer cont")])]
    out = lm.loglikelihood(reqs, disable_tqdm=True)
    assert len(out) == 2
    assert all(isinstance(lp, float) and isinstance(g, bool) for lp, g in out)

    rolling = lm.loglikelihood_rolling(
        [Instance(request_type="loglikelihood_rolling", doc={},
                  arguments=("hello rolling world, longer than one window",),
                  idx=0)],
        disable_tqdm=True)
    assert len(rolling) == 1 and isinstance(rolling[0], float)

    with pytest.raises(NotImplementedError):
        lm.generate_until([], disable_tqdm=True)

    # Our hand-rolled windows match lm_eval's reference utilities.
    from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

    toks = list(range(10))
    ref = [make_disjoint_window(w) for w in get_rolling_token_windows(
        toks, prefix_token=99, max_seq_len=4, context_len=1)]
    ref = [(list(c), list(t)) for c, t in ref]
    assert disjoint_rolling_windows(toks, prefix_token=99, max_length=4) == ref
