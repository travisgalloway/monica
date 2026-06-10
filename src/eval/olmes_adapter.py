"""Tier-2 evaluation: OLMES / lm-evaluation-harness adapter.

OLMES inherits its model abstraction from EleutherAI's lm-evaluation-harness.
This module implements the harness's model class (the loglikelihood-style
methods) over `ModelInterface.forward`, with the same split as `val_loss`: a
pure-numpy scoring core (`score_continuation`, `disjoint_rolling_windows`) that
is testable anywhere, and a thin lm-eval shell built by `make_lm_eval_adapter`.

lm-eval is a heavy optional dependency (and some versions of it pull in torch),
so it is imported ONLY inside the factory — this module stays above the seam
(guarded by tests/test_import_guard.py).

The classic trap here is the loglikelihood token-indexing off-by-one:
`forward` logits at position i predict the token at position i+1, so the model
input is `(ctx + cont)[:-1]` and the continuation is scored by the LAST
`len(cont)` logit rows. See `score_continuation`.

For a 100M model, absolute scores will be poor — judge the harness by whether
it runs end to end (scripts/eval_olmes.py), not by leaderboard position.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..model.interface import ModelInterface


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Stable log-softmax over the last axis (float64 internally)."""
    x = np.asarray(logits, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def score_continuation(
    model: ModelInterface,
    ctx_tokens: Sequence[int],
    cont_tokens: Sequence[int],
    *,
    max_length: int,
    to_numpy=np.asarray,
) -> Tuple[float, bool]:
    """Return (sum log P(cont | ctx), is_greedy) for one context/continuation pair.

    Indexing: with `whole = ctx + cont`, the model sees `whole[:-1]` (the final
    continuation token is never fed — its logit comes from the position before
    it), and `logits[i]` predicts `whole[i + 1]`, so the continuation is scored
    by `logits[-len(cont):]` against targets `whole[-len(cont):]`.

    Inputs longer than `max_length` are left-truncated, always keeping the full
    continuation and at least one context token. Requests run one at a time
    (batch=1); right-padded batching would be a safe future optimization since
    the model is causal, but the POC bar is "runs end to end".
    """
    ctx, cont = list(ctx_tokens), list(cont_tokens)
    if not ctx or not cont:
        raise ValueError("context and continuation must each be non-empty")
    if len(cont) > max_length:
        raise ValueError(f"continuation length {len(cont)} exceeds max_length {max_length}")

    # Keep max_length + 1 tokens so the model input whole[:-1] is <= max_length
    # and the token preceding the first continuation token survives truncation.
    whole = (ctx + cont)[-(max_length + 1):]
    inp = np.asarray(whole[:-1], dtype=np.int64)[None, :]
    logits = to_numpy(model.forward(inp))[0]  # (len(whole)-1, V)

    cont_logits = np.asarray(logits[-len(cont):], dtype=np.float64)
    targets = np.asarray(whole[-len(cont):])
    logprob = float(_log_softmax(cont_logits)[np.arange(len(targets)), targets].sum())
    is_greedy = bool((cont_logits.argmax(axis=-1) == targets).all())
    return logprob, is_greedy


def disjoint_rolling_windows(
    tokens: Sequence[int], prefix_token: int, max_length: int,
) -> List[Tuple[List[int], List[int]]]:
    """Disjoint (ctx, cont) windows covering `tokens`; each token scored once.

    Matches lm-eval's make_disjoint_window(get_rolling_token_windows(...,
    context_len=1)): the first window is conditioned on the prefix (EOT) token,
    full windows on the single preceding token, and the final short window on
    as many preceding tokens as fit — once the document exceeds one window,
    every window's ctx + cont spans exactly max_length + 1 tokens.
    """
    tokens = list(tokens)
    windows = []
    for start in range(0, len(tokens), max_length):
        cont = tokens[start:start + max_length]
        ctx = ([prefix_token] if start == 0
               else tokens[start - (max_length + 1 - len(cont)):start])
        windows.append((ctx, cont))
    return windows


def make_lm_eval_adapter(
    model: ModelInterface,
    tokenizer,
    *,
    max_length: int | None = None,
    to_numpy=np.asarray,
):
    """Build an lm-eval `LM` over `ModelInterface.forward`.

    `tokenizer` is an HF tokenizer (or ByteTokenizer-like: anything with
    `.encode`). `to_numpy` converts backend logits to numpy, as in
    `val_loss.evaluate`. Imports lm_eval (and transitively torch) lazily.
    """
    from lm_eval.api.model import TemplateLM  # lazy: pulls torch

    seq_limit = max_length or model.config.seq_len

    class _MonicaLM(TemplateLM):
        @property
        def eot_token_id(self) -> int:
            eos = getattr(tokenizer, "eos_token_id", None)
            return 0 if eos is None else int(eos)  # ByteTokenizer: NUL byte

        @property
        def max_length(self) -> int:
            return seq_limit

        def tok_encode(self, string: str, **kwargs) -> List[int]:
            try:
                return tokenizer.encode(string, add_special_tokens=False)
            except TypeError:  # ByteTokenizer takes no kwargs
                return tokenizer.encode(string)

        def _loglikelihood_tokens(self, requests, **kwargs):
            return [
                score_continuation(model, ctx_toks, cont_toks,
                                   max_length=seq_limit, to_numpy=to_numpy)
                for _key, ctx_toks, cont_toks in requests
            ]

        def loglikelihood_rolling(self, requests, **kwargs):
            results = []
            for (string,) in (req.args for req in requests):
                results.append(sum(
                    score_continuation(model, ctx, cont,
                                       max_length=seq_limit, to_numpy=to_numpy)[0]
                    for ctx, cont in disjoint_rolling_windows(
                        self.tok_encode(string), self.eot_token_id, seq_limit)
                ))
            return results

        def generate_until(self, requests, **kwargs):
            raise NotImplementedError(
                "generate_until is not implemented: HellaSwag/ARC/PIQA are "
                "loglikelihood (multiple-choice) tasks and never call it. "
                "Implement over ModelInterface.step when generative tasks are "
                "needed.")

    return _MonicaLM()
