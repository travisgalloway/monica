"""Self-speculative decoding primitives (#52) — portable, numpy only.

Draft-and-verify decoding accelerates autoregressive generation: a cheap drafter
proposes the next few tokens, the real model verifies them in one batched pass, and
verification keeps the output identical to plain decoding. This module holds the two
BACKEND-FREE pieces — the drafter and the accept rule — so they are testable without mlx.
The stateful verifier pass and the timing live in `scripts/spec_decode.py` (it advances
the model's `step` recurrence on the backend).

GREEDY ONLY: the accept rule (`first_mismatch`) compares draft tokens against the
verifier's *argmax*, so the preserved output is the GREEDY decode. It is NOT
distribution-preserving for temperature>0 / top-p sampling — that needs the Leviathan
et al. rejection-sampling rule, which this does not implement. Drive it greedily.

The drafter here is **prompt-lookup** (a.k.a. n-gram / self-speculative): it proposes
continuations by finding where the current context's tail recurred earlier and copying
what followed. It needs no second trained model — the "self-speculative variant avoids
training a second model" the issue calls for — and any wrong guess is simply rejected by
the verifier, so it can never change the output, only the speed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def propose(context: Sequence[int], gamma: int, max_n: int = 8) -> list[int]:
    """Prompt-lookup draft: up to `gamma` tokens continuing `context`.

    Tries the longest tail first: for n from min(max_n, len-1) down to 1, take the last
    n tokens as a pattern and search for its most recent EARLIER occurrence in the
    context; on a hit, copy the up-to-`gamma` tokens that followed it. Returns `[]` when
    no tail recurs (the caller then takes one ordinary step). Longer matched patterns are
    preferred because they predict the continuation more reliably.
    """
    ctx = [int(t) for t in context]
    L = len(ctx)
    if L < 2 or gamma <= 0:
        return []
    for n in range(min(max_n, L - 1), 0, -1):
        pattern = ctx[L - n:]
        # Most recent earlier occurrence: scan start positions right-to-left, excluding
        # the tail occurrence itself (search space is ctx[: L - n]).
        for start in range(L - n - 1, -1, -1):
            if ctx[start:start + n] == pattern:
                draft = ctx[start + n:start + n + gamma]
                if draft:
                    return draft
                break  # this pattern recurs only at the very end — try a shorter one
    return []


def first_mismatch(draft: Sequence[int], verifier_preds: Sequence[int]) -> int:
    """Number of leading draft tokens the verifier agrees with (greedy acceptance).

    `verifier_preds[i]` is the verifier's greedy next token at position i given the
    accepted prefix `draft[:i]`. The accepted count is the first i where they differ
    (or `len(draft)` if all agree) — exactly the prefix plain greedy decoding would also
    have produced, which is what makes speculative decoding distribution-preserving.
    """
    m = 0
    for d, p in zip(draft, verifier_preds):
        if int(d) != int(p):
            break
        m += 1
    return m


def prune_draft_trajectory(
    context: Sequence[int],
    draft: Sequence[int],
    critic_evaluator: Any,
    *,
    threshold: float = 0.70,
    model: Any | None = None,
    to_numpy: Callable[[Any], np.ndarray] = np.asarray,
) -> tuple[list[int], dict[str, Any]]:
    """Evaluate intermediate draft tokens using the critic head to abort flawed draft trajectories early (#388).

    At each intermediate position j in the draft, evaluates P(clean) for context + draft[:j].
    If P(clean) < threshold, the trajectory is aborted at position j-1, pruning all subsequent
    draft tokens before external verifier compute is spent.

    Args:
        context: Prior token sequence.
        draft: Proposed draft tokens from propose().
        critic_evaluator: Callable, DecisionCriticHead, or pre-computed probabilities.
        threshold: P(clean) acceptance threshold (default 0.70).
        model: Optional model providing forward_hidden or forward_with_critics.
        to_numpy: Backend-to-numpy conversion callable.

    Returns:
        pruned_draft: Truncated draft tokens prefix that passed the critic filter.
        telemetry: Dict with critic_latency_ms, tokens_evaluated, tokens_aborted,
                   abort_position, and probs.
    """
    if not draft:
        return [], {
            "critic_latency_ms": 0.0,
            "tokens_evaluated": 0,
            "tokens_aborted": 0,
            "abort_position": None,
            "probs": [],
        }

    t0 = time.perf_counter()
    probs: list[float] = []
    abort_pos: int | None = None
    pruned_draft: list[int] = list(draft)
    tokens_aborted: int = 0
    ctx = list(context)

    for j in range(1, len(draft) + 1):
        cand = ctx + list(draft[:j])
        prob = 0.5
        if callable(critic_evaluator):
            try:
                res = critic_evaluator(cand)
                prob = float(res.get("prob", res)) if isinstance(res, dict) else float(res)
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                prob = 0.5
        elif isinstance(critic_evaluator, (list, tuple)):
            idx = j - 1
            prob = float(critic_evaluator[idx]) if idx < len(critic_evaluator) else 0.5
        elif model is not None and hasattr(model, "forward_with_critics"):
            try:
                batch = np.array([cand], dtype=np.int32)
                _, critic_out = model.forward_with_critics(batch, critic_names=["noul"])
                prob = float(critic_out["noul"]["prob"])
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                prob = 0.5
        elif model is not None and hasattr(model, "forward_hidden"):
            try:
                batch = np.array([cand], dtype=np.int32)
                h_seq = model.forward_hidden(batch)
                h_last = to_numpy(h_seq)[0, -1, :]
                head = critic_evaluator or getattr(model, "critic_heads", {}).get("noul")
                if head is not None:
                    res = head.predict(h_last) if hasattr(head, "predict") else head.predict_noul(h_last)
                    prob = float(res.get("prob", 0.5))
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                prob = 0.5
        elif critic_evaluator is not None and hasattr(critic_evaluator, "predict_noul"):
            try:
                # If dummy or pre-computed hidden states are available
                res = critic_evaluator.predict_noul(np.zeros((1, getattr(critic_evaluator.config, 'd_model', 64)), dtype=np.float32))
                prob = float(res.get("prob", 0.5))
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                prob = 0.5

        probs.append(prob)
        if prob < threshold:
            abort_pos = j - 1
            pruned_draft = list(draft[:j - 1])
            tokens_aborted = len(draft) - len(pruned_draft)
            break

    latency_ms = (time.perf_counter() - t0) * 1000.0
    telemetry = {
        "critic_latency_ms": latency_ms,
        "tokens_evaluated": len(probs),
        "tokens_aborted": tokens_aborted,
        "abort_position": abort_pos,
        "probs": probs,
    }
    return pruned_draft, telemetry
