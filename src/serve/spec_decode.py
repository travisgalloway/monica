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

from typing import List, Sequence


def propose(context: Sequence[int], gamma: int, max_n: int = 8) -> List[int]:
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
