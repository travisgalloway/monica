"""Shared generation core: prompt ids -> sampled continuation ids (portable).

One generation loop, two consumers: the CLI (`scripts/generate.py`) streams the
decoded tokens to a user, and the lm-eval adapter (`src/eval/olmes_adapter.py`)
collects them for generative tasks. It drives the model through two `SessionStore`
primitives only — `prefill(session_id, prompt_ids)` for the prompt (#165) and
`step(session_id, token)` for each decoded token — so it adds no state handling of its
own.

Above the seam: only numpy + the `SessionStore` API. Backend logits are converted to
numpy via the injected `to_numpy` (as in `src/eval/val_loss.py`); the `sampler`
chooses the next id; `stop_fn` lets a caller end generation on a decoded stop string
without baking a tokenizer in here; `on_token` streams ids as they are produced.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np


def generate(
    store,
    session_id: str,
    prompt_ids: Sequence[int],
    *,
    sampler: Callable[..., int],
    to_numpy: Callable[[object], np.ndarray] = np.asarray,
    max_new_tokens: int = 128,
    eos_id: Optional[int] = None,
    stop_fn: Optional[Callable[[List[int]], bool]] = None,
    on_token: Optional[Callable[[int], None]] = None,
    pass_context: bool = False,
) -> List[int]:
    """Generate up to `max_new_tokens` continuation ids for `session_id`.

    Prefill: `store.prefill` consumes the whole prompt in ONE parallel scan and returns
    the last position's logits, which seed the first sample (#165) — replacing the old
    one-`step`-per-prompt-token loop. That makes `session_id` a **fresh** session's id:
    prefill seeds attention RoPE from position 0, so it may not extend a session that has
    already consumed tokens (every caller in the tree already does `store.create(sid)`
    immediately before). Then loop: sample the next id, record/stream it, feed it back
    through `store.step`, and stop on `eos_id`, on reaching `max_new_tokens`, or when
    `stop_fn(generated)` is True. `prompt_ids` must be non-empty (the recurrence needs a
    token to advance on). Returns only the generated ids (not the prompt).

    `pass_context=True` calls `sampler(logits, previous_tokens=prompt + generated)` so a
    repetition-aware sampler can penalize already-emitted tokens; the default keeps the
    bare `sampler(logits)` contract the lm-eval adapter relies on.
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")

    prompt = [int(t) for t in prompt_ids]
    logits = store.prefill(session_id, prompt)

    generated: List[int] = []
    for _ in range(max_new_tokens):
        row = to_numpy(logits)[0]  # (1, vocab) -> (vocab,)
        nxt = sampler(row, previous_tokens=prompt + generated) if pass_context \
            else sampler(row)
        if eos_id is not None and nxt == eos_id:
            break
        generated.append(nxt)
        if on_token is not None:
            on_token(nxt)
        # Feed the emitted token back BEFORE the stop check so the session state always
        # reflects every id in `generated` (a caller can resume generation in-session).
        logits = store.step(session_id, nxt)
        if stop_fn is not None and stop_fn(generated):
            break

    return generated
