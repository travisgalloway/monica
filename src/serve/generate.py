"""Shared generation core: prompt ids -> sampled continuation ids (portable).

One generation loop, two consumers: the CLI (`scripts/generate.py`) streams the
decoded tokens to a user, and the lm-eval adapter (`src/eval/olmes_adapter.py`)
collects them for generative tasks. It drives the model purely through
`SessionStore.step(session_id, token) -> logits` — the proven seam primitive — so it
adds no new state handling.

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
    sampler: Callable[[np.ndarray], int],
    to_numpy: Callable[[object], np.ndarray] = np.asarray,
    max_new_tokens: int = 128,
    eos_id: Optional[int] = None,
    stop_fn: Optional[Callable[[List[int]], bool]] = None,
    on_token: Optional[Callable[[int], None]] = None,
) -> List[int]:
    """Generate up to `max_new_tokens` continuation ids for `session_id`.

    Prefill: feed every prompt id through `store.step` (the last one's logits seed the
    first sample). Then loop: sample the next id, record/stream it, feed it back, and
    stop on `eos_id`, on reaching `max_new_tokens`, or when `stop_fn(generated)` is
    True. `prompt_ids` must be non-empty (the recurrence needs a token to advance on).
    Returns only the generated ids (not the prompt).
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")

    logits = None
    for tok in prompt_ids:
        logits = store.step(session_id, int(tok))

    generated: List[int] = []
    for _ in range(max_new_tokens):
        nxt = sampler(to_numpy(logits)[0])  # (1, vocab) -> (vocab,)
        if eos_id is not None and nxt == eos_id:
            break
        generated.append(nxt)
        if on_token is not None:
            on_token(nxt)
        if stop_fn is not None and stop_fn(generated):
            break
        logits = store.step(session_id, nxt)

    return generated
