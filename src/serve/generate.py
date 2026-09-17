"""Shared generation core: prompt ids -> sampled continuation ids (portable).

One generation loop, two consumers: the CLI (`scripts/generate.py`) streams the
decoded tokens to a user, and the lm-eval adapter (`src/eval/olmes_adapter.py`)
collects them for generative tasks. It drives the model through two `SessionStore`
primitives only — `prefill(session_id, prompt_ids)` for the prompt (#165) and
`step(session_id, token)` for each decoded token — so it adds no state handling of its
own. `prefill=False` routes the prompt through `step` instead, which is how the stateful
REPL (`scripts/generate.py --interactive`, #305) extends a live or rewound session that
`SessionStore.prefill` refuses.

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
    prefill: bool = True,
    sampler_hook: Optional[Callable[..., Optional[np.ndarray]] | Sequence[Callable[..., Optional[np.ndarray]]]] = None,
    sampler_hooks: Optional[Sequence[Callable[..., Optional[np.ndarray]]]] = None,
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

    `prefill=False` seeds the loop by `step`-ing each prompt id instead. The two paths are
    equivalent by construction — same state, same final logits, gated by
    `src/conformance/prefill_decode_parity.py` — so the default is byte-identical to the
    single-scan path and no existing caller changes behavior. The option exists because a
    **live** session cannot be prefilled: `SessionStore.prefill` is fresh-session-only
    (`src/serve/sessions.py:111`), and a session restored from a rewind snapshot has an
    unknowable token position above the seam (`set_state` marks it `None`), so prefill
    refuses it rather than mis-seeding attention RoPE from position 0. The stateful
    continuation REPL (`scripts/generate.py --interactive`) uses `prefill=False` for every
    turn after the first and after every rewind.

    `pass_context=True` calls `sampler(logits, previous_tokens=prompt + generated)` so a
    repetition-aware sampler can penalize already-emitted tokens; the default keeps the
    bare `sampler(logits)` contract the lm-eval adapter relies on.

    `sampler_hook` / `sampler_hooks` (#200): optional callback(s) executed before each
    sampling step. Each hook receives `(logits, previous_tokens=...)` if `pass_context`
    is True, or `(logits)`. If a hook returns a non-None array, it replaces the logits
    for subsequent hooks and the sampler; returning None leaves logits untouched
    (observational/telemetry hook).
    """
    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")

    prompt = [int(t) for t in prompt_ids]
    if prefill:
        logits = store.prefill(session_id, prompt)
    else:
        for tok_id in prompt:
            logits = store.step(session_id, tok_id)

    hooks: List[Callable] = []
    if sampler_hook is not None:
        if isinstance(sampler_hook, Sequence) and not isinstance(sampler_hook, (str, bytes)):
            hooks.extend(sampler_hook)
        else:
            hooks.append(sampler_hook)
    if sampler_hooks is not None:
        hooks.extend(sampler_hooks)

    generated: List[int] = []
    for _ in range(max_new_tokens):
        row = to_numpy(logits)[0]  # (1, vocab) -> (vocab,)
        for hook in hooks:
            ctx = prompt + generated
            try:
                mod = hook(row, previous_tokens=ctx) if pass_context else hook(row)
            except TypeError:
                mod = hook(row)
            if mod is not None:
                row = np.asarray(mod)
        if pass_context:
            try:
                nxt = sampler(row, previous_tokens=prompt + generated)
            except TypeError:
                nxt = sampler(row)
        else:
            nxt = sampler(row)
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


custom_generate = generate


def create_repair_sampler(
    ban_table: Optional[dict] = None,
    *,
    base_sampler: Optional[Callable[..., int]] = None,
    masker: Optional[object] = None,
    decode_fn: Optional[Callable[[Sequence[int]], str]] = None,
    prompt_text: str = "",
    temperature: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    eos_ids: Optional[set[int]] = None,
) -> Callable[..., int]:
    """Wrap a sampler for validate-and-rollback and constrained decode (#201).

    Attaches to `generate(..., pass_context=True)`. When called with
    `(logits, previous_tokens=prompt + generated)`, it queries `ban_table`
    for tokens banned at this history (setting banned logits to -inf via
    `src/serve/sampling.py::sample(..., banned_ids=...)`) and applies
    `masker.mask_for(...)` when fast-loop completion masking is active.
    """
    from .sampling import sample

    _ban_table = ban_table if ban_table is not None else {}

    def repair_sampler(logits: np.ndarray, previous_tokens: Optional[Sequence[int]] = None) -> int:
        banned = None
        if previous_tokens is not None and _ban_table:
            key = tuple(previous_tokens)
            banned = _ban_table.get(key)
            if not banned:
                for prefix_len in range(len(previous_tokens) + 1):
                    sub_key = tuple(previous_tokens[prefix_len:])
                    if sub_key in _ban_table:
                        banned = _ban_table[sub_key]
                        break

        allowed = None
        if masker is not None and decode_fn is not None and previous_tokens is not None:
            text = prompt_text + decode_fn(previous_tokens)
            allowed = masker.mask_for(text, vocab_size=int(np.asarray(logits).size))
            if allowed is not None and eos_ids:
                allowed = sorted(set(allowed) | eos_ids)

        if base_sampler is not None and banned is None and allowed is None:
            try:
                return base_sampler(logits, previous_tokens=previous_tokens)
            except TypeError:
                return base_sampler(logits)

        return sample(
            logits,
            temperature=temperature,
            rng=rng,
            previous_tokens=previous_tokens,
            allowed_ids=allowed,
            banned_ids=list(banned) if banned else None,
        )

    return repair_sampler
