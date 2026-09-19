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


class AdaptiveReasoningGate:
    """Adaptive dual-mode reasoning gate for code generation (#362).

    Suppresses <think> tokens during inline FIM completions to minimize latency,
    while permitting structured multi-step reasoning traces (<think> ... </think>)
    bounded by token limits during instruction-driven refactoring and chat.
    """

    def __init__(
        self,
        prompt_ids: Sequence[int],
        *,
        mode: Optional[str] = None,
        fim_prefix_ids: Optional[int | Sequence[int] | set[int]] = (1,),
        think_token_ids: Optional[int | Sequence[int] | set[int]] = None,
        think_close_ids: Optional[int | Sequence[int] | set[int]] = None,
        max_reasoning_tokens: Optional[int] = None,
        decode_fn: Optional[Callable[[Sequence[int]], str]] = None,
    ):
        self.prompt_ids = [int(t) for t in prompt_ids]
        self.max_reasoning_tokens = max_reasoning_tokens
        self.decode_fn = decode_fn

        def _to_set(val):
            if val is None:
                return set()
            if isinstance(val, int):
                return {int(val)}
            return {int(v) for v in val}

        self.fim_prefix_ids = _to_set(fim_prefix_ids)
        self.think_token_ids = _to_set(think_token_ids)
        self.think_close_ids = _to_set(think_close_ids)

        self.is_fim = self._detect_fim()
        if mode is not None:
            self.mode = mode.lower()
        elif self.is_fim:
            self.mode = "direct"
        else:
            self.mode = "reasoning"

        self.in_reasoning = False
        self.reasoning_tokens_count = 0
        self.reasoning_completed = False

    def _detect_fim(self) -> bool:
        """Check whether prompt_ids contains a FIM prefix sentinel."""
        if any(fid in self.prompt_ids for fid in self.fim_prefix_ids):
            return True
        if self.decode_fn is not None:
            try:
                if "<|fim_prefix|>" in self.decode_fn(self.prompt_ids):
                    return True
            except Exception:
                pass
        fim_bytes = list(b"<|fim_prefix|>")
        if len(self.prompt_ids) >= len(fim_bytes):
            for i in range(len(self.prompt_ids) - len(fim_bytes) + 1):
                if self.prompt_ids[i : i + len(fim_bytes)] == fim_bytes:
                    return True
        return False

    def filter_logits(self, logits: np.ndarray, generated: Sequence[int]) -> np.ndarray:
        """Filter logits based on mode and reasoning token limits."""
        row = np.array(logits, copy=True)
        vocab_size = row.size

        # Direct / FIM mode: strictly disallow reasoning tokens (<think>)
        if self.mode == "direct" or self.is_fim:
            for tid in self.think_token_ids:
                if 0 <= tid < vocab_size:
                    row[tid] = -np.inf
            return row

        # Instruction / reasoning mode:
        # If reasoning already finished in this turn, forbid reopening <think>
        if self.reasoning_completed:
            for tid in self.think_token_ids:
                if 0 <= tid < vocab_size:
                    row[tid] = -np.inf

        # If inside reasoning trace and token limit reached, force close or suppress
        if self.in_reasoning and self.max_reasoning_tokens is not None:
            if self.reasoning_tokens_count >= self.max_reasoning_tokens:
                if self.think_close_ids:
                    # Cleanly force </think>
                    mask = np.full(vocab_size, -np.inf, dtype=np.float32)
                    for cid in self.think_close_ids:
                        if 0 <= cid < vocab_size:
                            mask[cid] = 100.0
                    return mask
                else:
                    for tid in self.think_token_ids:
                        if 0 <= tid < vocab_size:
                            row[tid] = -np.inf

        return row

    def record_token(self, token_id: int) -> None:
        """Advance reasoning state upon emitting token_id."""
        if token_id in self.think_token_ids:
            self.in_reasoning = True
        elif token_id in self.think_close_ids:
            self.in_reasoning = False
            self.reasoning_completed = True
        elif self.in_reasoning:
            self.reasoning_tokens_count += 1

    def __call__(self, logits: np.ndarray, previous_tokens: Optional[Sequence[int]] = None) -> np.ndarray:
        generated = (
            previous_tokens[len(self.prompt_ids):]
            if previous_tokens is not None and len(previous_tokens) >= len(self.prompt_ids)
            else []
        )
        return self.filter_logits(logits, generated)


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
    reasoning_mode: Optional[str] = None,
    fim_prefix_id: Optional[int | Sequence[int]] = 1,
    think_token_id: Optional[int | Sequence[int]] = None,
    think_close_id: Optional[int | Sequence[int]] = None,
    max_reasoning_tokens: Optional[int] = None,
    adaptive_reasoning: bool = True,
    decode_fn: Optional[Callable[[Sequence[int]], str]] = None,
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

    `adaptive_reasoning` (#362): dual-mode gate suppressing `<think>` tokens during FIM
    while permitting structured reasoning traces up to `max_reasoning_tokens` during
    instruction/chat tasks.
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

    gate: Optional[AdaptiveReasoningGate] = None
    if adaptive_reasoning and (
        reasoning_mode is not None
        or think_token_id is not None
        or (fim_prefix_id is not None and fim_prefix_id in prompt)
    ):
        gate = AdaptiveReasoningGate(
            prompt_ids=prompt,
            mode=reasoning_mode,
            fim_prefix_ids=fim_prefix_id,
            think_token_ids=think_token_id,
            think_close_ids=think_close_id,
            max_reasoning_tokens=max_reasoning_tokens,
            decode_fn=decode_fn,
        )

    generated: List[int] = []
    for _ in range(max_new_tokens):
        row = to_numpy(logits)[0]  # (1, vocab) -> (vocab,)
        if gate is not None:
            row = gate.filter_logits(row, generated)
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
        if gate is not None:
            gate.record_token(nxt)
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
    grammar_masker: Optional[object] = None,
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

        grammar_allowed = None
        if grammar_masker is not None and decode_fn is not None and previous_tokens is not None:
            text = prompt_text + decode_fn(previous_tokens)
            grammar_allowed = grammar_masker.mask_for(text, vocab_size=int(np.asarray(logits).size))
            if grammar_allowed is not None and eos_ids:
                if getattr(grammar_masker, "can_end", lambda t: True)(text):
                    grammar_allowed = sorted(set(grammar_allowed) | eos_ids)

        if base_sampler is not None and banned is None and allowed is None and grammar_allowed is None:
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
            grammar_allowed_ids=grammar_allowed,
            banned_ids=list(banned) if banned else None,
        )

    return repair_sampler
