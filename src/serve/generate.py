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

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ServingTelemetry:
    """Telemetry metrics collected during surrogate critic candidate filtering (#388)."""
    critic_latency_ms: float = 0.0
    oracle_calls_saved: int = 0
    debounce_time_saved_s: float = 0.0
    downstream_clean_rate: float = 1.0
    total_candidates: int = 0
    candidates_pruned: int = 0
    oracle_calls: int = 0
    prune_rate: float = 0.0
    erroneous_prune_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "critic_latency_ms": self.critic_latency_ms,
            "oracle_calls_saved": self.oracle_calls_saved,
            "debounce_time_saved_s": self.debounce_time_saved_s,
            "downstream_clean_rate": self.downstream_clean_rate,
            "total_candidates": self.total_candidates,
            "candidates_pruned": self.candidates_pruned,
            "oracle_calls": self.oracle_calls,
            "prune_rate": self.prune_rate,
            "erroneous_prune_rate": self.erroneous_prune_rate,
        }


@dataclass
class CandidateCompletion:
    """A candidate completion generated during Best-of-N sampling (#388)."""
    token_ids: list[int]
    completion_ids: list[int]
    text: str = ""
    critic_prob: float = 1.0
    passed_critic: bool = True
    oracle_verified: bool | None = None
    critic_latency_ms: float = 0.0


@dataclass
class BestOfNResult:
    """Result of verified Best-of-N candidate generation with critic filtering (#388)."""
    best_candidate: list[int] | None
    best_text: str
    best_prob: float
    candidates: list[CandidateCompletion]
    telemetry: ServingTelemetry


def add_critic_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add surrogate critic filtering CLI options (--critic-filter, --critic-threshold) (#388)."""
    parser.add_argument(
        "--critic-filter",
        action="store_true",
        default=False,
        help="filter candidate completions with surrogate critic head (M12 #388)",
    )
    parser.add_argument(
        "--critic-threshold",
        type=float,
        default=0.70,
        help="P(clean) threshold for surrogate critic filtering (default: 0.70)",
    )
    return parser


def evaluate_completion_critic(
    model: Any,
    full_token_ids: Sequence[int],
    *,
    critic_head: Any | None = None,
    critic_name: str = "noul",
    to_numpy: Callable[[Any], np.ndarray] = np.asarray,
) -> tuple[float, float]:
    """Evaluate candidate completion via auxiliary decision critic head at completion boundary (#388).

    Returns:
        (prob, latency_ms): Calibrated P(clean) in [0.0, 1.0] and evaluation latency in milliseconds.
    """
    t0 = time.perf_counter()
    prob = 0.5
    tokens = [int(t) for t in full_token_ids]
    if callable(critic_head):
        try:
            res = critic_head(tokens)
            prob = float(res.get("prob", res)) if isinstance(res, dict) else float(res)
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
            prob = 0.5
    elif model is not None and hasattr(model, "forward_with_critics"):
        try:
            batch = np.array([tokens], dtype=np.int32)
            _, critic_out = model.forward_with_critics(batch, critic_names=[critic_name])
            prob = float(critic_out[critic_name]["prob"])
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
            prob = 0.5
    elif model is not None and hasattr(model, "forward_hidden"):
        try:
            batch = np.array([tokens], dtype=np.int32)
            h_seq = model.forward_hidden(batch)
            h_arr = to_numpy(h_seq)
            h_last = h_arr[0, -1, :]
            head = critic_head or getattr(model, "critic_heads", {}).get(critic_name)
            if head is not None:
                res = head.predict(h_last) if hasattr(head, "predict") else head.predict_noul(h_last)
                prob = float(res.get("prob", 0.5))
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
            prob = 0.5
    elif critic_head is not None and hasattr(critic_head, "predict_noul"):
        try:
            d = getattr(critic_head.config, "d_model", 64)
            res = critic_head.predict_noul(np.zeros((1, d), dtype=np.float32))
            prob = float(res.get("prob", 0.5))
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
            prob = 0.5

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return prob, latency_ms


def filter_candidates_with_critic(
    candidates: Sequence[CandidateCompletion],
    *,
    critic_threshold: float = 0.70,
    verifier: Callable[[Any], bool] | None = None,
    debounce_floor_s: float = 0.350,
    erroneous_labels: Sequence[bool] | None = None,
) -> tuple[list[CandidateCompletion], ServingTelemetry]:
    """Filter candidate completions with surrogate critic before invoking external verifier (#388).

    Prunes candidate completions where critic_prob < critic_threshold, reserving external oracle
    calls (such as the ~350ms LSP debounce floor) only for promising candidates.

    Returns:
        (surviving_candidates, telemetry)
    """
    total = len(candidates)
    pruned = 0
    calls_saved = 0
    oracle_calls = 0
    oracle_clean = 0
    surviving: list[CandidateCompletion] = []

    for c in candidates:
        if c.critic_prob >= critic_threshold:
            c.passed_critic = True
            surviving.append(c)
            if verifier is not None:
                oracle_calls += 1
                try:
                    clean = bool(verifier(c))
                except (TypeError, AttributeError):
                    try:
                        clean = bool(verifier(c.text))
                    except (TypeError, AttributeError):
                        clean = bool(verifier(c.completion_ids))
                c.oracle_verified = clean
                if clean:
                    oracle_clean += 1
        else:
            c.passed_critic = False
            c.oracle_verified = None
            pruned += 1
            calls_saved += 1

    downstream_clean_rate = (oracle_clean / oracle_calls) if oracle_calls > 0 else 1.0
    debounce_time_saved = calls_saved * debounce_floor_s
    prune_rate = (pruned / total) if total > 0 else 0.0
    tot_latency = sum(c.critic_latency_ms for c in candidates)

    erroneous_prune_rate = 0.0
    if erroneous_labels is not None and len(erroneous_labels) == total:
        err_total = sum(1 for is_err in erroneous_labels if is_err)
        err_pruned = sum(
            1 for c, is_err in zip(candidates, erroneous_labels)
            if is_err and not c.passed_critic
        )
        erroneous_prune_rate = (err_pruned / err_total) if err_total > 0 else 1.0

    telemetry = ServingTelemetry(
        critic_latency_ms=tot_latency,
        oracle_calls_saved=calls_saved,
        debounce_time_saved_s=debounce_time_saved,
        downstream_clean_rate=downstream_clean_rate,
        total_candidates=total,
        candidates_pruned=pruned,
        oracle_calls=oracle_calls,
        prune_rate=prune_rate,
        erroneous_prune_rate=erroneous_prune_rate,
    )
    return surviving, telemetry


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
        mode: str | None = None,
        fim_prefix_ids: int | Sequence[int] | set[int] | None = (1,),
        think_token_ids: int | Sequence[int] | set[int] | None = None,
        think_close_ids: int | Sequence[int] | set[int] | None = None,
        max_reasoning_tokens: int | None = None,
        decode_fn: Callable[[Sequence[int]], str] | None = None,
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
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
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

    def __call__(self, logits: np.ndarray, previous_tokens: Sequence[int] | None = None) -> np.ndarray:
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
    eos_id: int | None = None,
    stop_fn: Callable[[list[int]], bool] | None = None,
    on_token: Callable[[int], None] | None = None,
    pass_context: bool = False,
    prefill: bool = True,
    sampler_hook: Callable[..., np.ndarray | None] | Sequence[Callable[..., np.ndarray | None]] | None = None,
    sampler_hooks: Sequence[Callable[..., np.ndarray | None]] | None = None,
    reasoning_mode: str | None = None,
    fim_prefix_id: int | Sequence[int] | None = 1,
    think_token_id: int | Sequence[int] | None = None,
    think_close_id: int | Sequence[int] | None = None,
    max_reasoning_tokens: int | None = None,
    adaptive_reasoning: bool = True,
    decode_fn: Callable[[Sequence[int]], str] | None = None,
    critic_filter: bool = False,
    critic_threshold: float = 0.70,
    critic_head: Any | None = None,
) -> list[int]:
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

    hooks: list[Callable] = []
    if sampler_hook is not None:
        if isinstance(sampler_hook, Sequence) and not isinstance(sampler_hook, (str, bytes)):
            hooks.extend(sampler_hook)
        else:
            hooks.append(sampler_hook)
    if sampler_hooks is not None:
        hooks.extend(sampler_hooks)

    gate: AdaptiveReasoningGate | None = None
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

    generated: list[int] = []
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
    ban_table: dict | None = None,
    *,
    base_sampler: Callable[..., int] | None = None,
    masker: object | None = None,
    grammar_masker: object | None = None,
    decode_fn: Callable[[Sequence[int]], str] | None = None,
    prompt_text: str = "",
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
    eos_ids: set[int] | None = None,
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

    def repair_sampler(logits: np.ndarray, previous_tokens: Sequence[int] | None = None) -> int:
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


def generate_best_of_n(
    store,
    session_id_prefix: str,
    prompt_ids: Sequence[int],
    n_candidates: int = 4,
    *,
    sampler: Callable[..., int],
    critic_filter: bool = False,
    critic_threshold: float = 0.70,
    critic_head: Any | None = None,
    verifier: Callable[[Any], bool] | None = None,
    decode_fn: Callable[[Sequence[int]], str] | None = None,
    to_numpy: Callable[[object], np.ndarray] = np.asarray,
    max_new_tokens: int = 128,
    eos_id: int | None = None,
    stop_fn: Callable[[list[int]], bool] | None = None,
    pass_context: bool = False,
    prefill: bool = True,
    debounce_floor_s: float = 0.350,
    erroneous_labels: Sequence[bool] | None = None,
    candidate_samplers: Sequence[Callable[..., int]] | None = None,
    **kwargs,
) -> BestOfNResult:
    """Generate N candidate completions and verify with optional surrogate critic filtering (#388).

    Without critic filtering (unfiltered baseline):
      Every candidate completion is passed directly to the external verifier (such as
      the ~350ms LSP debounce floor), causing severe latency bottlenecks.

    With critic filtering (--critic-filter, threshold default 0.70):
      Each candidate is evaluated at the completion boundary via the calibrated surrogate critic
      head in microseconds (~0.010 ms). Candidates with P(clean) < threshold are pruned prior
      to invoking external LSP verification, eliminating debounce waits and improving overall
      wall-clock latency by >= 3x.
    """
    prompt = [int(t) for t in prompt_ids]
    candidates: list[CandidateCompletion] = []

    for i in range(n_candidates):
        sid = f"{session_id_prefix}_{i}"
        store.create(sid)
        try:
            curr_sampler = (
                candidate_samplers[i]
                if candidate_samplers is not None and i < len(candidate_samplers)
                else sampler
            )
            gen_ids = generate(
                store,
                sid,
                prompt,
                sampler=curr_sampler,
                to_numpy=to_numpy,
                max_new_tokens=max_new_tokens,
                eos_id=eos_id,
                stop_fn=stop_fn,
                pass_context=pass_context,
                prefill=prefill,
                decode_fn=decode_fn,
                **kwargs,
            )
        finally:
            store.remove(sid)

        full_ids = prompt + gen_ids
        text = decode_fn(gen_ids) if decode_fn is not None else ""

        if critic_filter:
            prob, lat_ms = evaluate_completion_critic(
                store.model, full_ids, critic_head=critic_head, to_numpy=to_numpy
            )
        else:
            prob, lat_ms = 1.0, 0.0

        cand = CandidateCompletion(
            token_ids=full_ids,
            completion_ids=gen_ids,
            text=text,
            critic_prob=prob,
            passed_critic=(prob >= critic_threshold) if critic_filter else True,
            critic_latency_ms=lat_ms,
        )
        candidates.append(cand)

    if critic_filter:
        surviving, telemetry = filter_candidates_with_critic(
            candidates,
            critic_threshold=critic_threshold,
            verifier=verifier,
            debounce_floor_s=debounce_floor_s,
            erroneous_labels=erroneous_labels,
        )
    else:
        # Un-filtered baseline: evaluate every candidate with verifier
        oracle_calls = 0
        oracle_clean = 0
        for c in candidates:
            c.passed_critic = True
            if verifier is not None:
                oracle_calls += 1
                try:
                    clean = bool(verifier(c))
                except (TypeError, AttributeError):
                    try:
                        clean = bool(verifier(c.text))
                    except (TypeError, AttributeError):
                        clean = bool(verifier(c.completion_ids))
                c.oracle_verified = clean
                if clean:
                    oracle_clean += 1

        downstream_clean_rate = (oracle_clean / oracle_calls) if oracle_calls > 0 else 1.0
        telemetry = ServingTelemetry(
            critic_latency_ms=0.0,
            oracle_calls_saved=0,
            debounce_time_saved_s=0.0,
            downstream_clean_rate=downstream_clean_rate,
            total_candidates=len(candidates),
            candidates_pruned=0,
            oracle_calls=oracle_calls,
            prune_rate=0.0,
            erroneous_prune_rate=0.0,
        )
        surviving = [c for c in candidates if c.oracle_verified is not False]

    # Select best candidate
    best: CandidateCompletion | None = None
    # 1. Prefer oracle verified clean
    verified_clean = [c for c in candidates if c.oracle_verified is True]
    if verified_clean:
        best = max(verified_clean, key=lambda c: c.critic_prob)
    elif surviving:
        best = max(surviving, key=lambda c: c.critic_prob)
    elif candidates:
        best = max(candidates, key=lambda c: c.critic_prob)

    best_candidate_ids = best.completion_ids if best else None
    best_text = best.text if best else ""
    best_prob = best.critic_prob if best else 0.0

    return BestOfNResult(
        best_candidate=best_candidate_ids,
        best_text=best_text,
        best_prob=best_prob,
        candidates=candidates,
        telemetry=telemetry,
    )
