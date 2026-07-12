"""The LSP-in-the-loop generation strategies (#199) — the core of the harness.

Three strategies, all returning a `GenResult`:

- `generate_baseline` — free-running generation, no diagnostics, no repair. The
  thing we're trying to beat.
- `generate_slow_loop` — checkpoint-and-repair generation. `repair="hard"` rolls
  back to the offending token and bans it, retrying in place; `repair="soft"`
  injects the diagnostic as a `// tsc: ...` comment above the current statement and
  regenerates it from scratch; `repair="both"` tries hard repair first and falls
  back to one soft-repair round once hard repair's retries are exhausted (soft
  repair's checkpoint-anchored rewrite is deliberately the *fallback*, not a
  parallel strategy — see the design doc).
- `generate_toolcall` — the same-model, same-greedy, text-mode equivalent of a
  tool-call round-trip: diagnostic injected as a comment, the whole completion
  regenerated from scratch, `k` rounds, no token-level banning. Implemented as
  `generate_slow_loop(repair="soft", budget="stmt", max_retries=k)` under a
  different strategy label — on this one-statement eval set the two are the same
  algorithm at different granularity (an acknowledged, expected overlap; see the
  design doc's risks section), and the hard-ban condition is what tests something a
  tool-call baseline structurally can't express.

The two-string invariant (see `docs/design/12-lsp-in-the-loop.md`): `context` is
what the model conditions on (may contain an injected `// tsc: ...` comment);
`GenResult.artifact = prompt + completion` is what gets scored and NEVER contains
an injected comment. `completion` always equals `context[generation_start:]` for
whichever attempt is currently live, because soft repair discards the prior
attempt's generated suffix outright rather than editing it in place.

ABOVE THE SEAM — stdlib + numpy only. No `mlx`/`torch` import anywhere in this
module (guarded by `tests/test_import_guard.py`); `lm: LMAdapter` and
`diagnose: DiagnoseFn` are injected callables, the established seam idiom
(`src/eval/bfcl_adapter.py::evaluate_bfcl`, `src/eval/probes.py::run_probes`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..serve.sampling import sample
from .diagnostics import (Diagnostic, SUPPRESSION_RE, close_open_delimiters,
                           filter_diagnostics, is_incomplete, statement_boundary,
                           strip_suggestion)
from .lm import LMAdapter, token_index_at

DiagnoseFn = Callable[[str], List[Diagnostic]]

_DEFAULT_MAX_RETRIES = 8
_DEFAULT_MAX_GEN_TOKENS = 200
_DEFAULT_BLOCK_SIZE = 96


@dataclass
class GenResult:
    strategy: str
    prompt: str
    completion: str            # ARTIFACT-visible generated text (never includes an injected comment)
    context: str                # final CONTEXT the lm ended in (prompt [+ injected comments] + completion)
    checkpoints: List[int] = field(default_factory=list)   # committed statement-start offsets, context coords
    events: List[dict] = field(default_factory=list)        # per-repair-action transcript (kind, code, offset, ...)
    n_generated_tokens: int = 0
    n_rollbacks: int = 0
    n_soft_repairs: int = 0
    n_retries: int = 0
    unrepaired: bool = False
    no_progress: bool = False
    reward_hack_detected: bool = False
    n_tsc_calls: int = 0
    tsc_wall_s: float = 0.0
    n_forward_tokens: int = 0
    n_forward_tokens_nocache: int = 0
    wall_s: float = 0.0

    @property
    def artifact(self) -> str:
        return self.prompt + self.completion


# --------------------------------------------------------------------------- #
# baseline: free-running, no diagnostics
# --------------------------------------------------------------------------- #

def generate_baseline(
    lm: LMAdapter,
    prompt: str,
    *,
    budget: str = "stmt",
    block_size: int = _DEFAULT_BLOCK_SIZE,
    max_gen_tokens: int = _DEFAULT_MAX_GEN_TOKENS,
    temperature: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> GenResult:
    """Generate a completion with no diagnostic feedback at all — the thing every
    repair strategy has to beat. `budget="stmt"` stops at the first statement
    boundary (or `max_gen_tokens` as a safety cap); `budget="block"` always
    generates exactly `block_size` tokens with no early stop.
    """
    if budget not in ("stmt", "block"):
        raise ValueError(f"unknown budget {budget!r}")
    t0 = time.monotonic()
    n_fwd0, n_fwd_nc0 = lm.n_forward_tokens, lm.n_forward_tokens_nocache

    target = block_size if budget == "block" else max_gen_tokens
    stop_at_boundary = budget == "stmt"

    logits = lm.reset(prompt)
    gen_ids: List[int] = []
    checkpoints: List[int] = []
    for _ in range(target):
        tok = sample(logits, temperature=temperature, rng=rng, previous_tokens=gen_ids)
        logits = lm.step(tok)
        gen_ids.append(tok)
        if stop_at_boundary:
            text = lm.decode(gen_ids)
            if statement_boundary(text) is not None:
                break
        else:
            text = lm.decode(gen_ids)
            b = statement_boundary(text)
            if b is not None and (not checkpoints or checkpoints[-1] != len(prompt) + b):
                checkpoints.append(len(prompt) + b)

    completion = lm.decode(gen_ids)
    return GenResult(
        strategy="baseline", prompt=prompt, completion=completion,
        context=prompt + completion, checkpoints=checkpoints,
        n_generated_tokens=len(gen_ids),
        n_forward_tokens=lm.n_forward_tokens - n_fwd0,
        n_forward_tokens_nocache=lm.n_forward_tokens_nocache - n_fwd_nc0,
        wall_s=time.monotonic() - t0,
    )


# --------------------------------------------------------------------------- #
# the repair loop
# --------------------------------------------------------------------------- #

def _format_repair_comment(diag: Diagnostic) -> str:
    return f"// tsc: {diag.code}: {diag.message}"


def _inject_comment(context: str, generation_start: int, comment: str) -> Tuple[str, int]:
    """Splice `comment` in on its own line above the partial statement (at the
    last newline before `generation_start`, or the very start of `context` if
    there is none). Returns `(new_context_prefix, new_generation_start)` — the
    caller regenerates from `new_generation_start` onward.
    """
    last_nl = context.rfind("\n", 0, generation_start)
    injection_point = last_nl + 1 if last_nl != -1 else 0
    new_prefix = context[:injection_point] + comment + "\n" + context[injection_point:generation_start]
    return new_prefix, len(new_prefix)


def _is_clean(diags: List[Diagnostic], generated_text: str) -> bool:
    if diags:
        return False
    if SUPPRESSION_RE.search(generated_text):
        return False
    return True


def generate_slow_loop(
    lm: LMAdapter,
    diagnose: DiagnoseFn,
    prompt: str,
    *,
    repair: str = "hard",
    budget: str = "stmt",
    block_size: int = _DEFAULT_BLOCK_SIZE,
    max_gen_tokens: int = _DEFAULT_MAX_GEN_TOKENS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    temperature: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    strip_suggestions: bool = False,
) -> GenResult:
    """Checkpoint-and-repair generation. See the module docstring for the three
    `repair` modes and the two-string (context/artifact) invariant.

    `budget="stmt"` checks and repairs exactly one statement. `budget="block"`
    generates `block_size` tokens total across possibly several statements,
    checking (and repairing) diagnostics at every statement boundary crossed —
    the checkpoint stack — and, if the token budget runs out mid-statement,
    virtually closes the dangling delimiters (`close_open_delimiters`) before the
    final check, so that last check sees compilable TS instead of a flood of
    "expected X" syntax noise from the unfinished tail.
    """
    if repair not in ("hard", "soft", "both"):
        raise ValueError(f"unknown repair strategy {repair!r}")
    if budget not in ("stmt", "block"):
        raise ValueError(f"unknown budget {budget!r}")

    t0 = time.monotonic()
    n_fwd0, n_fwd_nc0 = lm.n_forward_tokens, lm.n_forward_tokens_nocache

    result = GenResult(strategy=f"slow-{repair}", prompt=prompt, completion="", context=prompt)
    strip = strip_suggestion if strip_suggestions else (lambda d: d)

    def _diag(source: str) -> List[Diagnostic]:
        t = time.monotonic()
        out = diagnose(source)
        result.tsc_wall_s += time.monotonic() - t
        result.n_tsc_calls += 1
        return [strip(d) for d in out]

    context = prompt          # everything before the LIVE segment's generation_start
    committed_completion = "" # ARTIFACT text already locked in from prior committed segments
    total_budget = block_size if budget == "block" else max_gen_tokens
    committed_tokens = 0

    while committed_tokens < total_budget:
        segment_start = len(context)  # context coords: this segment's checkpoint
        result.checkpoints.append(segment_start)
        remaining = total_budget - committed_tokens

        logits = lm.reset(context)
        gen_ids: List[int] = []
        logits_history = [logits]
        ban_table: Dict[Tuple[int, ...], set] = {}

        # --- generate this segment, up to `remaining` tokens or a statement boundary ---
        def _extend_to_boundary_or_budget(n_max: int) -> Tuple[bool, bool]:
            """Returns (hit_boundary, hit_budget)."""
            nonlocal logits
            for _ in range(n_max):
                key = tuple(gen_ids)
                banned = ban_table.get(key)
                step_logits = logits
                if banned:
                    step_logits = step_logits.copy()
                    step_logits[list(banned)] = -np.inf
                tok = sample(step_logits, temperature=temperature, rng=rng,
                             previous_tokens=gen_ids)
                logits = lm.step(tok)
                gen_ids.append(tok)
                logits_history.append(logits)
                text = lm.decode(gen_ids)
                if statement_boundary(text) is not None:
                    return True, False
            return False, True

        hit_boundary, hit_budget = _extend_to_boundary_or_budget(remaining)
        segment_is_final_partial = hit_budget and not hit_boundary  # budget ran out mid-statement

        n_retry_rounds = 0
        while True:
            gen_text = lm.decode(gen_ids)
            check_text = close_open_delimiters(gen_text) if segment_is_final_partial else gen_text
            source = context + check_text
            if hit_boundary:
                # The model itself emitted an explicit terminator and generation
                # for this attempt has genuinely stopped -- there is no "one more
                # token might still be coming" to wait out, so the whole segment
                # (including its last token) counts as committed. Withholding the
                # last token here (as the mid-generation case below does) would
                # create a permanent blind spot for single-token segments, where
                # the last token IS the only token.
                frontier = segment_start + len(gen_text)
            else:
                # Budget ran out mid-statement (segment_is_final_partial): the last
                # real token may genuinely still be "forming" -- keep withholding
                # it. Virtually-appended closer characters sit past this frontier
                # by construction, so their own syntax noise is dropped for free.
                frontier = segment_start + (len(lm.decode(gen_ids[:-1])) if gen_ids else 0)

            raw = _diag(source)
            filtered = filter_diagnostics(raw, frontier=frontier, generation_start=segment_start)
            if hit_boundary and not filtered:
                # filter_diagnostics unconditionally drops TS1xxx as mid-generation
                # "still typing" noise -- correct while more tokens might still be
                # coming, but this segment just reached a genuine statement
                # boundary: there ISN'T any more text coming for this attempt, so a
                # committed TS1xxx (e.g. `u.)` -- "Identifier expected") is a real
                # defect, not noise, and must not be waved through as clean.
                committed_incomplete = [d for d in raw if is_incomplete(d.code) and d.offset < frontier]
                if committed_incomplete:
                    filtered = [replace(d, offset=max(d.offset, segment_start))
                                for d in committed_incomplete]
            if SUPPRESSION_RE.search(gen_text):
                result.reward_hack_detected = True
            clean = _is_clean(filtered, gen_text)

            if clean or n_retry_rounds >= max_retries:
                if not clean:
                    result.unrepaired = True
                break

            diag = filtered[0]
            n_retry_rounds += 1
            result.n_retries += 1

            # "both": hard repair gets every round except the last, which falls back
            # to one soft-repair round once hard repair's retries are exhausted —
            # soft repair is deliberately the fallback, not a parallel strategy.
            use_soft = (repair == "soft") or (repair == "both" and n_retry_rounds >= max_retries)
            if repair == "hard" or (repair == "both" and not use_soft):
                # --- hard repair: roll back to the token containing the diagnostic ---
                offsets = [segment_start + len(lm.decode(gen_ids[:k])) for k in range(len(gen_ids))]
                if not offsets:
                    # Nothing generated yet in this segment (diagnostic is purely
                    # prompt-caused) — nothing to roll back to; fall through to a
                    # fresh sample at the (empty) prefix instead of crashing.
                    tok_idx = 0
                else:
                    tok_idx = token_index_at(offsets, diag.offset)
                n_to_rollback = len(gen_ids) - tok_idx
                if n_to_rollback > 0:
                    lm.rollback(n_to_rollback)
                    result.n_rollbacks += 1
                banned_tok = gen_ids[tok_idx] if tok_idx < len(gen_ids) else None
                gen_ids = gen_ids[:tok_idx]
                logits_history = logits_history[: tok_idx + 1]
                logits = logits_history[-1]
                if banned_tok is not None:
                    ban_table.setdefault(tuple(gen_ids), set()).add(banned_tok)
                result.events.append({"kind": "hard_repair", "code": diag.code,
                                       "offset": diag.offset, "rolled_back_to": tok_idx})
                hit_boundary, hit_budget = _extend_to_boundary_or_budget(
                    total_budget - committed_tokens - len(gen_ids))
                segment_is_final_partial = hit_budget and not hit_boundary
            else:
                # --- soft repair: inject the diagnostic, regenerate from scratch ---
                comment = _format_repair_comment(diag)
                new_context, new_start = _inject_comment(context, segment_start, comment)
                old_text = gen_text

                context = new_context
                segment_start = new_start
                logits = lm.reset(context)
                gen_ids = []
                logits_history = [logits]
                ban_table = {}
                hit_boundary, hit_budget = _extend_to_boundary_or_budget(
                    total_budget - committed_tokens)
                segment_is_final_partial = hit_budget and not hit_boundary

                result.n_soft_repairs += 1
                result.events.append({"kind": "soft_repair", "code": diag.code,
                                       "offset": diag.offset, "comment": comment})

                new_text = lm.decode(gen_ids)
                if new_text == old_text:
                    result.no_progress = True
                    result.unrepaired = True
                    break

        # --- commit this segment ---
        gen_text = lm.decode(gen_ids)
        committed_completion += gen_text
        committed_tokens += len(gen_ids)
        context = context + gen_text

        if result.no_progress or result.unrepaired or segment_is_final_partial or budget == "stmt":
            break

    result.completion = committed_completion
    result.context = context
    result.n_generated_tokens = committed_tokens
    result.n_forward_tokens = lm.n_forward_tokens - n_fwd0
    result.n_forward_tokens_nocache = lm.n_forward_tokens_nocache - n_fwd_nc0
    result.wall_s = time.monotonic() - t0
    return result


# --------------------------------------------------------------------------- #
# tool-call baseline
# --------------------------------------------------------------------------- #

def generate_toolcall(
    lm: LMAdapter,
    diagnose: DiagnoseFn,
    prompt: str,
    *,
    k: int = 1,
    budget: str = "stmt",
    block_size: int = _DEFAULT_BLOCK_SIZE,
    max_gen_tokens: int = _DEFAULT_MAX_GEN_TOKENS,
    temperature: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    strip_suggestions: bool = False,
) -> GenResult:
    """The same-model, same-greedy, text-mode equivalent of a tool-call round
    trip: diagnostic injected as a comment, the whole completion regenerated from
    scratch, `k` rounds. See the module docstring for why this shares
    `generate_slow_loop`'s soft-repair machinery under a distinct label.

    `budget`/`block_size`/`max_gen_tokens` MUST be threaded through to
    `generate_slow_loop` rather than hardcoded to `"stmt"` — a caller comparing
    strategies under `budget="block"` needs every strategy free-running to the
    same token budget, or a "toolcall" that quietly stops at the first statement
    boundary looks artificially cheap and artificially clean for reasons that have
    nothing to do with tool-call repair actually working.
    """
    result = generate_slow_loop(
        lm, diagnose, prompt, repair="soft", budget=budget, block_size=block_size,
        max_gen_tokens=max_gen_tokens, max_retries=k, temperature=temperature, rng=rng,
        strip_suggestions=strip_suggestions,
    )
    result.strategy = f"toolcall-k{k}"
    return result
