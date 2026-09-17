"""Discrete diffusion generation with optional LSP discriminator guidance (#203 / #204).

Implements iterative parallel denoising for code completion across two cells:
1. `diffusion_baseline`: unguided iterative discrete diffusion denoising. Tokens are
   gradually unmasked according to a cosine or linear noise schedule without compiler
   feedback.
2. `diffusion_guided`: LSP-guided sampling (the #203 discriminator concept). Inside the
   sampler loop, before committing intermediate tokens, candidate text is passed to
   the LSP oracle (`tsc --noEmit`). Diagnostic spans are mapped to token positions,
   offending tokens are forced back to <mask\\> (remasked), and their committed logits
   are penalized according to a guidance ramp schedule.

ABOVE THE SEAM — stdlib + numpy only. No `mlx`/`torch` import anywhere in this module
(guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..serve.sampling import sample
from .diagnostics import Diagnostic, filter_diagnostics, statement_boundary
from .harness import GenResult, _first_stop
from .lm import LMAdapter, offset_map, token_index_at

DiagnoseFn = Callable[[str], List[Diagnostic]]

_DEFAULT_MAX_GEN_TOKENS = 200
_DEFAULT_CANVAS_LEN = 16


@dataclass
class DiffusionConfig:
    """Hyperparameters for discrete diffusion generation and discriminator guidance."""
    n_steps: int = 8
    canvas_len: int = _DEFAULT_CANVAS_LEN
    guidance_scale: float = 15.0
    schedule: str = "cosine"  # "cosine" | "linear"
    check_interval: int = 1
    start_guidance_step: int = 1
    temperature: float = 0.0
    mask_token_id: Optional[int] = None
    stop_strings: Optional[Sequence[str]] = (";", "\n", "}")


def _mask_schedule(step: int, total_steps: int, total_len: int, schedule: str = "cosine") -> int:
    """Number of positions that should remain masked at `step` (0-indexed)."""
    if step >= total_steps - 1:
        return 0
    t = float(step + 1) / float(total_steps)
    if schedule == "linear":
        ratio = 1.0 - t
    else:  # cosine schedule
        ratio = float(np.cos(0.5 * np.pi * t))
    return max(0, min(total_len, int(np.floor(total_len * ratio))))


def generate_diffusion(
    lm: LMAdapter,
    diagnose: DiagnoseFn,
    prompt: str,
    *,
    guided: bool = False,
    config: Optional[DiffusionConfig] = None,
    budget: str = "stmt",
    max_gen_tokens: int = _DEFAULT_MAX_GEN_TOKENS,
    temperature: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    stop_strings: Optional[Sequence[str]] = (";", "\n", "}"),
) -> GenResult:
    """Generate completion using discrete masked diffusion with optional LSP guidance.

    In unguided mode (`guided=False`), tokens are iteratively committed by model
    confidence across `n_steps`.

    In guided mode (`guided=True`), the discriminator checks candidate text with `diagnose()`,
    maps diagnostic spans back to token positions, and forces faulty tokens back to mask
    while penalizing their logits, directly testing the #203 hypothesis.
    """
    if config is None:
        config = DiffusionConfig(temperature=temperature)

    t0 = time.monotonic()
    n_fwd0 = lm.n_forward_tokens
    n_fwd_nc0 = lm.n_forward_tokens_nocache

    strategy_name = "diffusion-guided" if guided else "diffusion-baseline"
    result = GenResult(
        strategy=strategy_name,
        prompt=prompt,
        completion="",
        context=prompt,
    )

    canvas_len = min(config.canvas_len, max_gen_tokens)
    canvas: List[Optional[int]] = [None] * canvas_len
    penalties: Dict[int, Dict[int, float]] = {pos: {} for pos in range(canvas_len)}

    prompt_ids = lm.encode(prompt)

    for step_idx in range(config.n_steps):
        target_masks = _mask_schedule(step_idx, config.n_steps, canvas_len, config.schedule)

        # 1. Evaluate logits and candidate tokens for all currently masked positions
        predictions: Dict[int, Tuple[int, float]] = {}

        # Autoregressive / causal adapters evaluate prefix up to position
        committed_prefix_tokens = [t for t in canvas if t is not None]
        prefix_text = prompt + lm.decode(committed_prefix_tokens)
        base_logits = lm.reset(prefix_text)

        masked_positions = [i for i, tok in enumerate(canvas) if tok is None]
        if not masked_positions:
            break

        curr_logits = base_logits.copy()
        for p in masked_positions:
            pos_penalties = penalties[p]
            if pos_penalties:
                curr_logits = curr_logits.copy()
                for banned_id, pen in pos_penalties.items():
                    if 0 <= banned_id < len(curr_logits):
                        curr_logits[banned_id] -= pen

            tok = sample(curr_logits, temperature=config.temperature, rng=rng)
            score = float(curr_logits[tok])
            predictions[p] = (tok, score)

            curr_logits = lm.step(tok)

        # 2. Rank masked positions by confidence and unmask according to schedule
        sorted_pos = sorted(masked_positions, key=lambda p: predictions[p][1], reverse=True)
        n_to_unmask = max(0, len(masked_positions) - target_masks)
        to_commit = sorted_pos[:n_to_unmask]

        for p in to_commit:
            canvas[p] = predictions[p][0]

        # 3. LSP Discriminator Guidance (#203)
        if guided and (step_idx >= config.start_guidance_step) and ((step_idx + 1) % config.check_interval == 0):
            cand_tokens = [canvas[p] if canvas[p] is not None else predictions.get(p, (ord(";"), 0.0))[0]
                           for p in range(canvas_len)]
            cand_completion = lm.decode(cand_tokens)

            if stop_strings:
                stop_idx = _first_stop(cand_completion, stop_strings)
                if stop_idx is not None:
                    cand_completion = cand_completion[: stop_idx + 1]

            cand_artifact = prompt + cand_completion
            t_tsc0 = time.monotonic()
            diags = diagnose(cand_artifact)
            result.tsc_wall_s += time.monotonic() - t_tsc0
            result.n_tsc_calls += 1

            real = filter_diagnostics(
                diags,
                frontier=len(cand_artifact),
                generation_start=len(prompt),
                source=cand_artifact,
            )

            if real:
                offsets = offset_map(lm, cand_artifact)
                step_weight = float(step_idx + 1) / float(config.n_steps)

                for d in real:
                    diag_offset = getattr(d, "offset", getattr(d, "start", len(prompt)))
                    tok_pos = token_index_at(offsets, diag_offset) - len(prompt_ids)
                    tok_pos = max(0, min(canvas_len - 1, tok_pos))

                    bad_tok = canvas[tok_pos] if canvas[tok_pos] is not None else predictions.get(tok_pos, (None, 0))[0]
                    if bad_tok is not None:
                        canvas[tok_pos] = None
                        penalties[tok_pos][bad_tok] = (
                            penalties[tok_pos].get(bad_tok, 0.0) + config.guidance_scale * step_weight
                        )
                        result.n_rollbacks += 1
                        result.events.append({
                            "kind": "guided_remask",
                            "step": step_idx,
                            "pos": tok_pos,
                            "token": bad_tok,
                            "code": d.code,
                            "message": d.message,
                        })

    final_tokens = [tok for tok in canvas if tok is not None]
    completion = lm.decode(final_tokens)

    if budget == "stmt":
        if stop_strings:
            stop_idx = _first_stop(completion, stop_strings)
            if stop_idx is not None:
                completion = completion[: stop_idx + 1]
        boundary = statement_boundary(completion)
        if boundary is not None:
            completion = completion[:boundary]

    result.completion = completion
    result.context = prompt + completion
    result.n_generated_tokens = len(final_tokens)
    result.n_forward_tokens = lm.n_forward_tokens - n_fwd0
    result.n_forward_tokens_nocache = lm.n_forward_tokens_nocache - n_fwd_nc0
    result.wall_s = time.monotonic() - t0
    return result
