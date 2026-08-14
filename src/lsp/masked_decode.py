"""#226 — the constrained-decode loop: `generate_masked`, the decode strategy for
the `masked` / `masked-null` / `masked-oracle` / `masked-oracle-null` arms (and,
with `masker=None`, the byte-identical `unconstrained` control).

Structurally `generate_baseline` (`harness.py:107`) with one added line per step:
`masker.mask_for(...)`, threaded into `sample(..., allowed_ids=...)`. `GenResult`,
`_eos_ids`, and `_first_stop` are imported from `.harness` rather than re-expressed
— same package, and a second copy of the EOS-set logic is exactly the silent-
divergence trap this repo keeps flagging.

A new module, not an edit to `generate_baseline`: the #194/#199 harness path is
load-bearing and stays untouched (see `harness.py`'s module docstring for why
`ts_service.py` is a new class rather than an extension of `ts_lsp.py` — same
reasoning here).

EOS ids are always folded into an active mask, so masking can never make a
generation unstoppable. With `masker=None` the loop is byte-identical to
`generate_baseline` — `sample(..., allowed_ids=None)` is an exact no-op — which is
what makes the `unconstrained` arm a true control.

ABOVE THE SEAM — stdlib + numpy only. No `mlx`/`torch` import anywhere in this
module (guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np

from ..serve.sampling import sample
from .completion_mask import CompletionMasker
from .diagnostics import statement_boundary
from .harness import GenResult, _eos_ids, _first_stop
from .lm import LMAdapter

_DEFAULT_MAX_GEN_TOKENS = 200
_DEFAULT_BLOCK_SIZE = 96


def generate_masked(
    lm: LMAdapter,
    masker: Optional[CompletionMasker],
    prompt: str,
    *,
    budget: str = "stmt",
    block_size: int = _DEFAULT_BLOCK_SIZE,
    max_gen_tokens: int = _DEFAULT_MAX_GEN_TOKENS,
    temperature: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    stop_strings: Optional[Sequence[str]] = None,
    strategy: Optional[str] = None,
) -> GenResult:
    """Free-running generation, exactly like `generate_baseline`, except each
    step consults `masker.mask_for(...)` (a `CompletionMasker`, or `None` for the
    unconstrained control) for the allowed next-token id set.

    `strategy` labels the result (for scoring/logging); when not given it is
    derived from whether masking is actually active, so a `masker=None` run is
    never silently labeled `"masked"`.
    """
    if strategy is None:
        strategy = "masked" if masker is not None else "unconstrained"
    if budget not in ("stmt", "block"):
        raise ValueError(f"unknown budget {budget!r}")
    t0 = time.monotonic()
    n_fwd0, n_fwd_nc0 = lm.n_forward_tokens, lm.n_forward_tokens_nocache

    target = block_size if budget == "block" else max_gen_tokens
    stop_at_boundary = budget == "stmt"

    logits = lm.reset(prompt)
    gen_ids: List[int] = []
    checkpoints: List[int] = []
    stop_at: Optional[int] = None
    eos = _eos_ids(lm) if not stop_at_boundary else set()   # EOS ends a real-code body
    gen_text = ""  # lm.decode(gen_ids), kept in sync so each step decodes only once
    for _ in range(target):
        allowed = None
        if masker is not None:
            allowed = masker.mask_for(prompt + gen_text, vocab_size=int(np.asarray(logits).size))
            if allowed is not None and eos:
                # EOS must always be reachable through an active mask -- masking
                # can never make a generation unstoppable. `allowed` is already
                # sorted/deduped by `mask_for`, so skip the set/sort when there's
                # no EOS set to fold in.
                allowed = sorted(set(allowed) | eos)
        tok = sample(logits, temperature=temperature, rng=rng, previous_tokens=gen_ids,
                     allowed_ids=allowed)
        if tok in eos:
            break
        logits = lm.step(tok)
        gen_ids.append(tok)
        gen_text = lm.decode(gen_ids)
        text = gen_text
        if stop_at_boundary:
            if statement_boundary(text) is not None:
                break
        else:
            stop_at = _first_stop(text, stop_strings)
            if stop_at is not None:
                break
            b = statement_boundary(text)
            if b is not None and (not checkpoints or checkpoints[-1] != len(prompt) + b):
                checkpoints.append(len(prompt) + b)

    completion = gen_text
    if stop_at is not None:
        completion = completion[:stop_at]
    result = GenResult(
        strategy=strategy, prompt=prompt, completion=completion,
        context=prompt + completion, checkpoints=checkpoints,
        n_generated_tokens=len(gen_ids),
        n_forward_tokens=lm.n_forward_tokens - n_fwd0,
        n_forward_tokens_nocache=lm.n_forward_tokens_nocache - n_fwd_nc0,
        wall_s=time.monotonic() - t0,
    )
    if masker is not None:
        result.n_mask_steps = masker.n_mask_steps
        result.n_mask_bypass = masker.n_mask_bypass
        result.n_completion_calls = masker.n_completion_calls
        result.mask_wall_s = masker.mask_wall_s
    return result
