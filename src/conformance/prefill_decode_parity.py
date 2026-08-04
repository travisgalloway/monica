"""prefill vs step parity (#165).

`prefill` (one SSD parallel scan over a whole prompt) and `step` (the one-token
recurrence) are two SEPARATE code paths that must leave the model in the SAME place.
`forward_step_parity` already guards the *logits*; what is new here is that prefill also
emits **state** — the SSD carry-out, the conv window, and the attention KV cache — and a
wrong state is a silent bug: the prompt's own logits still look right, and every token
decoded afterwards is subtly wrong in a way no training loss surfaces (serving would
just generate slightly-off continuations forever).

So this checks four things, in increasing strength:
  1. `prefill` full logits == `forward` logits (the prefill traversal has not drifted
     from the training path — it bypasses `_layer_fns`/`_forward_compute`).
  2. `prefill(last_only=True)` == the last row of (1) (the head shortcut is honest).
  3. **State parity**: prefill's `State` vs the state produced by stepping all `L` tokens
     from `init_state(B)`, compared ELEMENT-WISE.
  4. **Prefill-then-decode vs pure step-by-step**: prefill a prompt, step the tail, and
     compare against stepping everything — the end-to-end property serving depends on.

Run in fp32 at ~1e-4 relative tolerance (see `docs/design/03-conformance.md`); bf16's
epsilon is too coarse to be meaningful. The check RETURNS the verdict; the caller asserts
on `result["ok"]`, so that assertion is the real gate. Portable — numpy plus the seam
only, so it runs on MLX and the torch/CUDA backend alike.
"""

from __future__ import annotations

from typing import Iterator, List

import numpy as np

from ..model.interface import ModelInterface


def _leaves(obj) -> Iterator:
    """Flatten the opaque nested list/tuple `State` into its array leaves.

    `State` is backend-defined above the seam, so this makes no structural assumption
    beyond "lists/tuples of arrays": MLX arrays and torch tensors are neither `list` nor
    `tuple`, so the walk terminates on them.
    """
    if isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _leaves(o)
    else:
        yield obj


def _state_diff(state_a, state_b, to_numpy, rtol: float = 1e-4, atol: float = 1e-5) -> tuple:
    """Max |a - b| over the two states' leaves, plus the first mismatching leaf index.

    A per-leaf index is what makes a failure diagnosable — it says "layer 3's conv
    window", not "something differs". Returns `(max_abs, first_bad_leaf)`. `rtol`/`atol`
    default to the module's standard tolerance but are threaded through from the caller
    so state-parity checks honor the same tolerance as the logit checks.
    """
    a = [np.asarray(to_numpy(x), dtype=np.float64) for x in _leaves(state_a)]
    b = [np.asarray(to_numpy(x), dtype=np.float64) for x in _leaves(state_b)]
    if len(a) != len(b):
        raise ValueError(
            f"prefill state has {len(a)} leaves, stepped state has {len(b)} — the two "
            "paths disagree on the state's STRUCTURE, not just its values.")
    max_abs, first_bad = 0.0, None
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape:
            raise ValueError(
                f"state leaf {i}: prefill shape {x.shape} != stepped shape {y.shape}")
        # `initial=0.0`: the MoE placeholder leaves are zero-sized, and `.max()` on an
        # empty array raises.
        d = float(np.max(np.abs(x - y), initial=0.0))
        if d > max_abs:
            max_abs = d
        if first_bad is None and not np.allclose(x, y, rtol=rtol, atol=atol):
            first_bad = i
    return max_abs, first_bad


def check_prefill_decode_parity(model: ModelInterface, token_batch: np.ndarray,
                                to_numpy=np.asarray, rtol: float = 1e-4,
                                atol: float = 1e-5, n_decode: int = 4) -> dict:
    """Check `prefill` agrees with `forward` and leaves the state `step` would have.

    `token_batch` is (batch, seq_len) ids; the last `n_decode` positions are held back as
    the decode tail for check 4, so `seq_len` must exceed `n_decode`. Returns
    `{max_abs_diff, max_abs_state_diff, first_bad_leaf, ok}` (does NOT raise) — the caller
    asserts on `ok`.
    """
    batch, seq_len = token_batch.shape
    if seq_len <= n_decode:
        raise ValueError(f"seq_len={seq_len} must exceed n_decode={n_decode}")

    # 1) prefill's full logits vs the training path's.
    fwd = np.asarray(to_numpy(model.forward(token_batch)), dtype=np.float64)   # (B,L,V)
    pre_logits, pre_state = model.prefill(token_batch)
    pre = np.asarray(to_numpy(pre_logits), dtype=np.float64)                   # (B,L,V)
    max_abs = float(np.max(np.abs(fwd - pre), initial=0.0))
    ok = bool(np.allclose(fwd, pre, rtol=rtol, atol=atol))

    # 2) last_only must be exactly the last position of the full logits.
    last_logits, _ = model.prefill(token_batch, last_only=True)
    last = np.asarray(to_numpy(last_logits), dtype=np.float64)                 # (B,V)
    max_abs = max(max_abs, float(np.max(np.abs(last - pre[:, -1]), initial=0.0)))
    ok = ok and bool(np.allclose(last, pre[:, -1], rtol=rtol, atol=atol))

    # 3) state parity: prefill's state vs stepping every token from init_state.
    stepped = model.init_state(batch)
    step_logits: List[np.ndarray] = []
    for t in range(seq_len):
        logits_t, stepped = model.step(token_batch[:, t], stepped)
        step_logits.append(np.asarray(to_numpy(logits_t), dtype=np.float64))
    max_state, first_bad = _state_diff(pre_state, stepped, to_numpy, rtol=rtol, atol=atol)
    ok = ok and first_bad is None

    # 4) prefill-then-decode vs pure step-by-step over the same ids. This is the property
    # serving actually relies on: a prompt consumed in one scan must decode identically to
    # a prompt consumed token by token.
    prompt, tail = token_batch[:, :-n_decode], token_batch[:, -n_decode:]
    _, state_a = model.prefill(prompt)
    mixed = []
    for t in range(n_decode):
        logits_t, state_a = model.step(tail[:, t], state_a)
        mixed.append(np.asarray(to_numpy(logits_t), dtype=np.float64))
    mixed_arr = np.stack(mixed, axis=1)                                   # (B,n_decode,V)
    pure = np.stack(step_logits[-n_decode:], axis=1)                      # (B,n_decode,V)
    max_abs = max(max_abs, float(np.max(np.abs(mixed_arr - pure), initial=0.0)))
    ok = ok and bool(np.allclose(mixed_arr, pure, rtol=rtol, atol=atol))

    return {"max_abs_diff": max_abs, "max_abs_state_diff": max_state,
            "first_bad_leaf": first_bad, "ok": bool(ok)}
