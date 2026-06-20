"""forward vs step parity (MILESTONE 1).

The training path (`forward`, parallel scan) and the inference path (`step`,
recurrence) are two SEPARATE code paths and must produce the same logits for the
same input. A mismatch here is a silent, nasty bug that the parallel-vs-sequential
scan check does NOT catch (that check validates only the scan, not train/infer
equivalence).

Run in fp32, ~1e-4 relative tolerance. Build the model, run a fixed batch through
`forward`, then feed the same tokens one at a time through `step` carrying state, and
compare the per-position logits. The check RETURNS the verdict (`{max_abs_diff, ok}`);
the caller asserts on `result["ok"]` (so that assertion is the real gate). Backend-
agnostic — drives the model only through `ModelInterface`, so it runs on MLX and the
torch/CUDA backend alike (see `tests/test_mlx_parity.py` / `tests/test_cuda_parity.py`).
"""

from __future__ import annotations

import numpy as np

from ..model.interface import ModelInterface


def check_forward_step_parity(model: ModelInterface, token_batch: np.ndarray,
                              to_numpy=np.asarray, rtol: float = 1e-4,
                              atol: float = 1e-5) -> dict:
    """Check forward (parallel) and step (recurrence) agree. Returns `{max_abs_diff, ok}`
    (does NOT raise) — the caller asserts on `ok`.

    `to_numpy` converts backend logits to numpy (identity by default; on MLX pass a
    converter). Requires a working backend model (MLX or torch/CUDA).
    """
    batch, seq_len = token_batch.shape
    parallel_logits = to_numpy(model.forward(token_batch))  # (B, T, V)

    state = model.init_state(batch)
    step_logits = []
    for t in range(seq_len):
        logits_t, state = model.step(token_batch[:, t], state)
        step_logits.append(to_numpy(logits_t))
    step_logits = np.stack(step_logits, axis=1)  # (B, T, V)

    diff = np.abs(parallel_logits.astype(np.float64) - step_logits.astype(np.float64))
    max_abs = float(diff.max())
    # Return the verdict (don't raise) so the CALLER's `assert result["ok"]` is the
    # actual gate — otherwise the assertion is decorative (the result could only ever
    # be ok=True if a raise gated it first).
    ok = bool(np.allclose(parallel_logits, step_logits, rtol=rtol, atol=atol))
    return {"max_abs_diff": max_abs, "ok": ok}
