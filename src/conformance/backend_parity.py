"""Backend parity (write at the CUDA scale-up) — SKELETON.

Fixed seed, fixed weights, fixed input batch. Run `forward` through both the MLX
and CUDA backends and assert agreement. Run the comparison in FP32 on BOTH sides:
bf16's machine epsilon (~8e-3) is larger than a meaningful tolerance, so comparing
low-precision paths yields false failures. In fp32 a tight tolerance (~1e-4
relative) is meaningful: within = correct port, beyond = a real math bug.

Requires both backends present, so this runs only where CUDA is available; until
then it stays a stub the seam can point at.
"""

from __future__ import annotations

import numpy as np

from ..model.interface import ModelInterface


def check_backend_parity(model_a: ModelInterface, model_b: ModelInterface,
                         token_batch: np.ndarray, to_numpy_a=np.asarray,
                         to_numpy_b=np.asarray, rtol: float = 1e-4,
                         atol: float = 1e-5) -> dict:
    """Assert two backends' `forward` agree in fp32 for identical weights+input.

    Caller is responsible for loading IDENTICAL portable weights into both models
    (via checkpoint.load_weights) before calling.
    """
    a = to_numpy_a(model_a.forward(token_batch)).astype(np.float64)
    b = to_numpy_b(model_b.forward(token_batch)).astype(np.float64)
    max_abs = float(np.abs(a - b).max())
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    if not ok:
        raise AssertionError(f"backend parity FAILED: max|diff|={max_abs:.3e}")
    return {"max_abs_diff": max_abs, "ok": ok}
