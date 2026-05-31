"""MLX-only parity tests (Milestone 1). Skipped where mlx is unavailable.

Two independent checks, both in fp32 (~1e-4 rel):
  * SSM scan parity: the chunked closed-form `parallel` must match a sequential
    reference built from one-step `recurrence`.
  * forward/step parity: whole-model `forward` (parallel) vs `step` (recurrence).
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel, SelectiveSSM
from src.conformance.forward_step_parity import check_forward_step_parity


def _np(a):
    return np.array(a)


def test_ssm_parallel_matches_sequential():
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, L, di = 2, cfg.seq_len, cfg.d_inner
    x = mx.random.normal((B, L, di)) * 0.1

    y_par = _np(ssm.parallel(x))

    # Sequential reference from the one-step recurrence.
    h = mx.zeros((B, di, cfg.d_state))
    ys = []
    for t in range(L):
        y_t, h = ssm.recurrence(x[:, t], h)
        ys.append(_np(y_t))
    y_seq = np.stack(ys, axis=1)

    max_abs = float(np.abs(y_par.astype(np.float64) - y_seq.astype(np.float64)).max())
    assert np.allclose(y_par, y_seq, rtol=1e-4, atol=1e-5), f"max|diff|={max_abs:.3e}"


def test_forward_step_parity_toy():
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    model = MLXMambaModel(cfg)
    B, L = 2, 32
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)
    result = check_forward_step_parity(model, tokens, to_numpy=_np, rtol=1e-4, atol=1e-5)
    assert result["ok"], result
