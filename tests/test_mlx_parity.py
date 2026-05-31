"""MLX-only parity tests (Milestone 1). Skipped where mlx is unavailable.

Checks, all in fp32 (~1e-4 rel):
  * SSM scan parity: the chunked closed-form `parallel` must match a sequential
    reference built from one-step `recurrence`.
  * SSM scan vs an INDEPENDENT reference: `parallel` must also match a
    from-scratch numpy sequential scan computed from the SSM's raw parameters
    (shares no code with `parallel` or `recurrence`). This catches a bug that
    would live identically in the discretization shared by the MLX paths, plus a
    long-context case guarding the chunked scan's fp32 overflow-safety.
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


def _seq_reference_numpy(ssm, x_np):
    """Fully independent fp64 sequential selective scan.

    Computes the projections, softplus, diagonal-A discretization and the
    recurrence FROM SCRATCH in numpy using only the SSM's raw parameters — it
    shares no code with `SelectiveSSM.parallel` or `.recurrence`, so a bug living
    identically in their shared `_project`/discretization cannot hide here.

    x_np: (B, L, d_inner) float -> (B, L, d_inner) float64.
    """
    cfg = ssm.config
    dt_rank, d_state = cfg.dt_rank_resolved, cfg.d_state

    # Raw params -> fp64. MLX nn.Linear stores weight as (out, in) and computes
    # `x @ W.T + b`, so we project with W.T.
    Wx = _np(ssm.x_proj.weight).astype(np.float64)    # (dt_rank+2*ds, di)
    Wdt = _np(ssm.dt_proj.weight).astype(np.float64)  # (di, dt_rank)
    bdt = _np(ssm.dt_proj.bias).astype(np.float64)    # (di,)
    A = -np.exp(_np(ssm.A_log).astype(np.float64))    # (di, ds)
    D = _np(ssm.D).astype(np.float64)                 # (di,)

    x = x_np.astype(np.float64)
    B_, L, di = x.shape

    proj = x @ Wx.T                                   # (B, L, dt_rank+2*ds)
    dt = proj[..., :dt_rank]
    Bm = proj[..., dt_rank:dt_rank + d_state]         # (B, L, ds)
    Cm = proj[..., dt_rank + d_state:]                # (B, L, ds)
    delta = np.logaddexp(dt @ Wdt.T + bdt, 0.0)       # softplus -> (B, L, di)

    y = np.zeros((B_, L, di), np.float64)
    h = np.zeros((B_, di, d_state), np.float64)
    for t in range(L):
        dlt = delta[:, t][..., None]                  # (B, di, 1)
        dA = np.exp(dlt * A[None])                     # (B, di, ds)
        dBu = dlt * Bm[:, t][:, None, :] * x[:, t][..., None]
        h = dA * h + dBu
        y[:, t] = np.sum(h * Cm[:, t][:, None, :], axis=-1) + x[:, t] * D
    return y


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


def test_ssm_parallel_matches_independent_reference():
    """`parallel` vs a from-scratch numpy scan (no shared code) at toy seq_len."""
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, L, di = 2, cfg.seq_len, cfg.d_inner
    x = mx.random.normal((B, L, di)) * 0.1

    y_par = _np(ssm.parallel(x)).astype(np.float64)
    y_ref = _seq_reference_numpy(ssm, _np(x))

    max_abs = float(np.abs(y_par - y_ref).max())
    assert np.allclose(y_par, y_ref, rtol=1e-4, atol=1e-5), f"max|diff|={max_abs:.3e}"


def test_ssm_parallel_overflow_safe_long_context():
    """Long context (L=512 -> 16 chunks of 32): the chunked scan must stay finite
    (the docstring's fp32 overflow-safety claim) and still match the independent
    reference."""
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, L, di = 2, 512, cfg.d_inner
    x = mx.random.normal((B, L, di)) * 0.1

    y_par = _np(ssm.parallel(x)).astype(np.float64)
    assert np.isfinite(y_par).all(), "chunked scan produced NaN/Inf at long context"

    y_ref = _seq_reference_numpy(ssm, _np(x))
    max_abs = float(np.abs(y_par - y_ref).max())
    assert np.allclose(y_par, y_ref, rtol=1e-4, atol=1e-5), f"max|diff|={max_abs:.3e}"


def test_forward_step_parity_toy():
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    model = MLXMambaModel(cfg)
    B, L = 2, 32
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)
    result = check_forward_step_parity(model, tokens, to_numpy=_np, rtol=1e-4, atol=1e-5)
    assert result["ok"], result
