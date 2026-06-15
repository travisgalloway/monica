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
    """Fully independent fp64 sequential scan of the scalar-A Mamba-2 / SSD SSM.

    Computes the projections, softplus, scalar-A (per-head) discretization and the
    recurrence FROM SCRATCH in numpy using only the SSM's raw parameters — it
    shares no code with `SelectiveSSM.parallel` (SSD matmul) or `.recurrence`, so a
    bug living identically in their shared `_project`/discretization cannot hide here.

    x_np: (B, L, d_inner) float -> (B, L, d_inner) float64.
    """
    cfg = ssm.config
    dt_rank, N = cfg.dt_rank_resolved, cfg.d_state
    H, P = cfg.n_heads, cfg.head_dim

    # Raw params -> fp64. MLX nn.Linear stores weight as (out, in), computes x @ W.T + b.
    Wx = _np(ssm.x_proj.weight).astype(np.float64)    # (dt_rank+2N, d_inner)
    Wdt = _np(ssm.dt_proj.weight).astype(np.float64)  # (H, dt_rank)
    bdt = _np(ssm.dt_proj.bias).astype(np.float64)    # (H,)
    a = -np.exp(_np(ssm.A_log).astype(np.float64))    # (H,) scalar decay per head
    D = _np(ssm.D).astype(np.float64)                 # (H,)

    x = x_np.astype(np.float64)
    B_, L, di = x.shape

    proj = x @ Wx.T                                   # (B, L, dt_rank+2N)
    Bm = proj[..., dt_rank:dt_rank + N]               # (B, L, N) shared across heads
    Cm = proj[..., dt_rank + N:]                      # (B, L, N)
    delta = np.logaddexp(proj[..., :dt_rank] @ Wdt.T + bdt, 0.0)  # softplus -> (B, L, H)

    X = x.reshape(B_, L, H, P)
    y = np.zeros((B_, L, H, P), np.float64)
    h = np.zeros((B_, H, P, N), np.float64)           # per-head state (P, N)
    for t in range(L):
        dA = np.exp(delta[:, t] * a)                  # (B, H)
        Xin = delta[:, t][..., None] * X[:, t]        # (B, H, P) input = dt * X
        dBx = Xin[..., None] * Bm[:, t][:, None, None, :]    # (B, H, P, N)
        h = dA[:, :, None, None] * h + dBx
        y[:, t] = np.sum(h * Cm[:, t][:, None, None, :], axis=-1)  # (B, H, P)
    y = y + X * D[None, None, :, None]
    return y.reshape(B_, L, di)


def test_ssm_parallel_matches_sequential():
    mx.random.seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, L, di = 2, cfg.seq_len, cfg.d_inner
    x = mx.random.normal((B, L, di)) * 0.1

    y_par = _np(ssm.parallel(x))

    # Sequential reference from the one-step recurrence. State is per-head (B,H,P,N).
    h = mx.zeros((B, cfg.n_heads, cfg.head_dim, cfg.d_state))
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


def test_forward_step_parity_hybrid():
    # The hybrid model interleaves attention blocks (RoPE + KV cache). forward (full
    # causal attention) and step (incremental cache) must still agree — the attention
    # path's train/infer equivalence, alongside the SSM's.
    mx.random.seed(0)
    cfg = load_config("config/toy-hybrid.yaml")
    model = MLXMambaModel(cfg)
    assert any(type(l).__name__ == "AttentionBlock" for l in model.layers)
    B, L = 2, 40
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)
    result = check_forward_step_parity(model, tokens, to_numpy=_np, rtol=1e-4, atol=1e-5)
    assert result["ok"], result
