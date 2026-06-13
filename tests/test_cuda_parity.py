"""torch-only parity tests for the CUDA backend (#36). Skipped where torch is absent.

The pure-PyTorch backend runs on CPU, so these run anywhere torch installs — no GPU.
Checks, all in fp32 (~1e-4 rel), mirroring tests/test_mlx_parity.py:
  * SSM scan parity: chunked `parallel` vs a sequential reference from `recurrence`.
  * SSM scan vs an INDEPENDENT from-scratch numpy scan (shares no code with the torch
    paths), plus a long-context case guarding the chunked scan's fp32 overflow-safety.
  * forward/step parity: whole-model `forward` (parallel) vs `step` (recurrence).
  * portable round-trip: save -> reload into a fresh instance -> identical logits.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import load_config
from src.model.cuda_backend import CUDAMambaModel, SelectiveSSM
from src.conformance.forward_step_parity import check_forward_step_parity


def _np(a):
    return a.detach().cpu().numpy()


def _seq_reference_numpy(ssm, x_np):
    """Fully independent fp64 sequential scan of the scalar-A Mamba-2 / SSD SSM.

    Computes projections, softplus, scalar-A (per-head) discretization and the
    recurrence FROM SCRATCH in numpy using only the SSM's raw parameters — shares no
    code with `SelectiveSSM.parallel` or `.recurrence`. x_np: (B,L,d_inner) ->
    (B,L,d_inner) float64.
    """
    cfg = ssm.config
    dt_rank, N = cfg.dt_rank_resolved, cfg.d_state
    H, P = cfg.n_heads, cfg.head_dim

    # torch nn.Linear stores weight as (out, in), computes x @ W.T + b — same as MLX.
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
    torch.manual_seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, L, di = 2, cfg.seq_len, cfg.d_inner
    x = torch.randn(B, L, di) * 0.1

    with torch.no_grad():
        y_par = _np(ssm.parallel(x))

        # Sequential reference from the one-step recurrence. State is (B,H,P,N).
        h = torch.zeros((B, cfg.n_heads, cfg.head_dim, cfg.d_state))
        ys = []
        for t in range(L):
            y_t, h = ssm.recurrence(x[:, t], h)
            ys.append(_np(y_t))
    y_seq = np.stack(ys, axis=1)

    max_abs = float(np.abs(y_par.astype(np.float64) - y_seq.astype(np.float64)).max())
    assert np.allclose(y_par, y_seq, rtol=1e-4, atol=1e-5), f"max|diff|={max_abs:.3e}"


def test_ssm_parallel_matches_independent_reference():
    """`parallel` vs a from-scratch numpy scan (no shared code) at toy seq_len."""
    torch.manual_seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, L, di = 2, cfg.seq_len, cfg.d_inner
    x = torch.randn(B, L, di) * 0.1

    with torch.no_grad():
        y_par = _np(ssm.parallel(x)).astype(np.float64)
    y_ref = _seq_reference_numpy(ssm, _np(x))

    max_abs = float(np.abs(y_par - y_ref).max())
    assert np.allclose(y_par, y_ref, rtol=1e-4, atol=1e-5), f"max|diff|={max_abs:.3e}"


def test_ssm_parallel_overflow_safe_long_context():
    """Long context (L=512 -> 16 chunks of 32): the chunked scan must stay finite and
    still match the independent reference."""
    torch.manual_seed(0)
    cfg = load_config("config/toy.yaml")
    ssm = SelectiveSSM(cfg)
    B, L, di = 2, 512, cfg.d_inner
    x = torch.randn(B, L, di) * 0.1

    with torch.no_grad():
        y_par = _np(ssm.parallel(x)).astype(np.float64)
    assert np.isfinite(y_par).all(), "chunked scan produced NaN/Inf at long context"

    y_ref = _seq_reference_numpy(ssm, _np(x))
    max_abs = float(np.abs(y_par - y_ref).max())
    assert np.allclose(y_par, y_ref, rtol=1e-4, atol=1e-5), f"max|diff|={max_abs:.3e}"


def test_forward_step_parity_toy():
    torch.manual_seed(0)
    cfg = load_config("config/toy.yaml")
    model = CUDAMambaModel(cfg)
    model.eval()
    B, L = 2, 32
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)
    with torch.no_grad():
        result = check_forward_step_parity(model, tokens, to_numpy=_np, rtol=1e-4, atol=1e-5)
    assert result["ok"], result


def test_portable_roundtrip(tmp_path):
    """save -> reload into a fresh instance -> identical logits (the portable bridge)."""
    torch.manual_seed(0)
    cfg = load_config("config/toy.yaml")
    model = CUDAMambaModel(cfg)
    model.eval()
    tokens = np.random.default_rng(1).integers(0, cfg.vocab_size, size=(2, 16)).astype(np.int32)
    with torch.no_grad():
        before = _np(model.forward(tokens))

    path = str(tmp_path / "weights.safetensors")
    model.save(path)

    reloaded = CUDAMambaModel(cfg)
    reloaded.load(path)
    reloaded.eval()
    with torch.no_grad():
        after = _np(reloaded.forward(tokens))

    max_abs = float(np.abs(before.astype(np.float64) - after.astype(np.float64)).max())
    assert np.allclose(before, after, rtol=0, atol=0), f"round-trip drift max|diff|={max_abs:.3e}"
