"""Backend parity (#38): MLX and torch agree, and portable weights round-trip.

Because the CUDA backend runs on torch-CPU, the cross-backend checks are runnable
entirely on a Mac (mlx + torch both present) — no GPU. They SKIP cleanly when either
backend is missing, so the suite stays green on single-backend hosts:
  * this Linux container (torch present, mlx not installable) — cross-backend tests
    skip; the torch-only harness self-check below still runs;
  * a CUDA host without mlx — same;
  * a Mac without torch — all skip.

All comparisons are fp32, ~1e-4 rel (the documented tolerance; bf16/fp16 epsilon is too
coarse to be meaningful), per src/conformance/backend_parity.py.
"""

import numpy as np
import pytest

try:
    import mlx.core  # noqa: F401
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from src.model.blocks import load_config
from src.conformance.backend_parity import check_backend_parity

CFG = "config/toy.yaml"


def _tokens(cfg, B=2, L=24, seed=0):
    return np.random.default_rng(seed).integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)


def _mlx_np(a):
    return np.array(a)


def _torch_np(a):
    return a.detach().cpu().numpy()


@pytest.mark.skipif(not (HAVE_MLX and HAVE_TORCH),
                    reason="needs both mlx and torch (run on a Mac)")
def test_backend_parity_mlx_vs_torch(tmp_path):
    """Identical portable weights in both backends -> `forward` agrees in fp32."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.cuda_backend import CUDAMambaModel

    cfg = load_config(CFG)
    # One source of weights -> both backends (torch is the source here; the round-trip
    # test below proves the other direction).
    torch.manual_seed(0)
    src = CUDAMambaModel(cfg)
    path = str(tmp_path / "weights.safetensors")
    src.save(path)

    mlx_m = MLXMambaModel(cfg)
    mlx_m.load(path)
    cuda_m = CUDAMambaModel(cfg)
    cuda_m.load(path)

    tokens = _tokens(cfg)
    with torch.no_grad():
        result = check_backend_parity(mlx_m, cuda_m, tokens,
                                      to_numpy_a=_mlx_np, to_numpy_b=_torch_np,
                                      rtol=1e-4, atol=1e-5)
    assert result["ok"], result


@pytest.mark.skipif(not (HAVE_MLX and HAVE_TORCH),
                    reason="needs both mlx and torch (run on a Mac)")
def test_portable_weights_roundtrip_both_directions(tmp_path):
    """MLX save -> torch _load_portable -> torch save -> load back into MLX; the MLX
    logits are unchanged. Proves the cross-backend bridge in both directions (a
    CUDA-trained model can come back to the Mac)."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.cuda_backend import CUDAMambaModel

    cfg = load_config(CFG)
    import mlx.core as mx
    mx.random.seed(0)
    mlx_src = MLXMambaModel(cfg)
    tokens = _tokens(cfg)
    before = _mlx_np(mlx_src.forward(tokens))

    p_mlx = str(tmp_path / "from_mlx.safetensors")
    mlx_src.save(p_mlx)                         # MLX -> portable
    bridge = CUDAMambaModel(cfg)
    bridge.load(p_mlx)                          # portable -> torch
    p_torch = str(tmp_path / "from_torch.safetensors")
    bridge.save(p_torch)                        # torch -> portable

    mlx_back = MLXMambaModel(cfg)
    mlx_back.load(p_torch)                      # portable -> MLX
    after = _mlx_np(mlx_back.forward(tokens))

    max_abs = float(np.abs(before.astype(np.float64) - after.astype(np.float64)).max())
    assert np.allclose(before, after, rtol=1e-4, atol=1e-5), f"round-trip drift {max_abs:.3e}"


@pytest.mark.skipif(not HAVE_TORCH, reason="needs torch")
def test_parity_harness_torch_self(tmp_path):
    """Runnable without mlx: identical weights in two torch instances pass the parity
    harness (exercises check_backend_parity + the to_numpy plumbing on this host)."""
    from src.model.cuda_backend import CUDAMambaModel

    cfg = load_config(CFG)
    torch.manual_seed(0)
    src = CUDAMambaModel(cfg)
    path = str(tmp_path / "weights.safetensors")
    src.save(path)

    a = CUDAMambaModel(cfg)
    a.load(path)
    b = CUDAMambaModel(cfg)
    b.load(path)

    tokens = _tokens(cfg)
    with torch.no_grad():
        result = check_backend_parity(a, b, tokens,
                                      to_numpy_a=_torch_np, to_numpy_b=_torch_np,
                                      rtol=1e-4, atol=1e-5)
    assert result["ok"], result
