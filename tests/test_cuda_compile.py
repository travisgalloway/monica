"""torch.compile of the CUDA student forward (#145). Torch CPU; skipped where torch
(or a working inductor/C-compiler) is absent.

The flag `MambaConfig.torch_compile` wraps `CUDAMambaModel._forward_compute` with
torch.compile. These guard two things:
  * compile changes NO numerics — compiled vs eager logits agree at fp32 ~1e-4, built
    from one shared state_dict (the same standard as tests/test_cuda_parity.py);
  * the compiled region traces through the grad-checkpoint wrapper (the riskiest
    interaction the #145 plan flagged) — a forward+backward yields finite grads.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import load_config
from src.model.cuda_backend import CUDAMambaModel


def _np(a):
    return a.detach().cpu().numpy()


@pytest.fixture(scope="module")
def _compile_works():
    """Skip the whole module if torch.compile can't run here (e.g. CI without a C
    compiler) — feature-detect on a trivial fn rather than assume."""
    try:
        f = torch.compile(lambda t: t * 2 + 1)
        f(torch.zeros(2))
    except Exception as e:  # pragma: no cover - environment-dependent
        pytest.skip(f"torch.compile unusable here: {type(e).__name__}: {e}")


def test_compiled_forward_matches_eager(_compile_works):
    """Compiled forward == eager forward at fp32 ~1e-4 (hybrid: Mamba + attention)."""
    torch.manual_seed(0)
    cfg = load_config("config/toy-hybrid.yaml")          # fp32, exercises both block types
    eager = CUDAMambaModel(cfg)
    eager.eval()

    cfg_c = load_config("config/toy-hybrid.yaml")
    cfg_c.torch_compile = True
    compiled = CUDAMambaModel(cfg_c)
    compiled.load_state_dict(eager.state_dict())         # share weights exactly
    compiled.eval()

    B, L = 2, cfg.seq_len
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)
    with torch.no_grad():
        y_eager = _np(eager.forward(tokens))
        y_comp = _np(compiled.forward(tokens))
    max_abs = float(np.max(np.abs(y_eager - y_comp)))
    assert np.allclose(y_eager, y_comp, rtol=1e-4, atol=1e-5), f"max|diff|={max_abs:.3e}"


def test_compiled_grad_checkpoint_backward_runs(_compile_works):
    """Compile + grad_checkpoint (use_reentrant=False) traces and backprops to finite
    grads — the interaction the #145 plan called out as the likely break point."""
    torch.manual_seed(0)
    cfg = load_config("config/toy-hybrid.yaml")
    cfg.torch_compile = True
    cfg.grad_checkpoint = True
    model = CUDAMambaModel(cfg)
    model.train()

    B, L = 2, cfg.seq_len
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)
    targets = torch.as_tensor(rng.integers(0, cfg.vocab_size, size=(B, L)), dtype=torch.long)

    logits = model.forward(tokens)                        # (B, L, V), fp32 head
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()

    assert torch.isfinite(loss), f"non-finite loss {loss}"
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients populated"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def test_compiled_forward_matches_eager_pure_mamba(_compile_works):
    """Compiled == eager at fp32 ~1e-4 for a PURE-Mamba config (no attention block)."""
    torch.manual_seed(0)
    cfg = load_config("config/toy.yaml")
    eager = CUDAMambaModel(cfg)
    eager.eval()

    cfg_c = load_config("config/toy.yaml")
    cfg_c.torch_compile = True
    compiled = CUDAMambaModel(cfg_c)
    compiled.load_state_dict(eager.state_dict())
    compiled.eval()

    B, L = 2, cfg.seq_len
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(B, L)).astype(np.int32)
    with torch.no_grad():
        y_eager = _np(eager.forward(tokens))
        y_comp = _np(compiled.forward(tokens))
    max_abs = float(np.max(np.abs(y_eager - y_comp)))
    assert np.allclose(y_eager, y_comp, rtol=1e-4, atol=1e-5), f"max|diff|={max_abs:.3e}"


def test_torch_compile_auto_resolves_eager_on_cpu():
    """Default (None) => AUTO: a CPU-built model must NOT compile (CPU is the parity
    surface). Asserted via the resolved `_compiled` decision flag."""
    cfg = load_config("config/toy.yaml")
    assert cfg.torch_compile is None          # tri-state default
    model = CUDAMambaModel(cfg)               # device defaults to CPU
    assert model._compiled is False


def test_torch_compile_explicit_true_compiles_on_cpu(_compile_works):
    """Explicit True is honored on ANY device (the parity tests rely on this)."""
    cfg = load_config("config/toy.yaml")
    cfg.torch_compile = True
    model = CUDAMambaModel(cfg)
    assert model._compiled is True
