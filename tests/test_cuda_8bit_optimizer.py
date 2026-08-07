"""8-bit AdamW moments (#214), an orthogonal axis to `optimizer` composing with the Muon
hybrid — see `MambaConfig.optimizer_8bit` and `backend.py::_cuda_backend._make_optimizer`'s
`_adamw` helper.

`bitsandbytes` ships CUDA-only prebuilt wheels (no CPU/macOS wheel at all), so its actual
8-bit numerics are UNVERIFIABLE without a real CUDA box with the `[cuda-8bit]` extra
installed — see docs/infrastructure.md's manual-verification checklist. What runs here,
everywhere, without a GPU: the config surface, the sizing-table consequence, the MLX
rejection, and — the load-bearing one — that a missing `bitsandbytes` install surfaces as
a legible `SystemExit` naming the `[cuda-8bit]` extra rather than a bare `ModuleNotFoundError`
or (worse) a silent 32-bit fallback. `tests/test_import_guard.py`'s `FORBIDDEN_ROOTS` proves
the import itself stays lazy (bitsandbytes never lands in `sys.modules` from importing
`src.model.backend` alone).
"""

import pytest

torch = pytest.importorskip("torch")

from src.model.backend import get_backend
from src.model.blocks import MambaConfig, load_config
from src.model.cuda_backend import CUDAMambaModel


def _cfg(**over):
    base = dict(d_model=64, n_layers=2, head_dim=16, d_state=16, vocab_size=256,
                seq_len=32, precision="fp32")
    base.update(over)
    return MambaConfig(**base)


# --------------------------------------------------------------------------- #
# Config surface
# --------------------------------------------------------------------------- #
def test_optimizer_8bit_defaults_off():
    cfg = _cfg()
    assert cfg.optimizer_8bit is False
    cfg.validate()                                  # must not raise


def test_optimizer_8bit_composes_with_muon():
    """Orthogonal axis, not a third `optimizer` enum value — `optimizer_8bit=True` must
    validate cleanly alongside `optimizer: muon` (the plan's stated composition point;
    the doc pass notes the memory payoff is small there, but it must still validate)."""
    cfg = _cfg(optimizer="muon", optimizer_8bit=True)
    cfg.validate()


def test_toy_moe_8bit_fixture_loads_and_validates():
    cfg = load_config("config/toy-moe-8bit.yaml")
    assert cfg.optimizer == "adamw"
    assert cfg.optimizer_8bit is True


# --------------------------------------------------------------------------- #
# Sizing-table consequence (#214's `optimizer_sizing_key` property)
# --------------------------------------------------------------------------- #
def test_optimizer_sizing_key_reflects_8bit():
    assert _cfg(optimizer_8bit=False).optimizer_sizing_key == "adamw_fp32"
    assert _cfg(optimizer_8bit=True).optimizer_sizing_key == "adam8bit_fp32"


# --------------------------------------------------------------------------- #
# MLX rejection — mirrors the existing Muon raise (backend.py::_mlx_backend)
# --------------------------------------------------------------------------- #
def test_mlx_backend_raises_on_optimizer_8bit():
    pytest.importorskip("mlx")
    cfg = _cfg(optimizer_8bit=True)
    backend = get_backend("mlx")
    model = backend.model_cls(cfg)
    with pytest.raises(NotImplementedError):
        backend.make_optimizer(model, 1e-3)


# --------------------------------------------------------------------------- #
# CUDA: legible SystemExit when bitsandbytes is absent (the box this suite runs on)
# --------------------------------------------------------------------------- #
def test_cuda_backend_raises_legible_systemexit_without_bitsandbytes():
    pytest.importorskip("torch")
    if _bitsandbytes_available():
        pytest.skip("bitsandbytes IS installed here — the absent-dependency path is "
                     "untestable on this box; see the manual pod checklist instead.")
    cfg = _cfg(optimizer_8bit=True)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    backend = get_backend("cuda")
    with pytest.raises(SystemExit, match=r"cuda-8bit"):
        backend.make_optimizer(model, 1e-3)


def test_cuda_backend_adamw_muon_hybrid_also_raises_without_bitsandbytes():
    """The `_adamw` helper is shared by both the `adamw` branch AND the `muon` branch's
    AdamW remainder side — this pins that the switch fires on the muon path too, not
    just the pure-adamw one (the whole point of factoring it into one helper)."""
    pytest.importorskip("torch")
    if _bitsandbytes_available():
        pytest.skip("bitsandbytes IS installed here — see the manual pod checklist instead.")
    cfg = _cfg(optimizer="muon", optimizer_8bit=True)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    backend = get_backend("cuda")
    with pytest.raises(SystemExit, match=r"cuda-8bit"):
        backend.make_optimizer(model, 1e-3)


def _bitsandbytes_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("bitsandbytes") is not None
