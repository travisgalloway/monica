"""fp8 MoE-expert linears via NVIDIA Transformer Engine (#214/#240, LIVE).

The `_te_linear_cls()` capability probe and the `fp8_experts` config-validation rules
are plain-CPU testable and run everywhere. The bf16-vs-fp8 forward-equivalence +
finite-grad-under-checkpoint acceptance tests need real Hopper+ hardware with
`transformer_engine` installed (the `cuda-fp8` extra) — they build a real CUDA MoE
model via `_Expert`/`MoEBlock` (#214) and exercise the fp8 GEMM/checkpoint wiring, so
they stay GPU-gated behind the `_hopper` fixture and SKIP everywhere else, including
CI's `cuda-cpu` job (CPU torch, no CUDA device at all).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.model.blocks import MambaConfig, load_config  # noqa: E402
from src.model.cuda_backend import (CUDAMambaModel, MoEBlock, _te_linear_cls,  # noqa: E402
                                    fp8_status)


@pytest.fixture(scope="module")
def _hopper():
    """Skip unless BOTH a Hopper+ (sm_90) CUDA device AND `transformer_engine` are
    present — `fp8_status()` is exactly that combined probe (mirrors `_compile_works`
    in test_cuda_compile.py). Building an fp8 model without this fixture would just
    silently fall back to bf16 `nn.Linear`, defeating the point of these tests. The
    probe/validate tests above do NOT use this fixture, since they must run (and
    pass) on the non-Hopper Mac/CI box too."""
    if not fp8_status():
        pytest.skip("no Hopper+ (sm_90) CUDA device with transformer_engine installed here")


def _moe_config(**over):
    base = dict(d_model=64, n_layers=4, head_dim=16, d_state=16, vocab_size=256,
                seq_len=32, precision="bf16", moe_every=2, n_experts=4, top_k=2)
    base.update(over)
    return MambaConfig(**base)


def test_te_linear_cls_returns_none_without_crashing():
    """The probe never raises, and returns None when TE is absent or the device is
    pre-Hopper — the expected state on the non-Hopper Mac/CI box. On a real Hopper+
    box with `transformer-engine` installed (the `cuda-fp8` extra) this would instead
    return the TE Linear class; that path is exercised by the (currently skipped)
    acceptance tests below once #214 gives it something to build."""
    cls = _te_linear_cls()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        assert cls is None
    assert fp8_status() == (cls is not None)


def test_fp8_experts_requires_moe():
    cfg = load_config("config/toy.yaml")           # no moe_every
    cfg.fp8_experts = True
    cfg.precision = "bf16"
    with pytest.raises(ValueError, match="MoE"):
        cfg.validate()


def test_fp8_experts_requires_bf16_or_fp16_precision():
    cfg = load_config("config/toy-moe.yaml")        # has moe_every/n_experts
    cfg.fp8_experts = True
    cfg.precision = "fp32"
    with pytest.raises(ValueError, match="precision"):
        cfg.validate()


def test_fp8_experts_accepts_bf16_moe():
    cfg = load_config("config/toy-moe.yaml")
    cfg.fp8_experts = True
    cfg.precision = "bf16"
    cfg.validate()                                  # must not raise
    assert cfg.fp8_experts is True


def test_fp8_experts_default_off_is_unaffected():
    """Sanity: fp8_experts=False (the default) never trips the new validate() rules,
    even with fp32 + no MoE (the toy default)."""
    cfg = load_config("config/toy.yaml")
    assert cfg.fp8_experts is False
    cfg.validate()


def test_fp8_expert_forward_matches_bf16(_hopper):
    """An fp8-expert MoE forward should be CLOSE to (not equal to) its bf16-expert
    twin, both built from the same weights so the only difference under test is the
    expert GEMM precision:
      * finite logits — fp8 GEMMs aren't a NaN factory
      * ROUTING bit-identical at ZERO tolerance — the router is a plain `nn.Linear`
        that always routes in fp32 (`MoEBlock._fp8_ctx` wraps the expert GEMMs only,
        never the router), so fp8 must not leak into the routing DECISION even though
        the expert VALUES it dispatches to differ
      * an fp8-honest logit bound (~10% of the logit scale, NOT the usual 1e-4 fp32
        parity bar — e4m3 has only 3 mantissa bits)
      * loss agreement within ~5% — the hard gate
    """
    cfg_bf16 = _moe_config()
    cfg_fp8 = _moe_config(fp8_experts=True)
    cfg_bf16.validate()
    cfg_fp8.validate()

    torch.manual_seed(0)
    model_bf16 = CUDAMambaModel(cfg_bf16, device="cuda")
    model_fp8 = CUDAMambaModel(cfg_fp8, device="cuda")
    # Same weights for both — an fp8 vs bf16 GEMM is the only variable under test; a
    # different random init would confound the comparison.
    model_fp8.load_state_dict(model_bf16.state_dict(), strict=True)
    model_bf16.eval()
    model_fp8.eval()
    model_bf16.set_moe_load_counting(True)
    model_fp8.set_moe_load_counting(True)

    tokens = np.random.default_rng(0).integers(
        0, cfg_bf16.vocab_size, size=(2, 32)).astype(np.int32)
    with torch.no_grad():
        logits_bf16 = model_bf16.forward(tokens)
        logits_fp8 = model_fp8.forward(tokens)

    assert torch.isfinite(logits_fp8).all()

    loads_bf16 = model_bf16.pop_moe_load()
    loads_fp8 = model_fp8.pop_moe_load()
    assert loads_fp8 == loads_bf16, (loads_fp8, loads_bf16)   # zero-tolerance routing identity

    scale = logits_bf16.abs().mean().item()
    max_diff = (logits_fp8 - logits_bf16).abs().max().item()
    assert max_diff < 0.10 * scale, (max_diff, scale)

    targets = torch.as_tensor(
        np.random.default_rng(1).integers(0, cfg_bf16.vocab_size, size=(2, 32)),
        dtype=torch.long, device=logits_bf16.device)
    loss_bf16 = F.cross_entropy(logits_bf16.reshape(-1, cfg_bf16.vocab_size), targets.reshape(-1))
    loss_fp8 = F.cross_entropy(logits_fp8.reshape(-1, cfg_bf16.vocab_size), targets.reshape(-1))
    assert abs(loss_fp8.item() - loss_bf16.item()) < 0.05 * abs(loss_bf16.item())


def test_fp8_expert_checkpoint_backward_finite_grads(_hopper, monkeypatch):
    """fp8 experts under grad-checkpoint must use `transformer_engine.pytorch.
    checkpoint`, NOT `torch.utils.checkpoint` (plain checkpoint's recompute pass
    double-updates the fp8 amax history TE uses to calibrate its scale). "Grads are
    finite" alone would also pass with the WRONG checkpoint function, so this also
    monkeypatches `te.checkpoint` with a call counter to prove it was actually
    selected — the sharp assertion this test exists for."""
    import transformer_engine.pytorch as te
    calls = {"n": 0}
    real_checkpoint = te.checkpoint

    def _counting_checkpoint(*args, **kwargs):
        calls["n"] += 1
        return real_checkpoint(*args, **kwargs)

    monkeypatch.setattr(te, "checkpoint", _counting_checkpoint)

    cfg = _moe_config(fp8_experts=True, grad_checkpoint=True)
    cfg.validate()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg, device="cuda")
    model.train()

    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 32)).astype(np.int32)
    logits = model.forward(tokens)
    logits.float().sum().backward()

    assert calls["n"] > 0, "te.checkpoint was never called — wrong checkpoint fn selected"

    block = next(l for l in model.layers if isinstance(l, MoEBlock))
    any_nonzero = False
    for expert in block.experts.values():
        for p in expert.parameters():
            assert p.grad is not None, "expert param has no grad (no-continue invariant broken)"
            assert torch.isfinite(p.grad).all()
            if torch.any(p.grad != 0):
                any_nonzero = True
    assert any_nonzero, "no routed expert received a non-zero grad"
