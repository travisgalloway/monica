"""fp8 MoE-expert linears via NVIDIA Transformer Engine (#240, DESIGN-ONLY).

#240 is blocked on #214 (CUDA MoE backend — no `_Expert`/`MoEBlock` exists in
`cuda_backend.py` yet), so only the pieces that don't need a CUDA MoE expert are
tested live here: the `_te_linear_cls()` capability probe and the `fp8_experts`
config-validation rules. The bf16-vs-fp8 forward-equivalence + finite-grad-under-
checkpoint acceptance tests (the real #240 payoff) are recorded as skipped
placeholders so the intent survives to the #214 landing, when they get un-skipped
and filled in against the real CUDA `_Expert`.

This whole module SKIPS its Hopper/TE-specific tests on the non-Hopper Mac/CI box
(no CUDA, no `transformer_engine` installed) — only the probe/validate tests run
there, and they must PASS everywhere (they assert graceful absence, not presence).
"""

import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import load_config
from src.model.cuda_backend import _te_linear_cls, fp8_status


@pytest.fixture(scope="module")
def _hopper():
    """Skip if there's no Hopper+ (sm_90) CUDA device — mirrors `_compile_works` in
    test_cuda_compile.py. TE-specific acceptance tests build on this; the probe test
    below does NOT use this fixture, since it must run (and pass) on the Mac box too."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("no Hopper+ (sm_90) CUDA device here")


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


@pytest.mark.skip(reason="BLOCKED-ON-#214: no CUDA MoE _Expert/MoEBlock to build fp8 "
                          "linears against yet; un-skip once #214 lands.")
def test_fp8_expert_forward_matches_bf16(_hopper):
    """Acceptance (deferred): a fp8-expert MoE forward should match the bf16-expert
    forward within an fp8-appropriate tolerance (NOT the 1e-4 fp32 parity bar — fp8
    e4m3 has ~2 decimal digits; expect ~1e-1 rel or a loss-delta bound). Needs #214's
    CUDA `_Expert`/`MoEBlock` to exist before this can build a model."""
    raise NotImplementedError


@pytest.mark.skip(reason="BLOCKED-ON-#214: no CUDA MoE _Expert/MoEBlock to build fp8 "
                          "linears against yet; un-skip once #214 lands.")
def test_fp8_expert_checkpoint_backward_finite_grads(_hopper):
    """Acceptance (deferred): fp8 experts under grad-checkpoint (using
    `transformer_engine.pytorch.checkpoint`, not `torch.utils.checkpoint` — see the
    `# BLOCKED-ON-#214` note in cuda_backend.py) should backprop to finite grads.
    Needs #214's CUDA `_Expert`/`MoEBlock` to exist before this can build a model."""
    raise NotImplementedError
