"""CUDA/PyTorch twin of tests/test_moe.py (#214): the CUDA MoE block, mirroring the MLX
dense reference plus the new grouped-gather routing.

Skipped where torch is unavailable (runs on CPU — no GPU needed, like the rest of the
`test_cuda_*` suite).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.model.blocks import MambaConfig  # noqa: E402
from src.model.cuda_backend import CUDAMambaModel, MambaBlock, MoEBlock  # noqa: E402


def _cfg(**over):
    base = dict(d_model=64, n_layers=4, head_dim=16, d_state=16, vocab_size=256,
                seq_len=32, precision="fp32", moe_every=2, n_experts=4, top_k=2)
    base.update(over)
    return MambaConfig(**base)


def _np(a):
    return a.detach().cpu().numpy()


def _moe_block(model):
    return next(l for l in model.layers if isinstance(l, MoEBlock))


# --------------------------------------------------------------------------- #
# Block types + param accounting
# --------------------------------------------------------------------------- #
def test_moe_model_builds_with_expected_block_types():
    cfg = _cfg()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    types = [type(l).__name__ for l in model.layers]
    assert types == ["MambaBlock", "MoEBlock", "MambaBlock", "MoEBlock"]


def test_moe_param_count_matches_formula():
    cfg = _cfg()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    actual = sum(int(v.size) for v in model._portable_state_dict().values())
    assert cfg.num_parameters() == actual


def test_moe_shared_expert_param_count_matches_formula():
    cfg = _cfg(n_shared_experts=1)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    block = _moe_block(model)
    assert len(block.shared_experts) == 1
    actual = sum(int(v.size) for v in model._portable_state_dict().values())
    assert cfg.num_parameters() == actual


# --------------------------------------------------------------------------- #
# forward/step parity (MoE is pointwise -> trivially exact, but pinned anyway)
# --------------------------------------------------------------------------- #
def test_moe_forward_step_parity():
    from src.conformance.forward_step_parity import check_forward_step_parity

    cfg = _cfg()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    model.eval()
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 32)).astype(np.int32)
    with torch.no_grad():
        result = check_forward_step_parity(model, tokens, to_numpy=_np, rtol=1e-4, atol=1e-5)
    assert result["ok"], result


@pytest.mark.parametrize("impl", ["dense", "gather"])
def test_moe_forward_step_parity_both_impls(impl):
    from src.conformance.forward_step_parity import check_forward_step_parity

    cfg = _cfg(moe_impl=impl)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    model.eval()
    tokens = np.random.default_rng(1).integers(0, cfg.vocab_size, size=(2, 32)).astype(np.int32)
    with torch.no_grad():
        result = check_forward_step_parity(model, tokens, to_numpy=_np, rtol=1e-4, atol=1e-5)
    assert result["ok"], result


# --------------------------------------------------------------------------- #
# init_state / forward_prefill / clone_state: the (B,0) placeholder round-trip
# --------------------------------------------------------------------------- #
def test_moe_state_placeholder_round_trip():
    cfg = _cfg()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    model.eval()
    moe_idx = 1
    assert cfg.is_moe_layer(moe_idx)

    state = model.init_state(2)
    conv, ssm = state[moe_idx]
    assert conv.shape == (2, 0) and ssm.shape == (2, 0)

    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 16)).astype(np.int32)
    with torch.no_grad():
        _, pf_state = model.prefill(tokens)
    pconv, pssm = pf_state[moe_idx]
    assert pconv.shape == (2, 0) and pssm.shape == (2, 0)

    cloned = model.clone_state(state)
    cconv, cssm = cloned[moe_idx]
    assert cconv.shape == (2, 0) and cssm.shape == (2, 0)
    assert cconv is not conv and cssm is not ssm     # independent snapshot, not aliased


# --------------------------------------------------------------------------- #
# mixing_matrices: MoE layers (like attention) are skipped -- no mixing_matrix exists
# --------------------------------------------------------------------------- #
def test_mixing_matrices_skips_moe_layers():
    cfg = _cfg()
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    model.eval()
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 16)).astype(np.int32)
    with torch.no_grad():
        out = model.mixing_matrices(tokens)
    indices = [i for i, _ in out]
    assert indices == [0, 2]                          # the Mamba layers only (1, 3 are MoE)
    for i, m in out:
        assert torch.isfinite(m).all()


# --------------------------------------------------------------------------- #
# Routing correctness
# --------------------------------------------------------------------------- #
def test_router_top_k_one_equals_argmax_expert():
    cfg = _cfg(top_k=1)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    model.eval()
    block = _moe_block(model)
    xn = torch.randn(5, cfg.d_model)
    cd = torch.float32
    with torch.no_grad():
        logits = _np(block.router(xn))
        chosen = logits.argmax(axis=-1)
        combined = _np(block._moe(xn))
        expert_outs = np.stack([_np(e(xn, cd)) for e in block.experts], axis=1)  # (5,E,d)
    for i, e in enumerate(chosen):
        assert np.allclose(combined[i], expert_outs[i, e], atol=1e-5)


@pytest.mark.parametrize("impl", ["dense", "gather"])
def test_router_keeps_exactly_k_on_ties(impl):
    """A zeroed router makes all experts tie; exactly top_k must be kept (ranks break
    ties by index), for BOTH compute strategies — the whole point of the gather kernel
    is to reproduce the dense reference's decision, not just its logits."""
    cfg = _cfg(moe_impl=impl)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    model.eval()
    block = _moe_block(model)
    with torch.no_grad():
        block.router.weight.zero_()
    xn = torch.randn(3, cfg.d_model)
    cd = torch.float32
    with torch.no_grad():
        combined = _np(block._moe(xn))
        expert_outs = np.stack([_np(e(xn, cd)) for e in block.experts], axis=1)  # (3,E,d)
    mean_first_k = expert_outs[:, :cfg.top_k, :].mean(axis=1)   # ranks break ties by index
    assert np.allclose(combined, mean_first_k, atol=1e-5)
    assert not np.allclose(combined, expert_outs.mean(axis=1), atol=1e-5)   # NOT all E


def test_moe_degenerate_single_expert_equals_hand_computed_swiglu():
    """E=1 (the dense-upcycle source, #214): softmax over a single logit is exactly 1.0
    in fp32, so the block's output must equal the raw SwiGLU computation bit-for-bit."""
    cfg = _cfg(n_experts=1, top_k=1)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    model.eval()
    block = _moe_block(model)
    xn = torch.randn(4, cfg.d_model)
    with torch.no_grad():
        combined = block._moe(xn)
        e = block.experts[0]
        g = xn @ e.gate.weight.t()
        u = xn @ e.up.weight.t()
        expected = (F.silu(g) * u) @ e.down.weight.t()
    assert torch.equal(combined, expected)


# --------------------------------------------------------------------------- #
# Tie-break contract (#214).
#
# MLX, Swift and this backend all rank with `argsort(argsort(-sel))`, whose ties break by
# ASCENDING EXPERT INDEX. `MoEBlock._moe` deliberately does not use `torch.topk`, whose
# tie order torch does not specify and which is observably platform-dependent: on the tie
# vector below, torch 2.12.0 returns {2,5,6} on macOS/arm64 but {0,5,6} on Linux/x86_64.
# That is WHY topk is avoided -- but it is rationale, not an invariant, and asserting
# `topk != reference` asserts on behaviour torch never promised (it fails whenever topk
# happens to coincide, which says nothing about this code). So the assertions below pin
# the two things that ARE contractual: the reference semantics, and the block obeying them.
# --------------------------------------------------------------------------- #
def test_double_argsort_breaks_ties_by_ascending_index():
    """The primitive MLX/Swift/CUDA all agree on. Ties -> lowest expert index wins."""
    sel = torch.tensor([1., 1., 1., 0., 0., 2., 2., 0.])
    order = torch.argsort(-sel, stable=True)
    assert set(order[:3].tolist()) == {0, 5, 6}
    # stable=True is the whole contract; without it the tie order is unspecified.
    assert order[:3].tolist() == [5, 6, 0]


@pytest.mark.parametrize("impl", ["dense", "gather"])
def test_moe_block_selection_obeys_double_argsort_tie_break(impl):
    """The real routing path, on a deliberate all-ties fixture.

    A zeroed router makes every logit 0, so `sel` is exactly the route bias and the
    top-k choice is decided purely by the tie-break. Both impls must land on {0,5,6}
    -- ascending index among equals -- for every token.
    """
    cfg = _cfg(n_experts=8, top_k=3, moe_impl=impl)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    block = _moe_block(model)
    with torch.no_grad():
        block.router.weight.zero_()
    block.set_route_bias([1., 1., 1., 0., 0., 2., 2., 0.])
    block.set_load_counting(True)

    x = torch.randn(2, 5, cfg.d_model)
    with torch.no_grad():
        block.forward_seq(x)

    loads = block.pop_load()
    n_tokens = 2 * 5
    assert [i for i, c in enumerate(loads) if c > 0] == [0, 5, 6]
    assert all(loads[i] == n_tokens for i in (0, 5, 6))
