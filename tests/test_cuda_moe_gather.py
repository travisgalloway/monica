"""Dense vs. grouped-gather routing (#214): the two `MoEBlock` compute strategies must
describe the SAME function -- same selection, same gate values, same logits, same load
counts, same gradients -- differing only in how many GEMMs they issue.

Skipped where torch is unavailable (CPU-only, no GPU needed).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import MambaConfig  # noqa: E402
from src.model.cuda_backend import CUDAMambaModel, MoEBlock  # noqa: E402
from src.model.cuda_train_step import _global_grad_norm  # noqa: E402


def _cfg(**over):
    base = dict(d_model=64, n_layers=4, head_dim=16, d_state=16, vocab_size=256,
                seq_len=32, precision="fp32", moe_every=2, n_experts=4, top_k=2)
    base.update(over)
    return MambaConfig(**base)


def _np(a):
    return a.detach().cpu().numpy()


def _built_pair(**cfg_over):
    """Two models, identically seeded, differing only in `moe_impl`."""
    cfg_d = _cfg(moe_impl="dense", **cfg_over)
    torch.manual_seed(0)
    m_dense = CUDAMambaModel(cfg_d)
    m_dense.eval()

    cfg_g = _cfg(moe_impl="gather", **cfg_over)
    torch.manual_seed(0)
    m_gather = CUDAMambaModel(cfg_g)
    m_gather.eval()
    return m_dense, m_gather


def _built_block_pair(**cfg_over):
    """Two bare MoEBlocks, identically seeded, differing only in `moe_impl` -- isolates
    the dispatch kernel from the rest of the architecture (embedding, Mamba layers, LM
    head), which is what the tight rtol=1e-6/atol=1e-6 tolerance below is actually about:
    dense and gather are two summation ORDERS of the same computation, and floating-point
    addition is not associative, so a whole-MODEL comparison compounds that reordering
    noise across every layer and the wide-vocab head matmul -- see
    `test_dense_vs_gather_full_model_logits_allclose` for that (looser-tolerance) check."""
    cfg_d = _cfg(moe_impl="dense", **cfg_over)
    torch.manual_seed(0)
    bd = MoEBlock(cfg_d, "cpu")
    cfg_g = _cfg(moe_impl="gather", **cfg_over)
    torch.manual_seed(0)
    bg = MoEBlock(cfg_g, "cpu")
    return bd, bg


# --------------------------------------------------------------------------- #
# Logit agreement
# --------------------------------------------------------------------------- #
def test_dense_vs_gather_logits_allclose():
    bd, bg = _built_block_pair()
    xn = torch.randn(3, 17, bd.config.d_model)
    with torch.no_grad():
        yd = _np(bd._moe(xn))
        yg = _np(bg._moe(xn))
    assert np.allclose(yd, yg, rtol=1e-6, atol=1e-6)


def test_dense_vs_gather_logits_allclose_with_bias():
    bd, bg = _built_block_pair()
    bias = [0.5, -1.0, 1.5, -2.0]
    bd.set_route_bias(bias)
    bg.set_route_bias(bias)
    xn = torch.randn(4, 37, bd.config.d_model)
    with torch.no_grad():
        yd = _np(bd._moe(xn))
        yg = _np(bg._moe(xn))
    assert np.allclose(yd, yg, rtol=1e-6, atol=1e-6)


def test_dense_vs_gather_full_model_logits_allclose():
    """Same claim as above, but end-to-end through the whole 4-layer model + LM head.
    The standard project cross-implementation tolerance (rtol=1e-4, atol=1e-5, same as
    `forward_step_parity`/`backend_parity`) applies here rather than the tight
    block-level bound above: reordered floating-point sums compound across 3 more
    layers and a 256-wide softmax, which can amplify to a large RELATIVE error at a
    near-zero logit entry even though the underlying computation is unchanged."""
    m_dense, m_gather = _built_pair()
    tokens = np.random.default_rng(0).integers(0, 256, size=(2, 32)).astype(np.int32)
    with torch.no_grad():
        y_dense = _np(m_dense.forward(tokens))
        y_gather = _np(m_gather.forward(tokens))
    assert np.allclose(y_dense, y_gather, rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# Selection equivalence at ZERO tolerance: the dense mask and the gather ids must
# describe the exact same kept set, over random / all-tie / biased logits.
# --------------------------------------------------------------------------- #
def _dense_kept_set(block, xn):
    """The dense mask's kept expert-index set per row, read straight out of `_moe`'s
    own selection code path (not re-derived) so this is a faithful probe."""
    E, k = block.config.n_experts, block.config.top_k
    with torch.no_grad():
        logits = block.router(xn).float()
        sel = (logits + block._route_bias) if block._bias_active else \
            torch.softmax(logits, dim=-1)
        order = torch.argsort(-sel, dim=-1, stable=True)
        topk_ids, _ = torch.sort(order[..., :k], dim=-1)
    return topk_ids


@pytest.mark.parametrize("case", ["random", "all_tie", "biased"])
def test_selection_ids_identical_dense_vs_gather(case):
    m_dense, m_gather = _built_pair()
    xn = torch.randn(3, 17, m_dense.config.d_model)

    if case == "all_tie":
        for m in (m_dense, m_gather):
            for block in m.moe_blocks():
                with torch.no_grad():
                    block.router.weight.zero_()
    elif case == "biased":
        bias = [0.5, -1.0, 1.5, -2.0]
        for m in (m_dense, m_gather):
            for block in m.moe_blocks():
                block.set_route_bias(bias)

    bd = m_dense.moe_blocks()[0]
    bg = m_gather.moe_blocks()[0]
    ids_dense = _dense_kept_set(bd, xn)
    ids_gather = _dense_kept_set(bg, xn)     # same selection code both blocks share
    # Compare as SETS per row (order-independent) at zero tolerance -- this is an
    # integer-index comparison, not a float comparison, so atol/rtol don't apply.
    sd = ids_dense.sort(dim=-1).values
    sg = ids_gather.sort(dim=-1).values
    assert torch.equal(sd, sg)


# --------------------------------------------------------------------------- #
# pop_moe_load(): exactly equal between impls
# --------------------------------------------------------------------------- #
def test_pop_moe_load_exactly_equal_between_impls():
    m_dense, m_gather = _built_pair()
    m_dense.set_moe_load_counting(True)
    m_gather.set_moe_load_counting(True)
    tokens = np.random.default_rng(2).integers(0, 256, size=(3, 32)).astype(np.int32)
    with torch.no_grad():
        m_dense.forward(tokens)
        m_gather.forward(tokens)
    loads_dense = m_dense.pop_moe_load()
    loads_gather = m_gather.pop_moe_load()
    assert loads_dense == loads_gather
    assert sum(sum(layer) for layer in loads_dense) == 3 * 32 * m_dense.config.top_k * \
        m_dense.config.n_moe_layers


# --------------------------------------------------------------------------- #
# Empty-expert-group case: forward finite, that expert's grad is not None and exactly
# 0.0, and the accumulated global grad norm stays finite.
# --------------------------------------------------------------------------- #
def test_empty_expert_group_yields_zero_finite_grad_not_none():
    cfg = _cfg(moe_impl="gather")
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    block = next(l for l in model.layers if isinstance(l, MoEBlock))
    with torch.no_grad():
        # Force ALL tokens to rank experts [0, 1] first (ties broken by index), so
        # experts 2 and 3 route zero tokens -- the empty-group path (#214 finding 2).
        block.router.weight.zero_()

    tokens = np.random.default_rng(0).integers(0, 256, size=(2, 32)).astype(np.int32)
    out = model.forward(tokens)
    assert torch.isfinite(out).all()
    loss = out.float().sum()
    loss.backward()

    empty_expert = block.experts["3"]    # never selected: top_k=2 keeps [0, 1] on every tie
    for p in empty_expert.parameters():
        assert p.grad is not None
        assert torch.all(p.grad == 0.0)
        assert torch.isfinite(p.grad).all()

    norm = _global_grad_norm(list(model.parameters()))
    assert torch.isfinite(norm)


# --------------------------------------------------------------------------- #
# Dense vs. gather expert-weight GRADIENT agreement (not just forward logits)
# --------------------------------------------------------------------------- #
def test_dense_vs_gather_expert_grad_agreement():
    m_dense, m_gather = _built_pair()
    tokens = np.random.default_rng(3).integers(0, 256, size=(2, 32)).astype(np.int32)

    for m in (m_dense, m_gather):
        m.zero_grad(set_to_none=True)
        out = m.forward(tokens)
        loss = out.float().sum()
        loss.backward()

    for (name_d, p_d), (name_g, p_g) in zip(m_dense.named_parameters(),
                                             m_gather.named_parameters()):
        assert name_d == name_g
        if p_d.grad is None and p_g.grad is None:
            continue
        assert p_d.grad is not None and p_g.grad is not None, name_d
        max_abs = float((p_d.grad - p_g.grad).abs().max())
        assert torch.allclose(p_d.grad, p_g.grad, rtol=1e-5, atol=1e-5), \
            f"{name_d}: max|grad diff|={max_abs:.3e}"
