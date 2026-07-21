"""CUDA/PyTorch Muon + hybrid optimizer (#237). Runs on CPU — no GPU needed.

Covers: Newton-Schulz orthogonality, the Adam sub-group bit-matching a stock AdamW,
optimizer-state resume through `HybridOptimizer`'s state_dict round-trip, and the
two-LR wiring (`group["lr"] * lr_scale`) that survives the scheduler clobbering
`group["lr"]` every step (see `_accumulate_and_step` in `cuda_train_step.py`).
"""

import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import is_muon_param, load_config
from src.model.backend import get_backend
from src.model.cuda_backend import CUDAMambaModel
from src.model.cuda_muon import HybridOptimizer, Muon, _newton_schulz5

TOY_MUON_CFG = "config/toy-muon.yaml"


def _orthogonality_check(G):
    # The standard Muon quintic coefficients (3.4445, -4.7750, 2.0315) do not converge
    # singular values exactly to 1 in 5 steps — by design they push them into a tight
    # band (empirically ~[0.68, 1.13] here), which is sufficient orthogonalization for
    # the optimizer's purpose. Assert the band, not exact unit singular values.
    X = _newton_schulz5(G, steps=5)
    sv = torch.linalg.svdvals(X)
    assert sv.min() > 0.5 and sv.max() < 1.25, sv


def test_newton_schulz5_orthogonal_tall():
    torch.manual_seed(0)
    _orthogonality_check(torch.randn(40, 10))


def test_newton_schulz5_orthogonal_wide():
    torch.manual_seed(0)
    _orthogonality_check(torch.randn(10, 40))


def test_newton_schulz5_runs_in_fp32_regardless_of_input_dtype():
    torch.manual_seed(0)
    G = torch.randn(20, 6, dtype=torch.float16)
    X = _newton_schulz5(G, steps=5)
    assert X.dtype == torch.float16                # returned in the param dtype
    sv = torch.linalg.svdvals(X.float())
    assert sv.min() > 0.5 and sv.max() < 1.25, sv


def test_adam_subgroup_bitmatches_stock_adamw():
    """The AdamW half of HybridOptimizer, driven through the scheduler-write pattern
    (`group["lr"] = lr` every step), must reproduce a stock AdamW exactly."""
    torch.manual_seed(0)
    p1 = torch.nn.Parameter(torch.randn(8))
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    ref1 = torch.nn.Parameter(p1.detach().clone())
    ref2 = torch.nn.Parameter(p2.detach().clone())

    hybrid = HybridOptimizer(torch.optim.AdamW([p1, p2], lr=1e-3), None)
    ref_opt = torch.optim.AdamW([ref1, ref2], lr=1e-3)

    for _ in range(5):
        for g in hybrid.param_groups:
            g["lr"] = 1e-3
        p1.grad = torch.full_like(p1, 0.1)
        p2.grad = torch.full_like(p2, 0.1)
        hybrid.step()
        hybrid.zero_grad()

        ref1.grad = torch.full_like(ref1, 0.1)
        ref2.grad = torch.full_like(ref2, 0.1)
        ref_opt.step()
        ref_opt.zero_grad()

    assert torch.equal(p1, ref1)
    assert torch.equal(p2, ref2)


def test_hybrid_state_dict_round_trip_resumes_exactly():
    """Save mid-run, rebuild fresh sub-optimizers, load_state_dict, and continue —
    the post-resume trajectory must match the uninterrupted reference exactly (fp32,
    fixed grads — mirrors the smoke gate's save/kill/resume protocol)."""
    torch.manual_seed(0)

    def fresh_params():
        torch.manual_seed(0)
        return (torch.nn.Parameter(torch.randn(8)),        # AdamW
                torch.nn.Parameter(torch.randn(6, 3)))      # Muon

    def make_hybrid(p_adam, p_muon):
        adam = torch.optim.AdamW([p_adam], lr=1e-2)
        muon = Muon([p_muon], lr=1e-2, lr_scale=2.0, momentum=0.9, ns_steps=5)
        return HybridOptimizer(adam, muon)

    def grads_for(step, p_adam, p_muon):
        g = torch.Generator().manual_seed(step)
        return (torch.randn(p_adam.shape, generator=g),
               torch.randn(p_muon.shape, generator=g))

    def run(p_adam, p_muon, hybrid, lo, hi):
        for s in range(lo, hi):
            ga, gm = grads_for(s, p_adam, p_muon)
            p_adam.grad, p_muon.grad = ga, gm
            for g in hybrid.param_groups:
                g["lr"] = 1e-2
            hybrid.step()
            hybrid.zero_grad()

    # Uninterrupted reference.
    ref_adam, ref_muon = fresh_params()
    ref_hybrid = make_hybrid(ref_adam, ref_muon)
    run(ref_adam, ref_muon, ref_hybrid, 0, 8)

    # Save/kill/resume at step 4.
    a_p, m_p = fresh_params()
    hybrid = make_hybrid(a_p, m_p)
    run(a_p, m_p, hybrid, 0, 4)
    sd = hybrid.state_dict()

    a_p2, m_p2 = torch.nn.Parameter(a_p.detach().clone()), torch.nn.Parameter(m_p.detach().clone())
    hybrid2 = make_hybrid(a_p2, m_p2)
    hybrid2.load_state_dict(sd)
    run(a_p2, m_p2, hybrid2, 4, 8)

    assert torch.allclose(a_p2, ref_adam, atol=1e-6)
    assert torch.allclose(m_p2, ref_muon, atol=1e-6)


def test_effective_lr_is_group_lr_times_lr_scale():
    """Pins the two-LR wiring: the scheduler writes a plain `group["lr"]` every step
    (see `_accumulate_and_step`), and Muon must derive its effective lr as
    `group["lr"] * lr_scale`, never persisting `muon_lr` directly into the group."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(6, 3))
    grad = torch.randn(6, 3)
    p.grad = grad.clone()

    muon = Muon([p], lr=1.0, lr_scale=2.5, momentum=0.9, ns_steps=5)
    hybrid = HybridOptimizer(None, muon)

    scheduled_lr = 0.02
    for g in hybrid.param_groups:
        g["lr"] = scheduled_lr
    before = p.detach().clone()
    hybrid.step()

    # Momentum buffer starts at zero, so after one step buf == grad exactly.
    ns = _newton_schulz5(grad, steps=5)
    scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
    expected = before - (scheduled_lr * 2.5) * scale * ns
    assert torch.allclose(p, expected, atol=1e-6)


def test_hybrid_optimizer_tolerates_empty_side():
    """An all-AdamW or all-Muon partition (one side empty) must not crash step/zero_grad/
    state_dict/load_state_dict."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(4, 4))
    p.grad = torch.ones_like(p)
    hybrid = HybridOptimizer(torch.optim.AdamW([p], lr=1e-3), None)
    hybrid.step()
    hybrid.zero_grad()
    sd = hybrid.state_dict()
    assert sd["muon"] is None
    hybrid.load_state_dict(sd)                       # no-op on the muon side, must not raise


def test_backend_partitions_real_model_params_via_is_muon_param():
    """End-to-end: `get_backend("cuda").make_optimizer` on a `optimizer: muon` config
    partitions the real CUDA model's named_parameters() exactly per `is_muon_param`."""
    cfg = load_config(TOY_MUON_CFG)
    assert cfg.optimizer == "muon"
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    backend = get_backend("cuda")
    opt = backend.make_optimizer(model, 1e-3)
    assert isinstance(opt, HybridOptimizer)

    muon_ids = {id(p) for g in opt.muon.param_groups for p in g["params"]}
    adam_ids = {id(p) for g in opt.adam.param_groups for p in g["params"]}
    for name, p in model.named_parameters():
        expected_muon = is_muon_param(name, p.ndim)
        assert (id(p) in muon_ids) == expected_muon, name
        assert (id(p) in adam_ids) == (not expected_muon), name

    # Sanity: a hybrid model has both attention and Mamba 2D matrices, so neither side
    # of the real partition is empty.
    assert muon_ids and adam_ids


def test_mlx_backend_raises_on_muon_config():
    pytest.importorskip("mlx")
    cfg = load_config(TOY_MUON_CFG)
    backend = get_backend("mlx")
    model = backend.model_cls(cfg)
    with pytest.raises(NotImplementedError):
        backend.make_optimizer(model, 1e-3)
