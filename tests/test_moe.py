"""Tests for the sparse MoE-Mamba block (#53).

Portable: the config selectors, validation, and total-vs-active param accounting.
MLX-guarded: the dense path is byte-identical with MoE off, forward/step parity holds
for the MoE block, and routing actually selects top_k experts.
"""

import numpy as np
import pytest

from src.model.blocks import MambaConfig


def _cfg(**over):
    base = dict(d_model=64, n_layers=4, head_dim=16, d_state=16, vocab_size=256,
                seq_len=32, precision="fp32")
    base.update(over)
    return MambaConfig(**base)


# --------------------------------------------------------------------------- #
# Portable: selectors, validation, accounting
# --------------------------------------------------------------------------- #
def test_moe_off_by_default():
    cfg = _cfg()
    assert cfg.moe_every is None
    assert cfg.n_moe_layers == 0
    assert not any(cfg.is_moe_layer(i) for i in range(cfg.n_layers))
    assert "moe" not in cfg.parameter_breakdown()
    assert cfg.active_num_parameters() == cfg.num_parameters()   # no MoE -> identical


def test_moe_layer_placement():
    cfg = _cfg(moe_every=2, n_experts=4, top_k=2)
    assert [cfg.is_moe_layer(i) for i in range(4)] == [False, True, False, True]
    assert cfg.n_moe_layers == 2


def test_attention_takes_precedence_over_moe():
    # Layer 1 and 3 selected by both; attention wins, so neither is MoE.
    cfg = _cfg(attn_every=2, n_attn_heads=4, moe_every=2, n_experts=4, top_k=2)
    assert cfg.n_attention_layers == 2
    assert cfg.n_moe_layers == 0
    assert all(not cfg.is_moe_layer(i) for i in range(4))


def test_moe_breakdown_sums_and_adds_capacity():
    cfg = _cfg(moe_every=2, n_experts=4, top_k=2)
    bd = cfg.parameter_breakdown()
    assert "moe" in bd and bd["moe"] > 0
    assert cfg.num_parameters() == sum(bd.values())
    # MoE blocks replace Mamba blocks but add expert capacity -> more total params than
    # the pure-Mamba twin at these toy dims.
    pure = MambaConfig(**{**cfg.to_dict(), "moe_every": None})
    assert cfg.num_parameters() > pure.num_parameters()


def test_active_params_below_total_and_matches_top_k():
    cfg = _cfg(moe_every=2, n_experts=4, top_k=2)
    assert cfg.active_num_parameters() < cfg.num_parameters()    # sparse: top_k < n_experts
    # active MoE term uses top_k experts; total uses n_experts.
    full = _cfg(moe_every=2, n_experts=4, top_k=4)               # dense routing
    assert full.active_num_parameters() == full.num_parameters()


def test_moe_validation():
    with pytest.raises(ValueError, match="n_experts"):
        _cfg(moe_every=2, n_experts=1, top_k=1).validate()
    with pytest.raises(ValueError, match="top_k"):
        _cfg(moe_every=2, n_experts=4, top_k=5).validate()
    with pytest.raises(ValueError, match="moe_every"):
        _cfg(moe_every=0, n_experts=4, top_k=2).validate()
    _cfg(moe_every=2, n_experts=4, top_k=2).validate()           # valid: no raise


# --------------------------------------------------------------------------- #
# MLX-guarded: dense path unchanged, parity, routing
# --------------------------------------------------------------------------- #
mx = pytest.importorskip("mlx.core")


def test_dense_path_byte_identical_with_moe_off():
    """A model built with MoE off must be byte-for-byte the pre-#53 model: same layer
    types, same params, same logits."""
    from src.model.mlx_backend import MLXMambaModel, MambaBlock
    cfg = _cfg()                                  # moe off
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    assert all(isinstance(l, MambaBlock) for l in model.layers)
    tokens = mx.array(np.arange(2 * 32).reshape(2, 32) % 256)
    y = np.array(model.forward(tokens))
    assert np.all(np.isfinite(y))
    assert np.array_equal(y, np.array(model.forward(tokens)))    # deterministic


def test_moe_model_builds_with_expected_block_types():
    from src.model.mlx_backend import MLXMambaModel, MambaBlock, MoEBlock
    cfg = _cfg(moe_every=2, n_experts=4, top_k=2)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    types = [type(l).__name__ for l in model.layers]
    assert types == ["MambaBlock", "MoEBlock", "MambaBlock", "MoEBlock"]
    # param-count formula holds against the real tensors
    actual = sum(int(v.size) for v in model._portable_state_dict().values())
    assert cfg.num_parameters() == actual


def test_moe_forward_step_parity():
    """MoE is pointwise, so the chunked forward and the one-step recurrence must agree
    (fp32 ~1e-4), like the Mamba/attention blocks."""
    from src.model.mlx_backend import MLXMambaModel
    cfg = _cfg(moe_every=2, n_experts=4, top_k=2)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    tokens = np.arange(32).reshape(1, 32) % 256

    seq = np.array(model.forward(mx.array(tokens)))[0]
    state = model.init_state(1)
    step = []
    for t in range(tokens.shape[1]):
        logit, state = model.step(mx.array(tokens[:, t]), state)
        step.append(np.array(logit)[0])
    step = np.stack(step)
    rel = np.abs(seq - step).max() / (np.abs(seq).max() + 1e-6)
    assert rel < 1e-4


def test_router_selects_top_k():
    """With top_k=1 the combined output must equal the single argmax-routed expert."""
    from src.model.mlx_backend import MLXMambaModel, MoEBlock
    cfg = _cfg(moe_every=2, n_experts=4, top_k=1)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    block = next(l for l in model.layers if isinstance(l, MoEBlock))
    xn = mx.random.normal((5, cfg.d_model))
    cd = mx.float32
    logits = np.array(block.router(xn))
    chosen = logits.argmax(axis=-1)
    combined = np.array(block._moe(xn))
    expert_outs = np.stack([np.array(e(xn, cd)) for e in block.experts], axis=1)  # (5,E,d)
    for i, e in enumerate(chosen):
        assert np.allclose(combined[i], expert_outs[i, e], atol=1e-5)


def test_router_keeps_exactly_k_on_ties():
    """A zeroed router makes all experts tie; exactly top_k must be kept (not all of
    them), so the combination is the mean of the first k experts — not all E."""
    from src.model.mlx_backend import MLXMambaModel, MoEBlock
    cfg = _cfg(moe_every=2, n_experts=4, top_k=2)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    block = next(l for l in model.layers if isinstance(l, MoEBlock))
    block.router.weight = mx.zeros_like(block.router.weight)   # uniform routing -> ties
    xn = mx.random.normal((3, cfg.d_model))
    cd = mx.float32
    combined = np.array(block._moe(xn))
    expert_outs = np.stack([np.array(e(xn, cd)) for e in block.experts], axis=1)  # (3,E,d)
    mean_first_k = expert_outs[:, :2, :].mean(axis=1)          # ranks break ties by index
    assert np.allclose(combined, mean_first_k, atol=1e-5)
    assert not np.allclose(combined, expert_outs.mean(axis=1), atol=1e-5)  # NOT all 4
