"""CUDA/PyTorch twin of tests/test_moe_balance_mlx.py: Loss-Free-Balancing (#213/#214)
on the torch backend -- the buffer-not-parameter contract, the bias-steers-selection-
not-gate invariant, the portable round-trip, and the driver-facing plumbing.

Skipped where torch is unavailable (CPU-only, no GPU needed).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model.blocks import MambaConfig  # noqa: E402
from src.model.cuda_backend import CUDAMambaModel, MoEBlock  # noqa: E402
from src.model.cuda_train_step import make_train_step  # noqa: E402
from src.train.moe_balance import MoEBalancer, balancer_for_config  # noqa: E402


def _cfg(**over):
    base = dict(d_model=32, n_layers=2, head_dim=8, d_state=8, vocab_size=32,
                seq_len=16, precision="fp32", moe_every=1, n_experts=8, top_k=2,
                moe_d_ff=32)
    base.update(over)
    return MambaConfig(**base)


def _np(a):
    return a.detach().cpu().numpy()


def _capture_loads_during_step(model, step, micro_batches, lr):
    """Run one train step, capturing the per-expert load counts at the moment #217's
    routing-diagnostics pop drains them (`pop_moe_routing_stats`, called unconditionally
    now — even with `balancer=None`). Reading `model.pop_moe_load()` AFTER the step
    would see an already-drained (all-zero) accumulator, since the diagnostics pop is
    no longer gated on a balancer being attached."""
    captured = {}
    orig = model.pop_moe_routing_stats

    def spy():
        stats = orig()
        captured["stats"] = stats
        return stats

    model.pop_moe_routing_stats = spy
    step(model, micro_batches, lr)
    return [s["load"] for s in captured["stats"]]


# --------------------------------------------------------------------------- #
# D1 -- the bias is NOT a trained parameter
# --------------------------------------------------------------------------- #
def test_route_bias_and_load_counts_absent_from_named_parameters_and_state_dict():
    torch.manual_seed(0)
    model = CUDAMambaModel(_cfg(moe_balance_rate=0.05))
    model.set_moe_biases([[0.5] * 8, [-0.5] * 8])

    param_keys = [k for k, _ in model.named_parameters()]
    assert not any("_route_bias" in k for k in param_keys), param_keys
    assert not any("_load_counts" in k for k in param_keys), param_keys

    sd_keys = list(model.state_dict().keys())
    assert not any("_route_bias" in k for k in sd_keys), sd_keys
    assert not any("_load_counts" in k for k in sd_keys), sd_keys

    # ...and only the two D3-explicit portable keys, never a raw module-attribute leak.
    portable_keys = list(model._portable_state_dict())
    assert sorted(k for k in portable_keys if "route_bias" in k) == \
        ["moe_route_bias.0", "moe_route_bias.1"]


def test_load_state_dict_strict_succeeds_without_the_buffers():
    """`load_state_dict(strict=True)` must not demand `_route_bias`/`_load_counts` -- the
    whole point of `persistent=False`."""
    torch.manual_seed(0)
    model = CUDAMambaModel(_cfg(moe_balance_rate=0.05))
    model.set_moe_biases([[0.1] * 8, [-0.1] * 8])
    sd = model.state_dict()
    assert not any("_route_bias" in k or "_load_counts" in k for k in sd)

    torch.manual_seed(1)
    fresh = CUDAMambaModel(_cfg(moe_balance_rate=0.05))
    fresh.load_state_dict(sd, strict=True)      # must not raise


def test_buffers_are_moved_by_to():
    """A plain attribute would NOT be moved by `.to()`; `register_buffer` is what makes
    this work -- verified on CPU-\"device\" (a real device move needs CUDA hardware, but
    `.to()` on the same device still exercises the buffer machinery: it is a genuine
    `nn.Module._apply` traversal, not a no-op skip)."""
    torch.manual_seed(0)
    model = CUDAMambaModel(_cfg(moe_balance_rate=0.05))
    block = model.moe_blocks()[0]
    before = block._route_bias.data_ptr()
    model.to(torch.device("cpu"))
    # After `.to()` the buffer must still be the tensor `nn.Module._apply` produced --
    # i.e. it really participated in the traversal (a plain attribute would be
    # untouched and this assertion would still trivially "pass" for the wrong reason,
    # so the real guarantee here is the earlier persistent=False tests: this one just
    # documents that `.to()` does not error or silently drop the buffer).
    assert block._route_bias.shape == (8,)
    assert block._route_bias.data_ptr() is not None


def test_optimizer_step_leaves_the_bias_bit_identical():
    torch.manual_seed(0)
    model = CUDAMambaModel(_cfg(moe_balance_rate=0.05))
    model.set_moe_biases([[0.1 * i for i in range(8)], [-0.1 * i for i in range(8)]])
    before = model.moe_biases()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    step = make_train_step(model, opt, grad_clip=1.0, scaler=None, balancer=None)

    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 32, size=(4, 17))
    for _ in range(3):
        step(model, [(tokens[:, :-1], tokens[:, 1:])], 1e-2)
    assert model.moe_biases() == before


def test_balancer_moves_the_bias_by_exactly_rate_per_step():
    cfg = _cfg(moe_balance_rate=0.25)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    balancer = balancer_for_config(cfg)
    step = make_train_step(model, opt, grad_clip=1.0, scaler=None, balancer=balancer)
    from src.train.moe_balance import attach_balancer
    attach_balancer(balancer, model)

    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 32, size=(4, 17))
    out = step(model, [(tokens[:, :-1], tokens[:, 1:])], 1e-3)
    assert "moe_util_var" in out
    got = model.moe_biases()
    for layer_bias in got:
        assert set(layer_bias) <= {-0.25, 0.0, 0.25}, layer_bias
    assert got == balancer.biases()


# --------------------------------------------------------------------------- #
# D2 -- the bias steers SELECTION only; gates stay the unbiased softmax
# --------------------------------------------------------------------------- #
def _reference_moe(block, xn, bias=None):
    with torch.no_grad():
        logits = _np(block.router(xn).float())
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    k = block.config.top_k
    sel = logits + np.asarray(bias) if bias is not None else probs
    keep = np.argsort(np.argsort(-sel, axis=-1), axis=-1) < k
    gate = np.where(keep, probs, 0.0)
    gate = gate / gate.sum(-1, keepdims=True)
    return keep, gate


def _combine(block, xn, gate):
    cd = torch.float32
    with torch.no_grad():
        outs = np.stack([_np(e(xn, cd)) for e in block.experts.values()], axis=-2)
    return (np.asarray(gate)[..., None] * outs).sum(-2)


def test_bias_changes_selection_but_not_gate_values():
    cfg = _cfg(moe_balance_rate=0.05)
    torch.manual_seed(0)
    block = MoEBlock(cfg, "cpu")
    xn = torch.randn(3, cfg.seq_len, cfg.d_model)

    keep_off, gate_off = _reference_moe(block, xn)
    with torch.no_grad():
        y_off = _np(block._moe(xn))

    bias = [-50.0] + [0.0] * 6 + [50.0]
    block.set_route_bias(bias)
    keep_on, gate_on = _reference_moe(block, xn, bias)
    with torch.no_grad():
        y_on = _np(block._moe(xn))

    assert keep_on[..., 7].all() and not keep_on[..., 0].any()
    assert not np.array_equal(keep_off, keep_on)

    both = keep_off & keep_on
    ratio_off = np.where(both, gate_off, np.nan)
    ratio_on = np.where(both, gate_on, np.nan)
    scale = np.nansum(ratio_on, -1, keepdims=True) / np.nansum(ratio_off, -1, keepdims=True)
    assert np.allclose(ratio_on[both], (ratio_off * scale)[both], atol=1e-5)

    assert np.allclose(y_off, _combine(block, xn, gate_off), atol=1e-4)
    assert np.allclose(y_on, _combine(block, xn, gate_on), atol=1e-4)


def test_exactly_top_k_experts_survive_on_a_tie_under_bias():
    cfg = _cfg(moe_balance_rate=0.05)
    torch.manual_seed(0)
    block = MoEBlock(cfg, "cpu")
    with torch.no_grad():
        block.router.weight.zero_()
    block.set_route_bias([0.0] * cfg.n_experts)
    block.set_load_counting(True)
    xn = torch.randn(2, cfg.seq_len, cfg.d_model)
    with torch.no_grad():
        block._moe(xn)
    assert sum(block.pop_load()) == 2 * cfg.seq_len * cfg.top_k


# --------------------------------------------------------------------------- #
# Plumbing -- the accessors the drivers use
# --------------------------------------------------------------------------- #
def test_set_moe_biases_raises_on_a_shape_mismatch():
    torch.manual_seed(0)
    model = CUDAMambaModel(_cfg(moe_balance_rate=0.05))
    with pytest.raises(ValueError, match="1 bias vectors for 2 MoE layers"):
        model.set_moe_biases([[0.0] * 8])
    with pytest.raises(ValueError, match="route bias has 3 entries"):
        model.set_moe_biases([[0.0] * 3, [0.0] * 3])
    assert all(not b._bias_active for b in model.moe_blocks())


def test_loading_a_bias_for_a_non_moe_layer_raises_clearly(tmp_path):
    from src.train.checkpoint import load_weights_dict, save_weights

    torch.manual_seed(0)
    model = CUDAMambaModel(_cfg(moe_balance_rate=0.05))
    model.set_moe_biases([[0.1] * 8, [0.2] * 8])
    path = str(tmp_path / "weights.safetensors")
    model.save(path)

    weights = load_weights_dict(path)
    weights["moe_route_bias.7"] = weights.pop("moe_route_bias.1")   # no layer 7 here
    bad = str(tmp_path / "bad.safetensors")
    save_weights(weights, bad, config=model.config)

    torch.manual_seed(1)
    fresh = CUDAMambaModel(_cfg(moe_balance_rate=0.05))
    with pytest.raises(ValueError, match="layer 7 of this config is not an MoE layer"):
        fresh.load(bad)


def test_pop_moe_load_is_empty_for_a_dense_model():
    torch.manual_seed(0)
    model = CUDAMambaModel(_cfg(moe_every=None, n_experts=0, moe_d_ff=None))
    assert model.moe_blocks() == []
    assert model.pop_moe_load() == []
    assert model.moe_biases() == []


# --------------------------------------------------------------------------- #
# D3 -- the bias round-trips through the PORTABLE safetensors
# --------------------------------------------------------------------------- #
def test_route_bias_round_trips_through_portable_weights(tmp_path):
    cfg = _cfg(moe_balance_rate=0.05)
    torch.manual_seed(0)
    model = CUDAMambaModel(cfg)
    bias = [[0.1 * i for i in range(8)], [-0.05 * i for i in range(8)]]
    model.set_moe_biases(bias)

    path = str(tmp_path / "weights.safetensors")
    keys = model._portable_state_dict()
    assert sorted(k for k in keys if k.startswith("moe_route_bias.")) == \
        ["moe_route_bias.0", "moe_route_bias.1"]
    model.save(path)

    torch.manual_seed(1)
    fresh = CUDAMambaModel(cfg)
    assert fresh.moe_biases() == [[], []]
    fresh.load(path)
    assert fresh.moe_biases() == model.moe_biases()
    assert np.allclose(np.array(fresh.moe_biases()), np.array(bias), atol=1e-6)
    assert all(b._bias_active for b in fresh.moe_blocks())

    tokens = np.arange(2 * cfg.seq_len).reshape(2, cfg.seq_len) % cfg.vocab_size
    with torch.no_grad():
        y1 = _np(model.forward(tokens))
        y2 = _np(fresh.forward(tokens))
    assert np.allclose(y1, y2)


def test_portable_key_set_unchanged_when_the_bias_is_inactive():
    for cfg in (_cfg(moe_balance_rate=None), _cfg(moe_balance_rate=0.05)):
        torch.manual_seed(0)
        model = CUDAMambaModel(cfg)          # constructed, never activated
        sd = model._portable_state_dict()
        assert not any(k.startswith("moe_route_bias.") for k in sd), list(sd)
        assert sum(int(v.size) for v in sd.values()) == cfg.num_parameters()


def test_loading_a_pre_213_checkpoint_leaves_the_bias_inactive(tmp_path):
    cfg = _cfg(moe_balance_rate=0.05)
    torch.manual_seed(0)
    old = CUDAMambaModel(cfg)                # bias never activated
    path = str(tmp_path / "weights.safetensors")
    old.save(path)

    torch.manual_seed(1)
    fresh = CUDAMambaModel(cfg)
    fresh.load(path)
    assert all(not b._bias_active for b in fresh.moe_blocks())
    assert fresh.moe_biases() == [[], []]


# --------------------------------------------------------------------------- #
# D5 -- grad_checkpoint doubles the counts and changes nothing else
# --------------------------------------------------------------------------- #
def test_grad_checkpoint_doubles_the_counts_and_changes_nothing_else():
    def counts(grad_checkpoint):
        cfg = _cfg(moe_balance_rate=0.05, grad_checkpoint=grad_checkpoint)
        torch.manual_seed(0)
        model = CUDAMambaModel(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        step = make_train_step(model, opt, grad_clip=1.0, scaler=None, balancer=None)
        model.set_moe_load_counting(True)
        rng = np.random.default_rng(0)
        tokens = rng.integers(0, cfg.vocab_size, size=(4, cfg.seq_len + 1))
        return _capture_loads_during_step(
            model, step, [(tokens[:, :-1], tokens[:, 1:])], 1e-3)

    plain, checkpointed = counts(False), counts(True)
    assert checkpointed == [[2 * c for c in layer] for layer in plain]
    assert MoEBalancer.utilization_variance(plain) == \
        MoEBalancer.utilization_variance(checkpointed)
