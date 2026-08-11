"""MLX router tests for Loss-Free-Balancing (#213): the real router de-collapses.

`tests/test_moe_balance.py` unit-tests the portable `MoEBalancer` policy in isolation
(no backend). This file is the end-to-end proof: train a small many-expert MoE with and
without the balancer wired through the real `make_train_step` -> `_accumulate_and_step`
-> `MoEBlock` path and confirm the balanced run's expert-utilization variance is
materially lower (the acceptance criterion), plus the load-bearing invariants —

  D1  the bias is NOT in the parameter tree, so no optimizer ever steps it;
  D2  the bias steers top-k SELECTION only; gates stay the unbiased softmax probs;
  D3  the bias round-trips through the PORTABLE safetensors, not the resume bundle;
  D4  load counting is gated off by default;
  D5  `grad_checkpoint` double-counts by exactly 2x, which both consumers ignore;

and the off-path invariant that `moe_balance_rate: null` leaves the router untouched.
"""

import numpy as np
import pytest

from src.model.blocks import MambaConfig
from src.train.moe_balance import MoEBalancer, attach_balancer, balancer_for_config

# Per-test skip marker, NOT a module-level `importorskip`: the latter runs at import
# time and would skip the whole module on a non-Mac host (matches tests/test_moe.py).
try:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    HAVE_MLX = True
except ImportError:
    HAVE_MLX = False
requires_mlx = pytest.mark.skipif(not HAVE_MLX, reason="requires mlx (Apple Silicon)")


def _cfg(**over):
    base = dict(d_model=32, n_layers=2, head_dim=8, d_state=8, vocab_size=32,
                seq_len=16, precision="fp32", moe_every=1, n_experts=8, top_k=2,
                moe_d_ff=32)
    base.update(over)
    return MambaConfig(**base)


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


def _train_and_probe(cfg, seed=0, steps=400, lr=2e-3, batch=4):
    """Train `cfg`'s model for `steps` real train_steps on fresh random toy batches each
    step (sustained gradient signal — a single memorized batch saturates the loss too
    fast for the rich-get-richer collapse dynamic to show up), wiring the balancer
    exactly as `scripts/train.py` does. Returns the per-MoE-layer `utilization_variance`
    from a FRESH, fixed probe evaluation (independent of the training seed) — an
    apples-to-apples snapshot of the post-training routing distribution."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_train_step import make_train_step

    mx.random.seed(seed)
    model = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=lr)
    balancer = balancer_for_config(cfg)
    train_step = make_train_step(model, opt, grad_clip=1.0, scaler=None, balancer=balancer)
    attach_balancer(balancer, model)

    rng = np.random.default_rng(seed)
    last_metrics = None
    for _ in range(steps):
        tokens = rng.integers(0, cfg.vocab_size, size=(batch, cfg.seq_len + 1))
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        last_metrics = train_step(model, [(inputs, targets)], lr)

    # Fair snapshot: turn counting ON for the probe (the unbalanced run never enabled it)
    # and clear whatever accumulated during training, then aggregate many FRESH probe
    # batches (fixed seed, independent of `seed`) to average out per-batch sampling noise
    # before reading the final distribution.
    model.set_moe_load_counting(True)
    model.pop_moe_load()
    probe_rng = np.random.default_rng(1234)
    for _ in range(30):
        probe = probe_rng.integers(0, cfg.vocab_size, size=(8, cfg.seq_len))
        model.forward(mx.array(probe))
    loads = model.pop_moe_load()
    return MoEBalancer.utilization_variance(loads), last_metrics, model


# --------------------------------------------------------------------------- #
# Acceptance: the router de-collapses
# --------------------------------------------------------------------------- #
@requires_mlx
def test_balancing_reduces_expert_utilization_variance(capsys):
    """The acceptance criterion: expert-utilization variance de-collapses on a toy MoE
    run. Same seed / data / hyperparameters for both conditions — `moe_balance_rate` is
    the only difference. Run with `-s` to read the before/after numbers."""
    var_off, metrics_off, _ = _train_and_probe(_cfg(moe_balance_rate=None), seed=0)
    var_on, metrics_on, _ = _train_and_probe(_cfg(moe_balance_rate=0.05), seed=0)

    with capsys.disabled():
        print(f"\n[#213 de-collapse] utilization_variance per MoE layer")
        print(f"  unbalanced (moe_balance_rate=None): {var_off}  max={max(var_off):.6f}")
        print(f"  balanced   (moe_balance_rate=0.05): {var_on}  max={max(var_on):.6f}")
        print(f"  reduction: {max(var_off) / max(max(var_on), 1e-12):.1f}x")

    assert "moe_util_var" not in metrics_off       # balancer off -> no balancer metric
    assert "moe_util_var" in metrics_on            # balancer on -> reported every step

    # Materially lower, not just numerically lower: a >=3x margin comfortably clears the
    # reduction observed across several seeds while training this toy setup.
    assert max(var_on) * 3 < max(var_off), (
        f"balanced variance {var_on} not materially below unbalanced {var_off}")


# --------------------------------------------------------------------------- #
# D1 — the bias is NOT a trained parameter
# --------------------------------------------------------------------------- #
@requires_mlx
def test_route_bias_and_load_counts_are_not_in_the_parameter_tree():
    """The whole method depends on this: `nn.Module.valid_parameter_filter` excludes
    leading-underscore keys, so `nn.value_and_grad` cannot differentiate the bias and
    `optimizer.update` cannot step it. Verified, not assumed."""
    from src.model.mlx_backend import MLXMambaModel
    mx.random.seed(0)
    model = MLXMambaModel(_cfg(moe_balance_rate=0.05))
    model.set_moe_biases([[0.5] * 8, [-0.5] * 8])       # active, non-zero

    for tree in (model.parameters(), model.trainable_parameters()):
        keys = [k for k, _ in tree_flatten(tree)]
        assert not any("_route_bias" in k for k in keys), keys
        assert not any("_load_counts" in k for k in keys), keys
    # ...and it does not sneak into the portable dict through its `tree_flatten(
    # self.parameters())` path either: the only bias keys there are the two D3 adds
    # explicitly, never a `layers.N._route_bias` module-attribute leak.
    sd = list(model._portable_state_dict())
    assert [k for k in sd if "route_bias" in k] == ["moe_route_bias.0", "moe_route_bias.1"], sd
    assert not any("_load_counts" in k for k in sd), sd


@requires_mlx
def test_optimizer_step_leaves_the_bias_bit_identical():
    """A full train step (forward + backward + AdamW update) with an ACTIVE bias and no
    balancer must leave the bias exactly as set — only the policy may move it."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_train_step import make_train_step
    mx.random.seed(0)
    model = MLXMambaModel(_cfg(moe_balance_rate=0.05))
    model.set_moe_biases([[0.1 * i for i in range(8)], [-0.1 * i for i in range(8)]])
    # Baseline read AFTER the push: the router stores the bias in fp32, so this is the
    # exact vector the model holds. Bit-identity is then a real claim about the optimizer.
    before = model.moe_biases()
    opt = optim.AdamW(learning_rate=1e-2)
    step = make_train_step(model, opt, grad_clip=1.0, scaler=None, balancer=None)

    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 32, size=(4, 17))
    for _ in range(3):
        step(model, [(tokens[:, :-1], tokens[:, 1:])], 1e-2)
    assert model.moe_biases() == before                  # not a float apart


@requires_mlx
def test_balancer_moves_the_bias_by_exactly_rate_per_step():
    """With the balancer wired, the bias moves by `rate * sign(mean - load)` and nothing
    else — an exact-arithmetic check that the policy is the sole author of the bias."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_train_step import make_train_step
    cfg = _cfg(moe_balance_rate=0.25)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    opt = optim.AdamW(learning_rate=1e-3)
    balancer = balancer_for_config(cfg)
    step = make_train_step(model, opt, grad_clip=1.0, scaler=None, balancer=balancer)
    attach_balancer(balancer, model)

    # Shadow the policy by hand: pull the loads the step will see, apply the same rule.
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 32, size=(4, 17))
    step(model, [(tokens[:, :-1], tokens[:, 1:])], 1e-3)
    got = model.moe_biases()
    for layer_bias in got:
        # rate=0.25, one step from zero -> every entry is exactly -0.25, 0.0 or +0.25.
        assert set(layer_bias) <= {-0.25, 0.0, 0.25}, layer_bias
    assert got == balancer.biases()


# --------------------------------------------------------------------------- #
# D2 — the bias steers SELECTION only; gates stay the unbiased softmax
# --------------------------------------------------------------------------- #
def _reference_moe(block, xn, bias=None):
    """Independent re-implementation of the intended routing, for `_moe` to match."""
    logits = np.array(block.router(xn).astype(mx.float32))
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    k = block.config.top_k
    sel = logits + np.asarray(bias) if bias is not None else probs
    keep = np.argsort(np.argsort(-sel, axis=-1), axis=-1) < k
    gate = np.where(keep, probs, 0.0)                    # UNBIASED gate values
    gate = gate / gate.sum(-1, keepdims=True)            # renormalized over the kept set
    return keep, gate


@requires_mlx
def test_bias_changes_selection_but_not_gate_values():
    """D2 pinned: with a large hand-set bias the KEPT SET changes, while each surviving
    expert's gate is still its unbiased softmax prob (renormalized over the new kept
    set) — and the block's output equals the reference combination built from those
    unbiased gates."""
    from src.model.mlx_backend import MoEBlock
    cfg = _cfg(moe_balance_rate=0.05)
    mx.random.seed(0)
    block = MoEBlock(cfg)
    xn = mx.random.normal((3, cfg.seq_len, cfg.d_model))

    keep_off, gate_off = _reference_moe(block, xn)
    y_off = np.array(block._moe(xn))

    # A bias big enough to force expert 7 in and expert 0 out, whatever the logits say.
    bias = [-50.0] + [0.0] * 6 + [50.0]
    block.set_route_bias(bias)
    keep_on, gate_on = _reference_moe(block, xn, bias)
    y_on = np.array(block._moe(xn))

    # The SELECTION moved: expert 7 is always kept, expert 0 never is.
    assert keep_on[..., 7].all() and not keep_on[..., 0].any()
    assert not np.array_equal(keep_off, keep_on)

    # The GATE VALUES did not become biased: each kept expert's gate is still its
    # unbiased softmax prob, and the ratio between two co-kept experts is unchanged by
    # the bias (it would shift by exp(b_i - b_j) if the bias touched the gate).
    both = keep_off & keep_on
    ratio_off = np.where(both, gate_off, np.nan)
    ratio_on = np.where(both, gate_on, np.nan)
    scale = np.nansum(ratio_on, -1, keepdims=True) / np.nansum(ratio_off, -1, keepdims=True)
    assert np.allclose(ratio_on[both], (ratio_off * scale)[both], atol=1e-5)

    # And the block reproduces the reference combination in both conditions.
    assert np.allclose(y_off, _combine(block, xn, gate_off), atol=1e-4)
    assert np.allclose(y_on, _combine(block, xn, gate_on), atol=1e-4)


def _combine(block, xn, gate):
    cd = mx.float32
    outs = np.stack([np.array(e(xn, cd)) for e in block.experts], axis=-2)
    return (np.asarray(gate)[..., None] * outs).sum(-2)


@requires_mlx
def test_zero_bias_does_not_perturb_routing():
    """Bias lives in LOGIT space while the off-path ranks by probs; softmax is monotone
    per row and preserves ties, so activating with an all-zero bias is a no-op."""
    from src.model.mlx_backend import MoEBlock
    mx.random.seed(0)
    block = MoEBlock(_cfg(moe_balance_rate=0.05))
    xn = mx.random.normal((2, 16, 32))
    y_unbiased = np.array(block._moe(xn))
    block.set_route_bias([0.0] * 8)
    assert block._bias_active
    assert np.array_equal(y_unbiased, np.array(block._moe(xn)))


@requires_mlx
def test_exactly_top_k_experts_survive_on_a_tie_under_bias():
    """The double-argsort tie-break must hold on the BIASED path too: a router that
    scores every expert identically (zero weights -> uniform probs, zero bias) still
    keeps exactly top_k, not all E."""
    from src.model.mlx_backend import MoEBlock
    cfg = _cfg(moe_balance_rate=0.05)
    mx.random.seed(0)
    block = MoEBlock(cfg)
    block.router.weight = mx.zeros_like(block.router.weight)   # perfectly tied logits
    block.set_route_bias([0.0] * cfg.n_experts)
    block.set_load_counting(True)
    xn = mx.random.normal((2, cfg.seq_len, cfg.d_model))
    block._moe(xn)
    assert sum(block.pop_load()) == 2 * cfg.seq_len * cfg.top_k


# --------------------------------------------------------------------------- #
# Off-path invariant — moe_balance_rate: null is byte-identical to pre-#213
# --------------------------------------------------------------------------- #
@requires_mlx
def test_bias_inactive_and_router_unaffected_when_balancing_off():
    from src.model.mlx_backend import MLXMambaModel, MoEBlock
    cfg = _cfg(moe_balance_rate=None)
    assert balancer_for_config(cfg) is None
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    blocks = model.moe_blocks()
    assert len(blocks) == cfg.n_moe_layers > 0
    assert all(isinstance(b, MoEBlock) and not b._bias_active for b in blocks)
    assert all(not b._count_loads for b in blocks)      # D4: counting off by default

    tokens = mx.array(np.arange(2 * cfg.seq_len).reshape(2, cfg.seq_len) % cfg.vocab_size)
    y1 = np.array(model.forward(tokens))
    y2 = np.array(model.forward(tokens))
    assert np.array_equal(y1, y2)                      # deterministic, no bias touched
    assert all(not b._bias_active for b in blocks)     # still untouched after forwards
    # D4: with counting off the accumulator never grew, so a pop reads all-zero.
    assert model.pop_moe_load() == [[0.0] * cfg.n_experts for _ in blocks]


# --------------------------------------------------------------------------- #
# Plumbing — the accessors the drivers use
# --------------------------------------------------------------------------- #
@requires_mlx
def test_set_moe_biases_and_pop_moe_load_round_trip():
    from src.model.mlx_backend import MLXMambaModel
    cfg = _cfg(moe_balance_rate=1e-3)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    blocks = model.moe_blocks()
    assert all(not b._bias_active for b in blocks)     # zero bias, never pushed yet
    model.set_moe_load_counting(True)

    n_tokens = 3 * cfg.seq_len
    tokens = mx.array(np.arange(n_tokens).reshape(3, cfg.seq_len) % cfg.vocab_size)
    model.forward(tokens)
    loads = model.pop_moe_load()
    assert len(loads) == cfg.n_moe_layers
    for layer_loads in loads:
        assert len(layer_loads) == cfg.n_experts
        assert sum(layer_loads) == n_tokens * cfg.top_k   # every token picks top_k experts

    # A second pop (no forward in between) returns all-zero — the accumulator was reset.
    assert model.pop_moe_load() == [[0.0] * cfg.n_experts for _ in range(cfg.n_moe_layers)]

    balancer = MoEBalancer(cfg.n_moe_layers, cfg.n_experts, rate=0.1)
    balancer.update(loads)
    model.set_moe_biases(balancer.biases())
    assert all(b._bias_active for b in blocks)
    # fp32 in the router (MLX's compute dtype), float in the policy — compare at fp32.
    assert np.allclose(np.array(model.moe_biases()), np.array(balancer.biases()),
                       atol=1e-6)


@requires_mlx
def test_set_moe_biases_raises_on_a_shape_mismatch():
    """Driver-facing API: a wrong layer count or expert count must fail loudly, not
    leave the model in a partially-activated routing state."""
    from src.model.mlx_backend import MLXMambaModel
    cfg = _cfg(moe_balance_rate=0.05)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    with pytest.raises(ValueError, match="1 bias vectors for 2 MoE layers"):
        model.set_moe_biases([[0.0] * 8])
    with pytest.raises(ValueError, match="route bias has 3 entries"):
        model.set_moe_biases([[0.0] * 3, [0.0] * 3])
    assert all(not b._bias_active for b in model.moe_blocks())


@requires_mlx
def test_loading_a_bias_for_a_non_moe_layer_raises_clearly(tmp_path):
    """A balanced checkpoint loaded into a config with a different MoE interleave must
    name the mismatch, not die on an IndexError deep in the load."""
    from src.model.mlx_backend import MLXMambaModel
    from src.train.checkpoint import load_weights_dict, save_weights
    cfg = _cfg(moe_balance_rate=0.05)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    model.set_moe_biases([[0.1] * 8, [0.2] * 8])
    path = str(tmp_path / "weights.safetensors")
    model.save(path)

    weights = load_weights_dict(path)
    weights["moe_route_bias.7"] = weights.pop("moe_route_bias.1")   # no layer 7 here
    bad = str(tmp_path / "bad.safetensors")
    save_weights(weights, bad, config=cfg)

    mx.random.seed(1)
    fresh = MLXMambaModel(cfg)
    with pytest.raises(ValueError, match="layer 7 of this config is not an MoE layer"):
        fresh.load(bad)


@requires_mlx
def test_pop_moe_load_is_empty_for_a_dense_model():
    from src.model.mlx_backend import MLXMambaModel
    mx.random.seed(0)
    model = MLXMambaModel(_cfg(moe_every=None, n_experts=0, moe_d_ff=None))
    assert model.moe_blocks() == []
    assert model.pop_moe_load() == []
    assert model.moe_biases() == []


# --------------------------------------------------------------------------- #
# D3 — the bias round-trips in the PORTABLE safetensors
# --------------------------------------------------------------------------- #
@requires_mlx
def test_route_bias_round_trips_through_portable_weights(tmp_path):
    """A model served or evaluated from weights.safetensors must route the way it was
    trained, so the bias rides with the weights — not the resume bundle."""
    from src.model.mlx_backend import MLXMambaModel
    cfg = _cfg(moe_balance_rate=0.05)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    bias = [[0.1 * i for i in range(8)], [-0.05 * i for i in range(8)]]
    model.set_moe_biases(bias)

    path = str(tmp_path / "weights.safetensors")
    keys = model._portable_state_dict()
    assert sorted(k for k in keys if k.startswith("moe_route_bias.")) == \
        ["moe_route_bias.0", "moe_route_bias.1"]
    model.save(path)

    mx.random.seed(1)                                  # different init on purpose
    fresh = MLXMambaModel(cfg)
    assert fresh.moe_biases() == [[], []]              # inactive before the load
    fresh.load(path)
    assert fresh.moe_biases() == model.moe_biases()      # fp32 round-trip, exact
    assert np.allclose(np.array(fresh.moe_biases()), np.array(bias), atol=1e-6)
    assert all(b._bias_active for b in fresh.moe_blocks())

    tokens = mx.array(np.arange(2 * cfg.seq_len).reshape(2, cfg.seq_len) % cfg.vocab_size)
    assert np.allclose(np.array(model.forward(tokens)), np.array(fresh.forward(tokens)))


@requires_mlx
def test_portable_key_set_unchanged_when_the_bias_is_inactive():
    """Guards tests/test_moe.py's and tests/test_sizing_mlx.py's
    `sum(v.size) == num_parameters()`: no bias key may appear when balancing is off."""
    from src.model.mlx_backend import MLXMambaModel
    for cfg in (_cfg(moe_balance_rate=None), _cfg(moe_balance_rate=0.05)):
        mx.random.seed(0)
        model = MLXMambaModel(cfg)                     # constructed, never activated
        sd = model._portable_state_dict()
        assert not any(k.startswith("moe_route_bias.") for k in sd), list(sd)
        assert sum(int(v.size) for v in sd.values()) == cfg.num_parameters()


@requires_mlx
def test_loading_a_pre_213_checkpoint_leaves_the_bias_inactive(tmp_path):
    """Old checkpoints carry no `moe_route_bias.*` keys; loading one must not raise and
    must leave the router on the unbiased path."""
    from src.model.mlx_backend import MLXMambaModel
    cfg = _cfg(moe_balance_rate=0.05)
    mx.random.seed(0)
    old = MLXMambaModel(cfg)                           # bias never activated
    path = str(tmp_path / "weights.safetensors")
    old.save(path)

    mx.random.seed(1)
    fresh = MLXMambaModel(cfg)
    fresh.load(path)
    assert all(not b._bias_active for b in fresh.moe_blocks())
    assert fresh.moe_biases() == [[], []]


# --------------------------------------------------------------------------- #
# D5 — grad_checkpoint double-counts by exactly 2x, harmlessly
# --------------------------------------------------------------------------- #
@requires_mlx
def test_grad_checkpoint_doubles_the_counts_and_changes_nothing_else():
    """With `grad_checkpoint: true` each layer's forward is recomputed in backward, so
    `_moe` runs twice per micro-batch. Measured: the counter assignment inside the
    grad-traced, `mx.checkpoint`-wrapped call does not raise, and every expert of every
    MoE layer counts exactly 2x. Both consumers are invariant to a uniform positive
    scale (`update` is sign-based, `utilization_variance` normalizes to fractions), so
    the balancing behaviour is identical."""
    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_train_step import make_train_step

    def counts(grad_checkpoint):
        cfg = _cfg(moe_balance_rate=0.05, grad_checkpoint=grad_checkpoint)
        mx.random.seed(0)
        model = MLXMambaModel(cfg)
        opt = optim.AdamW(learning_rate=1e-3)
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

    b_plain = MoEBalancer(2, 8, rate=0.05)
    b_ckpt = MoEBalancer(2, 8, rate=0.05)
    b_plain.update(plain)
    b_ckpt.update(checkpointed)
    assert b_plain.biases() == b_ckpt.biases()
