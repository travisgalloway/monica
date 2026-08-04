"""MoEBalancer policy — portable (no backend); Loss-Free-Balancing (#213) number machine."""

from src.model.blocks import MambaConfig
from src.train.moe_balance import MoEBalancer, attach_balancer, balancer_for_config


def _cfg(**over):
    base = dict(d_model=64, n_layers=4, head_dim=16, d_state=16, vocab_size=256,
                seq_len=32, precision="fp32")
    base.update(over)
    return MambaConfig(**base)


# --------------------------------------------------------------------------- #
# utilization_variance
# --------------------------------------------------------------------------- #
def test_utilization_variance_zero_for_perfectly_uniform_load():
    loads = [[10.0, 10.0, 10.0, 10.0]]
    assert MoEBalancer.utilization_variance(loads) == [0.0]


def test_utilization_variance_positive_for_collapsed_load():
    loads = [[40.0, 0.0, 0.0, 0.0]]
    var = MoEBalancer.utilization_variance(loads)[0]
    assert var > 0.0


def test_utilization_variance_zero_total_reports_zero_not_nan():
    loads = [[0.0, 0.0, 0.0, 0.0]]
    assert MoEBalancer.utilization_variance(loads) == [0.0]


def test_utilization_variance_per_layer():
    loads = [[10.0, 10.0], [40.0, 0.0]]
    var_uniform, var_collapsed = MoEBalancer.utilization_variance(loads)
    assert var_uniform == 0.0
    assert var_collapsed > var_uniform


def test_utilization_variance_invariant_to_uniform_scale():
    """The property that makes the `grad_checkpoint` double-count harmless (#213 D5):
    the metric reads load FRACTIONS, so scaling every count by 2 changes nothing."""
    loads = [[4.0, 13.0, 12.0, 3.0]]
    doubled = [[2 * x for x in loads[0]]]
    assert MoEBalancer.utilization_variance(loads) == MoEBalancer.utilization_variance(doubled)


# --------------------------------------------------------------------------- #
# update() — bias moves toward starved experts, away from overloaded ones
# --------------------------------------------------------------------------- #
def test_skewed_load_moves_bias_toward_starved_experts():
    b = MoEBalancer(n_layers=1, n_experts=2, rate=0.1)
    b.update([[10.0, 0.0]])          # expert 0 overloaded, expert 1 starved
    bias0, bias1 = b.biases()[0]
    assert bias1 > 0.0                # starved expert gains bias
    assert bias0 < 0.0                # overloaded expert loses bias
    assert bias1 == -bias0            # symmetric two-expert case, rate=0.1 either side


def test_update_is_a_noop_on_perfectly_balanced_load():
    b = MoEBalancer(n_layers=1, n_experts=3, rate=0.1)
    b.update([[5.0, 5.0, 5.0]])       # every expert exactly at the mean -> sign(0) == 0
    assert b.biases() == [[0.0, 0.0, 0.0]]


def test_update_covers_every_layer_independently():
    b = MoEBalancer(n_layers=2, n_experts=2, rate=0.1)
    b.update([[10.0, 0.0], [0.0, 10.0]])   # layer 0 and layer 1 skewed opposite ways
    (l0_0, l0_1), (l1_0, l1_1) = b.biases()
    assert l0_1 > 0.0 and l0_0 < 0.0
    assert l1_0 > 0.0 and l1_1 < 0.0


def test_update_is_invariant_to_uniform_scale_on_loads():
    """The other half of #213 D5: `update` reads only `sign(mean - load_i)`, so a
    doubled count vector (grad_checkpoint's recompute) produces the identical bias."""
    a = MoEBalancer(n_layers=1, n_experts=4, rate=0.1)
    b = MoEBalancer(n_layers=1, n_experts=4, rate=0.1)
    loads = [[4.0, 13.0, 12.0, 3.0]]
    a.update(loads)
    b.update([[2 * x for x in loads[0]]])
    assert a.biases() == b.biases()


def test_update_ignores_empty_layer_loads():
    b = MoEBalancer(n_layers=2, n_experts=2, rate=0.1)
    b.update([[], [10.0, 0.0]])
    assert b.biases()[0] == [0.0, 0.0]
    assert b.biases()[1][0] < 0.0


def test_update_raises_on_a_shape_mismatch():
    """A short/long `loads` means the model and the policy disagree about the MoE
    layout; `zip`-truncating it would silently stop steering some layer."""
    import pytest
    b = MoEBalancer(n_layers=2, n_experts=3, rate=0.1)
    with pytest.raises(ValueError, match="loads has 1 layers"):
        b.update([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match=r"loads\[1\] has 2 entries"):
        b.update([[1.0, 2.0, 3.0], [1.0, 2.0]])
    assert b.biases() == [[0.0] * 3, [0.0] * 3]        # nothing partially applied


def test_load_state_dict_raises_on_a_wrong_shaped_bias():
    """A PRESENT but wrong-shaped bias is a mismatched checkpoint, not a no-op."""
    import pytest
    b = MoEBalancer(n_layers=2, n_experts=3, rate=0.1)
    with pytest.raises(ValueError, match="bias has 1 layers"):
        b.load_state_dict({"bias": [[0.1, 0.2, 0.3]]})
    with pytest.raises(ValueError, match=r"bias\[0\] has 4 entries"):
        b.load_state_dict({"bias": [[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0]]})


def test_load_state_dict_fills_an_empty_row_with_zeros():
    """An inactive router reads back as `[]`; adopting it must still leave a usable
    `[n_layers][n_experts]` bias rather than a ragged one."""
    b = MoEBalancer(n_layers=2, n_experts=3, rate=0.1)
    b.load_state_dict({"bias": [[0.1, 0.2, 0.3], []]})
    assert b.biases() == [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]]


def test_repeated_update_reduces_utilization_variance():
    """A toy closed-loop sanity check of the intended dynamics: if routing load shifts
    toward whichever expert currently holds the larger bias (the effect the router's
    biased-selection path is meant to produce), repeated `update()` calls converge the
    load distribution toward uniform — the de-collapse the real MLX router exercises."""
    n_experts = 4
    balancer = MoEBalancer(n_layers=1, n_experts=n_experts, rate=0.1)
    loads = [[100.0, 0.0, 0.0, 0.0]]   # fully collapsed onto expert 0
    first_var = MoEBalancer.utilization_variance(loads)[0]

    for _ in range(200):
        balancer.update(loads)
        bias = balancer.biases()[0]
        total = sum(loads[0])
        # Toy feedback: route mass proportional to (1 + bias), clipped nonnegative.
        weights = [max(0.0, 1.0 + bias[i]) for i in range(n_experts)]
        s = sum(weights) or 1.0
        loads = [[total * w / s for w in weights]]

    final_var = MoEBalancer.utilization_variance(loads)[0]
    assert final_var < first_var
    assert final_var < 0.01           # converges close to uniform


# --------------------------------------------------------------------------- #
# state_dict round trip / resume-safety
# --------------------------------------------------------------------------- #
def test_state_dict_round_trip():
    b = MoEBalancer(n_layers=2, n_experts=3, rate=0.1)
    b.update([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    snap = b.state_dict()

    restored = MoEBalancer(n_layers=2, n_experts=3, rate=0.1)
    restored.load_state_dict(snap)
    assert restored.biases() == b.biases()
    # And it keeps evolving identically from there.
    b.update([[5.0, 5.0, 0.0], [0.0, 5.0, 5.0]])
    restored.update([[5.0, 5.0, 0.0], [0.0, 5.0, 5.0]])
    assert restored.biases() == b.biases()


def test_load_state_dict_seeds_from_a_model_read_back():
    """The D3 resume path: the bias rides in the portable weights, so the driver seeds
    the policy with `{"bias": model.moe_biases()}` rather than a checkpoint bundle."""
    b = MoEBalancer(n_layers=2, n_experts=3, rate=0.1)
    from_model = [[0.1, -0.2, 0.3], [-0.4, 0.5, 0.0]]
    b.load_state_dict({"bias": from_model})
    assert b.biases() == from_model
    b.biases()[0][0] = 99.0                       # biases() hands back copies
    assert b.biases() == from_model


def test_load_empty_state_is_noop():
    b = MoEBalancer(n_layers=1, n_experts=2, rate=0.1)
    b.update([[10.0, 0.0]])
    before = b.biases()
    b.load_state_dict({})
    assert b.biases() == before
    b.load_state_dict(None)            # resume with no balancer bundle
    assert b.biases() == before
    b.load_state_dict({"bias": None})  # bundle present, key nulled
    assert b.biases() == before
    b.load_state_dict({"bias": []})    # a dense/never-balanced model read back
    assert b.biases() == before


# --------------------------------------------------------------------------- #
# balancer_for_config — the single source of truth for config->balancer wiring
# --------------------------------------------------------------------------- #
def test_balancer_for_config_none_for_dense_config():
    cfg = _cfg()                       # moe_every is None
    assert balancer_for_config(cfg) is None


def test_balancer_for_config_none_when_balance_rate_unset():
    # MoE is on, but moe_balance_rate is the None default -> balancing OFF.
    cfg = _cfg(moe_every=2, n_experts=4, top_k=2)
    assert cfg.moe_balance_rate is None
    assert balancer_for_config(cfg) is None


def test_balancer_for_config_sized_balancer_when_rate_set():
    cfg = _cfg(moe_every=2, n_experts=4, top_k=2, moe_balance_rate=1e-3)
    b = balancer_for_config(cfg)
    assert isinstance(b, MoEBalancer)
    assert b.n_layers == cfg.n_moe_layers == 2
    assert b.n_experts == cfg.n_experts == 4
    assert b.rate == 1e-3
    assert b.biases() == [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


def test_toy_moe_config_ships_with_balancing_off():
    """config/toy-moe.yaml must keep balancing OFF so every pre-#213 MoE test stays
    byte-identical."""
    from src.model.blocks import load_config
    cfg = load_config("config/toy-moe.yaml")
    assert cfg.moe_balance_rate is None
    assert balancer_for_config(cfg) is None


def test_validate_rejects_non_positive_balance_rate():
    import pytest
    for bad in (0.0, -1e-3):
        with pytest.raises(ValueError, match="moe_balance_rate"):
            _cfg(moe_every=2, n_experts=4, top_k=2, moe_balance_rate=bad).validate()
    _cfg(moe_every=2, n_experts=4, top_k=2, moe_balance_rate=1e-3).validate()   # no raise


# --------------------------------------------------------------------------- #
# attach_balancer — driver wiring, duck-typed so it needs no backend
# --------------------------------------------------------------------------- #
class _FakeModel:
    """The three-method slice of the backend model `attach_balancer` touches."""

    def __init__(self, biases):
        self._biases = biases
        self.pushed = None
        self.counting = False

    def moe_biases(self):
        return [list(b) for b in self._biases]

    def set_moe_biases(self, biases):
        self.pushed = [list(b) for b in biases]

    def set_moe_load_counting(self, flag):
        self.counting = bool(flag)


def test_attach_balancer_is_a_noop_when_balancing_off():
    m = _FakeModel([[], []])
    attach_balancer(None, m)
    assert m.pushed is None and m.counting is False


def test_attach_balancer_pushes_zero_bias_and_enables_counting_on_a_fresh_model():
    m = _FakeModel([[], []])                       # no bias in the weights yet
    b = MoEBalancer(n_layers=2, n_experts=3, rate=0.1)
    attach_balancer(b, m)
    assert b.biases() == [[0.0] * 3, [0.0] * 3]    # empty read-back left the init alone
    assert m.pushed == [[0.0] * 3, [0.0] * 3]      # pushed before the first step
    assert m.counting is True


def test_attach_balancer_adopts_the_bias_that_came_with_the_weights():
    loaded = [[0.1, -0.2, 0.3], [-0.4, 0.5, 0.0]]
    m = _FakeModel(loaded)
    b = MoEBalancer(n_layers=2, n_experts=3, rate=0.1)
    attach_balancer(b, m)
    assert b.biases() == loaded                    # policy seeded FROM the model (D3)
    assert m.pushed == loaded
    assert m.counting is True


def test_attach_balancer_ignores_a_partially_biased_read_back():
    """A model where only some MoE blocks carry a bias is not a usable seed (shape
    mismatch); keep the zero init rather than truncating the policy."""
    m = _FakeModel([[0.1, -0.2, 0.3], []])
    b = MoEBalancer(n_layers=2, n_experts=3, rate=0.1)
    attach_balancer(b, m)
    assert b.biases() == [[0.0] * 3, [0.0] * 3]
