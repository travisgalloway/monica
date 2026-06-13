"""dt-projection bias init verification (Milestone 1, issue #5). Skipped where
mlx is unavailable (Apple Silicon only).

The dt-bias init is LOAD-BEARING: `SelectiveSSM._init_dt_bias` samples dt
log-uniformly in [dt_min, dt_max], clamps to dt_init_floor, and sets the dt_proj
bias to inverse-softplus(dt). Three checks:

  * `test_dt_bias_timescales_in_range` — the initialized timescales have the
    right shape, sit in [dt_init_floor, dt_max], and the inverse-softplus is a
    true inverse of softplus (criteria 1 & 2 of the issue).
  * `test_dt_bias_loss_decreases` — with the proper init the toy model's
    training loss falls over ~40 steps (criterion 3: "loss decreases").
  * `test_dt_bias_enables_long_range_memory` — load-bearing demonstration: with
    the proper (log-uniform) init an early input materially influences a distant
    output, i.e. the SSM carries state across the whole sequence (the
    prerequisite for recall). A naive *constant* dt init retains far less. This
    is the deterministic, robust form of "load-bearing": the toy-LM training
    loss does NOT separate inits because that task is purely local (the conv +
    skip term handle it) — the dt-init's value is long-range recall, which this
    forward-pass probe isolates.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel, SelectiveSSM
from src.model.mlx_train_step import make_train_step
from src.train.schedule import CosineSchedule
from src.data.loader import PackedLoader

TOY_CFG = "config/toy.yaml"


def _np(a):
    return np.array(a)


def _softplus(x):
    return mx.logaddexp(x, mx.zeros_like(x))


def _fixed_batches(cfg, n, train_bin, batch_size=8):
    """A deterministic, pre-materialized batch stream (shuffle off => batch at
    step s is identical across runs), mirroring scripts/smoke_test.py."""
    loader = PackedLoader(train_bin, cfg.seq_len, batch_size, shuffle=False,
                          vocab_size=cfg.vocab_size)
    out = []
    for inp, tgt in loader.epoch():
        out.append((inp, tgt))
        if len(out) == n:
            break
    assert len(out) == n, f"need {n} batches, got {len(out)}"
    return out


def _train_losses(model, batches, *, base_lr=3e-4):
    n = len(batches)
    sched = CosineSchedule(base_lr=base_lr, warmup_steps=max(1, n // 6), total_steps=n)
    opt = optim.AdamW(learning_rate=base_lr)
    step = make_train_step(model, opt, grad_clip=1.0, scaler=None)
    return [step(model, [(inp, tgt)], sched.lr_at(s))["loss"]
            for s, (inp, tgt) in enumerate(batches)]


def test_dt_bias_timescales_in_range():
    mx.random.seed(0)
    cfg = load_config(TOY_CFG)
    ssm = SelectiveSSM(cfg)
    bias = ssm.dt_proj.bias
    dt = _np(_softplus(bias))

    assert bias.shape == (cfg.n_heads,)        # Mamba-2: one dt per head
    assert np.all(dt > 0.0)
    # log-uniform in [dt_min, dt_max], clamped up to dt_init_floor. The floor
    # (1e-4) is below dt_min (1e-3) here, so the effective range is [dt_min, dt_max].
    assert dt.min() >= cfg.dt_init_floor - 1e-6
    assert dt.min() >= cfg.dt_min - 1e-4
    assert dt.max() <= cfg.dt_max + 1e-4

    # inverse-softplus self-consistency: inv_softplus(softplus(bias)) == bias.
    dt_mx = _softplus(bias)
    recon = dt_mx + mx.log(-mx.expm1(-dt_mx))
    assert np.max(np.abs(_np(recon) - _np(bias))) < 1e-4


def test_dt_bias_loss_decreases(toy_train_bin):
    cfg = load_config(TOY_CFG)
    batches = _fixed_batches(cfg, 40, toy_train_bin)
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    losses = _train_losses(model, batches)
    print(f"\n[loss-decreases] step0={losses[0]:.4f} -> "
          f"step{len(losses) - 1}={losses[-1]:.4f}")
    assert np.isfinite(losses[0]) and np.isfinite(losses[-1])
    assert losses[-1] < 0.9 * losses[0]


def _memory_influence(bias_const=None):
    """How much an input at t=0 influences the SSM output at t=L-1, measured as
    a differential so it isolates state-carried memory.

    Two sequences are identical except for the input at t=0, with the SAME query
    input at the readout step t=L-1 (zeros in between, so the state just decays
    per the dt init across the gap). B/C/delta are input-dependent, so a query at
    the readout is required to read the state out at all. The norm of the change
    in y[L-1] is the influence of the t=0 input carried across the full sequence.

    bias_const=None uses the proper init; a float pins dt_proj.bias to a constant
    (a naive, single-timescale baseline)."""
    cfg = load_config(TOY_CFG)
    L, di = cfg.seq_len, cfg.d_inner
    mx.random.seed(0)
    ssm = SelectiveSSM(cfg)
    if bias_const is not None:
        ssm.dt_proj.bias = mx.full(ssm.dt_proj.bias.shape, float(bias_const))

    rng = np.random.default_rng(1)
    query = rng.standard_normal((di,)).astype(np.float32)
    v0 = rng.standard_normal((di,)).astype(np.float32)
    xa = np.zeros((1, L, di), np.float32); xa[0, -1] = query; xa[0, 0] = v0
    xb = np.zeros((1, L, di), np.float32); xb[0, -1] = query
    ya = _np(ssm.parallel(mx.array(xa)))
    yb = _np(ssm.parallel(mx.array(xb)))
    return float(np.linalg.norm(ya[0, -1] - yb[0, -1]))


def test_dt_bias_enables_long_range_memory():
    """Load-bearing demonstration. The proper log-uniform init spans timescales
    down to dt~dt_min, whose slow decay (dA = exp(dt*A) ~ 1) retains information
    across the whole sequence. A constant mid-range dt (bias=-3 => dt~0.05) has a
    single fast timescale and forgets. So the proper init carries an early input
    to a distant output far more strongly than the naive constant baseline."""
    proper = _memory_influence(bias_const=None)
    constant = _memory_influence(bias_const=-3.0)  # dt ~ softplus(-3) ~ 0.05, in-range but single-scale
    print(f"\n[memory] proper init influence@last  = {proper:.4e}")
    print(f"[memory] constant-dt influence@last  = {constant:.4e}")
    print(f"[memory] ratio proper/constant       = {proper / max(constant, 1e-12):.1f}x")
    assert proper > 1e-3, "proper init should carry real long-range memory"
    assert proper > 5.0 * constant, "log-uniform spread should beat a single timescale"
