"""DynamicLossScaler policy — portable (no backend); just the number machine."""

from src.train.loss_scale import DynamicLossScaler, scaler_for_precision


def test_overflow_backs_off_and_resets_counter():
    s = DynamicLossScaler(init_scale=1024.0, backoff=0.5, growth_interval=3)
    # Two clean steps build toward growth...
    s.update(overflow=False)
    s.update(overflow=False)
    assert s.scale == 1024.0
    # ...then an overflow halves the scale and resets the good-step counter.
    s.update(overflow=True)
    assert s.scale == 512.0
    # Counter was reset: two more clean steps still short of growth_interval=3.
    s.update(overflow=False)
    s.update(overflow=False)
    assert s.scale == 512.0


def test_grows_after_growth_interval_clean_steps():
    s = DynamicLossScaler(init_scale=1024.0, growth_factor=2.0, growth_interval=3)
    s.update(overflow=False)
    s.update(overflow=False)
    assert s.scale == 1024.0
    s.update(overflow=False)            # third clean step -> grow
    assert s.scale == 2048.0


def test_min_scale_floor():
    s = DynamicLossScaler(init_scale=2.0, backoff=0.5, min_scale=1.0)
    s.update(overflow=True)
    assert s.scale == 1.0
    s.update(overflow=True)             # never drops below the floor
    assert s.scale == 1.0


def test_state_dict_round_trip():
    s = DynamicLossScaler(init_scale=1024.0, growth_interval=5)
    s.update(overflow=False)
    s.update(overflow=False)
    snap = s.state_dict()

    restored = DynamicLossScaler(init_scale=1.0, growth_interval=5)
    restored.load_state_dict(snap)
    assert restored.scale == s.scale
    # The good-step counter survives too: 3 more clean steps reach growth at the same point.
    for _ in range(3):
        s.update(overflow=False)
        restored.update(overflow=False)
    assert restored.scale == s.scale


def test_load_empty_state_is_noop():
    s = DynamicLossScaler(init_scale=512.0)
    s.load_state_dict({})
    assert s.scale == 512.0
    s.load_state_dict(None)             # resume with no scaler bundle
    assert s.scale == 512.0


# --- precision -> scaler wiring (issue #3, acceptance criterion 3) --------------
def test_scaler_for_precision_fp16_enables_scaling():
    s = scaler_for_precision("fp16", init_scale=4096.0)
    assert isinstance(s, DynamicLossScaler)
    assert s.scale == 4096.0             # the requested init scale is honored


def test_scaler_for_precision_fp16_default_init():
    s = scaler_for_precision("fp16")
    assert isinstance(s, DynamicLossScaler)
    assert s.scale == 2.0 ** 13          # matches DynamicLossScaler's default


def test_scaler_for_precision_skips_scaling_for_fp32_and_bf16():
    # bf16's wide exponent range and fp32 don't need (and must not get) loss scaling.
    assert scaler_for_precision("fp32") is None
    assert scaler_for_precision("bf16") is None
