"""The precision -> tolerance contract (#266): a genuinely different, looser gate for
fp16/bf16 than the fp32 one, derived from measurement rather than picked to make a run
go green.

Portable — numpy-only, no backend import (see `tests/test_import_guard.py`'s
`PORTABLE_MODULES`). `swift/engine/Sources/monica-parity/main.swift`'s `precisionBands`
mirrors this table by hand (Swift cannot import Python); `tests/test_lowp_parity_band.py`
T5 walks every checked-in fixture and asserts the RESOLVED band for the 8 pre-#266
fixtures is unchanged, which is the structural guard against the two tables drifting
apart.

Why fp32 stays at `1e-4`/`1e-5` (`docs/design/03-conformance.md` §"Why fp32, ~1e-4"):
bf16/fp16's own machine epsilon is larger than that band, so reusing it for a
low-precision comparison yields false failures — the low-precision contract has to be a
DIFFERENT one, not a widened fp32 one.

## Derivation (measured on this host — Python MLX; see the calibration scripts in
## `tests/test_lowp_parity_band.py`)

### Step 1 — the phenomenon the band has to tolerate

The Swift gate tolerates *cross-implementation disagreement*: two mathematically
equivalent implementations of the same math **at the same dtype**, differing only in
accumulation order/kernel. The best available proxy, obtainable without running
mlx-swift: Python's own `forward` (SSD chunked matmul) vs stacked `step` (recurrence)
disagreement at that dtype (`check_forward_step_parity`).

Measured `max|d|` on the three toy configs:

| config          | fp32    | fp16     | bf16     |
|-----------------|---------|----------|----------|
| toy.yaml        | 1.43e-6 | 1.58e-3  | 9.58e-3  |
| toy-moe.yaml    | 1.43e-6 | 2.93e-3  | 2.76e-2  |
| toy-hybrid.yaml | 1.43e-6 | 2.41e-3  | 1.72e-2  |

Normalising by the dtype's unit roundoff `u = 2^-(significand bits)`:

| dtype | u        | worst measured max\|d\| | max\|d\| / u |
|-------|----------|--------------------------|---------------|
| fp32  | 5.96e-8  | 1.43e-6                  | 24            |
| fp16  | 4.883e-4 | 2.93e-3                  | **6.0**       |
| bf16  | 3.906e-3 | 2.76e-2                  | **7.1**       |

The two low-precision dtypes land on the same coefficient (6-7 * u) across configs of
different depth/attention/MoE — the accumulation-depth factor is a property of the
network, not the dtype, so it cancels. That is the empirical law the band below is
derived from.

### Step 2 — the derived elementwise band

    atol(dtype) = 64 * u        rtol(dtype) = 8 * u

64 * u is ~9x the measured 6-7 * u coefficient. `rtol = atol / 8` so the relative term
contributes at most one `atol` at the largest logit (|logit| ~= 8), keeping the band
absolute-dominated where the measurement says the error actually lives.

| precision | u        | rtol   | atol   | headroom over measured |
|-----------|----------|--------|--------|-------------------------|
| fp32      | 5.96e-8  | 1e-4   | 1e-5   | ~70x (unchanged)        |
| fp16      | 4.883e-4 | 4e-3   | 3e-2   | ~10x                    |
| bf16      | 3.906e-3 | 3e-2   | 2.5e-1 | ~9x                     |

### Step 3 — the elementwise band is nearly vacuous at low precision

Injecting a relative perturbation into one weight tensor and measuring the resulting
logit shift shows the elementwise band only catches a weight defect of >=0.6% at fp32,
but only >=33% at fp16 and >=330% at bf16 (Δlogit is linear in the defect; see
`tests/test_lowp_parity_band.py`'s T3 for the reproduction). No elementwise `allclose`
predicate can be the LOAD-BEARING low-precision gate.

### Step 4 — the load-bearing tier: mean KL

Mean KL is a mean over positions of a SUM over vocab; rounding noise is zero-mean and
largely cancels in that sum, while a systematic defect contributes coherently.

An earlier pass at this derivation measured the forward-vs-step KL noise floor on
`toy.yaml` ALONE (fp16 2.63e-9, bf16 1.90e-7, ~1.2e-2 * u^2) and set the threshold at
`0.1 * u^2`. That threshold turned out NOT to generalise: exporting `toy-moe-fp16`
against it (#266 implementation) hit the exporter's own anti-vacuity refusal — the MoE
config's noise floor is ~40-60x higher than `toy.yaml`'s, because forward (chunked scan)
and step (recurrence) do not just accumulate floating-point rounding differently inside
otherwise-identical math the way a pure Mamba layer's do; a near-tie router selection can
land a DIFFERENT top-k expert set between the two paths at low precision, which is a
coherent (non-cancelling) divergence, not noise. Re-measured on all three configs
(the same method Step 1 already uses to establish the elementwise band is
config-invariant):

| config          | fp16 noise-floor mean KL | bf16 noise-floor mean KL |
|-----------------|----------------------------|-----------------------------|
| toy.yaml        | 1.70e-9                    | 2.19e-7                     |
| toy-moe.yaml    | **1.07e-7** (worst)         | **7.35e-6** (worst)          |
| toy-hybrid.yaml | 9.03e-8                    | 6.24e-6                     |

Taking the WORST case across configs (`toy-moe.yaml` at both dtypes) with the same ~8x
headroom multiplier the original derivation used:

    mean_kl(dtype) <= 8 * worst_measured_floor(dtype)

| precision | worst measured floor | mean-KL threshold | headroom |
|-----------|------------------------|---------------------|----------|
| fp16      | 1.07e-7                | **9e-7**            | ~8.4x    |
| bf16      | 7.35e-6                | **6e-5**            | ~8.2x    |

Re-measuring the detection floor with a weight-perturbation sweep on `toy-moe.yaml` (the
calibration config) at these corrected thresholds: the KL tier now catches a weight
defect of roughly >= 2.8% at fp16 and >= 3% at bf16 (interpolated from measured KL at
defects 1e-2/1e-1; the sweep's raw numbers are reproduced in
`tests/test_lowp_parity_band.py`'s T3). Against the elementwise floors of 33%/330%, the
KL tier is still ~12x more sensitive at fp16 and ~110x more sensitive at bf16 — smaller
margins than the toy.yaml-only calibration implied, but still clearly load-bearing, and
still the evidence this gate can actually fail — which is why the low-precision contract
stays structurally different (elementwise coarse + KL load-bearing) rather than merely a
looser version of the fp32 one.

Top-1 agreement is kept at >= 0.99 (matching #168's int8 threshold) as a cheap coarse
tier, not a discriminating one — it measured 1.0000 in every case including a 10% defect.

## The two exactness guards that must scale with dtype

`greedy_ids` and `load.{i}` (MoE counts) are compared EXACTLY at every precision — never
with a tolerance, at any dtype — but the MARGIN that makes an exact comparison safe (not
flaky on a near-tie) must widen with the dtype's own noise floor, or the guard becomes
either vacuous (too loose) or spuriously flaky (too tight, i.e. left at the fp32 value).
`greedy_margin_floor` and `ROUTE_MARGIN_MIN` are that scaling.
"""

from __future__ import annotations

# Unit roundoff (machine epsilon at 1.0), 2^-(significand bits). float32 has a 24-bit
# significand (23 explicit + implicit leading 1), float16 an 11-bit one, bfloat16 an
# 8-bit one.
UNIT_ROUNDOFF: dict[str, float] = {
    "fp32": 2.0 ** -24,
    "fp16": 2.0 ** -11,
    "bf16": 2.0 ** -8,
}

# The elementwise (rtol, atol) pair per precision. fp32's values are UNCHANGED from the
# pre-#266 gate (see the module docstring's Step 2) — this table is the single place both
# Python and (by hand-mirrored copy) Swift read them from.
PARITY_TOLERANCES: dict[str, tuple[float, float]] = {
    "fp32": (1e-4, 1e-5),
    "fp16": (4e-3, 3e-2),
    "bf16": (3e-2, 2.5e-1),
}

# The load-bearing distributional tier for low precision only (Step 4). fp32 has no entry
# here — its elementwise band is already the load-bearing gate. Calibrated on the WORST
# of the three toy configs (toy-moe.yaml, both dtypes) — see the module docstring for why
# a toy.yaml-only calibration undershot this by ~40-60x.
LOWP_MEAN_KL_MAX: dict[str, float] = {
    "fp16": 9e-7,
    "bf16": 6e-5,
}

# A cheap, coarse tier for every low-precision fixture — not discriminating on its own
# (see the module docstring), kept because it is nearly free to compute alongside KL.
LOWP_TOP1_MIN: float = 0.99


def band_for(precision: str) -> tuple[float, float]:
    """The elementwise `(rtol, atol)` pair for `precision`. Unknown precision is a hard
    error (`KeyError`) — a fixture the caller cannot classify must never fall back to a
    default, the same anti-BLIND rule `main.swift` follows for `meta.precision`."""
    return PARITY_TOLERANCES[precision]


def greedy_margin_floor(precision: str, top1: float) -> float:
    """The minimum greedy-decode margin (`top1_logit - top2_logit`) required for
    cross-implementation argmax equality to be well-posed at `precision`, mirroring the
    exporter's existing `atol_gen + rtol_gen * |top1|` rule but generalised across dtype
    and doubled for headroom (`greedy_margin_min` observed 1.45-6.31 across existing fp32
    fixtures against a ~5e-4 floor there, so the `2x` factor costs nothing at fp32 and
    gives low precision the same relative safety margin)."""
    rtol, atol = band_for(precision)
    return 2.0 * (atol + rtol * abs(top1))


def route_margin_min(precision: str) -> float:
    """The minimum MoE router selection-score margin (k-th vs (k+1)-th) below which
    `load.{i}`'s EXACT per-expert count comparison is a live near-tie hazard.
    `128 * u` recovers the pre-#266 fp32 constant exactly (`128 * 2^-24 = 7.629e-6`,
    which the pre-existing `< 1e-5` refusal already approximated) and generalises it to
    fp16 (6.3e-2) and bf16 (5e-1). The `max(1e-5, ...)` floor keeps fp32 at its historical
    value even though `128 * u_fp32 < 1e-5`."""
    return max(1e-5, 128.0 * UNIT_ROUNDOFF[precision])
