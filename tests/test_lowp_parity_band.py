"""The #266 fp16/bf16 parity contract: anti-vacuity + falsification (T2/T3/T4), plus the
calibration reproduction (T1) and the structural fp32-non-regression guarantee (T5).

Portable pieces (T5) run with no MLX and no Swift at all — `pytest.importorskip("mlx.core")`
is called LOCALLY inside each MLX-dependent test, not at module scope, precisely so T5
still runs on the portable Linux CI job (see `tests/test_bench_context.py`/
`test_cuda_muon.py` for the same intra-file pattern).

See `src/conformance/tolerances.py`'s module docstring for the full derivation this file
verifies — including the correction made while writing this test: the mean-KL threshold
was first calibrated on `toy.yaml` alone (giving `0.1 * u^2`), which turned out to be
~40-60x too tight for `toy-moe.yaml`/`toy-hybrid.yaml` (MoE routing and attention
introduce coherent, not purely-random, forward-vs-step divergence at low precision) — the
threshold now used (`LOWP_MEAN_KL_MAX`) is calibrated on the WORST of the three configs,
matching the same worst-case-across-configs methodology the elementwise band already
used successfully.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.conformance.tolerances import (
    LOWP_MEAN_KL_MAX, PARITY_TOLERANCES, UNIT_ROUNDOFF, band_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "swift" / "engine" / "Fixtures"

# The three toy configs the elementwise band (Step 1) and the KL threshold's
# worst-case recalibration were both measured against.
CALIBRATION_CONFIGS = ["toy.yaml", "toy-moe.yaml", "toy-hybrid.yaml"]


def _build_model_and_tokens(config_name: str, precision: str, seed: int = 0):
    """MLX-only helper (local import — see the module docstring on why `mx` is never
    imported at module scope here). Mirrors `scripts/export_parity_fixture.py`'s own
    token construction exactly, so these measurements are reproducible against the real
    exporter's numbers."""
    import mlx.core as mx

    # #298 mitigation (buffer-cache reuse corrupting an unrelated later computation) —
    # same as scripts/export_parity_fixture.py and tests/test_parity_fixture_export.py.
    mx.clear_cache()
    mx.set_cache_limit(0)

    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel

    mx.random.seed(seed)
    cfg = load_config(str(REPO_ROOT / "config" / config_name))
    cfg.precision = precision
    cfg.validate()
    model = MLXMambaModel(cfg)
    tokens = np.random.default_rng(seed).integers(
        0, cfg.vocab_size, size=(2, 40)).astype(np.int32)
    return model, cfg, tokens


def _forward_step_np(model, tokens):
    """`(forward_logits, step_logits)` as float32 numpy, casting to float32 INSIDE MLX
    before the numpy conversion — this local MLX build (0.32.0) cannot convert a
    `bfloat16` array to numpy via the buffer protocol directly (mirrors
    `scripts/export_parity_fixture.py`'s `_np_f32` helper; duplicated here rather than
    imported so this test has no import-time dependency on that script)."""
    import mlx.core as mx

    fwd = np.array(model.forward(tokens).astype(mx.float32), dtype=np.float32)
    state = model.init_state(tokens.shape[0])
    steps = []
    for t in range(tokens.shape[1]):
        logits_t, state = model.step(tokens[:, t], state)
        steps.append(np.array(logits_t.astype(mx.float32), dtype=np.float32))
    step = np.stack(steps, axis=1)
    return fwd, step


def _mean_kl(a: np.ndarray, b: np.ndarray) -> float:
    from src.conformance.quant_parity import check_quant_parity
    return check_quant_parity(
        a, b, bits=0, thresholds={"top1": 0.0, "kl": float("inf")})["mean_kl"]


# --- T1: the calibration reproduces --------------------------------------------------

def test_t1_elementwise_coefficient_reproduces():
    """The elementwise max|d| / u coefficient (Step 1) sits in a broad [1, 20] sanity
    envelope across every (config, low-precision dtype) pair — loose enough to tolerate
    MLX-version noise, tight enough to catch a gross regression (e.g. a 100x drift that
    would mean the derivation no longer describes this repo's model). If this ever
    legitimately moves outside the envelope, the bands in `tolerances.py` must be
    RE-DERIVED, not silently adjusted around the new number."""
    pytest.importorskip("mlx.core")
    for config_name in CALIBRATION_CONFIGS:
        for precision in ("fp16", "bf16"):
            model, cfg, tokens = _build_model_and_tokens(config_name, precision)
            fwd, step = _forward_step_np(model, tokens)
            max_abs = float(np.abs(fwd.astype(np.float64) - step.astype(np.float64)).max())
            u = UNIT_ROUNDOFF[precision]
            coeff = max_abs / u
            assert 1.0 <= coeff <= 20.0, (
                f"{config_name} {precision}: max|d|/u = {coeff:.2f} is outside the "
                "[1, 20] sanity envelope the atol=64*u/rtol=8*u derivation assumes — "
                "re-derive the bands in src/conformance/tolerances.py, do not adjust "
                "them to match")


# --- T2: anti-vacuity — every checked-in low-precision fixture's OWN noise floor sits
# comfortably (>=4x) inside its own gate ------------------------------------------------

# (config, precision, seed, moe_bias) for each checked-in low-precision fixture, mirroring
# swift/engine/Fixtures/README.md's regeneration commands exactly.
_LOWP_FIXTURES = [
    ("toy.yaml", "fp16", 0, False),
    ("toy.yaml", "bf16", 0, False),
    ("toy-moe.yaml", "fp16", 15, True),
    ("toy-hybrid.yaml", "fp16", 0, False),
]


def test_t2_anti_vacuity_every_lowp_fixture_sits_inside_its_gate():
    """Freshly recomputes (not just re-reads meta.json) each checked-in low-precision
    fixture's forward-vs-step max|d| and mean_kl, and asserts both sit `<= band/4` — the
    same anti-vacuity ceiling `scripts/export_parity_fixture.py` refuses to write past.
    A fixture whose own noise floor were NOT comfortably inside its gate would make the
    whole low-precision contract vacuous."""
    mx = pytest.importorskip("mlx.core")
    for config_name, precision, seed, moe_bias in _LOWP_FIXTURES:
        model, cfg, tokens = _build_model_and_tokens(config_name, precision, seed=seed)
        if moe_bias:
            blocks = model.moe_blocks()
            model.set_moe_biases([
                [0.5 * ((-1) ** e) * (e + 1) for e in range(cfg.n_experts)]
                for _ in blocks
            ])
        fwd, step = _forward_step_np(model, tokens)
        max_abs = float(np.abs(fwd.astype(np.float64) - step.astype(np.float64)).max())
        kl = _mean_kl(fwd, step)

        rtol, atol = band_for(precision)
        kl_max = LOWP_MEAN_KL_MAX[precision]
        assert max_abs <= atol / 4, (
            f"{config_name} {precision}: forward/step max|d|={max_abs:.3e} is not "
            f"comfortably (>=4x) inside atol={atol:.3e}")
        assert kl <= kl_max / 4, (
            f"{config_name} {precision}: forward/step mean_kl={kl:.3e} is not "
            f"comfortably (>=4x) inside kl_max={kl_max:.3e}")


# --- T3: falsification — the KL tier actually rejects a real defect -------------------

def test_t3_kl_tier_rejects_a_real_defect():
    """`toy-moe.yaml` (the config `LOWP_MEAN_KL_MAX` is calibrated FROM — the worst-case
    noise floor of the three) gets `layers.0.out_proj.weight` perturbed by a relative
    0.5 and the KL tier must REJECT the resulting model at both fp16 and bf16, with
    plenty of headroom over the threshold — proof the gate can fail, not just pass.

    Detection floors (see `tolerances.py`'s docstring): this tier catches roughly a
    >=2.8% (fp16) / >=3% (bf16) weight defect on `toy-moe.yaml`, vs the elementwise
    tier's >=33%/>=330% — ~12x and ~110x more sensitive respectively. Smaller margins
    than a `toy.yaml`-only calibration would have implied, but still clearly
    load-bearing.
    """
    mx = pytest.importorskip("mlx.core")
    for precision in ("fp16", "bf16"):
        model, cfg, tokens = _build_model_and_tokens("toy-moe.yaml", precision)
        base_fwd, _ = _forward_step_np(model, tokens)

        sd = model._portable_state_dict()
        target = "layers.0.out_proj.weight"
        assert target in sd, f"expected {target} in toy-moe.yaml's state dict"
        perturbed_sd = {k: np.array(v, copy=True) for k, v in sd.items()}
        rng = np.random.default_rng(0)
        w = perturbed_sd[target]
        perturbed_sd[target] = w * (1 + 0.5 * rng.standard_normal(w.shape))

        perturbed_model, _, _ = _build_model_and_tokens("toy-moe.yaml", precision)
        perturbed_model._load_portable(perturbed_sd)
        perturbed_fwd = np.array(
            perturbed_model.forward(tokens).astype(mx.float32), dtype=np.float32)

        kl = _mean_kl(base_fwd, perturbed_fwd)
        kl_max = LOWP_MEAN_KL_MAX[precision]
        assert kl > kl_max, (
            f"toy-moe.yaml {precision}: a 50% relative weight defect produced "
            f"mean_kl={kl:.3e}, which the KL tier (threshold={kl_max:.3e}) FAILED to "
            "reject — the gate would be vacuous")


# --- T4: the fp32 gate has NOT regressed into a vacuous one ----------------------------

def test_t4_fp32_gate_still_fails_when_perturbed():
    """The explicit "did #266 accidentally loosen the strict gate" check: a relative 1e-2
    perturbation of `layers.0.out_proj.weight` on `toy.yaml` at fp32 must still FAIL
    `allclose(rtol=1e-4, atol=1e-5)` — the fp32 band in `tolerances.py` is byte-identical
    to the pre-#266 constants, so this must hold exactly as it always has."""
    pytest.importorskip("mlx.core")
    rtol, atol = band_for("fp32")
    assert (rtol, atol) == PARITY_TOLERANCES["fp32"] == (1e-4, 1e-5)

    model, cfg, tokens = _build_model_and_tokens("toy.yaml", "fp32")
    base_fwd, _ = _forward_step_np(model, tokens)

    sd = model._portable_state_dict()
    perturbed_sd = {k: np.array(v, copy=True) for k, v in sd.items()}
    rng = np.random.default_rng(0)
    w = perturbed_sd["layers.0.out_proj.weight"]
    perturbed_sd["layers.0.out_proj.weight"] = w * (1 + 1e-2 * rng.standard_normal(w.shape))

    perturbed_model, _, _ = _build_model_and_tokens("toy.yaml", "fp32")
    perturbed_model._load_portable(perturbed_sd)
    import mlx.core as mx
    perturbed_fwd = np.array(
        perturbed_model.forward(tokens).astype(mx.float32), dtype=np.float32)

    max_abs = float(np.abs(base_fwd.astype(np.float64)
                           - perturbed_fwd.astype(np.float64)).max())
    ok = bool(np.allclose(base_fwd, perturbed_fwd, rtol=rtol, atol=atol))
    print(f"T4: 1e-2 defect on toy.yaml fp32 -> max|Δlogit|={max_abs:.3e} "
          f"(band rtol={rtol} atol={atol}) allclose={ok}")
    assert not ok, (
        f"toy.yaml fp32: a 1% relative weight defect (max|Δlogit|={max_abs:.3e}) PASSED "
        f"allclose(rtol={rtol}, atol={atol}) — the strict fp32 gate has regressed into a "
        "vacuous one")


# --- T5: the structural fp32-non-regression guarantee — PORTABLE, no MLX --------------

# The 8 pre-#266 fixtures' expected resolved band (precision, quant_bits) -> (rtol, atol),
# exactly reproducing today's `main.swift` behaviour before this issue touched anything.
_PRE_266_EXPECTED_BANDS = {
    "toy": (1e-4, 1e-5),
    "toy-hybrid": (1e-4, 1e-5),
    "toy-moe": (1e-4, 1e-5),
    "toy-moe-biased": (1e-4, 1e-5),
    "toy-short": (1e-4, 1e-5),
    "toy-gen": (1e-4, 1e-5),
    "toy-moe-int8": (2e-2, 2e-2),
    "toy-moe-int4": (2e-2, 2e-2),
}


def test_t5_structural_fp32_guarantee():
    """PORTABLE — imports nothing but `json`/`pathlib`/`src.conformance.tolerances`, so
    it runs on the Linux CI job with no Swift and no MLX at all, guarding the Swift
    `precisionBands` table from drifting away from this Python one.

    Walks every `swift/engine/Fixtures/*/meta.json` and asserts:
      (a) every fixture declares a `precision` this module's table recognises;
      (b) no fixture with `precision == "fp32"` and no `quant_bits` carries an `rtol` or
          `atol` override key (mechanism 2 of the fp32 guarantee);
      (c) the RESOLVED band for every one of the 8 pre-#266 fixtures equals its value on
          `main` today — 1e-4/1e-5, or 2e-2/2e-2 for the two quantized ones.
    """
    assert FIXTURES_DIR.is_dir(), f"missing fixtures directory {FIXTURES_DIR}"
    seen = set()
    for meta_path in sorted(FIXTURES_DIR.glob("*/meta.json")):
        fixture_name = meta_path.parent.name
        seen.add(fixture_name)
        meta = json.loads(meta_path.read_text())

        precision = meta.get("precision")
        assert precision in PARITY_TOLERANCES, (
            f"{fixture_name}: meta.json's precision={precision!r} is not a recognised "
            f"key of PARITY_TOLERANCES ({sorted(PARITY_TOLERANCES)}) — a fixture the "
            "runner cannot classify must never read green")

        quant_bits = meta.get("quant_bits")
        has_override = ("rtol" in meta) or ("atol" in meta)
        if precision == "fp32" and quant_bits is None:
            assert not has_override, (
                f"{fixture_name}: precision=fp32, quant_bits=None, but meta.json "
                f"carries an rtol/atol override — this is exactly the back door "
                "mechanism 2 of the fp32 guarantee exists to close")

        band_rtol, band_atol = band_for(precision)
        resolved_rtol = max(band_rtol, meta.get("rtol") or 0)
        resolved_atol = max(band_atol, meta.get("atol") or 0)

        if fixture_name in _PRE_266_EXPECTED_BANDS:
            expected_rtol, expected_atol = _PRE_266_EXPECTED_BANDS[fixture_name]
            assert (resolved_rtol, resolved_atol) == (expected_rtol, expected_atol), (
                f"{fixture_name}: resolved band ({resolved_rtol}, {resolved_atol}) != "
                f"its value on main today ({expected_rtol}, {expected_atol}) — a "
                "pre-#266 fixture's gate must not have moved")

    missing = set(_PRE_266_EXPECTED_BANDS) - seen
    assert not missing, f"expected pre-#266 fixtures not found on disk: {sorted(missing)}"


def test_t5_train_rtol_train_atol_keys_are_not_in_scope():
    """#195/#293 introduces SEPARATE `train_rtol`/`train_atol` keys for the training-step
    comparison — not on `main` yet. This module and the override guard in `main.swift`
    key on the literal names `rtol`/`atol` only, so #293 landing cannot collide with
    #266. Documents the invariant rather than asserting anything about #293 (which this
    repo cannot see yet)."""
    assert "train_rtol" not in PARITY_TOLERANCES and "train_atol" not in PARITY_TOLERANCES
