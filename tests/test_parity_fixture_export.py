"""The checked-in Swift parity fixtures are not stale (#166).

`swift/engine/Fixtures/` holds frozen Python-MLX logits that `monica-parity` gates the
Swift port against. That makes them a CONTRACT, and contracts rot: a future change to
`src/model/mlx_backend.py`'s math would leave the Swift gate happily green against an
oracle that no longer describes this repo's model.

So: re-export `toy` into a tmpdir with today's backend and assert it still reproduces the
CHECKED-IN reference at the same fp32 tolerance the Swift runner uses. `toy` only — it is
the cheapest fixture and shares every code path in it with the others' Mamba layers.

MLX-gated, like the other ~19 mlx-only test files: runs on `full-macos`, skips elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
# #298: this local MLX build (0.32.0) has been observed to silently corrupt a computation
# when many independently-constructed models allocate/free buffers of the same size class
# within one process — exactly what `build_fixture` (re-invoked below) and the fresh
# `MLXMambaModel` construction in the route-bias test do inside an already-populated
# pytest process. One shared mitigation, defined in `scripts/export_parity_fixture.py`.
from scripts.export_parity_fixture import disable_buffer_cache_for_process  # noqa: E402

disable_buffer_cache_for_process()

from safetensors.numpy import load_file            # noqa: E402

from scripts.export_parity_fixture import build_fixture   # noqa: E402
# The #298 regeneration policy, appended to most staleness-failure messages below (the
# exact-count assertions using assert_array_equal/plain assert are the exception — see
# individual call sites). It lives in the portable module rather than being restated here,
# so the rule a person hits on a red gate is the same rule wherever they hit it (see
# tests/test_fixture_digest.py, which pins it).
from src.conformance.fixture_digest import REGEN_ADVICE   # noqa: E402
from src.conformance.tolerances import band_for           # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "swift" / "engine" / "Fixtures" / "toy"
MOE_FIXTURE = FIXTURE.parent / "toy-moe"
MOE_BIASED_FIXTURE = FIXTURE.parent / "toy-moe-biased"
# The same tolerance monica-parity applies, with the checked-in array as the reference
# operand (np.allclose is asymmetric in rtol).
RTOL, ATOL = 1e-4, 1e-5


def test_checked_in_toy_fixture_matches_todays_backend(tmp_path):
    assert FIXTURE.is_dir(), f"missing checked-in fixture {FIXTURE}"
    meta_ref = load_file(str(FIXTURE / "reference.safetensors"))

    build_fixture("config/toy.yaml", str(tmp_path / "toy"), batch=2, seq=129,
                  packed_doc_lengths="Q,2*Q,5")
    fresh = load_file(str(tmp_path / "toy" / "reference.safetensors"))

    for key in ("forward_logits", "step_logits"):
        assert np.allclose(fresh[key], meta_ref[key], rtol=RTOL, atol=ATOL), (
            f"{key} drifted from the checked-in Swift-parity oracle "
            f"(max|d| = {np.abs(fresh[key] - meta_ref[key]).max():.3e}). If the backend "
            "change is intended, regenerate the fixtures — see "
            "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`." + REGEN_ADVICE
        )

    # #264: extend the staleness check to every hidden.{i} and mixing.{i} key. A silently
    # DROPPED oracle key (the exporter's loop shrinking, a layer-count regression) is the
    # same class of failure as a drifted one, so the key sets must match exactly, not just
    # a subset comparison.
    ref_keys = set(meta_ref.keys())
    fresh_keys = set(fresh.keys())
    assert fresh_keys == ref_keys, (
        f"reference.safetensors key set drifted from the checked-in Swift-parity oracle "
        f"(missing={sorted(ref_keys - fresh_keys)}, extra={sorted(fresh_keys - ref_keys)}). "
        "If the backend change is intended, regenerate the fixtures — see "
        "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`." + REGEN_ADVICE
    )
    for key in sorted(ref_keys):
        if not (key.startswith("hidden.") or key.startswith("mixing.")):
            continue
        assert np.allclose(fresh[key], meta_ref[key], rtol=RTOL, atol=ATOL), (
            f"reference.safetensors[{key!r}] drifted from the checked-in Swift-parity "
            f"oracle (max|d| = {np.abs(fresh[key] - meta_ref[key]).max():.3e}). If the "
            "backend change is intended, regenerate the fixtures — see "
            "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`." + REGEN_ADVICE
        )

    # #167: extend the staleness check to the greedy-id oracle. Unlike the logits above,
    # this is EXACT int equality, not a tolerance — `monica-parity`'s AC1 check is exact too,
    # so a fixture that silently drifted here would rot just as quietly as a logit fixture
    # would without the check above.
    gen_ref = load_file(str(FIXTURE / "generation.safetensors"))
    fresh_gen = load_file(str(tmp_path / "toy" / "generation.safetensors"))
    np.testing.assert_array_equal(
        fresh_gen["greedy_ids"], gen_ref["greedy_ids"],
        err_msg="greedy_ids drifted from the checked-in Swift-parity oracle. If the backend "
                "change is intended, regenerate the fixtures — see "
                "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`." + REGEN_ADVICE)
    np.testing.assert_array_equal(fresh_gen["prompt_ids"], gen_ref["prompt_ids"])

    # #169: extend the staleness check to the prefill state-handoff oracle — the same
    # rationale as the greedy-id extension above, applied to prefill.safetensors's logits
    # AND its per-leaf state (a state leaf drifting silently is exactly the "generation
    # subtly wrong after the prompt" bug #169 exists to catch).
    pre_ref = load_file(str(FIXTURE / "prefill.safetensors"))
    fresh_pre = load_file(str(tmp_path / "toy" / "prefill.safetensors"))
    assert set(fresh_pre.keys()) == set(pre_ref.keys()), (
        "prefill.safetensors keys drifted — the layer structure (Mamba/Attention/MoE state "
        "slots) no longer matches the checked-in oracle.")
    for key in pre_ref:
        assert np.allclose(fresh_pre[key], pre_ref[key], rtol=RTOL, atol=ATOL), (
            f"prefill.safetensors[{key!r}] drifted from the checked-in Swift-parity oracle "
            f"(max|d| = {np.abs(fresh_pre[key] - pre_ref[key]).max():.3e}). If the backend "
            "change is intended, regenerate the fixtures — see "
            "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`." + REGEN_ADVICE
        )

    # #68/#263: extend the staleness check to the packed (seg_ids) oracle — same
    # rationale as the extensions above, applied to monica-parity's P6 fixture. The
    # tokens/seg_ids/doc_lengths are EXACT int equality (the packing is deterministic
    # from the seed, same as the tokens themselves); packed_logits is a tolerance
    # comparison, same as every other logit array here.
    packed_ref = load_file(str(FIXTURE / "packed.safetensors"))
    fresh_packed = load_file(str(tmp_path / "toy" / "packed.safetensors"))
    assert set(fresh_packed.keys()) == set(packed_ref.keys()), (
        "packed.safetensors keys drifted from the checked-in Swift-parity oracle.")
    for key in ("packed_tokens", "packed_seg_ids", "doc_lengths"):
        np.testing.assert_array_equal(
            fresh_packed[key], packed_ref[key],
            err_msg=f"packed.safetensors[{key!r}] drifted from the checked-in "
                    "Swift-parity oracle. If the backend change is intended, regenerate "
                    "the fixtures — see swift/engine/Fixtures/README.md — and re-run "
                    "`swift run monica-parity`.")
    assert np.allclose(fresh_packed["packed_logits"], packed_ref["packed_logits"],
                       rtol=RTOL, atol=ATOL), (
        "packed.safetensors['packed_logits'] drifted from the checked-in Swift-parity "
        f"oracle (max|d| = "
        f"{np.abs(fresh_packed['packed_logits'] - packed_ref['packed_logits']).max():.3e})"
        ". If the backend change is intended, regenerate the fixtures — see "
        "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`." + REGEN_ADVICE
    )


def test_checked_in_tokens_are_reproducible():
    """The token batch is derived from the seed, so a fixture's inputs must be
    regenerable without the file. If this drifts, every reference above is unanchored."""
    tokens = load_file(str(FIXTURE / "inputs.safetensors"))["tokens"]
    expected = np.random.default_rng(0).integers(0, 256, size=(2, 129)).astype(np.int32)
    np.testing.assert_array_equal(tokens, expected)


FP16_FIXTURE = FIXTURE.parent / "toy-fp16"


def test_checked_in_toy_fp16_fixture_matches_todays_backend(tmp_path):
    """#266: the low-precision oracle rots exactly like the fp32 one does — a future
    change to `mlx_backend.py`'s math must be caught here, not first noticed as a
    stale-fixture failure in Swift CI. Compared at fp16's OWN band (not fp32's 1e-4/1e-5,
    which would be meaningless here — see src/conformance/tolerances.py), matching
    `monica-parity`'s own resolved band for this fixture.

    If fp16 re-export ever proves non-deterministic in CI (see #298), the documented
    fallback is to downgrade this to 'recorded measurements agree within 2x' rather than
    deleting the check — T6 in tests/test_lowp_parity_band.py's docstring already
    confirmed reproducibility on this host at the time #266 landed."""
    assert FP16_FIXTURE.is_dir(), f"missing checked-in fixture {FP16_FIXTURE}"
    meta_ref = load_file(str(FP16_FIXTURE / "reference.safetensors"))
    meta_json = json.loads((FP16_FIXTURE / "meta.json").read_text())
    assert meta_json["precision"] == "fp16"
    rtol, atol = band_for("fp16")

    build_fixture("config/toy.yaml", str(tmp_path / "toy-fp16"), batch=2, seq=40,
                  precision="fp16")
    fresh = load_file(str(tmp_path / "toy-fp16" / "reference.safetensors"))

    assert set(fresh.keys()) == set(meta_ref.keys()), (
        f"toy-fp16 reference.safetensors key set drifted (missing="
        f"{sorted(set(meta_ref) - set(fresh))}, extra={sorted(set(fresh) - set(meta_ref))})")
    for key in sorted(meta_ref.keys()):
        assert np.allclose(fresh[key], meta_ref[key], rtol=rtol, atol=atol), (
            f"toy-fp16 reference.safetensors[{key!r}] drifted from the checked-in "
            f"Swift-parity oracle (max|d| = "
            f"{np.abs(fresh[key] - meta_ref[key]).max():.3e}, band rtol={rtol} "
            f"atol={atol}). If the backend change is intended, regenerate the fixtures "
            "— see swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`." + REGEN_ADVICE
        )


def test_checked_in_toy_moe_fixture_load_counts_match_todays_backend(tmp_path):
    """#265: the load-count oracle (`load.{i}`) rots exactly like the logit oracle does —
    a future change to `_moe`'s routing math (`mlx_backend.py`) must be caught here, not
    first noticed as a stale-fixture failure in Swift CI.

    Counts are EXACT integers (`assert_array_equal`, never `allclose`/`rtol`/`atol`) —
    the #265 rule that this port must never blur the distinction between the logit
    oracle (a tolerance comparison) and the count oracle (an exact one)."""
    assert MOE_FIXTURE.is_dir(), f"missing checked-in fixture {MOE_FIXTURE}"
    meta_ref = load_file(str(MOE_FIXTURE / "reference.safetensors"))

    build_fixture("config/toy-moe.yaml", str(tmp_path / "toy-moe"), batch=2, seq=40,
                  packed_doc_lengths="Q,7,Q+3")
    fresh = load_file(str(tmp_path / "toy-moe" / "reference.safetensors"))

    # A dropped oracle key rots as quietly as a drifted one (the #264 rule this test file
    # already applies to hidden.*/mixing.*) — so the key sets must match exactly.
    ref_keys = set(meta_ref.keys())
    fresh_keys = set(fresh.keys())
    assert fresh_keys == ref_keys, (
        f"reference.safetensors key set drifted from the checked-in toy-moe oracle "
        f"(missing={sorted(ref_keys - fresh_keys)}, extra={sorted(fresh_keys - ref_keys)}). "
        "If the backend change is intended, regenerate the fixture — see "
        "swift/engine/Fixtures/README.md.")

    for key in ("load.1", "load.3"):
        np.testing.assert_array_equal(
            fresh[key], meta_ref[key],
            err_msg=f"reference.safetensors[{key!r}] drifted from the checked-in toy-moe "
                    "load-count oracle. Counts are exact integers — this must be exact "
                    "equality, never a tolerance comparison. If the backend change is "
                    "intended, regenerate the fixture — see swift/engine/Fixtures/README.md.")

    fresh_meta = json.loads((tmp_path / "toy-moe" / "meta.json").read_text())
    assert fresh_meta.get("moe_load_layers") == [1, 3], fresh_meta.get("moe_load_layers")
    margin = fresh_meta.get("moe_route_margin_min")
    assert margin is not None and margin > 1e-5, (
        f"moe_route_margin_min={margin!r} must be present and above the exact-comparison "
        "hazard threshold (1e-5) — see export_parity_fixture.py's #265 section.")


def test_route_bias_write_lands_in_the_logits_and_the_counts():
    """The strongest available proof that #265's `set_moe_biases` WRITE path actually
    lands: build a fresh model from `toy-moe`'s (unbiased) checked-in weights, push
    `toy-moe-biased`'s own checked-in route-bias vectors into it via `set_moe_biases`, and
    assert BOTH the forward logits and the load counts reproduce `toy-moe-biased`'s
    checked-in oracle — not merely that the call doesn't raise. The bias is read out of
    `toy-moe-biased`'s `weights.safetensors` rather than hard-coded, so this can never
    drift from the fixture it is checked against."""
    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel

    assert MOE_FIXTURE.is_dir() and MOE_BIASED_FIXTURE.is_dir()

    cfg = load_config("config/toy-moe.yaml")
    model = MLXMambaModel(cfg)
    model.load(str(MOE_FIXTURE / "weights.safetensors"))   # toy-moe: no bias keys, unbiased

    biased_weights = load_file(str(MOE_BIASED_FIXTURE / "weights.safetensors"))
    bias_layers = sorted(
        int(k.split(".", 1)[1]) for k in biased_weights if k.startswith("moe_route_bias."))
    assert bias_layers == [i for i, l in enumerate(model.layers)
                           if l.__class__.__name__ == "MoEBlock"], bias_layers
    biases = [biased_weights[f"moe_route_bias.{i}"].tolist() for i in bias_layers]
    model.set_moe_biases(biases)

    tokens = load_file(str(MOE_FIXTURE / "inputs.safetensors"))["tokens"]
    model.set_moe_load_counting(True)
    model.pop_moe_load()      # drain (nothing has run yet on this fresh model — hygiene)
    logits = np.array(model.forward(tokens), dtype=np.float32)
    loads = model.pop_moe_load()
    model.set_moe_load_counting(False)

    biased_ref = load_file(str(MOE_BIASED_FIXTURE / "reference.safetensors"))
    assert np.allclose(logits, biased_ref["forward_logits"], rtol=RTOL, atol=ATOL), (
        f"forward_logits after set_moe_biases do not reproduce toy-moe-biased's checked-in "
        f"oracle (max|d| = {np.abs(logits - biased_ref['forward_logits']).max():.3e}) — the "
        "route-bias write path did not land.")

    for layer_idx, counts in zip(bias_layers, loads):
        np.testing.assert_array_equal(
            np.asarray(counts, dtype=np.float32), biased_ref[f"load.{layer_idx}"],
            err_msg=f"load counts for layer {layer_idx} after set_moe_biases do not "
                    "reproduce toy-moe-biased's checked-in oracle exactly — the route-bias "
                    "write path did not land in the routing decision.")
