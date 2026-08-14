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

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from safetensors.numpy import load_file            # noqa: E402

from scripts.export_parity_fixture import build_fixture   # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "swift" / "engine" / "Fixtures" / "toy"
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
            "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`."
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
                "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`.")
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
            "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`."
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
        "swift/engine/Fixtures/README.md — and re-run `swift run monica-parity`."
    )


def test_checked_in_tokens_are_reproducible():
    """The token batch is derived from the seed, so a fixture's inputs must be
    regenerable without the file. If this drifts, every reference above is unanchored."""
    tokens = load_file(str(FIXTURE / "inputs.safetensors"))["tokens"]
    expected = np.random.default_rng(0).integers(0, 256, size=(2, 129)).astype(np.int32)
    np.testing.assert_array_equal(tokens, expected)
