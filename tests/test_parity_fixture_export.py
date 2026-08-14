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

    build_fixture("config/toy.yaml", str(tmp_path / "toy"), batch=2, seq=129)
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


def test_checked_in_tokens_are_reproducible():
    """The token batch is derived from the seed, so a fixture's inputs must be
    regenerable without the file. If this drifts, every reference above is unanchored."""
    tokens = load_file(str(FIXTURE / "inputs.safetensors"))["tokens"]
    expected = np.random.default_rng(0).integers(0, 256, size=(2, 129)).astype(np.int32)
    np.testing.assert_array_equal(tokens, expected)
