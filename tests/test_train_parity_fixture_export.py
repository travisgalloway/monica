"""The checked-in Swift TRAINING parity fixtures are not stale (#195).

Mirrors `tests/test_parity_fixture_export.py` exactly, extended to `train.safetensors`: a
future change to `src/model/mlx_train_step.py`'s math (or `mlx_backend.py`'s, which the
train step forwards through) would otherwise leave `monica-train`'s fixture gate happily
green against a K-step trajectory oracle that no longer describes this repo's train step.

`toy` only, like the forward/step staleness guard — cheapest fixture, shares every code
path in it with the other two training fixtures (`toy-hybrid`'s attention, `toy-moe`'s
router/expert backward). MLX-gated: runs on `full-macos`, skips elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
# #298: this local MLX build (0.32.0) has been observed to silently corrupt a computation
# when many independently-constructed models allocate/free buffers of the same size class
# within one process — exactly what `build_fixture` (re-invoked below, with `--train-steps`
# constructing multiple MLX models in-process) does inside an already-populated pytest
# process. One shared mitigation, defined in `scripts/export_parity_fixture.py`.
from scripts.export_parity_fixture import disable_buffer_cache_for_process  # noqa: E402

disable_buffer_cache_for_process()

from safetensors.numpy import load_file            # noqa: E402

from scripts.export_parity_fixture import build_fixture   # noqa: E402

FIXTURE = Path(__file__).resolve().parents[1] / "swift" / "engine" / "Fixtures" / "toy"


def test_checked_in_toy_train_fixture_matches_todays_train_step(tmp_path):
    assert FIXTURE.is_dir(), f"missing checked-in fixture {FIXTURE}"
    ref = load_file(str(FIXTURE / "train.safetensors"))
    # The training gate's own tolerance, read from the checked-in fixture's own
    # `meta.json` (the same `train_rtol`/`train_atol` `monica-train` gates against) rather
    # than duplicated here as constants — deliberately NOT the fp32 `rtol=1e-4/atol=1e-5`
    # forward/step gate's constants. A training step is not a pure forward quantity:
    # gradients accumulate error through the whole backward graph and AdamW moments
    # compound it across steps.
    meta = json.loads((FIXTURE / "meta.json").read_text())
    rtol, atol = meta["train_rtol"], meta["train_atol"]

    build_fixture("config/toy.yaml", str(tmp_path / "toy"), batch=2, seq=129, train_steps=3)
    fresh = load_file(str(tmp_path / "toy" / "train.safetensors"))

    assert set(fresh.keys()) == set(ref.keys()), (
        "train.safetensors keys drifted — the micro-batch/step/parameter structure no "
        "longer matches the checked-in training oracle. If the change is intended, "
        "regenerate the fixtures — see swift/engine/Fixtures/README.md.")
    for key in ref:
        assert np.allclose(fresh[key], ref[key], rtol=rtol, atol=atol), (
            f"train.safetensors[{key!r}] drifted from the checked-in Swift training-parity "
            f"oracle (max|d| = {np.abs(fresh[key] - ref[key]).max():.3e}). If the "
            "mlx_train_step.py/mlx_backend.py change is intended, regenerate the fixtures "
            "— see swift/engine/Fixtures/README.md — and re-run `swift run monica-train`."
        )
