"""Swift -> Python checkpoint round-trip gate (#196).

`monica-parity`'s round-trip section (#196) proves the Swift WRITER produces a
checkpoint that Swift itself can read back byte-for-byte — but that claim, made
entirely inside a Swift binary, does not discharge the issue's other acceptance
direction: "a checkpoint written by Swift loads and runs in Python." Only a Python
process can prove that. This script closes it.

    python scripts/check_swift_checkpoint.py \\
        --roundtrip-dir <dir written by `monica-parity --roundtrip-out`> \\
        --fixtures swift/engine/Fixtures

For each fixture subdirectory found under `--roundtrip-dir` (one per checked-in
fixture, written by `monica-parity`'s round-trip section (a)):

1. `load_config_sidecar` the Swift-written weights and the fixture's own weights;
   assert the two `MambaConfig`s are equal FIELD-FOR-FIELD (`dataclasses.asdict`
   equality) — this is where sidecar fidelity actually gets gated on the Python side
   (the Swift side only proves Swift-to-Swift; this proves Swift's writer produced
   something Python's OWN reader reconstructs identically).
2. `load_quant_sidecar` both; assert equal (`None == None` for the five fp fixtures).
3. For an fp fixture: build the MLX backend model from the round-tripped config,
   `check_weight_keys` its round-tripped weights against a freshly-built model's own
   `_portable_state_dict()`, load them, run `forward`, and compare logits against the
   fixture's `reference.safetensors` at the fixture's own rtol/atol (falling back to
   the repo-wide fp32 gate's `1e-4`/`1e-5`).
4. For a quantized fixture (`toy-moe-int8`/`toy-moe-int4`, `meta.json` carries
   `quant_bits`): steps 1-2 above still run (pure JSON, no model construction needed),
   but step 3 is explicitly SKIPPED, not silently passed. `src/eval/quantize.py`'s own
   docstring says quantization here is a fake-quant MEASUREMENT spike, not a servable
   format — the Python MLX backend (`src/model/mlx_backend.py`) never builds an
   `nn.QuantizedLinear`/`nn.QuantizedEmbedding`, so there is no loader reachable from a
   plain `MLXMambaModel` + `_load_portable` that can consume packed
   `.weight`/`.scales`/`.biases` tensors. The packed-tensor bit-identity and the
   quant-block re-decode ARE gated — on the Swift side, by `monica-parity`'s round
   trip (a) (exact tensor comparison, which covers packed tensors like any other) and
   (d) (`QuantSpec` re-decode). This script prints an explicit SKIP line rather than
   silently doing nothing, per the standing rule that a checker which cannot observe
   its target must never read green.

An EMPTY `--roundtrip-dir` (no subdirectories) is itself a FAILURE, not a pass, for
the same reason: a checker pointed at nothing must say so loudly, not report OK
because it found zero problems in zero fixtures.

MLX-only where it touches the backend (imported lazily inside `_check_one`, mirroring
`scripts/smoke_test.py`'s "keep MLX-only imports local" convention) — the argument
parsing and directory walk above are portable.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

# The repo-wide fp32 parity gate's constants (`swift/engine/Sources/monica-parity/
# main.swift`'s `rtol`/`atol`), used as the fallback when a fixture's `meta.json`
# carries no override — matching monica-parity's own `meta.rtol ?? rtol` behavior.
_DEFAULT_RTOL = 1e-4
_DEFAULT_ATOL = 1e-5


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load Swift-written portable checkpoints in Python/MLX and verify "
                    "they reproduce each fixture's config and reference logits (#196).")
    p.add_argument("--roundtrip-dir", required=True, type=Path,
                   help="directory of Swift-written checkpoints, one subdirectory per "
                        "fixture: <dir>/<fixture>/weights.safetensors[.config.json] "
                        "(written by `monica-parity --roundtrip-out <dir>`)")
    p.add_argument("--fixtures", required=True, type=Path,
                   help="the checked-in fixtures directory, e.g. swift/engine/Fixtures")
    return p.parse_args()


def _check_one(name: str, rt_weights: Path, fixture_dir: Path) -> tuple[str, str]:
    """Check one fixture's Swift-written round-trip checkpoint. Returns
    `(status, message)` with `status` in `{"OK", "FAIL", "SKIP"}`. Never raises for an
    ordinary check failure (those become `"FAIL"`); the caller still guards against a
    genuinely unexpected exception so one bad fixture can't silently truncate the run.
    """
    from src.model.mlx_backend import MLXMambaModel  # local: MLX-only (see module docstring)
    from src.train.checkpoint import (
        check_weight_keys,
        load_config_sidecar,
        load_quant_sidecar,
        load_weights_dict,
    )

    fixture_weights = fixture_dir / "weights.safetensors"
    if not fixture_weights.exists():
        return "FAIL", f"no fixture weights.safetensors at {fixture_weights}"

    meta_path = fixture_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    quant_bits = meta.get("quant_bits")

    # --- 1. sidecar fidelity: Swift's writer must reproduce the SAME MambaConfig ------
    rt_cfg = load_config_sidecar(str(rt_weights))
    fx_cfg = load_config_sidecar(str(fixture_weights))
    if rt_cfg is None or fx_cfg is None:
        return "FAIL", (f"missing MambaConfig sidecar (roundtrip="
                        f"{'present' if rt_cfg is not None else 'MISSING'}, fixture="
                        f"{'present' if fx_cfg is not None else 'MISSING'})")
    rt_dict, fx_dict = dataclasses.asdict(rt_cfg), dataclasses.asdict(fx_cfg)
    if rt_dict != fx_dict:
        diff = {k: (rt_dict[k], fx_dict[k]) for k in rt_dict
                if rt_dict[k] != fx_dict.get(k)}
        return "FAIL", f"sidecar MambaConfig fields differ (roundtrip, fixture): {diff}"

    # --- 2. the quant block rides the same sidecar; must survive independently -------
    rt_quant = load_quant_sidecar(str(rt_weights))
    fx_quant = load_quant_sidecar(str(fixture_weights))
    if rt_quant != fx_quant:
        return "FAIL", f"quant sidecar differs: roundtrip={rt_quant} fixture={fx_quant}"

    if quant_bits is not None:
        # See module docstring: no Python loader for packed checkpoints exists to skip
        # TO — this is a real, explained SKIP, not a placeholder for future work.
        return "SKIP", (
            f"quant_bits={quant_bits} — Python's MLX backend has no "
            "QuantizedLinear/QuantizedEmbedding loader reachable from a plain "
            "MLXMambaModel; sidecar fidelity (checks 1-2 above) still gates this "
            "fixture, and the packed-tensor / quant-block round trip is gated on the "
            "Swift side (monica-parity round trip (a)/(d))")

    # --- 3. build + load the round-tripped fp checkpoint, compare against the oracle -
    model = MLXMambaModel(rt_cfg)
    rt_weights_dict = load_weights_dict(str(rt_weights))
    try:
        check_weight_keys(rt_weights_dict, model._portable_state_dict(),
                          where=f"{name} (Swift round trip)")
    except ValueError as e:
        return "FAIL", f"weight-key check failed: {e}"
    model._load_portable(rt_weights_dict)

    inputs = load_weights_dict(str(fixture_dir / "inputs.safetensors"))
    reference = load_weights_dict(str(fixture_dir / "reference.safetensors"))
    tokens = inputs["tokens"]
    ref_forward = reference["forward_logits"]

    logits = np.array(model.forward(tokens), dtype=np.float32)
    rtol = float(meta.get("rtol", _DEFAULT_RTOL))
    atol = float(meta.get("atol", _DEFAULT_ATOL))
    if not np.allclose(logits, ref_forward, rtol=rtol, atol=atol):
        max_abs = float(np.abs(
            logits.astype(np.float64) - ref_forward.astype(np.float64)).max())
        return "FAIL", (f"forward logits vs reference.safetensors differ "
                        f"(max|d|={max_abs:.3e}, rtol={rtol} atol={atol})")

    return "OK", f"sidecar + key-set + forward logits match (rtol={rtol} atol={atol})"


def main() -> None:
    args = _parse_args()

    if not args.roundtrip_dir.is_dir():
        print(f"FAIL: --roundtrip-dir {args.roundtrip_dir} does not exist")
        sys.exit(1)

    subdirs = sorted(p for p in args.roundtrip_dir.iterdir() if p.is_dir())
    if not subdirs:
        # A checker that cannot see its target must never read green: an empty
        # round-trip dir means the Swift emit step didn't run (or wrote somewhere this
        # script can't see), not that every fixture silently passed.
        print(f"FAIL: {args.roundtrip_dir} contains no fixture subdirectories — the "
              "Swift emit step (`monica-parity --roundtrip-out`) did not run, or wrote "
              "somewhere this script isn't looking")
        sys.exit(1)

    results: list[tuple[str, str, str]] = []
    for sub in subdirs:
        name = sub.name
        rt_weights = sub / "weights.safetensors"
        fixture_dir = args.fixtures / name
        if not rt_weights.exists():
            results.append((name, "FAIL", f"no weights.safetensors under {sub}"))
            continue
        if not fixture_dir.is_dir():
            results.append(
                (name, "FAIL", f"no matching fixture directory at {fixture_dir}"))
            continue
        try:
            status, msg = _check_one(name, rt_weights, fixture_dir)
        except Exception as e:  # a single bad fixture must not silently end the run
            status, msg = "FAIL", f"threw: {e!r}"
        results.append((name, status, msg))

    for name, status, msg in results:
        print(f"{status}: {name} — {msg}")

    n_ok = sum(1 for _, s, _ in results if s == "OK")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\n{len(results)} fixture(s) checked: {n_ok} OK, {n_skip} SKIP, {n_fail} FAIL")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
