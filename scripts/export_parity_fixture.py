"""Export a Swift/MLX logit-parity fixture (#166).

Writes a self-contained fixture directory that `swift/engine`'s `monica-parity` runner
consumes: the portable weights, the exact token batch, and the Python MLX backend's
reference logits for BOTH code paths (`forward` and stacked per-token `step`), plus each
layer's output so a Swift mismatch can be localized to a single block instead of a
whole-model logit diff.

    python scripts/export_parity_fixture.py --config config/toy.yaml \\
        --out swift/engine/Fixtures/toy --batch 2 --seq 129 [--precision fp32] [--moe-bias]

The checked-in fixtures under `swift/engine/Fixtures/` ARE the Swift contract, so this
script is also the regeneration command recorded there — and
`tests/test_parity_fixture_export.py` re-runs it against the checked-in `toy` reference so
a future change to `mlx_backend.py`'s math cannot silently leave the Swift gate testing a
stale oracle.

MLX-only (it builds the MLX backend), so it does not run on a Linux/CUDA host.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from src.conformance.forward_step_parity import check_forward_step_parity
from src.model.blocks import load_config


def build_fixture(config_path: str, out_dir: str, *, batch: int, seq: int,
                  precision: str | None = None, moe_bias: bool = False,
                  seed: int = 0) -> dict:
    """Build and write one fixture directory. Returns the `meta.json` contents."""
    import mlx.core as mx  # local import: this script is MLX-only, like scripts/smoke_test.py
    from safetensors.numpy import save_file

    from src.model.mlx_backend import MLXMambaModel

    mx.random.seed(seed)
    cfg = load_config(config_path)
    if precision is not None:
        # The harness gates fp32 (poc.yaml is fp16). Override BEFORE `model.save`, so the
        # emitted sidecar is self-describing and the Swift loader needs no override of
        # its own.
        cfg.precision = precision
        cfg.validate()

    model = MLXMambaModel(cfg)

    if moe_bias:
        # One fixture carries `moe_route_bias.*` keys, so the biased-ranking branch AND the
        # loader's pop path are both exercised. A fixed, deliberately asymmetric vector:
        # a uniform bias would not change the ranking and would gate nothing.
        blocks = model.moe_blocks()
        if not blocks:
            raise SystemExit("--moe-bias needs a config with MoE layers")
        model.set_moe_biases([
            [0.5 * ((-1) ** e) * (e + 1) for e in range(cfg.n_experts)]
            for _ in blocks
        ])

    # The exact token construction used by tests/test_mlx_parity.py.
    tokens = np.random.default_rng(seed).integers(
        0, cfg.vocab_size, size=(batch, seq)).astype(np.int32)

    # A fixture that fails PYTHON's own forward/step gate must never reach disk — it would
    # make the Swift gate compare against a reference that is itself internally inconsistent.
    verdict = check_forward_step_parity(model, tokens, to_numpy=np.array)
    if not verdict["ok"]:
        raise SystemExit(
            f"refusing to write {out_dir}: the Python model fails its own forward/step "
            f"parity check (max_abs_diff={verdict['max_abs_diff']:.3e})")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    weights_path = str(out / "weights.safetensors")
    model.save(weights_path)            # weights.safetensors + .config.json sidecar

    save_file({"tokens": tokens}, str(out / "inputs.safetensors"))

    forward_logits = np.array(model.forward(tokens), dtype=np.float32)
    state = model.init_state(batch)
    steps = []
    for t in range(seq):
        logits_t, state = model.step(tokens[:, t], state)
        steps.append(np.array(logits_t, dtype=np.float32))
    step_logits = np.stack(steps, axis=1)

    ref = {"forward_logits": forward_logits, "step_logits": step_logits}
    # Per-layer hidden states: a deliberate debugging investment. A whole-model logit
    # mismatch is nearly impossible to localize by hand across 24 layers; "first divergence
    # at layer 7" is the difference between an afternoon and a week. ~200 KB at toy scale.
    for i, h in enumerate(model.hidden_states(tokens)):
        ref[f"hidden.{i}"] = np.array(h, dtype=np.float32)
    save_file(ref, str(out / "reference.safetensors"))

    meta = {
        "config": os.path.basename(config_path),
        "batch": batch,
        "seq": seq,
        "seed": seed,
        "precision": cfg.precision,
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "moe_bias": moe_bias,
        "forward_step_max_abs_diff": verdict["max_abs_diff"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a config/*.yaml")
    p.add_argument("--out", required=True, help="fixture directory to write")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=40)
    p.add_argument("--precision", default=None, choices=["fp32", "fp16", "bf16"],
                   help="override the config's precision (the harness gates fp32)")
    p.add_argument("--moe-bias", action="store_true",
                   help="activate a per-expert route bias (#213) so the fixture carries "
                        "moe_route_bias.* keys and exercises the biased-ranking branch")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta = build_fixture(args.config, args.out, batch=args.batch, seq=args.seq,
                         precision=args.precision, moe_bias=args.moe_bias, seed=args.seed)
    print(f"wrote {args.out}: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
