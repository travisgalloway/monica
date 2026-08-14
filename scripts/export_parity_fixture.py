"""Export a Swift/MLX logit-parity fixture (#166 / #167).

Writes a self-contained fixture directory that `swift/engine`'s `monica-parity` runner
consumes: the portable weights, the exact token batch, and the Python MLX backend's
reference logits for BOTH code paths (`forward` and stacked per-token `step`), plus each
layer's output so a Swift mismatch can be localized to a single block instead of a
whole-model logit diff. Also writes `generation.safetensors` (#167 AC1): a greedy-decode id
oracle (`prompt_ids`, `greedy_ids`, `margins`) that `monica-parity` compares its own greedy
decode against, exactly.

    python scripts/export_parity_fixture.py --config config/toy.yaml \\
        --out swift/engine/Fixtures/toy --batch 2 --seq 129 \\
        [--precision fp32] [--vocab-size N] [--gen-steps 16] [--moe-bias]

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
from functools import partial
from pathlib import Path

import numpy as np

from src.conformance.forward_step_parity import check_forward_step_parity
from src.model.blocks import load_config


def build_fixture(config_path: str, out_dir: str, *, batch: int, seq: int,
                  precision: str | None = None, moe_bias: bool = False,
                  seed: int = 0, vocab_size: int | None = None,
                  gen_steps: int = 16) -> dict:
    """Build and write one fixture directory. Returns the `meta.json` contents."""
    import mlx.core as mx  # local import: this script is MLX-only, like scripts/smoke_test.py
    from safetensors.numpy import save_file

    from src.model.mlx_backend import MLXMambaModel
    from src.serve import sampling
    from src.serve.generate import generate as py_generate
    from src.serve.sessions import SessionStore

    mx.random.seed(seed)
    cfg = load_config(config_path)
    if precision is not None:
        # The harness gates fp32 (poc.yaml is fp16). Override BEFORE `model.save`, so the
        # emitted sidecar is self-describing and the Swift loader needs no override of
        # its own.
        cfg.precision = precision
        cfg.validate()
    if vocab_size is not None:
        # Alongside --precision: override BEFORE `model.save` so the sidecar stays
        # self-describing and the Swift loader needs no override of its own. `toy-gen`
        # (#167) uses this — a `monica-tokenize`-trained tokenizer's vocab (256 base bytes +
        # specials) cannot fit `toy.yaml`'s 256-wide model, so the fixture widens the model
        # instead of shrinking the tokenizer.
        cfg.vocab_size = vocab_size
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

    # --- generation.safetensors: the greedy-id parity oracle (#167 AC1) --------------------
    # `prompt_ids` is the first `min(8, L)` ids of the token batch's first row; `greedy_ids`
    # are `gen_steps` ids produced by driving `model.step` token-by-token (the same shape
    # `monica-parity` reproduces in Swift) with temperature=0 (argmax, first-max-on-tie).
    rtol_gen, atol_gen = 1e-4, 1e-5
    prompt_len = min(8, seq)
    prompt_ids = [int(t) for t in tokens[0, :prompt_len].tolist()]

    state = model.init_state(1)
    logits = None
    for t in prompt_ids:
        logits, state = model.step(np.array([t], dtype=np.int64), state)

    greedy_ids: list[int] = []
    margins: list[float] = []
    top1s: list[float] = []
    for _ in range(gen_steps):
        row = np.array(logits, dtype=np.float32).reshape(-1)
        nxt = int(np.argmax(row))
        top1 = float(row[nxt])
        row2 = row.copy()
        row2[nxt] = -np.inf
        top2 = float(row2.max())
        margins.append(top1 - top2)
        top1s.append(top1)
        greedy_ids.append(nxt)
        logits, state = model.step(np.array([nxt], dtype=np.int64), state)

    # A near-tie greedy argmax makes cross-implementation id equality ill-posed: two fp32
    # implementations that agree to 1e-4 relative can still cross a near-zero margin and pick
    # different tokens. Refusing to write a flaky fixture is the point — see the module
    # docstring's rationale and swift/engine/Fixtures/README.md.
    bad_steps = [i for i, (m, t1) in enumerate(zip(margins, top1s))
                if m < atol_gen + rtol_gen * abs(t1)]
    if bad_steps:
        raise SystemExit(
            f"refusing to write {out_dir}: greedy margin at step(s) {bad_steps} is inside the "
            f"parity band (atol={atol_gen} + rtol={rtol_gen}*|top1|) — a near-tie argmax makes "
            "cross-implementation id equality flaky. Bump --seed or --gen-steps and retry "
            f"(margins={margins})")

    # --- prefill parity: assert the SessionStore/generate.py path agrees, id-for-id ---------
    # Ties the fixture to "the same token ids as scripts/generate.py greedy", not to an
    # exporter-private step loop.
    store = SessionStore(model, max_concurrent=1)
    store.create("fixture")
    greedy_sampler = partial(sampling.sample, temperature=0.0)
    via_generate = py_generate(store, "fixture", prompt_ids, sampler=greedy_sampler,
                               to_numpy=np.array, max_new_tokens=gen_steps)
    store.remove("fixture")
    if via_generate != greedy_ids:
        raise SystemExit(
            f"refusing to write {out_dir}: step-driven greedy_ids {greedy_ids} disagree with "
            f"src.serve.generate's prefill-based path {via_generate} — the two Python code "
            "paths (step-by-step vs one-shot prefill) must produce identical greedy ids "
            "(src/conformance/prefill_decode_parity.py's contract)")

    save_file({
        "prompt_ids": np.array(prompt_ids, dtype=np.int32),
        "greedy_ids": np.array(greedy_ids, dtype=np.int32),
        "margins": np.array(margins, dtype=np.float32),
    }, str(out / "generation.safetensors"))

    meta = {
        "config": os.path.basename(config_path),
        "batch": batch,
        "seq": seq,
        "seed": seed,
        "precision": cfg.precision,
        "vocab_size": cfg.vocab_size,
        "gen_steps": gen_steps,
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "moe_bias": moe_bias,
        "forward_step_max_abs_diff": verdict["max_abs_diff"],
        "greedy_margin_min": min(margins) if margins else None,
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
    p.add_argument("--vocab-size", type=int, default=None,
                   help="override the config's vocab_size (#167's toy-gen: a "
                        "monica-tokenize-trained tokenizer's vocab needs a wider model than "
                        "toy.yaml's default 256)")
    p.add_argument("--gen-steps", type=int, default=16,
                   help="greedy-decode steps written to generation.safetensors (#167 AC1)")
    p.add_argument("--moe-bias", action="store_true",
                   help="activate a per-expert route bias (#213) so the fixture carries "
                        "moe_route_bias.* keys and exercises the biased-ranking branch")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta = build_fixture(args.config, args.out, batch=args.batch, seq=args.seq,
                         precision=args.precision, moe_bias=args.moe_bias, seed=args.seed,
                         vocab_size=args.vocab_size, gen_steps=args.gen_steps)
    print(f"wrote {args.out}: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
