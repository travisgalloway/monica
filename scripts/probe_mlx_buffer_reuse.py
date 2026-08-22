"""Measure MLX 0.32.0's buffer-reuse corruption (#298) — the instrument, not a gate.

#298: on this repo's MLX 0.32.0 Apple-Silicon build, repeated `MLXMambaModel` construction
plus computation within ONE process has been observed to silently corrupt an unrelated
later computation — deterministic per process, correct shapes, no exception, no log. The
defect is upstream; this repo cannot fix it. What it can do is *measure* it, and #298's
definition of done is a set of written, measured answers rather than a cure.

This script is where those numbers come from. It is a probe (precedent:
`scripts/probe_ts_lsp_debounce.py`), NOT part of the test suite: it is slow, it is
deliberately allowed to fail, and its output is transcribed into
`docs/design/14-inference-engine.md` §"MLX 0.32.0 buffer reuse (#298)".

    .venv/bin/python scripts/probe_mlx_buffer_reuse.py --pattern mixing   --trials 20
    .venv/bin/python scripts/probe_mlx_buffer_reuse.py --pattern hotpath  --trials 30
    .venv/bin/python scripts/probe_mlx_buffer_reuse.py --pattern export   --trials 20
    .venv/bin/python scripts/probe_mlx_buffer_reuse.py --pattern throughput --trials 30

## Patterns

`mixing`   — the reproducer shape #264 root-caused, with the `mx.eval(h)` barrier REMOVED.
             Many freshly-constructed models each run the mixing-matrix accessor (which
             forks a lazy `h` into two independent consumers without materialising it) and
             then `prefill(..., last_only=True)`. This is the pattern with a *known*
             positive control, so it is the probe's own self-check.
`hotpath`  — the training/inference shape: `forward` / stacked `step` / `prefill` /
             `make_train_step` over many freshly-constructed models, no interpretability
             accessors, i.e. no lazy fork. This is what the hot-path decision rests on.
`export`   — N in-process `build_fixture` runs, reporting the `mixing.{i}` drift
             distribution against the checked-in `toy` oracle. This is what the `mixing.N`
             near-zero escalation rests on.
`throughput` — the cost side: repeated forward + train_step timed at both cache settings,
             so "keep it off the hot path" is grounded rather than asserted.

## The anti-blind rule

MLX is deterministic given identical inputs and this graph has no dropout and no sampling,
so every trial from the same seed must be bit-identical to trial 0. A difference is
corruption, full stop — there is no tolerance to argue about.

`--pattern mixing` is the probe's POSITIVE CONTROL. If it does not reproduce with the
barrier removed and the cache on, the instrument cannot detect the defect on this host, so
a clean `hotpath` result is *no evidence at all*. In that case the probe prints `BLIND` and
exits non-zero, rather than reporting "clean" — a check that cannot observe its target must
say so.

`--report-only` (used by the `mlx-buffer-reuse-probe` CI job in
`.github/workflows/scheduled-parity.yml`) records `"blind": true` plus `"blind_reason"` in
the emitted JSON and exits **0** instead of 3, so a scheduled run captures the measurement
without a red job. It suppresses *only* the BLIND exit — a probe that cannot run at all
(no MLX wheel, unloadable config, import error) still fails loudly, because that is not a
measurement. `"blind"` is present on every run, blind or not, so a consumer never has to
read blindness out of a missing key.

MLX-only, like `scripts/smoke_test.py`: it does not run on a Linux/CUDA host.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOY_FIXTURE = REPO_ROOT / "swift" / "engine" / "Fixtures" / "toy"


def _tokens(cfg, seq: int, seed: int = 0):
    return np.random.default_rng(seed).integers(
        0, cfg.vocab_size, size=(2, seq)).astype(np.int32)


def _mixing_matrices_no_barrier(model, tokens):
    """`MLXMambaModel.mixing_matrices` with the #264 `mx.eval(h)` barrier REMOVED.

    Kept here rather than monkeypatched into the backend so the probe cannot accidentally
    leave the production accessor unbarriered; it is a verbatim copy of the loop minus the
    one `mx.eval(h)` line. This is the positive control's whole content — the lazy `h`
    forking into `layer.mixing_matrix(h)` and `layer_fn(h)` without being materialised.
    """
    import mlx.core as mx

    from src.model.mlx_backend import _cast

    h = _cast(model.embedding(mx.array(tokens)), model._cd)
    out = []
    for i, (layer, layer_fn) in enumerate(zip(model.layers, model._layer_fns)):
        if not model.config.is_attention_layer(i) and not model.config.is_moe_layer(i):
            out.append((i, layer.mixing_matrix(h).mean(axis=1)))
        h = layer_fn(h)
    return out


def _fresh_model(cfg, seed: int):
    import mlx.core as mx

    from src.model.mlx_backend import MLXMambaModel

    mx.random.seed(seed)
    return MLXMambaModel(cfg)


_USE_BARRIER = False


def _run_mixing(cfg, tokens, seed: int):
    """One `mixing` trial: read the mixing matrices through the UNBARRIERED accessor, then
    do an unrelated `prefill` on the same model. #264's finding is that the corruption
    lands on the *later, unrelated* call, not on the values read here."""
    import mlx.core as mx

    model = _fresh_model(cfg, seed)
    # `--barrier` swaps the unbarriered copy for the SHIPPED accessor, so the same trial
    # loop measures whether #264's `mx.eval(h)` fix actually holds rather than assuming it.
    mats = (model.mixing_matrices(mx.array(tokens)) if _USE_BARRIER
            else _mixing_matrices_no_barrier(model, tokens))
    mx.eval([m for _, m in mats])
    logits, _ = model.prefill(mx.array(tokens), last_only=True)
    mx.eval(logits)
    return np.array(logits)


def _run_hotpath(cfg, tokens, seed: int):
    """One `hotpath` trial: exactly the surfaces training and inference use — `forward`,
    stacked `step`, `prefill`, and a real `make_train_step` — and no interpretability
    accessor, so `h` is never forked into two consumers."""
    import mlx.core as mx

    model = _fresh_model(cfg, seed)
    tok = mx.array(tokens)

    fwd = model.forward(tok)
    mx.eval(fwd)

    # prefill over the prompt, then three stacked `step`s off the returned state — the
    # serving path, and the one #264 saw corrupted (`prefill(..., last_only=True)`).
    pre_logits, state = model.prefill(tok[:, :-4])
    mx.eval(pre_logits)
    step_logits = []
    for j in range(3):
        lg, state = model.step(tok[:, -4 + j], state)
        step_logits.append(lg)
    mx.eval(step_logits)

    last_only, _ = model.prefill(tok[:, :-1], last_only=True)
    mx.eval(last_only)

    stats = _train_step_stats(model, cfg, tokens)

    return np.concatenate([
        np.array(fwd).ravel(),
        np.array(pre_logits).ravel(),
        np.concatenate([np.array(lg).ravel() for lg in step_logits]),
        np.array(last_only).ravel(),
        np.array([stats["loss"], stats["grad_norm"]], dtype=np.float32),
    ])


def _train_step_stats(model, cfg, tokens) -> dict:
    """One real `make_train_step` step — the production primitive `src/train/loop.py` is
    injected with, so the hot-path pattern exercises backward as well as forward."""
    import mlx.optimizers as optim

    from src.model.mlx_train_step import make_train_step

    train_step = make_train_step(model, optim.AdamW(learning_rate=1e-3))
    inputs = tokens[:, :-1].astype(np.int32)
    targets = tokens[:, 1:].astype(np.int32)
    return train_step(model, [(inputs, targets)], 1e-3)


def _clean_reference(cfg, tokens, seed: int):
    """The known-good value for the `mixing` pattern: `prefill(..., last_only=True)` on a
    freshly-constructed model with NO interpretability accessor called at all.

    Trial 0 is NOT a safe reference. Measured on this host (toy-moe.yaml, seq 129): with
    `mixing_matrices` in the loop, roughly half of the trials — including trial 0 — return
    a corrupted prefill, so comparing trials to trial 0 reports the *clean* trials as the
    failures and undercounts by ~2x. Without the accessor the same loop is bit-identical
    8/8, which is what makes this the reference and not just another sample.
    """
    import mlx.core as mx

    model = _fresh_model(cfg, seed)
    logits, _ = model.prefill(mx.array(tokens), last_only=True)
    mx.eval(logits)
    return np.array(logits)


def _bit_identical_trials(runner, cfg, tokens, trials: int, seed: int,
                          reference_runner=None) -> dict:
    """Run `runner` `trials` times in ONE process and compare every result to a reference.

    fp32 MLX is deterministic given identical inputs, and this graph has no dropout and no
    sampling, so anything but bit-identity is corruption. `max|d|` is reported only to
    characterise a failure, never as a tolerance to pass under.

    `reference_runner` (defaulting to `runner` itself, i.e. trial 0) produces the value
    every trial is compared against. It is re-run at the END as well, and disagreement
    between the two reference computations makes the whole run BLIND — a reference that
    moved cannot adjudicate anything.
    """
    reference_runner = reference_runner or runner
    ref = reference_runner(cfg, tokens, seed)

    failures, drifts = 0, []
    for _ in range(trials):
        got = runner(cfg, tokens, seed)
        if got.shape != ref.shape:
            failures += 1
            drifts.append(float("inf"))
            continue
        d = float(np.max(np.abs(got - ref)))
        drifts.append(d)
        if d != 0.0:
            failures += 1

    ref_again = reference_runner(cfg, tokens, seed)
    ref_stable = bool(ref.shape == ref_again.shape
                      and np.max(np.abs(ref - ref_again)) == 0.0)
    return {
        "trials": trials,
        "failures": failures,
        "failure_rate": failures / trials,
        "max_drift": max(drifts) if drifts else 0.0,
        "median_drift": statistics.median(drifts) if drifts else 0.0,
        "reference_stable": ref_stable,
    }


def _run_export(trials: int, seq: int) -> dict:
    """`mixing.{i}` drift across `trials` fresh in-process `build_fixture` runs, measured
    against the CHECKED-IN toy oracle — the exact comparison
    `tests/test_parity_fixture_export.py` makes, which is the one that flaked in CI
    (`mixing.1 max|d| = 2.694e-05` on run 31806256071).

    `forward_logits` rides along as a control: it shares the export but not the near-zero
    structure, so if only `mixing.*` drifts the cause is the comparison's atol floor rather
    than the export.
    """
    from safetensors.numpy import load_file

    from scripts.export_parity_fixture import build_fixture

    ref = load_file(str(TOY_FIXTURE / "reference.safetensors"))
    keys = sorted(k for k in ref if k.startswith("mixing.")) + ["forward_logits"]
    per_key = {k: [] for k in keys}

    with tempfile.TemporaryDirectory() as td:
        for t in range(trials):
            out = Path(td) / f"toy-{t}"
            build_fixture("config/toy.yaml", str(out), batch=2, seq=seq,
                          packed_doc_lengths="Q,2*Q,5")
            fresh = load_file(str(out / "reference.safetensors"))
            for k in keys:
                per_key[k].append(float(np.max(np.abs(fresh[k] - ref[k]))))

    return {
        "trials": trials,
        "vs": str(TOY_FIXTURE.relative_to(REPO_ROOT)),
        "per_key": {k: {"max": max(v), "median": statistics.median(v),
                        "nonzero_trials": sum(1 for x in v if x != 0.0)}
                    for k, v in per_key.items()},
        # The near-zero structure the escalation comment points at: the causal mixing
        # matrix's strict upper triangle is EXACTLY zero, so at |b| ~ 0 the np.allclose
        # band collapses to atol with no rtol headroom at all.
        "mixing_abs_max_in_oracle": {
            k: float(np.max(np.abs(ref[k]))) for k in keys if k.startswith("mixing.")},
    }


def _run_throughput(cfg, tokens, trials: int, seed: int) -> dict:
    """Wall-clock cost of `set_cache_limit(0)`: the same forward + train_step loop timed
    with the buffer cache on and off, in two fresh sub-measurements within this process.

    Toy scale, so this is an *upper bound on the shape* of the cost rather than the poc
    number — the allocator penalty is largest where allocation dominates compute, which is
    exactly the toy regime.
    """
    import mlx.core as mx
    import mlx.optimizers as optim

    from scripts.export_parity_fixture import disable_buffer_cache
    from src.model.mlx_train_step import make_train_step

    def _timed() -> float:
        model = _fresh_model(cfg, seed)
        train_step = make_train_step(model, optim.AdamW(learning_rate=1e-3))
        inputs = tokens[:, :-1].astype(np.int32)
        targets = tokens[:, 1:].astype(np.int32)
        train_step(model, [(inputs, targets)], 1e-3)          # warm-up, uncounted
        t0 = time.perf_counter()
        for _ in range(trials):
            out = model.forward(mx.array(tokens))
            mx.eval(out)
            train_step(model, [(inputs, targets)], 1e-3)
        return time.perf_counter() - t0

    cache_on = _timed()
    with disable_buffer_cache():
        cache_off = _timed()

    return {
        "trials": trials,
        "seconds_cache_on": cache_on,
        "seconds_cache_limit_zero": cache_off,
        "slowdown_pct": 100.0 * (cache_off - cache_on) / cache_on,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", required=True,
                   choices=["mixing", "hotpath", "export", "throughput"])
    p.add_argument("--cache-limit", default="default", choices=["default", "zero"],
                   help="'default' leaves MLX's buffer-cache pool alone (the state the "
                        "defect needs); 'zero' applies the #264 mitigation")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seq", type=int, default=129)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="config/toy-moe.yaml",
                   help="the DEFAULT is toy-moe.yaml, not toy.yaml, because that is where "
                        "the positive control actually reproduces on this host — an "
                        "instrument that defaults to a blind configuration is worse than "
                        "no instrument (see the --pattern mixing sweep in design/14)")
    p.add_argument("--barrier", action="store_true",
                   help="--pattern mixing: use the SHIPPED `MLXMambaModel.mixing_matrices` "
                        "(with #264's mx.eval(h) barrier) instead of the unbarriered copy, "
                        "to measure whether the shipped fix holds")
    p.add_argument("--json", action="store_true", help="emit the result dict as JSON only")
    p.add_argument("--report-only", action="store_true",
                   help="record BLIND in the result dict ('blind': true, with "
                        "'blind_reason') and exit 0 instead of 3. For a CI step that must "
                        "CAPTURE the measurement without turning the job red — a hosted "
                        "runner whose shape does not reproduce the defect is a legitimate "
                        "measurement, not a build failure. It suppresses ONLY the BLIND "
                        "exit: a probe that cannot run at all (no MLX, bad config, import "
                        "error) still fails, because that is not a measurement.")
    args = p.parse_args()

    import mlx.core as mx

    from scripts.export_parity_fixture import disable_buffer_cache_for_process
    from src.model.blocks import load_config

    if args.cache_limit == "zero":
        disable_buffer_cache_for_process()

    global _USE_BARRIER
    _USE_BARRIER = args.barrier

    cfg = load_config(args.config)
    tokens = _tokens(cfg, args.seq, args.seed)

    if args.pattern == "mixing":
        # The reference is deliberately NOT trial 0 — see `_clean_reference`.
        result = _bit_identical_trials(_run_mixing, cfg, tokens, args.trials, args.seed,
                                       reference_runner=_clean_reference)
    elif args.pattern == "hotpath":
        # No accessor-free variant exists to compare against here: the hot path IS the
        # clean variant, so trial 0 is the reference and `reference_stable` is the check.
        result = _bit_identical_trials(_run_hotpath, cfg, tokens, args.trials, args.seed)
    elif args.pattern == "export":
        result = _run_export(args.trials, args.seq)
    else:
        result = _run_throughput(cfg, tokens, args.trials, args.seed)

    # `export` always builds toy.yaml (it compares against the checked-in `toy` oracle),
    # so it reports that rather than --config, which it ignores.
    config_label = "config/toy.yaml" if args.pattern == "export" else args.config
    result = {"pattern": args.pattern, "cache_limit": args.cache_limit,
              "mlx_version": mx.__version__, "config": config_label, **result}

    # The anti-blind rule, decided BEFORE the JSON is printed so `blind`/`blind_reason` ride
    # in the emitted dict rather than only in stderr-adjacent prose. `mixing` with the cache
    # ON is the positive control: it is the ONE pattern known to reproduce. If it does not,
    # this instrument cannot see the defect on this host, and every other pattern's clean
    # result is unknown rather than good news.
    blind_reason = None
    if args.pattern in ("mixing", "hotpath") and not result.get("reference_stable", True):
        blind_reason = (
            "the reference value itself moved between its two computations, so no trial "
            "verdict in this run means anything. Do not report it.")
    elif (args.pattern == "mixing" and args.cache_limit == "default"
            and not args.barrier and result["failures"] == 0):
        blind_reason = (
            "the positive control did NOT reproduce on this host "
            f"({result['trials']} trials, mlx {mx.__version__}, {args.config}, "
            f"seq={args.seq}). A clean `hotpath` result is therefore NOT evidence "
            "that the hot path is immune — the instrument cannot detect the defect "
            "at all here. Do not report this run as 'clean'. Reproduction is "
            "shape-dependent: toy-moe.yaml and toy-hybrid.yaml at seq=129 DO "
            "reproduce on the M1 Pro dev host; toy.yaml and seq=512 do not.")

    # `blind` is recorded on EVERY run, not only the blind ones, so a consumer reading the
    # JSON never has to infer blindness from a missing key — absence would be indistinguish-
    # able from "this build of the probe predates the flag", which is the same BLIND-reads-
    # healthy failure the flag exists to make visible.
    result["blind"] = blind_reason is not None
    result["blind_reason"] = blind_reason

    print(json.dumps(result, indent=2))

    # Under --json, stdout must be PARSEABLE JSON and nothing else — the CI job `tee`s it
    # straight into a .json file the verdict step reads back. Every human-readable line
    # below therefore goes to stderr in that mode; it is still shown in the workflow log,
    # it just stops corrupting the machine-readable half. (`--json` claimed "JSON only"
    # before this and did not deliver it.)
    def say(text: str) -> None:
        print(text, file=sys.stderr if args.json else sys.stdout)

    if blind_reason is not None:
        say(f"BLIND: {blind_reason}")
        # --report-only suppresses ONLY this exit. Everything above — an MLX that will not
        # import, a config that will not load, a pattern that raises — has already failed
        # loudly by the time we get here, and stays failing.
        if not args.report_only:
            raise SystemExit(3)
        say("--report-only: exiting 0 so the measurement is captured; the BLIND verdict "
            "above is the result, NOT a clean bill of health.")
        return

    if args.pattern == "mixing" and args.cache_limit == "default" and not args.barrier:
        say(f"positive control reproduced: {result['failures']}/{result['trials']} "
            "trials diverged from the accessor-free reference — the instrument can see "
            "the defect.")


if __name__ == "__main__":
    main()
