"""Context-length / throughput / memory bench harness — issue #104.

Measures, for two same-dims arms built from one config, how prefill/decode tok/s and
peak memory behave as context length grows:

  * ``ssm``  — the config as given (poc-qwen.yaml is pure Mamba-2: constant state size,
    independent of how many tokens have been consumed).
  * ``attn`` — ``dataclasses.replace(cfg, attn_every=1)``, i.e. every layer becomes
    causal-MHA+RoPE (`AttentionBlock`) — a same-dims transformer whose KV cache grows
    linearly with context length.

This is a THROUGHPUT/MEMORY benchmark, not a quality eval: tok/s, peak GB, and the
analytic per-token state size are architecture properties, not weight properties, so it
runs meaningfully on a RANDOM-INIT model (``--weights`` is optional, for when a trained
checkpoint is available — note a checkpoint trained on the ``ssm`` arm's Mamba weights
cannot be loaded into the ``attn`` arm, since the parameter shapes/names differ; the
``attn`` arm is therefore always random-init here). It measures the constant-memory-
per-token claim that is the mechanical core of #104's final headline metric; that
headline number (post-trained ~1B student vs a same-size transformer) still needs the
post-trained student (see the #65 tracker) — this harness is the reusable measurement
tool for that comparison, runnable today.

Both arms use the SAME recurrence primitive (`model.step`, batch=1) — a token-by-token
loop, exactly the constant-memory decode path `scripts/bench_train_step.py --mode
decode` already benchmarks for a single arm; this script generalizes that loop across
context lengths and a second (attention) arm.

RoPE positions are computed fresh each step from ``mx.arange(t, t+1)`` (no fixed-size
table — see ``_rope_cos_sin`` in ``src/model/mlx_backend.py``), so stepping past
``cfg.seq_len`` does not index out of bounds; the sweep is free to exceed the
training-time ``seq_len``.

    .venv/bin/python scripts/bench_context.py \\
        --config config/poc-qwen.yaml --lengths 512,1024,2048 --decode-tokens 64 \\
        --max-attn-length 2048

MLX imports are kept local so the module stays importable (e.g. for --help) on any
host, mirroring ``scripts/bench_train_step.py``.
"""

from __future__ import annotations

import argparse
import csv as csv_module
import dataclasses
import platform
import time
from pathlib import Path

ARMS = ("ssm", "attn")


def _parse_lengths(s: str) -> list[int]:
    try:
        lengths = [int(x) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"--lengths must be comma-separated ints, got {s!r}") from e
    if not lengths:
        raise ValueError("--lengths must contain at least one positive int")
    if any(l <= 0 for l in lengths):
        raise ValueError(f"--lengths must all be positive, got {lengths}")
    return lengths


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="e.g. config/poc-qwen.yaml (pure Mamba-2 — the ssm arm)")
    ap.add_argument("--weights", type=Path, default=None,
                    help="optional safetensors checkpoint for the ssm arm only "
                         "(random init otherwise; throughput/memory are weight-independent)")
    ap.add_argument("--lengths", type=_parse_lengths, default=_parse_lengths("512,1024,2048"),
                    help="comma-separated context lengths to sweep, e.g. 512,1024,2048")
    ap.add_argument("--decode-tokens", type=int, default=64,
                    help="sustained-decode steps timed after prefill (default 64)")
    ap.add_argument("--max-attn-length", type=int, default=None,
                    help="skip the attn arm above this length (guards against a slow/OOM "
                         "quadratic-KV-cache run); unset = no cap")
    ap.add_argument("--csv", type=Path, default=None, help="optional path to write CSV rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=4,
                    help="untimed steps run first to prime compilation (default 4)")
    args = ap.parse_args()
    if args.decode_tokens < 1:
        ap.error("--decode-tokens must be >= 1")
    if args.warmup_steps < 1:
        ap.error("--warmup-steps must be >= 1")
    if args.max_attn_length is not None and args.max_attn_length < 1:
        ap.error("--max-attn-length must be >= 1")
    return args


def arm_config(cfg, arm: str):
    """Build the per-arm MambaConfig. 'ssm': `cfg` unchanged. 'attn': every layer becomes
    causal-MHA+RoPE (`dataclasses.replace(cfg, attn_every=1)`) — zero new model code,
    the same pattern `scripts/retrieval_probe.py` uses for its pure-attention arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    if arm == "ssm":
        return cfg
    c = dataclasses.replace(cfg, attn_every=1)
    c.validate()
    return c


def analytic_state_bytes(cfg, length: int) -> int:
    """Architecture-derived recurrent-state size (bytes) at context length `length`.

    Pure-Mamba configs (`attn_every` unset — the ssm arm): `per_session_state_bytes`
    (src/serve/sessions.py) is exact and independent of `length` — the "constant memory
    per token" claim #104 exists to demonstrate. That function only models Mamba
    layers, so it is not meaningful for the attn arm; there we compute the KV cache
    size directly (`AttentionBlock.step` caches (k, v), each (B, H, T, Dh) fp32,
    growing by one token per step), which DOES grow linearly with `length`.
    """
    from src.serve.sessions import per_session_state_bytes

    if not cfg.attn_every:
        return per_session_state_bytes(cfg, conservative_fp32=False)
    n_attn_layers = cfg.n_layers // cfg.attn_every
    return n_attn_layers * 2 * cfg.n_attn_heads_resolved * cfg.attn_head_dim * length * 4


def measure(model, mx, length: int, decode_tokens: int, *, seed: int = 0,
            warmup_steps: int = 4) -> dict:
    """Prefill `length` tokens then sustain-decode `decode_tokens` more, one token at a
    time through the `model.step` recurrence (constant-mem decode primitive). A short
    untimed warmup on a throwaway state runs first to prime MLX's compiled graphs
    (mirrors `bench_train_step.py`'s separated first-call timing). Returns
    `{prefill_tok_s, decode_tok_s, peak_gb}`.
    """
    import numpy as np

    vocab = model.config.vocab_size
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=length + decode_tokens).astype(np.int64)

    warm_state = model.init_state(1)
    n_warm = min(warmup_steps, length + decode_tokens)
    warm_tokens = rng.integers(0, vocab, size=n_warm).astype(np.int64)
    last = None
    for i in range(n_warm):
        last, warm_state = model.step(warm_tokens[i:i + 1], warm_state)
        mx.eval(last, warm_state)

    mx.reset_peak_memory()
    state = model.init_state(1)

    t0 = time.perf_counter()
    for i in range(length):
        last, state = model.step(tokens[i:i + 1], state)
        mx.eval(last, state)
    prefill_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(length, length + decode_tokens):
        last, state = model.step(tokens[i:i + 1], state)
        mx.eval(last, state)
    decode_s = time.perf_counter() - t0

    peak_gb = mx.get_peak_memory() / 2**30

    if not bool(mx.all(mx.isfinite(last)).item()):
        raise RuntimeError(f"non-finite logits at length={length} — run is degenerate")

    return {
        "prefill_tok_s": length / prefill_s,
        "decode_tok_s": decode_tokens / decode_s,
        "peak_gb": peak_gb,
    }


def run_sweep(model_cls, mx, cfg, lengths, decode_tokens, *, weights=None,
              max_attn_length=None, seed=0, warmup_steps=4) -> list[dict]:
    """Run both arms across `lengths`; returns a list of result-row dicts. `model_cls`
    is the model constructor (`MLXMambaModel`), injected so this stays testable without
    a top-level mlx import."""
    rows = []
    for arm in ARMS:
        acfg = arm_config(cfg, arm)
        model = model_cls(acfg)
        if weights is not None and arm == "ssm":
            model.load(str(weights))
        mx.eval(model.parameters())
        for length in lengths:
            if arm == "attn" and max_attn_length is not None and length > max_attn_length:
                print(f"[skip] arm=attn length={length} exceeds "
                      f"--max-attn-length={max_attn_length} (would risk a slow/OOM run "
                      "growing the KV cache) — not measured")
                continue
            m = measure(model, mx, length, decode_tokens, seed=seed, warmup_steps=warmup_steps)
            rows.append({
                "arm": arm,
                "length": length,
                "prefill_tok_s": m["prefill_tok_s"],
                "decode_tok_s": m["decode_tok_s"],
                "peak_gb": m["peak_gb"],
                "state_bytes": analytic_state_bytes(acfg, length),
            })
    return rows


def _print_table(rows: list[dict]) -> None:
    header = f"{'arm':<6} {'length':>8} {'prefill tok/s':>14} {'decode tok/s':>13} {'peak GB':>9} {'state MB':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['arm']:<6} {r['length']:>8} {r['prefill_tok_s']:>14,.1f} "
              f"{r['decode_tok_s']:>13,.1f} {r['peak_gb']:>9.3f} "
              f"{r['state_bytes'] / 2**20:>10.3f}")

    ssm_rows = [r for r in rows if r["arm"] == "ssm"]
    attn_rows = [r for r in rows if r["arm"] == "attn"]
    if ssm_rows and attn_rows:
        ssm_state = ssm_rows[0]["state_bytes"]
        print(f"\n[summary] ssm state is flat at {ssm_state / 2**20:.3f} MB across all "
              f"swept lengths (architecture-constant); attn state grows from "
              f"{attn_rows[0]['state_bytes'] / 2**20:.3f} MB (L={attn_rows[0]['length']}) "
              f"to {attn_rows[-1]['state_bytes'] / 2**20:.3f} MB (L={attn_rows[-1]['length']}).")
        # A 10% margin keeps this from firing on ordinary measurement jitter (batch-1
        # decode timing is noisy run to run) — it should flag a real, sustained
        # degradation, not the first length where attn happens to measure slightly slower.
        margin = 1.10
        ssm_peak, ssm_decode = ssm_rows[0]["peak_gb"], ssm_rows[0]["decode_tok_s"]
        for r in attn_rows:
            if r["peak_gb"] > margin * ssm_peak or r["decode_tok_s"] < ssm_decode / margin:
                print(f"[summary] crossover: at length={r['length']}, attn peak memory "
                      f"({r['peak_gb']:.3f} GB) or decode tok/s ({r['decode_tok_s']:,.1f}) "
                      f"has degraded >10% past ssm's ({ssm_peak:.3f} GB / "
                      f"{ssm_decode:,.1f} tok/s).")
                break
        else:
            print("[summary] no crossover observed in the swept range — attn stayed "
                  "within ssm's peak-mem/decode-tok/s envelope (10% margin) at these lengths.")


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=["arm", "length", "prefill_tok_s",
                                                       "decode_tok_s", "peak_gb", "state_bytes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] wrote {len(rows)} rows to {path}")


def main() -> None:
    args = _parse_args()

    # MLX-only imports kept local so the module stays importable (e.g. for --help) on
    # any host — mirrors scripts/bench_train_step.py.
    try:
        import mlx.core as mx
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/bench_context.py ...\n"
            "(mlx installs only on Apple Silicon via the '[mlx]' extra; a bare "
            "`python` likely points at a different interpreter.)"
        ) from e

    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel

    mx.random.seed(args.seed)
    cfg = load_config(str(args.config))
    print(f"[bench_context] mlx {mx.__version__}  {platform.platform()}")
    print(f"[bench_context] config={args.config}  d_model={cfg.d_model}  n_layers={cfg.n_layers}  "
          f"vocab={cfg.vocab_size}  lengths={args.lengths}  decode_tokens={args.decode_tokens}  "
          f"weights={args.weights or '(random init)'}\n")

    rows = run_sweep(MLXMambaModel, mx, cfg, args.lengths, args.decode_tokens,
                     weights=args.weights, max_attn_length=args.max_attn_length,
                     seed=args.seed, warmup_steps=args.warmup_steps)
    _print_table(rows)
    if args.csv:
        _write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
