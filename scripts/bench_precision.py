"""Precision micro-benchmark (Apple Silicon / MLX) — the reproducer for issue #3.

Settles the poc training precision *empirically* on this Metal GPU rather than
assuming bf16. It times the matmul (GEMM) workload that dominates a Mamba-2 forward
pass — the per-layer in/x/out projections plus the tied LM head — in fp32, fp16, and
bf16, and reports achieved TFLOP/s and a tokens/sec-equivalent for each. The shapes
come from the model config (default `config/poc.yaml`), so the numbers are
representative of the real run, not a generic square GEMM.

This is op-level on purpose: it reproduces the kind of measurement the precision
decision rests on (see config/poc.yaml + docs/design/07-configs-and-decisions.md)
without requiring the model itself to compute in low precision. The model still runs
fp32; `precision` only toggles the fp16 loss scaler.

    .venv/bin/python scripts/bench_precision.py                       # config/poc.yaml
    .venv/bin/python scripts/bench_precision.py --batch 8 --iters 100

MLX imports are kept local so the module stays importable for `--help` on any host.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--batch", type=int, default=4, help="sequences per GEMM (rows = batch*seq)")
    ap.add_argument("--seq", type=int, default=None, help="sequence length (default: config seq_len)")
    ap.add_argument("--iters", type=int, default=50, help="timed iterations per dtype")
    ap.add_argument("--warmup", type=int, default=10, help="untimed warmup iterations per dtype")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _gemm_shapes(cfg, tokens: int):
    """Unique (K, N, count) GEMMs of one Mamba-2 forward, weighted by multiplicity.

    Per layer: in_proj (d_model -> 2*d_inner), x_proj (d_inner -> dt_rank+2*d_state),
    out_proj (d_inner -> d_model). Once for the whole model: the tied head
    (d_model -> vocab). The selective-scan einsums are not GEMMs against a weight and
    are a small share of FLOPs at poc dims, so the projections + head are the
    representative throughput workload."""
    dm, di = cfg.d_model, cfg.d_inner
    nL = cfg.n_layers
    x_out = cfg.dt_rank_resolved + 2 * cfg.d_state
    return [
        ("in_proj",  dm, 2 * di, nL),
        ("x_proj",   di, x_out,  nL),
        ("out_proj", di, dm,     nL),
        ("lm_head",  dm, cfg.vocab_size, 1),
    ]


def main() -> None:
    args = _parse_args()
    import mlx.core as mx

    from src.model.blocks import load_config

    cfg = load_config(str(args.config))
    seq = args.seq if args.seq is not None else cfg.seq_len
    tokens = args.batch * seq                       # rows M of every GEMM
    shapes = _gemm_shapes(cfg, tokens)

    # FLOPs of one full forward pass over `tokens` rows: 2*M*K*N per GEMM, x count.
    flops_per_iter = sum(2 * tokens * K * N * count for _, K, N, count in shapes)

    dtypes = [("fp32", mx.float32), ("fp16", mx.float16), ("bf16", mx.bfloat16)]
    print(f"[bench] config={args.config}  d_model={cfg.d_model}  d_inner={cfg.d_inner}  "
          f"n_layers={cfg.n_layers}  vocab={cfg.vocab_size}")
    print(f"[bench] rows(M)={tokens} (batch {args.batch} x seq {seq})  "
          f"iters={args.iters} (+{args.warmup} warmup)  "
          f"FLOPs/iter={flops_per_iter / 1e9:.2f} GFLOP\n")

    results = []
    for name, dt in dtypes:
        mx.random.seed(args.seed)
        # One A and W per unique GEMM, reused across its `count` calls (throughput is
        # what we measure, not a dimensionally-chained activation). Scaled small so
        # fp16 accumulation stays well inside its range — this is a speed test, the
        # finiteness assert below guards against a degenerate (overflowed) run.
        ops = []
        for _, K, N, count in shapes:
            A = (mx.random.uniform(shape=(tokens, K)) * 0.2 - 0.1).astype(dt)
            W = (mx.random.uniform(shape=(K, N)) * 0.2 - 0.1).astype(dt)
            ops.append((A, W, count))
        mx.eval([A for A, _, _ in ops] + [W for _, W, _ in ops])

        def run_iter():
            outs = []
            for A, W, count in ops:
                for _ in range(count):
                    outs.append(A @ W)
            mx.eval(outs)
            return outs

        for _ in range(args.warmup):
            run_iter()

        t0 = time.perf_counter()
        last = None
        for _ in range(args.iters):
            last = run_iter()
        elapsed = time.perf_counter() - t0

        if not bool(mx.all(mx.isfinite(last[0])).item()):
            raise RuntimeError(f"{name}: non-finite GEMM output — benchmark is degenerate")

        tflops = (flops_per_iter * args.iters) / elapsed / 1e12
        tok_per_s = (tokens * args.iters) / elapsed
        results.append((name, tflops, tok_per_s))

    fp32_tflops = next(t for n, t, _ in results if n == "fp32")
    print(f"{'dtype':<6} {'TFLOP/s':>9} {'tokens/s':>12} {'vs fp32':>9}")
    print("-" * 40)
    for name, tflops, tok in results:
        print(f"{name:<6} {tflops:>9.2f} {tok:>12,.0f} {tflops / fp32_tflops:>8.2f}x")

    best = max(results, key=lambda r: r[1])
    fp16_t = next(t for n, t, _ in results if n == "fp16")
    bf16_t = next(t for n, t, _ in results if n == "bf16")
    print(f"\n[verdict] fastest: {best[0]} ({best[1]:.2f} TFLOP/s).  "
          f"fp16 is {fp16_t / bf16_t:.2f}x bf16 and {fp16_t / fp32_tflops:.2f}x fp32.")


if __name__ == "__main__":
    main()
