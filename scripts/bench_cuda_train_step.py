"""CUDA / PyTorch train-step benchmark — the torch mirror of `bench_train_step.py`.

Times the *production* CUDA training path — `CUDAMambaModel` + `torch.optim.AdamW` +
`scaler_for_precision` + the real `make_train_step` closure (including the per-step
`.item()` overflow sync) — so a RunPod CUDA box has trustworthy s/step, tokens/s, and
peak-GPU-memory numbers before committing to a real run. It wires the model exactly as
`scripts/train.py --backend cuda` does, through `get_backend("cuda")`.

Develop it on the Mac FIRST (`--device cpu`, the default fallback): torch-CPU is in the
`[cuda]` extra and exercises the identical code path — slow, but it proves the script
works so the GPU run later is a no-surprise repeat. On the pod, `--device auto` picks
CUDA when available.

    # Mac dry-run (CPU; tiny so it finishes):
    .venv/bin/python scripts/bench_cuda_train_step.py --config config/toy.yaml \\
        --batch 2 --grad-accum 2 --warmup 1 --iters 3

    # RunPod GPU, standard protocol:
    .venv/bin/python scripts/bench_cuda_train_step.py --config config/poc.yaml \\
        --batch 32 --grad-accum 4 --warmup 3 --iters 10

torch imports are kept local so the module stays importable for `--help` on any host.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import platform
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                    help="auto: cuda if available, else cpu (the Mac dev fallback)")
    ap.add_argument("--batch", type=int, default=32, help="sequences per micro-batch")
    ap.add_argument("--seq", type=int, default=None, help="sequence length (default: config seq_len)")
    ap.add_argument("--grad-accum", type=int, default=4, help="micro-batches per optimizer step")
    ap.add_argument("--warmup", type=int, default=3, help="untimed warmup steps (first reported separately)")
    ap.add_argument("--iters", type=int, default=10, help="timed optimizer steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate (matches train.py --base-lr default)")
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="override SSD scan chunk length Q (default: config value, else 64)")
    args = ap.parse_args()
    # Fail fast on inputs that would crash or report nonsense (same as bench_train_step).
    if args.iters < 1:
        ap.error("--iters must be >= 1")
    if args.warmup < 1:
        ap.error("--warmup must be >= 1 (the first call is always run untimed)")
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    if args.grad_accum < 1:
        ap.error("--grad-accum must be >= 1")
    if args.seq is not None and args.seq < 1:
        ap.error("--seq must be >= 1")
    if args.chunk_size is not None and args.chunk_size < 1:
        ap.error("--chunk-size must be >= 1")
    return args


def main() -> None:
    args = _parse_args()

    # torch-only imports kept local so the seam stays clean and --help works anywhere.
    try:
        import torch
    except ModuleNotFoundError as e:
        if e.name != "torch":
            raise
        raise SystemExit(
            "torch not found — install the CUDA backend extra on a Linux/CUDA host:\n"
            "    pip install -e '.[cuda]'\n"
            "(torch is omitted from the default deps; mlx is the Apple-Silicon backend.)"
        ) from e
    import numpy as np

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.train.loss_scale import scaler_for_precision

    cfg = load_config(str(args.config))
    if args.chunk_size is not None:
        cfg = dataclasses.replace(cfg, chunk_size=args.chunk_size)
        cfg.validate()
    if cfg.n_moe_layers > 0:
        raise SystemExit(
            "MoE-Mamba (#53) is MLX-only; the CUDA backend can't build it. Bench a dense "
            "config (e.g. config/poc.yaml / config/toy.yaml).")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False.")
    on_cuda = device == "cuda"

    seq = args.seq if args.seq is not None else cfg.seq_len
    tokens_per_step = args.batch * seq * args.grad_accum

    print(f"[bench/cuda] torch {torch.__version__}  device={device}  {platform.platform()}")
    if on_cuda:
        print(f"[bench/cuda] gpu={torch.cuda.get_device_name(0)}")
    print(f"[bench/cuda] config={args.config}  d_model={cfg.d_model}  d_inner={cfg.d_inner}  "
          f"n_layers={cfg.n_layers}  vocab={cfg.vocab_size}  precision={cfg.precision}  "
          f"grad_checkpoint={cfg.grad_checkpoint}  chunk_size={cfg.chunk_size or 64}")
    print(f"[bench/cuda] batch={args.batch} x seq={seq} x grad_accum={args.grad_accum} "
          f"= {tokens_per_step} tokens/step  iters={args.iters} (+{args.warmup} warmup)\n")

    # The exact wiring of scripts/train.py --backend cuda — measures the production path.
    backend = get_backend("cuda")
    backend.seed(args.seed)
    model = backend.model_cls(cfg)  # backend picks cuda:0 / cpu internally (mirrors train.py)
    opt = backend.make_optimizer(model, args.lr)
    scaler = scaler_for_precision(cfg.precision)
    train_step = backend.make_train_step(model, opt, grad_clip=1.0, scaler=scaler)

    rng = np.random.default_rng(args.seed)
    # Mirror PackedLoader exactly: numpy int64 ids; the backend converts inside forward.
    micro_batches = [
        (rng.integers(0, cfg.vocab_size, size=(args.batch, seq)).astype(np.int64),
         rng.integers(0, cfg.vocab_size, size=(args.batch, seq)).astype(np.int64))
        for _ in range(args.grad_accum)
    ]

    def _sync() -> None:
        if on_cuda:
            torch.cuda.synchronize()  # the step queues async work; wait before stopping the clock

    # First call timed separately — it includes lazy CUDA init / first-kernel compile.
    t0 = time.perf_counter()
    out = train_step(model, micro_batches, args.lr)
    _sync()
    first_call_s = time.perf_counter() - t0
    for _ in range(args.warmup - 1):
        out = train_step(model, micro_batches, args.lr)
    _sync()

    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    skipped = 0
    for _ in range(args.iters):
        t0 = time.perf_counter()
        out = train_step(model, micro_batches, args.lr)
        _sync()
        times.append(time.perf_counter() - t0)
        skipped += int(out.get("skipped", False))

    if not math.isfinite(out["loss"]):
        raise RuntimeError("non-finite loss — run is degenerate")

    mean_s = sum(times) / len(times)
    min_s = min(times)
    print(f"first call      {first_call_s:>10.3f} s   (init/compile, excluded from stats)")
    print(f"s/step          {mean_s:>10.3f} mean   {min_s:.3f} min   over {args.iters} iters")
    print(f"tokens/s        {tokens_per_step / mean_s:>10,.0f}   ({tokens_per_step} tokens/step)")
    if on_cuda:
        print(f"peak memory     {torch.cuda.max_memory_allocated() / 2**30:>10.2f} GB")
    else:
        print(f"peak memory     {'n/a':>10}   (cpu dev run — GPU peak only on --device cuda)")
    line = f"loss {out['loss']:.4f}   grad_norm {out['grad_norm']:.4f}"
    if scaler:
        line += f"   loss_scale {out['loss_scale']:.0f}   skipped {skipped}/{args.iters}"
    print(line)


if __name__ == "__main__":
    main()
