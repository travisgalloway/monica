"""Milestone-4 SMOKE TEST — the gate (runs on Apple Silicon).

The single most important test in the project. Most projects silently break at
checkpoint resume and dataloading, not in the model. Do NOT proceed past this gate
until resume is verifiably exact and eval runs.

Procedure (toy model, tiny data, FIXED SEED, fp32 => effectively exact):
  1. Reference run: train N steps uninterrupted; record the loss trajectory.
  2. Interrupted run (same seed, same fixed batch stream, same LR schedule):
     train N/2 steps, SAVE portable weights + a within-backend resume bundle
     (optimizer state + step), tear the model/optimizer down, REBUILD, LOAD, and
     train the rest.
  3. Assert the post-resume trajectory matches the reference within tolerance.
  4. Run a held-out val-perplexity eval end to end.

Determinism note: we drive training over a PRE-MATERIALIZED fixed batch list so
the batch at global step s is identical in both runs (independent of where the
"kill" falls) — the resume exactness check would otherwise be confounded by data
ordering. The portable loop (train.loop) is exercised separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/toy.yaml"))
    ap.add_argument("--data", type=Path, required=True, help="dir with train.bin/val.bin")
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto",
                    help="hardware backend (auto: try mlx, fall back to cuda/torch)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/smoke"))
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--compile", action="store_true",
                    help="also run ONE torch.compile'd training step (CUDA + .[cuda-fast] "
                         "pod) and assert it does not hard-error; the main resume-exactness "
                         "run stays eager. Verifies the Triton graph-break path on hardware.")
    args = ap.parse_args()

    # Backend selection stays behind the seam factory; only portable modules are
    # imported at module scope.
    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.data.loader import PackedLoader
    from src.train.schedule import CosineSchedule
    from src.train.checkpoint import CheckpointStore
    from src.eval.val_loss import evaluate

    backend = get_backend(args.backend)
    cfg = load_config(str(args.config))
    assert cfg.precision == "fp32", "smoke test requires fp32 for exact resume"
    # Pin the CUDA gate to a DENSE config (e.g. config/toy.yaml). MoE-Mamba (#53) is
    # MLX-only, so CUDAMambaModel refuses to build it — fail here with that reason
    # rather than deep inside model construction. (toy-moe.yaml is also fp32, so the
    # precision assert above would not catch it.)
    if backend.name == "cuda":
        assert cfg.n_moe_layers == 0, (
            f"CUDA smoke gate needs a dense config (got {args.config} with "
            f"{cfg.n_moe_layers} MoE layers); MoE is MLX-only. Use config/toy.yaml.")
    N = args.steps
    half = N // 2
    np_to = backend.to_numpy

    # --- fixed batch stream (shuffle off => batch at step s is deterministic) ---
    train_loader = PackedLoader(args.data / "train.bin", cfg.seq_len,
                                args.batch_size, shuffle=False)
    batches = []
    for inp, tgt in train_loader.epoch():
        batches.append((inp, tgt))
        if len(batches) == N:
            break
    assert len(batches) == N, f"need {N} batches, got {len(batches)}"
    sched = CosineSchedule(base_lr=3e-4, warmup_steps=max(1, N // 6), total_steps=N)

    def fresh_model_opt():
        backend.seed(args.seed)                   # identical init weights each run
        model = backend.model_cls(cfg)
        opt = backend.make_optimizer(model, sched.base_lr)
        return model, opt, backend.make_train_step(model, opt, grad_clip=1.0, scaler=None)

    def run_window(model, step_fn, lo, hi, into):
        for s in range(lo, hi):
            inp, tgt = batches[s]
            # New step contract takes a list of micro-batches; one batch == grad_accum 1.
            into[s] = step_fn(model, [(inp, tgt)], sched.lr_at(s))["loss"]

    # --- 1) reference run -------------------------------------------------------
    ref = {}
    model, opt, step_fn = fresh_model_opt()
    run_window(model, step_fn, 0, N, ref)
    print(f"[reference] step0 loss={ref[0]:.5f}  step{N-1} loss={ref[N-1]:.5f}")

    # --- 2) interrupted run: train half, checkpoint, KILL, rebuild, resume ------
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(str(out / "resume"))

    res = {}
    model_a, opt_a, step_fn_a = fresh_model_opt()
    run_window(model_a, step_fn_a, 0, half, res)
    store.save(step=half, loss_scale_state=None,
               weights_serializer=lambda p: model_a.save(p),    # portable weights
               optimizer_serializer=lambda p: backend.save_optimizer(opt_a, p))
    del model_a, opt_a, step_fn_a                               # "kill" the process state

    backend.seed(args.seed + 999)              # different RNG: weights come from disk
    model_b = backend.model_cls(cfg)
    opt_b = backend.make_optimizer(model_b, sched.base_lr)
    meta = store.load(weights_deserializer=lambda p: model_b.load(p),
                      optimizer_deserializer=lambda p: backend.load_optimizer(opt_b, p))
    start = meta["step"]
    step_fn_b = backend.make_train_step(model_b, opt_b, grad_clip=1.0)
    run_window(model_b, step_fn_b, start, N, res)
    print(f"[resumed]   resumed at step={start}  step{N-1} loss={res[N-1]:.5f}")

    # --- 3) trajectory match ----------------------------------------------------
    diffs = [abs(ref[s] - res[s]) for s in range(half, N)]
    max_diff = max(diffs)
    print(f"[match] post-resume max|loss diff| over steps {half}..{N-1} = {max_diff:.3e}")
    if max_diff > args.atol:
        raise SystemExit(
            f"SMOKE TEST FAILED: resume not exact (max|diff|={max_diff:.3e} > {args.atol}).")

    # --- 4) held-out val-perplexity eval ---------------------------------------
    val_loader = PackedLoader(args.data / "val.bin", cfg.seq_len,
                              args.batch_size, shuffle=False, drop_last=False)
    val = evaluate(model_b, val_loader, max_batches=4, to_numpy=np_to)
    print(f"[eval] val_loss={val['val_loss']:.4f}  val_perplexity={val['val_perplexity']:.4f}")

    # --- 5) optional: one torch.compile'd step (#239) ---------------------------
    # Isolated from the resume math above (fresh model, own optimizer/step_fn) so
    # --compile can never perturb the gated eager comparison; this only asserts the
    # compiled Triton graph-break path runs without a hard error on real hardware.
    if args.compile:
        if backend.name != "cuda":
            print("[compile] skipped (only meaningful on the CUDA backend)")
        else:
            import dataclasses
            import math as _math

            ccfg = dataclasses.replace(cfg, torch_compile=True)
            backend.seed(args.seed)
            cmodel = backend.model_cls(ccfg)
            copt = backend.make_optimizer(cmodel, sched.base_lr)
            cstep = backend.make_train_step(cmodel, copt, grad_clip=1.0)
            inp, tgt = batches[0]
            out_c = cstep(cmodel, [(inp, tgt)], sched.lr_at(0))
            assert _math.isfinite(float(out_c["loss"])), \
                f"compiled step loss not finite: {out_c['loss']}"
            print(f"[compile] one compiled step OK  loss={float(out_c['loss']):.5f}")

    print("\nSMOKE TEST PASSED ✅  resume is exact and eval runs.")


if __name__ == "__main__":
    main()
