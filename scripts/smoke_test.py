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
    ap.add_argument("--curriculum", type=str, default=None,
                    help="phase 7 length curriculum (#216), "
                         "'until_frac:seq_len[:batch_size],...'. Default: a two-stage "
                         "ramp derived from the config (seq_len//2 -> seq_len).")
    ap.add_argument("--curriculum-steps", type=int, default=20,
                    help="phase 7: optimizer steps for the curriculum kill/resume gate")
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

    # --- 6) prefill + a few decode steps (#165) ---------------------------------
    # Isolated from the resume math above the same way --compile is: this drives model_b
    # only through prefill/step/init_state, never the optimizer, so the gated comparison
    # cannot be perturbed. Unconditional (no flag, no hardware need) — it is pure fp32 toy
    # math on both backends, and a silent prefill/decode divergence is exactly the kind of
    # bug this gate exists to catch: the prompt's own logits stay right while every token
    # decoded afterwards is subtly wrong.
    import contextlib

    from src.conformance.prefill_decode_parity import check_prefill_decode_parity

    guard = contextlib.nullcontext()
    if backend.name == "cuda":
        import torch
        guard = torch.no_grad()

    n_decode = 4
    prompt_len = 16
    with guard:
        prompt = batches[0][0][:2, :prompt_len]           # (2, 16) inputs from the stream
        par = check_prefill_decode_parity(model_b, prompt, to_numpy=np_to,
                                          rtol=1e-4, atol=1e-5, n_decode=n_decode)
        if not par["ok"]:
            raise SystemExit(f"SMOKE TEST FAILED: prefill/decode parity broken: {par}")

        # ...and end to end: prefill a prompt, then decode greedily off its state.
        logits, state = model_b.prefill(prompt, last_only=True)
        rows = [np_to(logits)]
        for _ in range(n_decode):
            nxt = rows[-1].argmax(axis=-1)                # (B,) greedy feedback
            logits, state = model_b.step(nxt, state)
            rows.append(np_to(logits))
    import numpy as _np
    if not all(_np.isfinite(r).all() for r in rows):
        raise SystemExit("SMOKE TEST FAILED: prefill/decode produced non-finite logits.")
    print(f"[prefill] L={prompt.shape[1]} + {n_decode} decode steps OK  "
          f"max|logit diff|={par['max_abs_diff']:.3e}  "
          f"max|state diff|={par['max_abs_state_diff']:.3e}")

    # --- 7) curriculum kill/resume gate (#216) ----------------------------------
    # Phases 1-6 drive `step_fn` directly over a PRE-MATERIALIZED batch list, which is
    # exactly what makes them insensitive to data ordering — and therefore blind to the
    # thing #216 changes. So this phase drives the REAL `train()` + `MicroBatchStream` +
    # `CheckpointStore` instead: the wiring between them is the new surface, and a
    # curriculum that silently failed to engage, or a resume that silently resampled,
    # would leave every other phase green.
    from src.train.curriculum import build_curriculum
    from src.train.loop import TrainConfig, train

    spec = args.curriculum or (f"0.5:{max(1, cfg.seq_len // 2)}:{2 * args.batch_size},"
                               f"1.0:{cfg.seq_len}:{args.batch_size}")
    curriculum = build_curriculum(spec, base_seq_len=cfg.seq_len,
                                  base_batch_size=args.batch_size, grad_accum=1,
                                  total_steps=args.curriculum_steps)
    cN = curriculum.total_steps
    boundary = curriculum.first_steps[1] if len(curriculum.stages) > 1 else cN // 2
    # Kill strictly AFTER the boundary so the resume starts INSIDE the later stage, with
    # a partial epoch — the case a boundary-aligned kill would not exercise.
    cK = min(cN - 1, boundary + 2)
    if cK <= 0 or cK >= cN:
        raise SystemExit(f"SMOKE TEST FAILED: phase 7 needs 0 < kill({cK}) < steps({cN}); "
                         "raise --curriculum-steps.")

    def curric_factory(seq_len, batch_size):
        return PackedLoader(args.data / "train.bin", seq_len, batch_size,
                            shuffle=True, seed=args.seed)

    ctcfg = TrainConfig(total_steps=cN, base_lr=3e-4, warmup_steps=max(1, cN // 6),
                        grad_accum=1, grad_clip=1.0, log_every=1, eval_every=10 ** 9,
                        ckpt_every=cK, out_dir=str(out), seed=args.seed)

    def curric_run(model, step_fn, logs, **kw):
        return train(model, None, ctcfg, step_fn, logger=logs.append,
                     curriculum=curriculum, loader_factory=curric_factory, **kw)

    # 7a) reference: one uninterrupted curriculum run.
    ref_logs = []
    cmodel, _, cstep = fresh_model_opt()
    curric_run(cmodel, cstep, ref_logs)
    ref_steps = [p for p in ref_logs if "event" not in p]

    # 7b) interrupted: kill AT the checkpoint (as a preempted pod does — the budget is
    # unchanged), tear down model/optimizer/stream, rebuild from the store, finish.
    class _Killed(Exception):
        pass

    cstore = CheckpointStore(str(out / "resume-curriculum"))
    saved = {}
    res_logs = []
    cmodel_a, copt_a, cstep_a = fresh_model_opt()

    def curric_ckpt(step, data_state):
        cstore.save(step=step, loss_scale_state=None, data_state=data_state,
                    weights_serializer=lambda p: cmodel_a.save(p),
                    optimizer_serializer=lambda p: backend.save_optimizer(copt_a, p))
        saved["step"] = step
        raise _Killed

    try:
        curric_run(cmodel_a, cstep_a, res_logs, on_checkpoint=curric_ckpt)
    except _Killed:
        pass
    if saved.get("step") != cK:
        raise SystemExit(f"SMOKE TEST FAILED: phase 7 checkpoint fired at "
                         f"{saved.get('step')}, expected {cK}.")
    del cmodel_a, copt_a, cstep_a                    # "kill" the process state

    backend.seed(args.seed + 999)                    # different RNG: weights from disk
    cmodel_b = backend.model_cls(cfg)
    copt_b = backend.make_optimizer(cmodel_b, 3e-4)
    cmeta = cstore.load(weights_deserializer=lambda p: cmodel_b.load(p),
                        optimizer_deserializer=lambda p: backend.load_optimizer(copt_b, p))
    if cmeta.get("data_state") is None:
        raise SystemExit("SMOKE TEST FAILED: phase 7 resume bundle carries no "
                         "data_state — the dataloader position was not committed.")
    curric_run(cmodel_b, backend.make_train_step(cmodel_b, copt_b, grad_clip=1.0),
               res_logs, start_step=int(cmeta["step"]),
               start_data_state=cmeta["data_state"])
    res_steps = [p for p in res_logs if "event" not in p]

    # 7c) the trajectory over the resumed window must match the reference. A resume that
    # resampled data would diverge here even though every counter looked healthy.
    ref_loss = {p["step"]: float(p["loss"]) for p in ref_steps}
    res_loss = {p["step"]: float(p["loss"]) for p in res_steps}
    missing = [s for s in range(cN) if s not in ref_loss or s not in res_loss]
    if missing:
        raise SystemExit(f"SMOKE TEST FAILED: phase 7 missing steps {missing[:5]}...")
    cdiffs = [abs(ref_loss[s] - res_loss[s]) for s in range(cK, cN)]
    cmax = max(cdiffs)
    print(f"[curriculum] {spec}  steps={cN}  boundary={boundary}  kill={cK}  "
          f"post-resume max|loss diff| over steps {cK}..{cN-1} = {cmax:.3e}")
    if cmax > args.atol:
        raise SystemExit(f"SMOKE TEST FAILED: curriculum resume not exact "
                         f"(max|diff|={cmax:.3e} > {args.atol}).")

    # 7d) ...and the curriculum actually ENGAGED. Without this a curriculum that silently
    # collapsed to one shape would sail through 7c, since a fixed-shape run resumes fine.
    for label, steps in (("reference", ref_steps), ("resumed", res_steps)):
        seq_lens = [p["seq_len"] for p in sorted(steps, key=lambda p: p["step"])]
        want = [curriculum.stage_at(s).seq_len for s in range(cN)]
        if seq_lens != want:
            raise SystemExit(f"SMOKE TEST FAILED: phase 7 {label} seq_len sequence "
                             f"{seq_lens} != expected {want} — the curriculum did not "
                             "engage.")
        if len(set(seq_lens)) < 2:
            raise SystemExit(f"SMOKE TEST FAILED: phase 7 {label} never changed seq_len; "
                             "the gate would be vacuous.")

    # 7e) ...and token accounting survived the kill (it is cumulative across stages, and
    # the resumed run must pick it up rather than restart from zero).
    want_tokens = sum(curriculum.tokens_at(s) for s in range(cN))
    got_tokens = max(p["tokens"] for p in res_steps)
    if got_tokens != want_tokens:
        raise SystemExit(f"SMOKE TEST FAILED: phase 7 cumulative tokens {got_tokens} != "
                         f"{want_tokens}.")
    print(f"[curriculum] stages engaged in both runs; tokens={got_tokens} "
          f"(cumulative across {len(curriculum.stages)} stages) OK")

    print("\nSMOKE TEST PASSED ✅  resume is exact and eval runs.")


if __name__ == "__main__":
    main()
