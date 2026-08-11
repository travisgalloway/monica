"""Scale-run training driver (Apple Silicon / MLX) — the M5 entry point.

Wires the portable training loop (`src.train.loop.train`) to the MLX backend for a
real run on `config/poc.yaml`: model + AdamW + (dynamic fp16) loss scaling + grad
accumulation, JSONL metrics, periodic checkpoints, and held-out val-perplexity eval.
Resume is exact-from-checkpoint (portable weights + within-backend optimizer/step/
loss-scale bundle).

Success for the POC is a smoothly decreasing `val_perplexity` curve in the metrics
file (`<out>/metrics.jsonl`) with a stable `grad_norm` — not a benchmark score.

Data prep (run once, see config/poc.yaml + docs/design/04-data-pipeline.md) produces
`<data>/train.bin` and `<data>/val.bin`. Example scale run:

    .venv/bin/python scripts/train.py --config config/poc.yaml --data data/split \\
        --out runs/poc --total-tokens 3_000_000_000 --batch-size 32 --grad-accum 4

Resume after an interruption:

    .venv/bin/python scripts/train.py --config config/poc.yaml --data data/split \\
        --out runs/poc --total-tokens 3_000_000_000 --batch-size 32 --grad-accum 4 \\
        --resume runs/poc/resume

The backend (MLX or CUDA/PyTorch) is selected via `--backend {auto,mlx,cuda}`; backend
imports stay behind `src.model.backend.get_backend`, so the module stays importable for
`--help` on any host.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--data", type=Path, required=True, help="dir with train.bin/val.bin")
    ap.add_argument("--out", type=Path, default=Path("runs/poc"))
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto",
                    help="hardware backend (auto: try mlx, fall back to cuda/torch)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--total-steps", type=int, help="number of optimizer steps")
    g.add_argument("--total-tokens", type=int,
                   help="target tokens; steps = tokens // (batch*seq*grad_accum)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--base-lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="default: total_steps // 100 (min 1)")
    ap.add_argument("--lr-schedule", choices=("cosine", "wsd"), default="cosine")
    ap.add_argument("--decay-frac", type=float, default=0.2,
                    help="WSD: fraction of total_steps spent decaying")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=50,
                    help="cap val batches per eval (None-like 0 = full val set)")
    ap.add_argument("--moe-diag-every", type=int, default=0,
                    help="#217: steps between MoE routing diagnostic passes (0 = off). "
                         "Only fires on steps that are also --log-every steps, so use a "
                         "multiple of it.")
    ap.add_argument("--moe-diag-domains", type=Path, default=None,
                    help="domains.json from scripts/build_domain_val_sets.py — the "
                         "per-domain val sets the expert histograms are measured over. "
                         "Packed training shards carry no domain tag (#252), so this is "
                         "usage on held-out samples DRAWN FROM each domain, not over the "
                         "live training mix.")
    ap.add_argument("--moe-kill-pair", type=str, default="typescript,math",
                    help="two domain names from domains.json whose expert histograms the "
                         "kill-criterion compares.")
    ap.add_argument("--moe-kill-overlap", type=float, default=0.90,
                    help="routing overlap at or above which the pair is NOT specializing. "
                         "Early in training routing is near-uniform by construction and "
                         "overlap is legitimately ~1.0 — this REPORTS, it never aborts.")
    ap.add_argument("--moe-diag-batches", type=int, default=8,
                    help="cap batches per domain in a diagnostic pass (this is a probe, "
                         "not an eval — keep it small).")
    ap.add_argument("--init-loss-scale", type=float, default=2.0 ** 13)
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume bundle dir; if omitted, auto-detects <out>/resume")
    ap.add_argument("--init", type=Path, default=None,
                    help="portable base weights to initialize from (e.g. a "
                         "sparse-upcycled checkpoint from scripts/upcycle.py) — NOT a "
                         "resume bundle: optimizer/step/loss-scale all start fresh. "
                         "IGNORED (with a printed note) when --resume is active; "
                         "--resume always wins.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curriculum", type=str, default=None,
                    help="length curriculum (#216): 'until_frac:seq_len[:batch_size],...' "
                         "e.g. '0.25:2048,0.5:4096,1.0:16384'. until_frac is CUMULATIVE. "
                         "Omitted batch_size is derived to hold tokens/step ~constant "
                         "against --batch-size x config seq_len. Default: no curriculum "
                         "(fixed shape, exactly the pre-#216 behavior).")
    ap.add_argument("--ignore-data-state", action="store_true",
                    help="discard a checkpoint's saved dataloader position and "
                         "reconstruct it from the step instead. ESCAPE HATCH only — it "
                         "is how you resume after deliberately changing the curriculum "
                         "or the token budget, and it will resample data.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # Backend selection stays behind the seam factory; only portable modules are
    # imported at module scope.
    import numpy as np

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.data.loader import PackedLoader
    from src.train.loss_scale import scaler_for_precision
    from src.train.moe_balance import attach_balancer, balancer_for_config
    from src.train.curriculum import build_curriculum
    from src.train.loop import TrainConfig, train
    from src.train.logging import JsonlLogger
    from src.train.checkpoint import CheckpointStore, check_weight_keys, load_weights_dict
    from src.eval.val_loss import evaluate

    backend = get_backend(args.backend)

    cfg = load_config(str(args.config))                 # validates (vocab < 2**32, head_dim | d_inner, ...)

    # #217: fail fast, before the model is built where possible — a bad routing-diagnostic
    # flag should not surface 200 steps into a multi-day run.
    if args.moe_diag_every and args.moe_diag_domains is None:
        raise SystemExit("--moe-diag-every > 0 requires --moe-diag-domains "
                         "(domains.json from scripts/build_domain_val_sets.py)")
    moe_kill_pair = tuple(s.strip() for s in args.moe_kill_pair.split(","))
    if len(moe_kill_pair) != 2:
        raise SystemExit(f"--moe-kill-pair must be exactly two comma-separated domain "
                         f"names, got {args.moe_kill_pair!r}")
    if args.moe_diag_every and cfg.n_moe_layers == 0:
        raise SystemExit(f"[moe-diag] {args.config} has no MoE layers (moe_every is "
                         "unset) — there is no router to diagnose.")
    if args.moe_diag_every and args.moe_diag_every % args.log_every != 0:
        print(f"[moe-diag] WARNING: --moe-diag-every={args.moe_diag_every} does not "
              f"divide --log-every={args.log_every} — the diagnostic pass will silently "
              "never fire (it only runs on steps that are also log_every steps).")

    tokens_per_step = args.batch_size * cfg.seq_len * args.grad_accum
    # Length curriculum (#216): the stage allocation, not the fixed shape, decides how
    # many optimizer steps the run has — tokens/step varies per stage, so the old
    # `tokens // (batch*seq*grad_accum)` division no longer describes the run.
    curriculum = None
    if args.curriculum:
        curriculum = build_curriculum(
            args.curriculum, base_seq_len=cfg.seq_len, base_batch_size=args.batch_size,
            grad_accum=args.grad_accum, total_tokens=args.total_tokens,
            total_steps=args.total_steps)
        total_steps = curriculum.total_steps
    else:
        total_steps = (args.total_steps if args.total_steps is not None
                       else max(1, args.total_tokens // tokens_per_step))
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, total_steps // 100)

    # --- model + optimizer + (dynamic) loss scaling ----------------------------
    backend.seed(args.seed)
    model = backend.model_cls(cfg)
    opt = backend.make_optimizer(model, args.base_lr)
    scaler = scaler_for_precision(cfg.precision, args.init_loss_scale)
    if scaler is None and cfg.precision != "fp32":
        print(f"[info] precision={cfg.precision!r}: training unscaled "
              "(loss scaling is fp16-only; expected for bf16)")
    # Loss-Free-Balancing (#213): None for dense/hybrid configs and for MoE configs with
    # moe_balance_rate unset (the default) — balancer_for_config is the single source of
    # truth for the off-switch. Both backends' make_train_step accept `balancer=` (#214);
    # None is a no-op on either.
    balancer = balancer_for_config(cfg)
    train_step = backend.make_train_step(model, opt, grad_clip=args.grad_clip, scaler=scaler,
                                         balancer=balancer)

    # --- data ------------------------------------------------------------------
    # ONE construction path for every stage, including stage 0 (`open_packed` returns a
    # read-only memmap, so a loader per stage costs no data copy).
    train_path = args.data / "train.bin"

    def loader_factory(seq_len: int, batch_size: int) -> PackedLoader:
        return PackedLoader(train_path, seq_len, batch_size, shuffle=True, seed=args.seed)

    if curriculum is not None:
        # Fail fast at startup, not deep inside the stream at a stage boundary 10 days in:
        # at long seq_len a small corpus can drop below one batch per epoch.
        for st in curriculum.stages:
            try:
                probe = loader_factory(st.seq_len, st.batch_size)
                n_chunks, n_tokens = probe.n_chunks, probe.n_tokens
                ok = len(probe) >= 1
            except ValueError:                  # too small for even one chunk
                n_chunks, n_tokens, ok = 0, None, False
            if not ok:
                raise SystemExit(
                    f"[curriculum] stage {st.index} (seq_len={st.seq_len}, "
                    f"batch_size={st.batch_size}) yields no batches per epoch: "
                    f"{n_tokens} tokens in {train_path} give {n_chunks} chunks, but a "
                    f"batch needs {st.batch_size}. This stage needs >= "
                    f"{st.batch_size * (st.seq_len + 1)} tokens.")
    train_loader = loader_factory(cfg.seq_len if curriculum is None
                                  else curriculum.stages[0].seq_len,
                                  args.batch_size if curriculum is None
                                  else curriculum.stages[0].batch_size)
    # The val loader stays PINNED at cfg.seq_len — deliberately NOT ramped. A val loss
    # measured at a changing context length is not comparable across a run, and would make
    # the POC's primary success signal (a smoothly improving curve / BPB) unreadable.
    val_loader = PackedLoader(args.data / "val.bin", cfg.seq_len,
                              args.batch_size, shuffle=False, drop_last=False)
    np_to = backend.to_numpy
    max_b = args.eval_batches or None
    val_eval = lambda m: evaluate(m, val_loader, max_batches=max_b, to_numpy=np_to)

    moe_diag = None
    if args.moe_diag_every:
        from src.eval.moe_routing import (expert_histograms, format_routing_report,
                                          kill_check, specialization_report)
        from src.eval.domain_bpb import load_domain_index
        diag_index = load_domain_index(args.moe_diag_domains)      # raises early if bad
        diag_domains = {n: e["packed"] for n, e in diag_index.items()}

        def moe_diag(m):
            hists = expert_histograms(m, diag_domains, batch_size=args.batch_size,
                                      seq_len=cfg.seq_len,
                                      max_batches=args.moe_diag_batches, to_numpy=np_to)
            report = specialization_report(hists)
            kill = kill_check(report, pair=moe_kill_pair, threshold=args.moe_kill_overlap)
            if kill["triggered"]:
                # Loud, and a machine-readable event line of its own — it does NOT abort
                # the run (that is the user's call; see --moe-kill-overlap's help). No
                # "step" key: the loop owns the step number and the payload this function
                # returns lands on the logged step anyway — this event line is the loud
                # one, the flat moe_kill_* fields below are the durable record. `logger`
                # is defined further down in main(); this closure only runs once training
                # starts, by which point it is bound in the enclosing scope.
                print(f"[moe-kill] {kill['pair']} routing overlap "
                      f"{kill['overlap']:.4f} >= {args.moe_kill_overlap:.4f}: the router "
                      f"is NOT specializing between these domains. Reporting only — the "
                      f"run continues. Early in training this is expected.")
                print(format_routing_report(report, kill))
                logger({"event": "moe_kill", "moe_kill_pair": kill["pair"],
                       "moe_kill_overlap": kill["overlap"],
                       "moe_kill_threshold": kill["threshold"],
                       "moe_kill_triggered": kill["triggered"]})
            return {"moe_domain_overlap": report["mean_overlap"],
                    "moe_domain_overlap_max": report["max_overlap"],
                    "moe_kill_pair": kill["pair"],
                    "moe_kill_overlap": kill["overlap"],
                    "moe_kill_triggered": kill["triggered"],
                    "moe_expert_hist": hists}

    # --- resume (double-buffered, crash-safe checkpoint store) -----------------
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    weights_path = str(out / "weights.safetensors")
    store = CheckpointStore(str(args.resume) if args.resume is not None
                            else str(out / "resume"))
    resuming = store.has_checkpoint()

    start_step = 0
    start_data_state = None
    if resuming:
        meta = store.load(weights_deserializer=lambda p: model.load(p),
                          optimizer_deserializer=lambda p: backend.load_optimizer(opt, p))
        start_step = int(meta["step"])
        if scaler is not None:
            scaler.load_state_dict(meta.get("loss_scale_state") or {})
        # #216: pre-#216 bundles have no `data_state` and fall back to reconstructing the
        # position from the step — correct only at a fixed shape, hence the printed note.
        start_data_state = meta.get("data_state")
        if start_data_state is None:
            data_state_note = "absent (legacy reconstruct from step)"
        elif args.ignore_data_state:
            data_state_note = "ignored (--ignore-data-state; will resample)"
        else:
            data_state_note = "present"
        print(f"[resume] from step {start_step} slot={meta['slot']} (out={out})  "
              f"data_state={data_state_note}")
        if args.init is not None:
            # --resume always wins (a resume bundle is the authoritative, in-progress
            # state of THIS run); printed, not silently dropped, so a stale --init left
            # on the command line after the first successful resume doesn't read as
            # "it did something."
            print(f"[init] IGNORED — --resume is active (from step {start_step}); "
                  f"--init={args.init} has no effect. --resume always wins over --init.")
    elif args.init is not None:
        init_weights = load_weights_dict(str(args.init))
        check_weight_keys(init_weights, model._portable_state_dict(),
                          where=f"--init {args.init}")
        model._load_portable(init_weights)
        print(f"[init] from {args.init}")
    # Loss-Free-Balancing (#213): seed the policy from the just-loaded weights (the bias
    # rides in the portable safetensors, D3), push it into the routers, and enable load
    # counting. No-op when balancing is off. `CheckpointStore.save` is unchanged — there
    # is deliberately no second copy of the bias in the resume bundle.
    attach_balancer(balancer, model)
    # #217: routing diagnostics are DECOUPLED from balancing. `attach_balancer` turns
    # counting on only when a balancer exists (moe_balance.py:171), and moe_balance_rate is
    # null in every config in the tree — so without this the diagnostics would observe
    # nothing and report a uniform, blind zero.
    if args.moe_diag_every:
        model.set_moe_load_counting(True)

    logger = JsonlLogger(str(out / "metrics.jsonl"), append=resuming)

    def on_checkpoint(step: int, data_state=None) -> None:
        # `data_state` rides INSIDE resume_meta.json, inside the slot, and is therefore
        # committed by the same atomic LATEST flip as the weights and the step (#216).
        store.save(step=step,
                   loss_scale_state=(scaler.state_dict() if scaler else None),
                   weights_serializer=lambda p: model.save(p),  # portable weights + config
                   optimizer_serializer=lambda p: backend.save_optimizer(opt, p),
                   data_state=data_state)

    tcfg = TrainConfig(
        total_steps=total_steps, base_lr=args.base_lr, warmup_steps=warmup,
        grad_accum=args.grad_accum, grad_clip=args.grad_clip,
        log_every=args.log_every, eval_every=args.eval_every,
        ckpt_every=args.ckpt_every, out_dir=str(out), seed=args.seed,
        lr_schedule=args.lr_schedule, decay_frac=args.decay_frac,
        moe_diag_every=args.moe_diag_every,
    )

    n_params = sum(int(np.asarray(v).size) for _, v in model._portable_state_dict().items())
    print(f"[run] params~{n_params/1e6:.1f}M  total_steps={total_steps}  warmup={warmup}  "
          f"tokens/step={tokens_per_step}  precision={cfg.precision}  "
          f"schedule={args.lr_schedule}")

    if curriculum is not None:
        print(f"[curriculum] {args.curriculum!r} -> {len(curriculum.stages)} stages, "
              f"total_steps={curriculum.total_steps} (authoritative; each stage gets at "
              f"least 1 step, so this can differ slightly from a requested --total-steps)")
        for line in curriculum.describe():
            print(f"[curriculum] {line}")
        print("[curriculum] val loss stays measured at the FIXED config seq_len "
              f"({cfg.seq_len}) so the curve remains comparable across stages.")
        # Sizing the recompile cost explicitly is a requirement of #216 — leaving it
        # implicit is how a bounded, expected stall gets misread as a hang. Note this
        # does NOT change the compile default; that is #239's decision.
        if backend.name == "cuda" and getattr(cfg, "torch_compile", False) is not False:
            print(f"[curriculum] CUDA torch.compile is ON: expect ONE inductor recompile "
                  f"at each of the {len(curriculum.stages) - 1} stage boundaries (tens of "
                  "seconds each, amortized over the stage; inductor promotes to dynamic "
                  "shapes after the 2nd distinct (B, L), so later boundaries are usually "
                  "free). Not a regression — each is preceded by an 'event: stage' log "
                  "line in metrics.jsonl.")

    result = train(model, train_loader, tcfg, train_step,
                   val_eval=val_eval, moe_diag=moe_diag, logger=logger,
                   on_checkpoint=on_checkpoint,
                   start_step=start_step, curriculum=curriculum,
                   loader_factory=loader_factory if curriculum is not None else None,
                   start_data_state=start_data_state,
                   ignore_data_state=args.ignore_data_state)

    # --- final checkpoint + full eval ------------------------------------------
    # The loop already checkpoints at `step == total_steps` when total_steps is a
    # multiple of ckpt_every; guard so we don't write (and re-serialize) it twice.
    if total_steps % tcfg.ckpt_every != 0:
        on_checkpoint(total_steps, result["data_state"])
    # Materialize the portable weights at the canonical out/weights.safetensors for
    # downstream eval/serving/distillation (the live copy otherwise lives in the slot).
    model.save(weights_path)
    final = evaluate(model, val_loader, max_batches=max_b, to_numpy=np_to)
    logger.close()
    print(f"[done] step={total_steps}  tokens={result['tokens_seen']}  "
          f"val_loss={final['val_loss']:.4f}  "
          f"val_perplexity={final['val_perplexity']:.4f}  metrics={out / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
