"""Distillation driver (#81) — train a Mamba-2 hybrid student from a frozen teacher.

The end-to-end M10 distill run for ONE manifest (a sweep iterates this over sibling manifests):
load the manifest, build the frozen conversion teacher (#93) and the student (its layout resolved
by `manifest_to_config`), initialize the student from the teacher (#99), then loop the manifest's
distillation stages (`mixing-match -> hidden-align -> logit-distill`, #100), swapping the
objective per stage on the shared backend-free training loop. Each stage gets a fresh optimizer
and its own crash-safe checkpoint area + metrics under `<out>/<stage>/`; the student weights
persist across stages (it is one model). Resume restarts the furthest-progressed stage exactly.

The `logit-distill` stage streams the cached teacher top-k (#94, `--teacher-outputs`) with zero
teacher inference; `hidden-align`/`mixing-match` recompute the teacher on the fly. Backend imports
stay behind `src.model.backend.get_backend`, so `--help` works on any host.

    # cloud distill run (CUDA), single manifest:
    .venv/bin/python scripts/distill.py --manifest config/manifests/student-1b-attn-hi.yaml \\
        --corpus data/poc-distill/split --teacher-outputs data/poc-distill/teacher-outputs/topk-logits \\
        --out runs/distill-hi --backend cuda --steps-per-stage 2000 --batch-size 8 --grad-accum 4

    # offline toy gate (byte vocab, synthetic teacher, fp32): see scripts/distill_smoke.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="student trial manifest")
    ap.add_argument("--corpus", type=Path, required=True, help="dir with train.bin/val.bin")
    ap.add_argument("--teacher-outputs", type=Path, default=None,
                    help="cached teacher top-k dir (required for the logit-distill stage)")
    ap.add_argument("--out", type=Path, default=Path("runs/distill"))
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto")
    ap.add_argument("--steps-per-stage", type=int, default=1000,
                    help="optimizer steps per distill stage (per-stage flags override)")
    ap.add_argument("--mixing-steps", type=int, default=None)
    ap.add_argument("--hidden-steps", type=int, default=None)
    ap.add_argument("--logit-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--base-lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=None, help="default: steps//100 (min 1)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=None, help="logit-distill: top-k to use (<= cached k)")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--ce-weight", type=float, default=0.1)
    ap.add_argument("--kl-weight", type=float, default=0.9)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=50)
    ap.add_argument("--init-loss-scale", type=float, default=2.0 ** 13)
    ap.add_argument("--resume", action="store_true", help="resume the furthest-progressed stage")
    ap.add_argument("--seed", type=int, default=0)
    # Toy/offline overrides (the smoke gate): manifest_to_config forces qwen25 vocab + bf16.
    ap.add_argument("--synthetic", action="store_true",
                    help="build a synthetic toy teacher (TeacherConfig.tiny) — offline tests only")
    ap.add_argument("--vocab", type=int, default=None, help="override student/teacher vocab (toy)")
    ap.add_argument("--precision", default=None, help="override precision, e.g. fp32 (toy)")
    return ap.parse_args()


def _teacher_config_for(model_id: str):
    """Known teacher repo id -> TeacherConfig (so logits slice to the tokenizer vocab)."""
    from src.model.teacher import TeacherConfig
    known = {"Qwen/Qwen3-4B-Thinking-2507": TeacherConfig.qwen3_4b_thinking,
             "open-r1/OpenR1-Distill-7B": TeacherConfig.openr1_distill_7b,
             "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": TeacherConfig.qwen_1_5b}
    return known[model_id]() if model_id in known else None


class _InputsLoader:
    """Wrap a `PackedLoader` to yield 1-tuples `(inputs,)` for stages whose teacher is recomputed
    on the fly (hidden-align / mixing-match). Mirrors the loader duck-type the loop needs
    (`batch_size`, `seq_len`, `__len__`, `epoch(reseed, skip_batches)`)."""

    def __init__(self, packed):
        self._p = packed
        self.batch_size = packed.batch_size
        self.seq_len = packed.seq_len

    def __len__(self):
        return len(self._p)

    def epoch(self, reseed=None, skip_batches=0):
        for inputs, _targets in self._p.epoch(reseed=reseed, skip_batches=skip_batches):
            yield (inputs,)


def main() -> None:
    args = _parse_args()

    import numpy as np

    from src.model.backend import get_backend
    from src.model.teacher import TeacherConfig
    from src.data.loader import PackedLoader
    from src.data.teacher_outputs import DistillLoader, read_teacher_meta
    from src.train.distill_manifest import (DistillStage, distill_stages, load_manifest,
                                            manifest_to_config)
    from src.train.loss_scale import scaler_for_precision
    from src.train.loop import TrainConfig, train
    from src.train.logging import JsonlLogger
    from src.train.checkpoint import CheckpointStore
    from src.eval.val_loss import evaluate

    backend = get_backend(args.backend)
    manifest = load_manifest(args.manifest)
    cfg = manifest_to_config(manifest)
    if args.vocab is not None:
        cfg.vocab_size = args.vocab
    if args.precision is not None:
        cfg.precision = args.precision
    cfg.validate()

    stages = distill_stages(manifest)
    if not stages:
        raise SystemExit(f"manifest {args.manifest} has no distillation stages")
    if DistillStage.LOGIT_DISTILL in stages and args.teacher_outputs is None:
        raise SystemExit("--teacher-outputs is required for the logit-distill stage")

    # --- teacher + student + init ---------------------------------------------
    backend.seed(args.seed)
    if args.synthetic or manifest.conversion_teacher == "synthetic":
        tcfg = TeacherConfig.tiny()
        if args.vocab is not None:
            tcfg.vocab_size = args.vocab
        teacher = backend.make_teacher(config=tcfg)
    else:
        mid = manifest.conversion_teacher
        teacher = backend.make_teacher(config=_teacher_config_for(mid), pretrained=mid)

    student = backend.model_cls(cfg)
    report = backend.init_student(student, teacher, manifest.init)
    print(f"[init] {report.method}: mapped {report.n_layers_mapped} layers, "
          f"frozen={report.frozen_layers}, trainable~{report.n_trainable_params/1e6:.1f}M, "
          f"frozen~{report.n_frozen_params/1e6:.1f}M")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    np_to = backend.to_numpy
    max_b = args.eval_batches or None
    val_loader = PackedLoader(args.corpus / "val.bin", cfg.seq_len, args.batch_size,
                              shuffle=False, drop_last=False)
    val_eval = lambda m: evaluate(m, val_loader, max_batches=max_b, to_numpy=np_to)

    _stage_steps = {DistillStage.MIXING_MATCH: args.mixing_steps,
                    DistillStage.HIDDEN_ALIGN: args.hidden_steps,
                    DistillStage.LOGIT_DISTILL: args.logit_steps}

    # --- resume detection: restart the furthest-progressed stage --------------
    resume_idx = 0
    if args.resume:
        for idx in range(len(stages) - 1, -1, -1):
            if CheckpointStore(str(out / stages[idx].value / "resume")).has_checkpoint():
                resume_idx = idx
                break

    for idx, stage in enumerate(stages):
        if idx < resume_idx:
            continue   # baked into the checkpoint we load for `resume_idx`

        steps = _stage_steps[stage] if _stage_steps[stage] is not None else args.steps_per_stage
        warmup = args.warmup_steps if args.warmup_steps is not None else max(1, steps // 100)
        stage_out = out / stage.value
        stage_out.mkdir(parents=True, exist_ok=True)

        opt = backend.make_optimizer(student, args.base_lr)      # fresh optimizer per stage
        scaler = scaler_for_precision(cfg.precision, args.init_loss_scale)
        teacher_arg = None if stage == DistillStage.LOGIT_DISTILL else teacher
        train_step = backend.make_distill_train_step(
            student, opt, stage=stage, teacher=teacher_arg, ce_weight=args.ce_weight,
            kl_weight=args.kl_weight, temperature=args.temperature,
            grad_clip=args.grad_clip, scaler=scaler)

        if stage == DistillStage.LOGIT_DISTILL:
            # Fail fast on a vocab mismatch: if the teacher-outputs were cached over a wider vocab
            # than the student's (e.g. an unknown conversion_teacher so the precompute used the
            # padded model vocab), the cached top-k indices can exceed cfg.vocab_size and the KL
            # gather would later fail with a cryptic index error. Passing vocab_size also makes the
            # loader's token guard catch out-of-range ids.
            tmeta = read_teacher_meta(args.teacher_outputs, "train")
            if tmeta["vocab_size"] > cfg.vocab_size:
                raise SystemExit(
                    f"teacher-outputs vocab_size {tmeta['vocab_size']} > student vocab "
                    f"{cfg.vocab_size}: cached top-k indices can exceed the student vocab. Re-run "
                    f"precompute_teacher with a teacher whose effective_vocab_size matches the "
                    f"student tokenizer (the #94 precompute guard enforces this for real runs).")
            loader = DistillLoader(args.corpus / "train.bin", args.teacher_outputs, "train",
                                   cfg.seq_len, args.batch_size, k=args.k, shuffle=True,
                                   seed=args.seed, vocab_size=cfg.vocab_size)
        else:
            loader = _InputsLoader(PackedLoader(args.corpus / "train.bin", cfg.seq_len,
                                                args.batch_size, shuffle=True, seed=args.seed))

        store = CheckpointStore(str(stage_out / "resume"))
        start_step = 0
        resuming = (idx == resume_idx and args.resume and store.has_checkpoint())
        if resuming:
            meta = store.load(weights_deserializer=lambda p: student.load(p),
                              optimizer_deserializer=lambda p: backend.load_optimizer(opt, p))
            start_step = int(meta["step"])
            if scaler is not None:
                scaler.load_state_dict(meta.get("loss_scale_state") or {})
            print(f"[resume] stage={stage.value} from step {start_step} slot={meta['slot']}")

        logger = JsonlLogger(str(stage_out / "metrics.jsonl"), append=resuming)

        def on_checkpoint(step: int, _store=store, _opt=opt, _scaler=scaler) -> None:
            _store.save(step=step,
                        loss_scale_state=(_scaler.state_dict() if _scaler else None),
                        weights_serializer=lambda p: student.save(p),
                        optimizer_serializer=lambda p: backend.save_optimizer(_opt, p))

        tcfg = TrainConfig(total_steps=steps, base_lr=args.base_lr, warmup_steps=warmup,
                           grad_accum=args.grad_accum, grad_clip=args.grad_clip,
                           log_every=args.log_every, eval_every=args.eval_every,
                           ckpt_every=args.ckpt_every, out_dir=str(stage_out), seed=args.seed)
        print(f"[stage] {stage.value}: {steps} steps (warmup {warmup})")
        train(student, loader, tcfg, train_step, val_eval=val_eval, logger=logger,
              on_checkpoint=on_checkpoint, start_step=start_step)
        if steps % tcfg.ckpt_every != 0:
            on_checkpoint(steps)
        logger.close()

    # --- final portable weights + eval ----------------------------------------
    weights_path = str(out / "weights.safetensors")
    student.save(weights_path)
    final = evaluate(student, val_loader, max_batches=max_b, to_numpy=np_to)
    print(f"[done] stages={[s.value for s in stages]}  val_loss={final['val_loss']:.4f}  "
          f"val_perplexity={final['val_perplexity']:.4f}  weights={weights_path}")


if __name__ == "__main__":
    main()
