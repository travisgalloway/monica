"""SFT driver (M9, Apple Silicon / MLX): instruction-tune the pretrained base.

Loads the pretrained POC weights as initialization (NOT a resume bundle — fresh AdamW),
then trains on response-masked instruction data (`HuggingFaceH4/no_robots`) with the
shared training loop: masked cross-entropy (`make_sft_train_step`), grad accumulation,
dynamic fp16 loss scaling, JSONL metrics, periodic checkpoints, and held-out masked
val-perplexity. Success is a falling masked `val_perplexity` and cleaner on-format chat
replies — not a benchmark score (100M ceiling).

Prep the data once (writes JSONL of response-masked records):

    .venv/bin/python -m src.data.sft_data --split train --out data/sft/train.jsonl
    .venv/bin/python -m src.data.sft_data --split test  --out data/sft/val.jsonl

Then fine-tune (init from the pretrained base):

    .venv/bin/python scripts/sft.py --config config/poc.yaml --data data/sft \\
        --init runs/poc/weights.safetensors --out runs/sft \\
        --epochs 2 --batch-size 8 --grad-accum 16 --base-lr 2e-5

Resume after an interruption (auto-detects <out>/resume):

    .venv/bin/python scripts/sft.py --config config/poc.yaml --data data/sft \\
        --out runs/sft --epochs 2 --batch-size 8 --grad-accum 16 --base-lr 2e-5

The `shared/sft/` corpora (#306)
--------------------------------
`--data` also takes a `shared/sft/tokenized/<tok>-<k>/` directory written by the #95/#96/#102
builders; `--corpus-form` picks which of its files to train, and the train/val split is made here
(seeded by `--seed`, sized by `--val-frac`) because the builders write one file per form:

    python -m src.data.instruct_sft  --sources handauthored --out-root data/shared
    python -m src.data.reasoning_sft --sources handauthored --out-root data/shared
    python -m src.data.tool_sft      --sources handauthored --out-root data/shared

    # instruct (#95) / reasoning traces (#96) / tool calls (#102) — one form:
    .venv/bin/python scripts/sft.py --config config/poc.yaml \\
        --data data/shared/sft/tokenized/qwen3-8k --corpus-form reasoning \\
        --init runs/poc/weights.safetensors --out runs/sft-reasoning

    # several forms in one run (rejected unless their manifests agree on
    # chat_eos/template/tokenizer/model_id/seq_len):
    .venv/bin/python scripts/sft.py --config config/poc.yaml \\
        --data data/shared/sft/tokenized/qwen3-8k --corpus-form instruct reasoning tool \\
        --init runs/poc/weights.safetensors --out runs/sft-mixed

`reasoning-packed/` is NOT trainable here — it is `shard.pack_atomic` output with no loss mask, so
it is a pretraining artifact and takes the pretraining driver instead:

    python -m src.data.split --shards data/shared/sft/tokenized/qwen3-8k/reasoning-packed \\
        --out data/reasoning-split --val-tokens 100000
    .venv/bin/python scripts/train.py --config config/poc.yaml --data data/reasoning-split

`--corpus-form reasoning-packed` fails with that recipe rather than inventing a mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    # Local, like main()'s imports: sft_corpus is portable (no backend), but keeping it here
    # preserves this script's "nothing is imported until it is used" shape.
    from src.data.sft_corpus import (AUTO as SFT_AUTO, FORMS as SFT_FORMS,
                                     GENERIC as SFT_GENERIC, PACKED_FORMS as SFT_PACKED_FORMS)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--data", type=Path, required=True,
                    help="dir with train.jsonl / val.jsonl (from src.data.sft_data), OR a "
                         "shared/sft/tokenized/<tok>-<k> dir (from src.data.{instruct,reasoning,"
                         "tool}_sft) — see --corpus-form")
    ap.add_argument("--corpus-form", nargs="+", default=[SFT_AUTO],
                    choices=tuple(SFT_FORMS) + tuple(SFT_PACKED_FORMS) + (SFT_AUTO, SFT_GENERIC),
                    help="which corpus file(s) under --data to train. 'auto' (default) uses "
                         "train.jsonl if present, else every shared/sft form it finds; "
                         "'generic' forces the train.jsonl/val.jsonl layout. Several masked "
                         "forms may be mixed. 'reasoning-packed' is rejected by name (no loss "
                         "mask — see the module docstring).")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="held-out fraction when the corpus has no val.jsonl of its own "
                         "(the shared/sft forms never do); split is seeded by --seed")
    ap.add_argument("--max-len", type=int, default=None,
                    help="drop records longer than this, never truncate (0 = keep all; "
                         "default: the config's seq_len). Applies to the shared/sft layout "
                         "only — a generic train.jsonl/val.jsonl pair is passed through "
                         "untouched.")
    ap.add_argument("--out", type=Path, default=Path("runs/sft"))
    ap.add_argument("--init", type=Path, default=Path("runs/poc/weights.safetensors"),
                    help="pretrained base weights to initialize from (fresh run only)")
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto")
    ap.add_argument("--epochs", type=int, default=2, help="passes over the SFT set")
    ap.add_argument("--total-steps", type=int, default=None,
                    help="override the epoch-derived optimizer-step count")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--base-lr", type=float, default=2e-5, help="low: ~1e-5..5e-5")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="default: total_steps // 20 (min 1)")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=30,
                    help="cap val batches per eval (0 = full val set)")
    ap.add_argument("--init-loss-scale", type=float, default=2.0 ** 13)
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume bundle dir; if omitted, auto-detects <out>/resume")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    import numpy as np

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.data.sft_corpus import resolve_sft_corpus
    from src.data.sft_loader import SFTLoader
    from src.train.loss_scale import scaler_for_precision
    from src.train.moe_balance import attach_balancer, balancer_for_config
    from src.train.loop import TrainConfig, train
    from src.train.logging import JsonlLogger
    from src.train.checkpoint import CheckpointStore, check_weight_keys, load_weights_dict
    from src.eval.val_loss import evaluate_masked

    backend = get_backend(args.backend)
    cfg = load_config(str(args.config))                 # validates (vocab < 2**32, head_dim | d_inner, ...)

    # --- data (response-masked instruction records) ----------------------------
    # #306: resolve --data to either the generic train.jsonl/val.jsonl pair (returned as paths,
    # so the pre-#306 invocation is unchanged) or the shared/sft forms (validated, concatenated
    # and split here). `reasoning-packed` is rejected inside with the pretraining recipe.
    max_len = cfg.seq_len if args.max_len is None else (args.max_len or None)
    corpus = resolve_sft_corpus(args.data, args.corpus_form, val_frac=args.val_frac,
                                seed=args.seed, max_len=max_len)
    print(f"[corpus] {corpus.summary()}")
    if corpus.is_generic:
        train_loader = SFTLoader(corpus.train_path, cfg.seq_len, args.batch_size,
                                 shuffle=True, seed=args.seed, vocab_size=cfg.vocab_size)
        val_loader = SFTLoader(corpus.val_path, cfg.seq_len, args.batch_size,
                               shuffle=False, drop_last=False, vocab_size=cfg.vocab_size)
    else:
        train_loader = SFTLoader(args.data, cfg.seq_len, args.batch_size, shuffle=True,
                                 seed=args.seed, vocab_size=cfg.vocab_size,
                                 records=corpus.train_records)
        val_loader = SFTLoader(args.data, cfg.seq_len, args.batch_size, shuffle=False,
                               drop_last=False, vocab_size=cfg.vocab_size,
                               records=corpus.val_records)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = args.total_steps or args.epochs * steps_per_epoch
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, total_steps // 20)

    # --- model + optimizer + (dynamic) loss scaling ----------------------------
    backend.seed(args.seed)
    model = backend.model_cls(cfg)
    opt = backend.make_optimizer(model, args.base_lr)
    scaler = scaler_for_precision(cfg.precision, args.init_loss_scale)
    # Loss-Free-Balancing (#213): see scripts/train.py for the off-switch. Both backends'
    # make_sft_train_step accept `balancer=` (#214); None is a no-op on either.
    balancer = balancer_for_config(cfg)
    train_step = backend.make_sft_train_step(model, opt, grad_clip=args.grad_clip,
                                             scaler=scaler, balancer=balancer)

    np_to = backend.to_numpy
    max_b = args.eval_batches or None
    val_eval = lambda m: evaluate_masked(m, val_loader, max_batches=max_b, to_numpy=np_to)

    # --- init / resume ---------------------------------------------------------
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    weights_path = str(out / "weights.safetensors")
    store = CheckpointStore(str(args.resume) if args.resume is not None
                            else str(out / "resume"))
    resuming = store.has_checkpoint()

    start_step = 0
    if resuming:
        meta = store.load(weights_deserializer=lambda p: model.load(p),
                          optimizer_deserializer=lambda p: backend.load_optimizer(opt, p))
        start_step = int(meta["step"])
        if scaler is not None:
            scaler.load_state_dict(meta.get("loss_scale_state") or {})
        print(f"[resume] from step {start_step} slot={meta['slot']} (out={out})")
    else:
        # #214: MLX's load is silently lenient (missing key -> stays at random init,
        # wrong shape -> silently rebound), so check explicitly before loading. Load the
        # safetensors once and reuse the dict for both the check and the load itself
        # (matches scripts/train.py's --init path) rather than reading it twice.
        init_weights = load_weights_dict(str(args.init))
        check_weight_keys(init_weights, model._portable_state_dict(),
                          where=f"--init {args.init}")
        model._load_portable(init_weights)                # initialize from pretrained base
        print(f"[init] from pretrained base {args.init}")
    # Loss-Free-Balancing (#213): adopt the bias that came in with the weights (D3), push
    # it into the routers, enable load counting. No-op when balancing is off.
    attach_balancer(balancer, model)

    logger = JsonlLogger(str(out / "metrics.jsonl"), append=resuming)

    # `data_state` is accepted and IGNORED: SFT resume stays step-based, and the #216
    # length curriculum applies to pretraining only. Persisting a position nothing reads
    # back would be worse than not persisting one.
    def on_checkpoint(step: int, data_state=None) -> None:
        store.save(step=step,
                   loss_scale_state=(scaler.state_dict() if scaler else None),
                   weights_serializer=lambda p: model.save(p),  # portable weights + config
                   optimizer_serializer=lambda p: backend.save_optimizer(opt, p))

    tcfg = TrainConfig(
        total_steps=total_steps, base_lr=args.base_lr, warmup_steps=warmup,
        grad_accum=args.grad_accum, grad_clip=args.grad_clip,
        log_every=args.log_every, eval_every=args.eval_every,
        ckpt_every=args.ckpt_every, out_dir=str(out), seed=args.seed,
    )

    n_params = sum(int(np.asarray(v).size) for _, v in model._portable_state_dict().items())
    print(f"[sft] params~{n_params/1e6:.1f}M  examples={len(train_loader.records)}  "
          f"total_steps={total_steps}  warmup={warmup}  precision={cfg.precision}")

    train(model, train_loader, tcfg, train_step,
          val_eval=val_eval, logger=logger, on_checkpoint=on_checkpoint,
          start_step=start_step)

    # Skip the terminal checkpoint if the loop already wrote one at total_steps.
    if total_steps % tcfg.ckpt_every != 0:
        on_checkpoint(total_steps)
    model.save(weights_path)            # canonical portable weights for downstream
    final = evaluate_masked(model, val_loader, max_batches=max_b, to_numpy=np_to)
    logger.close()
    print(f"[done] step={total_steps}  val_loss={final['val_loss']:.4f}  "
          f"val_perplexity={final['val_perplexity']:.4f}  metrics={out / 'metrics.jsonl'}")


if __name__ == "__main__":
    main()
