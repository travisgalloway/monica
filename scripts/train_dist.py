"""FSDP2 + expert-parallel `torchrun` entry point (#271) — the CUDA-only twin of
`scripts/train.py`, for a distributed run.

    torchrun --standalone --nproc_per_node=N scripts/train_dist.py \\
        --config config/toy-moe-dist.yaml --data <fresh toy split> --ep-size E \\
        --out runs/toy-moe-dist --total-steps 20

`N` ranks split into `dp_size = N // ep_size` data-parallel FSDP2 replicas, each an
`ep_size`-way expert-parallel shard (`ParallelConfig`, `src/train/parallel.py`).
`ep_size` must divide both `N` and the config's `n_experts`.

Deliberately a SEPARATE script from `scripts/train.py`, not a `--distributed` flag
bolted onto it: `scripts/train.py`'s single-GPU path stays byte-identical (it never
imports `torch.distributed`), and this script owns the process-group lifecycle
(`init_distributed`/`shutdown`) end to end. It constructs the model directly via
`CUDAMambaModel(cfg, ep_size=..., ep_rank=...)` rather than going through
`src.model.backend.get_backend`'s single-rank factory closures, which have no notion of
a process group or an EP shard.

CUDA-only in practice (see `.claude/plans/issue-271.md`'s host-constraint note): this
module imports `torch` and the CUDA backend directly at module scope, matching
`cuda_backend.py`/`cuda_train_step.py`'s own seam placement — it is NOT meant to be
imported on an MLX host. It runs on CPU/gloo too (every `[GLOO-SIM]` test in
`tests/test_cuda_distributed.py`, and this script's own manual `--nproc_per_node=2` CPU
check), which is what makes the mechanism provable without a GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/toy-moe-dist.yaml"))
    ap.add_argument("--data", type=Path, required=True, help="dir with train.bin/val.bin")
    ap.add_argument("--out", type=Path, default=Path("runs/dist"))
    ap.add_argument("--ep-size", type=int, default=1,
                    help="expert-parallel degree; must divide both --nproc_per_node "
                         "and the config's n_experts")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--total-steps", type=int)
    g.add_argument("--total-tokens", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--base-lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="0 disables periodic val eval (default here, unlike "
                         "scripts/train.py — a toy distributed smoke run cares about "
                         "'did it train and checkpoint', not a val curve)")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--reshard-after-forward", action="store_true",
                    help="ZeRO-3-equivalent (default: ZeRO-2, keep params materialized "
                         "after forward — see cuda_distributed.wrap_backbone)")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    import numpy as np
    import torch

    from src.model.blocks import load_config
    from src.model.cuda_backend import CUDAMambaModel
    from src.model.cuda_muon import HybridOptimizer, Muon
    from src.model.cuda_train_step import make_train_step
    from src.model.cuda_distributed import (
        init_distributed, shutdown, barrier, is_primary, get_rank, get_world_size,
        build_mesh, wrap_backbone, ep_group, make_load_reduce_fn,
        gather_portable_state_dict, save_resume_dcp, load_resume_dcp, has_resume_dcp,
    )
    from src.data.loader import PackedLoader
    from src.train.parallel import ParallelConfig
    from src.train.moe_balance import attach_balancer, balancer_for_config
    from src.train.loop import TrainConfig, train
    from src.train.logging import JsonlLogger
    from src.train.checkpoint import CheckpointStore, save_weights

    init_distributed()
    rank, world_size = get_rank(), get_world_size()
    try:
        cfg = load_config(str(args.config))
        pcfg = ParallelConfig(world_size=world_size, ep_size=args.ep_size,
                              n_experts=cfg.n_experts if cfg.n_moe_layers else None)

        torch.manual_seed(args.seed)   # identical seed on every rank: same init pre-shard
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CUDAMambaModel(cfg, device=device, ep_size=pcfg.ep_size, ep_rank=rank % pcfg.ep_size)

        mesh = build_mesh(dp_size=pcfg.dp_size, ep_size=pcfg.ep_size)
        wrap_backbone(model, mesh["dp"], reshard_after_forward=args.reshard_after_forward)
        if pcfg.ep_size > 1:
            model.set_ep_group(ep_group(mesh))

        base_lr = args.base_lr
        muon_params, adam_params = [], []
        if cfg.optimizer == "muon":
            from src.model.blocks import is_muon_param
            for name, p in model.named_parameters():
                (muon_params if is_muon_param(name, p.ndim) else adam_params).append(p)
            muon_lr = cfg.muon_lr if cfg.muon_lr is not None else base_lr
            adam = torch.optim.AdamW(adam_params, lr=base_lr) if adam_params else None
            muon = (Muon(muon_params, lr=base_lr, lr_scale=muon_lr / base_lr,
                         momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps)
                   if muon_params else None)
            opt = HybridOptimizer(adam, muon)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=base_lr)

        balancer = balancer_for_config(cfg)
        load_reduce = make_load_reduce_fn() if world_size > 1 else None
        train_step = make_train_step(model, opt, grad_clip=args.grad_clip,
                                     balancer=balancer, load_reduce=load_reduce)

        train_path = args.data / "train.bin"
        train_loader = PackedLoader(train_path, cfg.seq_len, args.batch_size,
                                    shuffle=True, seed=args.seed)
        val_loader = None
        val_eval = None
        if args.eval_every:
            from src.eval.val_loss import evaluate
            val_loader = PackedLoader(args.data / "val.bin", cfg.seq_len, args.batch_size,
                                      shuffle=False, drop_last=False)
            np_to = lambda a: a.detach().to("cpu").numpy()   # noqa: E731
            val_eval = lambda m: evaluate(m, val_loader, to_numpy=np_to)   # noqa: E731

        total_steps = (args.total_steps if args.total_steps is not None
                       else max(1, args.total_tokens // (args.batch_size * cfg.seq_len
                                                          * args.grad_accum)))
        warmup = args.warmup_steps if args.warmup_steps is not None else max(1, total_steps // 100)

        out = args.out
        if is_primary():
            out.mkdir(parents=True, exist_ok=True)
        barrier()
        resume_root = str(args.resume) if args.resume is not None else str(out / "resume")
        store = CheckpointStore(resume_root)

        start_step = 0
        if has_resume_dcp(resume_root):
            meta = load_resume_dcp(model, opt, resume_root, ep_size=pcfg.ep_size)
            start_step = int(meta["step"])
            if is_primary():
                print(f"[resume] from step {start_step} (world_size={world_size}, "
                     f"ep_size={pcfg.ep_size})")
        attach_balancer(balancer, model)

        logger = JsonlLogger(str(out / "metrics.jsonl"), append=(start_step > 0)) \
            if is_primary() else (lambda payload: None)

        def on_checkpoint(step: int, data_state=None) -> None:
            save_resume_dcp(model, opt, resume_root, step=step, world_size=world_size,
                            ep_size=pcfg.ep_size)
            sd = gather_portable_state_dict(model, ep_process_group=(
                ep_group(mesh) if pcfg.ep_size > 1 else None))
            store.save(step=step, loss_scale_state=None,
                      weights_serializer=lambda p: save_weights(sd, p, config=cfg),
                      optimizer_serializer=lambda p: None,   # optimizer state lives in DCP above
                      is_primary=is_primary(), barrier=barrier, rank=rank)

        tcfg = TrainConfig(total_steps=total_steps, base_lr=base_lr, warmup_steps=warmup,
                           grad_accum=args.grad_accum, grad_clip=args.grad_clip,
                           log_every=args.log_every, eval_every=(args.eval_every or total_steps + 1),
                           ckpt_every=args.ckpt_every, out_dir=str(out), seed=args.seed)

        if is_primary():
            print(f"[dist] world_size={world_size} dp_size={pcfg.dp_size} "
                 f"ep_size={pcfg.ep_size} rank={rank} total_steps={total_steps}")

        result = train(model, train_loader, tcfg, train_step, val_eval=val_eval,
                       logger=logger, on_checkpoint=on_checkpoint, start_step=start_step)

        if total_steps % tcfg.ckpt_every != 0:
            on_checkpoint(total_steps, result.get("data_state"))
        if is_primary():
            print(f"[done] step={total_steps} tokens={result['tokens_seen']} "
                 f"metrics={out / 'metrics.jsonl'}")
    finally:
        barrier()
        shutdown()


if __name__ == "__main__":
    main()
