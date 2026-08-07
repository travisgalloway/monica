"""Sparse-upcycle driver (#214): turn a dense (`n_experts=1`) checkpoint into a real MoE
one, exact at step 0.

Standalone, offline, and imports NO hardware backend at all (not even to build a model)
— the transform (`src.train.upcycle.upcycle_dense_to_moe`) operates purely on portable
`{name: np.ndarray}` weight dicts. Explicit and one-shot ON PURPOSE (issue #214 rejected
doing this implicitly inside `train.py --init`): a mistyped `--init` there would silently
RESHAPE a checkpoint into something plausible-looking rather than raise, and a run could
burn a full training budget on a wrong init before anyone noticed the loss curve was
merely "fine", not right.

    # Inspect the plan without touching disk (no --src-config needed if <src> has a
    # <src>.config.json sidecar, written by save_weights since day one):
    python scripts/upcycle.py --src runs/dense/weights.safetensors \\
        --config config/toy-moe-fine.yaml --out /tmp/upcycled.safetensors \\
        --seed 214 --dry-run

    # Actually write it:
    python scripts/upcycle.py --src runs/dense/weights.safetensors \\
        --config config/toy-moe-fine.yaml --out runs/upcycled/weights.safetensors --seed 214

Writes `<out>` (+ its own `<out>.config.json` sidecar, via `save_weights`) and a
`<out>.upcycle.json` manifest recording exactly what produced it (source path + sha256,
seed, and the init knobs) so the run is reproducible without re-deriving anything from
the weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="dense (n_experts=1) portable weights (.safetensors)")
    ap.add_argument("--config", type=Path, required=True,
                    help="TARGET MoE config (n_experts>1) to upcycle into")
    ap.add_argument("--out", type=Path, required=True,
                    help="output .safetensors path (+ .config.json sidecar written "
                         "by save_weights, + a .upcycle.json manifest)")
    ap.add_argument("--src-config", type=Path, default=None,
                    help="source config yaml; default reads <src>.config.json (the "
                         "sidecar save_weights has always written)")
    ap.add_argument("--seed", type=int, required=True,
                    help="RNG seed for the fresh router (and any synthesized shared "
                         "expert) init — required so a run is always reproducible")
    ap.add_argument("--router-init-scale", type=float, default=1.0,
                    help="router init bound scale: U(+-scale/sqrt(d_model))")
    ap.add_argument("--shared-expert-init", choices=("zero_down", "forbid"),
                    default="zero_down",
                    help="how to init a target shared expert with no source "
                         "counterpart: zero_down (default, step-0 exact) or forbid "
                         "(raise instead of guessing)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit before touching --src or --out")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    from src.model.blocks import load_config
    from src.train.checkpoint import (
        load_config_sidecar, load_weights_dict, save_weights,
    )
    from src.train.upcycle import (
        check_upcycle_compatible, upcycle_dense_to_moe, upcycle_manifest,
    )
    from src.eval.code_suite import sha256_file

    dst_cfg = load_config(str(args.config))
    if args.src_config is not None:
        src_cfg = load_config(str(args.src_config))
    else:
        src_cfg = load_config_sidecar(str(args.src))
        if src_cfg is None:
            raise SystemExit(
                f"no {args.src}.config.json sidecar found, and --src-config was not "
                "given — pass --src-config <yaml> to name the source config explicitly."
            )

    # Before ANY weights IO (the potentially-large safetensors read): a bad pairing
    # should fail in milliseconds, not after loading gigabytes of tensors.
    check_upcycle_compatible(src_cfg, dst_cfg)

    moe_layers = sorted(i for i in range(dst_cfg.n_layers) if dst_cfg.is_moe_layer(i))
    print(f"[upcycle] src={args.src}  config={args.config}")
    print(f"[upcycle] MoE layers: {moe_layers}")
    print(f"[upcycle] experts per MoE layer: 1 -> {dst_cfg.n_experts}  "
          f"(top_k {src_cfg.top_k} -> {dst_cfg.top_k})")
    print(f"[upcycle] router: [1, {dst_cfg.d_model}] -> "
          f"[{dst_cfg.n_experts}, {dst_cfg.d_model}]  (fresh U(+-"
          f"{args.router_init_scale}/sqrt({dst_cfg.d_model})), seed={args.seed})")
    if dst_cfg.n_shared_experts:
        print(f"[upcycle] shared experts per MoE layer: {src_cfg.n_shared_experts} -> "
              f"{dst_cfg.n_shared_experts}  (new ones: {args.shared_expert_init})")
    print(f"[upcycle] num_parameters: {src_cfg.num_parameters():,} -> "
          f"{dst_cfg.num_parameters():,}")
    print(f"[upcycle] active_num_parameters: {src_cfg.active_num_parameters():,} -> "
          f"{dst_cfg.active_num_parameters():,}")

    if args.dry_run:
        print("[upcycle] --dry-run: exiting before reading --src or writing --out")
        return

    weights = load_weights_dict(str(args.src))
    out = upcycle_dense_to_moe(
        weights, src_cfg, dst_cfg, seed=args.seed,
        router_init_scale=args.router_init_scale,
        shared_expert_init=args.shared_expert_init,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_weights(out, str(args.out), config=dst_cfg)   # writes <out>.config.json too

    src_sha256 = sha256_file(str(args.src))
    manifest = upcycle_manifest(
        src=str(args.src), src_sha256=src_sha256, seed=args.seed,
        router_init_scale=args.router_init_scale,
        shared_expert_init=args.shared_expert_init, src_cfg=src_cfg, dst_cfg=dst_cfg,
    )
    import json
    Path(str(args.out) + ".upcycle.json").write_text(json.dumps(manifest, indent=2))
    print(f"[upcycle] wrote {args.out} (+.config.json, +.upcycle.json)  "
          f"src_sha256={src_sha256[:12]}...")


if __name__ == "__main__":
    main()
