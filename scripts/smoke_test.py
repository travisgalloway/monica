"""Milestone-4 SMOKE TEST — the gate (runs on Apple Silicon). SKELETON.

The single most important test in the project. Most projects silently break at
checkpoint resume and dataloading, not in the model. Do NOT proceed past this gate
until resume is verifiably exact and eval runs.

Procedure (toy model, tiny data, ~50 steps, FIXED SEED):
  1. Train uninterrupted for N steps; record the loss trajectory  (reference run).
  2. Fresh run with the SAME seed: train N/2 steps, save a checkpoint + resume
     bundle, KILL the process, RESUME from the checkpoint, train the rest.
  3. Assert the post-resume trajectory matches the reference within tolerance
     (fp32 toy => effectively exact). This is how you prove resume is correct.
  4. Run a held-out val-perplexity eval end to end (eval.val_loss.evaluate).

This file imports the MLX backend, so it runs on Apple Silicon only. Everything it
orchestrates (loader, schedule, checkpoint, val_loss) is already backend-free and
unit-tested on any host.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# NOTE: imported here (not at top) to keep the intent clear that the heavy lifting
# is MLX-only and wired on the Mac.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/toy.yaml"))
    ap.add_argument("--data", type=Path, required=True, help="dir with train.bin/val.bin")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/smoke"))
    args = ap.parse_args()

    # TODO[mac]: implement using:
    #   from src.model.blocks import load_config
    #   from src.model.mlx_backend import MLXMambaModel
    #   from src.data.loader import PackedLoader
    #   from src.train.loop import train, TrainConfig          (+ MLX train_step)
    #   from src.train.checkpoint import save_weights/save_resume/load_resume
    #   from src.eval.val_loss import evaluate
    # 1) reference run  2) interrupted+resumed run  3) trajectory match assert
    # 4) final val-perplexity eval. Fail loudly on any mismatch.
    raise NotImplementedError(
        "Wire the smoke test on Apple Silicon once mlx_backend + MLX train_step exist. "
        "Gate: post-resume trajectory must match the reference within tolerance."
    )


if __name__ == "__main__":
    main()
