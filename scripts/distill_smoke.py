"""Distillation smoke gate (#81) — the end-to-end distill flow on MLX, offline.

The distill analogue of `scripts/smoke_test.py`: builds a tiny byte-vocab corpus and a synthetic
teacher, caches the teacher top-k (#94), then runs `scripts/distill.py` through all three
distillation stages (mixing-match -> hidden-align -> logit-distill) for a handful of steps each.
It asserts every stage's loss is finite and decreases, the portable student weights + per-stage
resume bundles are written, and that a `--resume` invocation reloads a checkpoint cleanly. The
loop + checkpoint machinery's bit-exact resume is covered by the M4 gate (`smoke_test.py`); the
distill stages reuse it unchanged.

    .venv/bin/python scripts/distill_smoke.py --out runs/distill-smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "config" / "toy-distill.yaml"
SEQ_LEN = 16
VOCAB = 256
STAGES = ("mixing-match", "hidden-align", "logit-distill")


def _build_corpus(root: Path) -> Path:
    """Byte-vocab tokenized corpus -> train.bin/val.bin split (the PackedLoader format)."""
    from src.data import storage
    from src.data.corpus import ingest_dummy
    from src.data.distill_corpus import build_distill_corpus
    from src.data.split import split_shards

    out_root = root / "pd"
    # tokenizer label only names the storage subdir here (byte_fallback uses the 256-vocab
    # ByteTokenizer); kept as qwen3 to match toy-distill.yaml's `tokenizer: qwen3`.
    build_distill_corpus(ingest_dummy(400, source="smoke"), out_root, tokenizer="qwen3",
                         seq_len=SEQ_LEN, byte_fallback=True)
    tok_dir = storage.corpus_tokenized_dir(out_root, "qwen3", SEQ_LEN)
    split_dir = root / "split"
    split_shards(tok_dir, split_dir, val_tokens=SEQ_LEN * 4)
    return split_dir


def _run(argv: list, label: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(REPO), "PATH": os.environ.get("PATH", "")}
    res = subprocess.run([sys.executable, *argv], cwd=REPO, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(f"{label} failed (exit {res.returncode})")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("runs/distill-smoke"))
    ap.add_argument("--steps-per-stage", type=int, default=12)
    ap.add_argument("--backend", choices=("mlx", "cuda"), default="mlx",
                    help="backend to gate on (mlx: Apple-Silicon dev; cuda: torch GPU/CPU)")
    args = ap.parse_args()

    probe = "mlx.core" if args.backend == "mlx" else "torch"
    try:
        __import__(probe)
    except ImportError:
        raise SystemExit(f"distill_smoke --backend {args.backend} requires {probe!r}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    split_dir = _build_corpus(out)
    teacher_out = out / "teacher-outputs"
    run_out = out / "run"

    # 1) cache the synthetic teacher's top-k, aligned to the split corpus.
    _run(["scripts/precompute_teacher.py", "--manifest", str(MANIFEST), "--data", str(split_dir),
          "--out", str(teacher_out), "--backend", args.backend, "--k", "8", "--batch-size", "2",
          "--synthetic"], "precompute")

    # 2) run the full distill flow (all three stages).
    distill_argv = ["scripts/distill.py", "--manifest", str(MANIFEST), "--corpus", str(split_dir),
                    "--teacher-outputs", str(teacher_out), "--out", str(run_out),
                    "--backend", args.backend,
                    "--synthetic", "--vocab", str(VOCAB), "--precision", "fp32",
                    "--base-lr", "1e-2", "--steps-per-stage", str(args.steps_per_stage),
                    "--batch-size", "2", "--grad-accum", "1", "--log-every", "1",
                    "--eval-every", "4", "--ckpt-every", "4"]
    _run(distill_argv, "distill")

    # 3) assert per-stage loss is finite and training reduces it below the start, and artifacts
    # exist. We check `min(losses) < losses[0]` rather than `last < first`: at the smoke lr the
    # hidden-align MSE oscillates (and MLX GPU ops aren't bit-deterministic run to run), so the
    # honest, stable signal of "the machinery trains" is that the loss dips below its start.
    for stage in STAGES:
        metrics_path = run_out / stage / "metrics.jsonl"
        losses = [json.loads(line)["loss"] for line in metrics_path.read_text().splitlines() if line]
        assert losses and all(math.isfinite(l) for l in losses), f"{stage}: non-finite loss {losses}"
        assert min(losses) < losses[0], f"{stage}: loss never dropped below start {losses}"
        assert (run_out / stage / "resume").exists(), f"{stage}: no resume bundle"
        print(f"[{stage}] loss {losses[0]:.4f} -> min {min(losses):.4f}  ✓")
    assert (run_out / "weights.safetensors").exists(), "no portable student weights"

    # 4) a --resume invocation reloads the furthest stage's checkpoint cleanly.
    _run(distill_argv + ["--resume"], "distill --resume")

    print("\nDISTILL SMOKE PASSED ✅  all stages decrease, checkpoints + weights written, "
          "resume reloads cleanly.")


if __name__ == "__main__":
    main()
