#!/usr/bin/env bash
# local_validate.sh — one offline command that exercises every local pipeline stage on Apple
# Silicon (or any host where the MLX backend imports). It is the "validate a change locally with
# speed and ease" entry point: each stage fails fast, so a regression anywhere stops the run.
#
# Stages (all offline — byte-fallback tokenizer, no network/HF/weights):
#   1. data     download(--dummy) -> tokenize(--byte-fallback) -> pack -> split
#   2. smoke     scripts/smoke_test.py on a FRESH byte split (resume-exactness + val eval, fp32)
#   3. train     scripts/train.py --config config/small.yaml  (a few real fp16 steps)
# (M10 distill/teacher stages were removed with #189.)
#
# Usage:   scripts/local_validate.sh
# Env:     PYTHON (default .venv/bin/python)   WORK (default runs/local-validate)
#          STEPS (default 20)                  KEEP=1 to keep the work dir
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
WORK="${WORK:-runs/local-validate}"
STEPS="${STEPS:-20}"
DATA="$WORK/data"

cd "$(dirname "$0")/.."   # repo root
echo "== local_validate :: python=$PY  work=$WORK  steps=$STEPS =="
# Safety: WORK is rm -rf'd, so refuse empty / root / home / any absolute path — a mis-set
# WORK (e.g. WORK=/) must never nuke the system. Keep it a relative path under the repo.
case "$WORK" in
  ""|"/"|"."|"./"|"~"*) echo "local_validate: refusing to rm unsafe WORK='$WORK'" >&2; exit 1 ;;
  /*) echo "local_validate: refusing to rm absolute WORK='$WORK' (use a repo-relative path)" >&2; exit 1 ;;
esac
rm -rf "$WORK"; mkdir -p "$WORK"

echo "== [1/3] data pipeline (offline byte fallback) =="
"$PY" -m src.data.download --dummy --out "$WORK/raw" --max-docs 8000
"$PY" -m src.data.tokenize --in "$WORK/raw/dummy.txt" --out "$WORK/ids.npy" --byte-fallback
"$PY" -m src.data.pack  --in "$WORK/ids.npy" --out "$WORK/packed.bin"
"$PY" -m src.data.split --packed "$WORK/packed.bin" --out "$DATA" --val-tokens 4000

echo "== [2/3] smoke gate (resume-exactness + val eval, fresh byte split) =="
"$PY" scripts/smoke_test.py --config config/toy.yaml --data "$DATA" \
    --steps "$STEPS" --batch-size 8 --out "$WORK/smoke"

echo "== [3/3] small.yaml training (real fp16 path, $STEPS steps) =="
"$PY" scripts/train.py --config config/small.yaml --data "$DATA" --out "$WORK/small" \
    --total-steps "$STEPS" --batch-size 8 --grad-accum 1 --eval-every "$STEPS" --ckpt-every 100000

echo "== local_validate: ALL STAGES PASSED =="
[ "${KEEP:-0}" = "1" ] || rm -rf "$WORK"
