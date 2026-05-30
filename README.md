# monica — Mamba POC (SSM on Mac Experiment)

A proof-of-concept Mamba (selective state-space) language model, developed and
validated on **Apple Silicon with MLX**, architected behind **one hardware seam**
so a successful POC migrates to **CUDA** for a larger run with minimal rewrite.

## The seam (most important rule)

All hardware-specific code lives behind `src/model/interface.py`
(`ModelInterface`). Everything above it — `data/`, `train/`, `serve/`, `eval/`,
`conformance/` — is portable Python that **never imports MLX or CUDA**. Only
`src/model/mlx_backend.py` and `src/model/cuda_backend.py` touch a hardware
library. `tests/test_import_guard.py` enforces this.

`ModelInterface`: `forward` (parallel training path) · `step` (recurrence inference
path) · `init_state` · `get_state`/`set_state` · `save`/`load` · `config`.

## Layout

```
config/{toy,poc}.yaml      model dims + run params (single source of truth)
src/model/                 interface (seam) · blocks (config) · mlx/cuda backends
src/data/                  download · tokenize · pack(uint16) · split · loader
src/train/                 loop · schedule (warmup+cosine) · checkpoint
src/eval/                  val_loss (Tier-1, primary) · olmes_adapter (deferred)
src/conformance/           forward_step_parity · backend_parity (fp32 guards)
src/serve/                 sessions · rewind (deferred)
scripts/smoke_test.py      the milestone-4 gate
tests/                     in-container unit tests
```

## Status — scaffold for milestones 1–4

This Linux repo holds the **structure + seam + configs + runnable data pipeline**,
with **skeletons/stubs** for MLX-specific bodies. MLX only runs on Apple Silicon,
so the model, training loop, and smoke test are **completed and run on a Mac**.

| Milestone | State | Where it runs |
|---|---|---|
| 1 Seam + toy MLX model | interface/config done; backend skeleton | Mac |
| 2 Data pipeline (tiny) | implemented + unit-tested | here (Linux) |
| 3 Minimal training loop | schedule/checkpoint/val done; loop skeleton | Mac |
| 4 Smoke test (gate) | skeleton | Mac |
| 5–8 POC scale, OLMES, serve/rewind, CUDA | deferred stubs | later |

**Locked decisions:** poc = d_model 768 / 24 layers / d_state 16 / seq 1024 /
~3B tokens (tied embedding mandatory). toy = d_model 64 / 2 layers / seq 128 /
fp32. Precision for poc (fp16 vs bf16) **confirmed on MLX in M1** — not assumed.
Conformance compares in **fp32** (~1e-4 rel). OLMES + serving/rewind deferred;
**POC success = a smoothly decreasing held-out val-perplexity curve**.

## Quickstart (this container)

```bash
pip install -e ".[dev,data]"      # mlx is Apple-Silicon only; omit on Linux
pytest                            # schedule, val_loss, data pipeline, import guard

# Data pipeline offline smoke (no network/tokenizer):
python -m src.data.download --dummy --out data/raw --max-docs 2000
python -m src.data.tokenize --in data/raw/dummy.txt --out data/ids.npy --byte-fallback
python -m src.data.pack --in data/ids.npy --out data/packed.bin
python -m src.data.split --packed data/packed.bin --out data/split --val-tokens 2000
```

On a Mac: `pip install mlx`, complete the `TODO[mac]` bodies in
`src/model/mlx_backend.py` + an MLX `train_step`, then run
`python scripts/smoke_test.py --data data/split` (the M4 gate) before scaling to
`config/poc.yaml`.
