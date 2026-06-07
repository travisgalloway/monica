# monica — Mamba POC (SSM on Mac Experiment)

A proof-of-concept Mamba (selective state-space) language model, developed and
validated on **Apple Silicon with MLX**, architected behind **one hardware seam**
so a successful POC migrates to **CUDA** for a larger run with minimal rewrite.

## The seam (most important rule)

All hardware-specific code lives behind `src/model/interface.py`
(`ModelInterface`). Everything above it — `data/`, `train/`, `serve/`, `eval/`,
`conformance/` — is portable Python that **never imports MLX or CUDA**. Only the
backend modules — `src/model/mlx_backend.py`, `src/model/mlx_train_step.py`, and
`src/model/cuda_backend.py` — touch a hardware library.
`tests/test_import_guard.py` enforces this.

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
docs/design/               design choices + rationale (start at docs/design/README.md)
docs/MAC_RUNBOOK.md        ordered build checklist (M0–M5)
```

## Status — M1–M4 done (verified on Apple Silicon)

The seam, configs, data pipeline, MLX model/backend, training loop, and smoke test
are **implemented and verified on Apple Silicon** — `pytest` → 20 passing on a Mac
(incl. the real MLX paths; on Linux the MLX-only tests skip and the rest pass) and
the M4 smoke gate passes end to end (exact save/kill/resume + held-out perplexity
eval). Remaining work is the scale-up (M5–M8). Progress is tracked in
[issue #2](https://github.com/travisgalloway/monica/issues/2).

| Milestone | State |
|---|---|
| 1 Seam + toy MLX model | done; MLX backend + forward/step parity verified |
| 2 Data pipeline (tiny) | done; unit-tested |
| 3 Minimal training loop | done; `train_step` + loop exercised by the smoke gate |
| 4 Smoke test (gate) | passing — resume exact, eval runs |
| 5–8 POC scale, OLMES, serve/rewind, CUDA | deferred stubs |

**Locked decisions:** poc = d_model 768 / 24 layers / d_state 16 / seq 1024 /
~3B tokens (tied embedding mandatory). toy = d_model 64 / 2 layers / seq 128 /
fp32. Precision for poc (fp16 vs bf16) **confirmed on MLX in M1** — not assumed.
Conformance compares in **fp32** (~1e-4 rel). OLMES + serving/rewind deferred;
**POC success = a smoothly decreasing held-out val-perplexity curve**.

## Quickstart

```bash
# Apple Silicon (full backend):
pip install -e ".[dev,data,mlx]"  # the mlx extra installs only on Apple Silicon

# Linux / CUDA host (portable layers only — omit the mlx extra):
pip install -e ".[dev,data]"

pytest                            # Mac: 20 passed. Linux: MLX-only tests
                                  # (pytest.importorskip("mlx.core")) are skipped,
                                  # not failed — the portable suite still runs.

# Data pipeline offline smoke (no network/tokenizer):
python -m src.data.download --dummy --out data/raw --max-docs 2000
python -m src.data.tokenize --in data/raw/dummy.txt --out data/ids.npy --byte-fallback
python -m src.data.pack --in data/ids.npy --out data/packed.bin
python -m src.data.split --packed data/packed.bin --out data/split --val-tokens 2000
```

Run the smoke gate (the M4 gate) to confirm the MLX backend, training loop, and
exact-resume all pass before scaling to `config/poc.yaml`:

```bash
python scripts/smoke_test.py --data data/split
```
