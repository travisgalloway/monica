# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A proof-of-concept Mamba (selective state-space) language model, developed and
validated on **Apple Silicon with MLX**, architected behind **one hardware seam** so
a successful POC can migrate to **CUDA** with minimal rewrite. POC success is defined
as a smoothly decreasing held-out validation-perplexity curve — not benchmark scores.

## Commands

```bash
# Install (Apple Silicon — the normal dev environment):
pip install -e ".[dev,data,mlx]"   # mlx requires Apple Silicon; omit on Linux/CUDA hosts

# Tests (uses the venv at .venv):
.venv/bin/python -m pytest                                   # full suite (36 tests)
.venv/bin/python -m pytest tests/test_mlx_parity.py          # one file
.venv/bin/python -m pytest tests/test_mlx_parity.py::test_forward_step_parity_toy  # one test
.venv/bin/python -m pytest -q -rs                            # quiet, report skips

# The M4 smoke gate — the most important check (resume exactness + eval):
.venv/bin/python scripts/smoke_test.py --data data/split

# Data pipeline offline smoke (no network/tokenizer; uses byte fallback):
python -m src.data.download --dummy --out data/raw --max-docs 2000
python -m src.data.tokenize --in data/raw/dummy.txt --out data/ids.npy --byte-fallback
python -m src.data.pack  --in data/ids.npy --out data/packed.bin
python -m src.data.split --packed data/packed.bin --out data/split --val-tokens 2000
```

There is no separate lint/format/build step — pytest is the gate. `mlx` is not
installable on Linux; on a non-Mac host the MLX backend simply won't import (by
design), and only the portable tests run.

## The seam — the most important architectural rule

All hardware-specific code lives behind `src/model/interface.py`
(`ModelInterface`). Everything above the seam — `src/data/`, `src/train/`,
`src/serve/`, `src/eval/`, `src/conformance/` — is **portable Python that must never
import `mlx` or `torch`/CUDA**. Only `src/model/mlx_backend.py`,
`src/model/mlx_train_step.py`, and `src/model/cuda_backend.py` may touch a hardware
library.

This is enforced by `tests/test_import_guard.py`, which imports every portable module
and asserts no backend leaked into `sys.modules`. **When adding code above the seam,
do not import a backend — and add new portable modules to that test's
`PORTABLE_MODULES` list.** Keep MLX-only imports local (inside functions), as
`scripts/smoke_test.py` does, when a portable-ish entry point needs the backend.

Consequences of the seam that shape how code is written:
- The training loop (`src/train/loop.py`) is backend-free and receives the
  backprop/optimizer primitive as an injected `train_step` callable
  (`TrainStepFn = (model, inputs, targets, lr) -> {loss, grad_norm}`). The MLX
  implementation is `make_train_step(...)` in `src/model/mlx_train_step.py`.
- The data loader yields **numpy**; the backend converts to its own array type inside
  `forward`. Eval (`src/eval/val_loss.py`) takes a `to_numpy` converter at the seam.
- Model `State` is opaque (`Any`) above the seam. In the MLX backend it is a per-layer
  list of `(conv_state, ssm_state)` tuples.

## Configuration is the single source of truth

Model dims and run params live in `config/toy.yaml` and `config/poc.yaml`, loaded into
`MambaConfig` (`src/model/blocks.py`). `MambaConfig.validate()` enforces cross-cutting
invariants (notably `vocab_size < 65536` for uint16 token packing). The YAML **comments
are the decision record** — read them before changing values. Key locked decisions:

- **toy.yaml** (smoke/correctness): tiny, `fp32` for bit-exact fixed-seed resume,
  `vocab_size 256` (byte-fallback tokenizer, offline).
- **poc.yaml** (~100M scale run): `vocab_size 50280` (OLMo-7B-hf, confirmed `<65536`),
  `precision fp16` + loss scaling (~18% faster than bf16 on Metal per the M1
  micro-benchmark — **do not assume bf16**), tied embedding **mandatory** (~38M of
  ~100M params).
- **dt-bias init** (`dt_min`/`dt_max`/`dt_init_floor`) is **load-bearing** — without
  the inverse-softplus init in `SelectiveSSM._init_dt_bias` the model fails to learn
  recall. These params are identical across both configs by design.

## Two compute paths must agree

The model has two separate implementations of the SSM that must produce identical
logits: `forward` (parallel chunked scan, training) and `step` (recurrence,
inference). The selective scan **always chunks** (default 32) because a single-pass
cumsum overflows fp32. Conformance (`src/conformance/`) guards this:
`forward_step_parity` (train vs infer) and `backend_parity` (MLX vs CUDA, deferred)
both compare in **fp32 at ~1e-4 rel** — bf16's epsilon is too coarse to be meaningful.

## Checkpointing: two deliberately separate concerns

`src/train/checkpoint.py` splits (1) **portable weights** (safetensors + config
sidecar — the cross-backend bridge) from (2) a **within-backend resume bundle** (step
+ RNG + optimizer state, via a backend-supplied serializer). They are not conflated:
weights port across backends; optimizer state does not need to (CUDA trains fresh).
The smoke gate stresses exactly this round-trip.

## Workflow

- Milestones M1–M8 are tracked in **GitHub issue #2** (the milestone tracker); each
  sub-issue references "Part of #2". M1–M4 are done and verified; M5 (scale run), and
  M6–M8 (OLMES eval, serving/rewind, CUDA backend) are deferred.
- `docs/design/` documents the design choices and rationale (start at
  `docs/design/README.md`). After completing a milestone, tick its box in issue #2.
- After finishing a milestone or backend change, run the smoke gate, not just pytest.
