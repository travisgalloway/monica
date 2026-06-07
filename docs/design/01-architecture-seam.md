# Architecture: the hardware seam

[← Index](README.md)

## The decision

All hardware-specific code lives behind **one abstraction**,
[`ModelInterface`](../../src/model/interface.py). Everything above the seam —
`data/`, `train/`, `serve/`, `eval/`, `conformance/` — is portable Python that
**never imports MLX or CUDA**. Only the backend modules
(`src/model/mlx_backend.py`, `src/model/mlx_train_step.py`,
`src/model/cuda_backend.py`) touch a hardware library.

## Why

The POC is developed and validated on Apple Silicon (MLX), but a successful POC is
meant to migrate to CUDA for a larger run **with minimal rewrite**. Confining every
hardware dependency to one swappable layer means the training loop, data pipeline,
checkpointing, and evaluation are written once and reused unchanged across backends.

From `src/model/interface.py`:

> THIS MODULE MUST NOT IMPORT ANY BACKEND (no `mlx`, no `torch`/CUDA). Everything
> above the seam (train/serve/eval/conformance) depends only on this interface and
> on `blocks.MambaConfig`. Each backend (`mlx_backend`, `cuda_backend`) provides a
> concrete subclass implementing exactly these methods.

## The contract

`ModelInterface` defines exactly six concerns:

| Method | Role |
|---|---|
| `forward(token_batch)` | Full-sequence **parallel** training path → logits `(B, T, vocab)` |
| `step(token, state)` | Single-token **recurrence** inference path; must agree with `forward` |
| `init_state(batch_size)` | Fresh, zeroed recurrent state |
| `get_state()` / `set_state(state)` | Snapshot / restore (for serving + rewind) |
| `save(path)` / `load(path)` | Persist weights in a portable format (safetensors) |
| `config` | The `MambaConfig` |

The two compute paths (`forward` and `step`) are separate implementations that must
produce identical logits — enforced by [conformance](03-conformance.md).

## Opaque state

`State` is typed as `Any` on purpose. From `src/model/interface.py`:

> State is intentionally typed as `Any`: its concrete representation is
> backend-specific (an MLX array tuple, a torch tensor, ...). Code above the seam
> treats it as an opaque, fixed-size blob that it can snapshot and restore.

In the MLX backend, the concrete state is a per-layer list of
`(conv_state, ssm_state)` tuples (see [model](02-model-ssm.md)), but nothing above
the seam knows or cares.

## Configuration is shared, not duplicated

`MambaConfig` ([`src/model/blocks.py`](../../src/model/blocks.py)) is the single
source of truth for model dimensions and run parameters, loaded from
`config/*.yaml`. It is backend-free and carries a `validate()` that enforces
cross-cutting invariants (e.g. the uint16 vocab bound — see
[data pipeline](04-data-pipeline.md)). Backends consume the same config object, so a
decision like the load-bearing dt-bias init is defined once and "carried into every
backend."

## Enforcement

The seam is not a convention — it is tested. `tests/test_import_guard.py` imports
every portable module in a subprocess and asserts neither `mlx` nor torch's CUDA
stack got pulled in:

> If importing the interface or any above-the-seam package pulls in `mlx` or
> torch's CUDA stack, the migration plan is broken.

The guarded set (`PORTABLE_MODULES`) covers the interface, config, the data
pipeline, schedule, checkpoint, the training loop, val_loss, and the
forward-step-parity conformance harness.

## Related

- [Model: the Mamba block + selective SSM](02-model-ssm.md) — what lives *below* the seam.
- [Training](05-training.md) — how the loop stays backend-free via an injected `train_step`.
- [Conformance](03-conformance.md) — how the two compute paths and two backends are kept honest.
