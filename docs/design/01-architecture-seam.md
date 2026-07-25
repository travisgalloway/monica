# Architecture: the hardware seam

[← Index](README.md)

## The decision

All hardware-specific code lives behind **one abstraction**,
[`ModelInterface`](../../src/model/interface.py). Everything above the seam —
`data/`, `train/`, `serve/`, `eval/`, `conformance/`, `lsp/` — is portable Python that
**never imports MLX or CUDA**. Exactly six modules touch a hardware library:

| Module | Role |
|---|---|
| `src/model/mlx_backend.py` | MLX model |
| `src/model/mlx_train_step.py` | MLX backprop/optimizer primitive |
| `src/model/cuda_backend.py` | CUDA/PyTorch model |
| `src/model/cuda_train_step.py` | CUDA backprop/optimizer primitive |
| `src/model/cuda_muon.py` | Muon + AdamW hybrid optimizer (#237) |
| `src/model/mlx_lm_adapter.py` | the LSP harness's `LMAdapter` (see [12-lsp-in-the-loop.md](12-lsp-in-the-loop.md)) |

`src/model/backend.py` is deliberately **not** on that list: it is the portable
backend-factory registry (`BackendSpec`, `make_optimizer`) and keeps its backend imports
**inside the factory closures**, so importing it above the seam pulls in nothing. That is
what lets `make_optimizer` select Muon-vs-AdamW without the training loop knowing a
backend exists.

**Outside this Python seam entirely:** the `swift/` native toolchain (the repo's first Swift
package — the code tokenizer `MonicaTokenizer` + `monica-tokenize` CLI, #191/#245). It is neither
above-seam Python nor a Python hardware backend; it builds and runs natively on macOS **and**
Linux/CUDA with bit-identical output and is not covered by the import guard below. It emits the
same `src/data/shard.py` shard layout, so the Python data/training path consumes its output
unchanged. This is the M13 "native, no-Python-runtime" engine direction (#163/#167).

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

`ModelInterface` defines exactly seven concerns:

| Method | Role |
|---|---|
| `forward(token_batch, seg_ids=None)` | Full-sequence **parallel** training path → logits `(B, T, vocab)`; `seg_ids` marks document boundaries within a packed sequence |
| `step(token, state)` | Single-token **recurrence** inference path; must agree with `forward` |
| `init_state(batch_size)` | Fresh, zeroed recurrent state |
| `get_state()` / `set_state(state)` | Snapshot / restore (for serving + rewind) |
| `clone_state(state)` | Independent snapshot of a state, safe to retain while stepping |
| `save(path)` / `load(path)` | Persist weights in a portable format (safetensors) |
| `config` | The `MambaConfig` |

`clone_state` is separate from `get_state` because the serving layer
(`serve/sessions`, `serve/rewind`) holds many states at once and snapshots them at turn
boundaries. On an immutable-array backend (MLX) a structural copy suffices; a backend whose
`step` mutates buffers in place **must** deep-copy here, or a retained snapshot silently
aliases a later step.

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
cross-cutting invariants (e.g. the vocab/packing-dtype bound — see
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
