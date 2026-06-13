# Conformance: fp32 parity

[← Index](README.md)

Conformance checks guard the seam's central promise: the model behaves identically
across its two compute paths, and (eventually) across its two backends. Both live in
[`src/conformance/`](../../src/conformance/) and both compare in **fp32 at ~1e-4
relative tolerance**.

## forward-vs-step parity

The training path (`forward`, parallel scan) and the inference path (`step`,
recurrence) are *separate implementations* of the same math. They must produce the
same logits for the same input. From
[`src/conformance/forward_step_parity.py`](../../src/conformance/forward_step_parity.py):

> The training path (`forward`, parallel scan) and the inference path (`step`,
> recurrence) are two SEPARATE code paths and must produce the same logits for the
> same input. A mismatch here is a silent, nasty bug that the parallel-vs-sequential
> scan check does NOT catch (that check validates only the scan, not train/infer
> equivalence).
>
> Run in fp32, ~1e-4 relative tolerance. Build the model, run a fixed batch through
> `forward`, then feed the same tokens one at a time through `step` carrying state,
> and assert the per-position logits agree.

`check_forward_step_parity(model, token_batch, to_numpy, rtol=1e-4, atol=1e-5)`
returns the max absolute diff and an `ok` flag. It is exercised on the toy MLX model
by `tests/test_mlx_parity.py::test_forward_step_parity_toy`, and end-to-end by the
[smoke gate](06-smoke-gate-and-eval.md).

This is distinct from the *scan* check (`test_mlx_parity.py`), which validates the
parallel scan against a sequential reference (and an independent from-scratch numpy
reference, plus a long-context overflow guard). The scan check proves the scan math;
the parity check proves train/infer equivalence. Both are needed.

## backend-vs-backend parity

The same weights and input must give the same logits on MLX and the CUDA/PyTorch
backend. From
[`src/conformance/backend_parity.py`](../../src/conformance/backend_parity.py):

> Fixed seed, fixed weights, fixed input batch. Run `forward` through both the MLX
> and CUDA backends and assert agreement. Run the comparison in FP32 on BOTH sides.

It loads identical [portable weights](05-training.md) into each backend (the seam's
`save`/`load`), then compares. Because the CUDA backend is **pure PyTorch and runs on
CPU**, this is runnable entirely on a Mac (mlx + torch-CPU both present) — no GPU
required. `tests/test_backend_parity.py` exercises it on the toy config, and also tests
the **portable-weights round-trip in both directions** (MLX → safetensors → torch →
safetensors → MLX, logits unchanged), which is what lets a CUDA-trained model come back
to the Mac. The only layout subtlety is the depthwise-conv weight: the portable format
is MLX-canonical `(out, k, in/groups)`, and the torch backend transposes to/from torch's
`(out, in/groups, k)` in `_portable_state_dict`/`_load_portable`.

## Why fp32, ~1e-4

bf16's machine epsilon is too coarse for a meaningful equivalence check. From
`backend_parity.py`:

> bf16's machine epsilon (~8e-3) is larger than a meaningful tolerance, so comparing
> low-precision paths yields false failures. In fp32 a tight tolerance (~1e-4
> relative) is meaningful: within = correct port, beyond = a real math bug.

So conformance always compares in fp32 regardless of the run precision — the goal is
to catch *math* bugs, not measure numerical noise.

## Status

`forward_step_parity` is active and passing on both backends (MLX, and the pure-PyTorch
CUDA backend on torch-CPU). `backend_parity` is implemented and exercised by
`tests/test_backend_parity.py`; the cross-backend cases need both backends present, so
they **skip cleanly** on a single-backend host (e.g. a Linux/CUDA box without mlx, or a
Mac without torch) and run in full on a Mac with torch-CPU installed.

## Related

- [Architecture: the hardware seam](01-architecture-seam.md)
- [Model: two compute paths](02-model-ssm.md)
- [Smoke gate & eval](06-smoke-gate-and-eval.md)
