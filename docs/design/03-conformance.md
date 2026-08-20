# Conformance: fp32 parity

[← Index](README.md)

Conformance checks guard the seam's central promise: the model behaves identically
across its two compute paths, across its two backends, and across document boundaries in
a packed sequence. All three live in [`src/conformance/`](../../src/conformance/) and all
three compare in **fp32 at ~1e-4 relative tolerance**.

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
required, which is why CI can gate it (see [Status](#status)).
`tests/test_backend_parity.py` exercises it on the toy config, and also tests
the **portable-weights round-trip in both directions** (MLX → safetensors → torch →
safetensors → MLX, logits unchanged), which is what lets a CUDA-trained model come back
to the Mac. The only layout subtlety is the depthwise-conv weight: the portable format
is MLX-canonical `(out, k, in/groups)`, and the torch backend transposes to/from torch's
`(out, in/groups, k)` in `_portable_state_dict`/`_load_portable`.

## document-boundary parity

When several documents are packed into one training sequence, recurrent SSM state (and
attention) must not bleed across the boundaries: a packed multi-document `forward` has to
equal running each document on its own. This is the gate for the `seg_ids` argument on
`ModelInterface.forward` (#68). From
[`src/conformance/doc_boundary_parity.py`](../../src/conformance/doc_boundary_parity.py):

> a silent leak across a boundary corrupts training in a way ordinary losses don't
> surface (mirrors `forward_step_parity` for the SSD scan).

`check_doc_boundary_parity(model, docs, chunk_size, to_numpy=..., pad_id=0, rtol=1e-4,
atol=1e-5)` pads each document up to a multiple of `chunk_size` (boundaries must be
**chunk-aligned**), packs them with `seg_ids`, and checks each document's logit slice
against its standalone forward. Returns `{max_abs_diff, ok, failed_doc}`.

## Why fp32, ~1e-4

bf16's machine epsilon is too coarse for a meaningful equivalence check. From
`backend_parity.py`:

> bf16's machine epsilon (~8e-3) is larger than a meaningful tolerance, so comparing
> low-precision paths yields false failures. In fp32 a tight tolerance (~1e-4
> relative) is meaningful: within = correct port, beyond = a real math bug.

So conformance always compares in fp32 regardless of the run precision — the goal is
to catch *math* bugs, not measure numerical noise. That claim stays true here: this
section's contract is the fp32/portable Python one (`forward_step_parity`,
`backend_parity`, `doc_boundary_parity`). What follows is a genuinely SEPARATE contract
for the Swift port's low-precision fixtures, not a loosening of this one.

## The low-precision contract (#266)

`swift/engine/`'s `monica-parity` gate runs the same forward-vs-step (and prefill/state)
comparisons this document describes, but at the model's ACTUAL serving precision — #167's
fast decode path runs at exactly that precision, and nothing gated it before #266. Reusing
the fp32 band there is not an option (the block quote above already explains why); this
section is the derivation for what replaces it. Full numbers and the reproduction scripts
live in `src/conformance/tolerances.py`'s module docstring and
`tests/test_lowp_parity_band.py` — this is the summary.

**Measurement, not a guess.** The elementwise band `atol(dtype) = 64u`, `rtol(dtype) = 8u`
(`u` = the dtype's unit roundoff) comes from measuring Python's OWN forward-vs-step
disagreement across three toy configs (plain Mamba, MoE, hybrid attention) at fp16/bf16:
the disagreement normalises to the same 6-7 * u coefficient regardless of config depth or
block mix, so a per-DTYPE constant (not per-fixture, not a function of layer count) is the
right shape for the gate. The chosen band sits ~9-10x above that measured coefficient.

**The elementwise band is nearly vacuous on its own.** At that headroom, the elementwise
tier only catches a weight defect of >=33% (fp16) / >=330% (bf16) — an honest headroom
over the dtype's noise floor is already comparable to the signal from a real defect. So it
stays in the contract as a coarse/format check, and a SEPARATE distributional tier (mean
KL between Python's reference and the Swift output, both `forward` and stacked `step`) is
the load-bearing one: `mean_kl(dtype) <= LOWP_MEAN_KL_MAX[dtype]`, calibrated on the WORST
of the three configs' own forward-vs-step KL noise floor (MoE routing and attention
introduce coherent, not purely-random, low-precision divergence — a `toy.yaml`-only
calibration undershot the real floor by 40-60x when first measured). At the corrected
threshold the KL tier catches a weight defect of roughly >=2.8% (fp16) / >=3% (bf16) on
the worst-case config — still ~12x/~110x more sensitive than the elementwise tier, so it
augments the fp32 gate's guarantee rather than replacing it: the low-precision contract
is real, but it is *weaker* than the fp32 one, on paths that only exist at low precision.

**Two exactness guards scale with dtype, everything else stays exact.** `greedy_ids`
(#167) and MoE `load.{i}` counts (#265) are compared EXACTLY at every precision, never
with a tolerance — but the MARGIN that makes an exact comparison safe from a near-tie
widens with the dtype's noise floor (`greedy_margin_floor`, `route_margin_min` in
`tolerances.py`), or the guard becomes either vacuous or spuriously flaky.

**Structurally prevented from leaking into the fp32 gate**, by construction, not by
convention: the band is looked up by the fixture's own declared `precision` (unknown/
missing is a hard failure, never a default); a per-fixture `rtol`/`atol` override is
accepted only when the fixture also declares `quant_bits` (#168's mechanism); and an
override can only loosen the dtype band, never tighten below it. `tests/
test_lowp_parity_band.py`'s T5 walks every checked-in fixture and asserts this holds —
portable, no MLX, runs on the Linux CI job.

## Status

`forward_step_parity` is active and passing on both backends (MLX, and the pure-PyTorch
CUDA backend on torch-CPU). `backend_parity` is implemented and exercised by
`tests/test_backend_parity.py`. `doc_boundary_parity` is implemented and has its own
dedicated tests.

**Cross-backend parity is automated (#303).** The `full-macos` job in
`.github/workflows/ci.yml` installs `.[dev,data,mlx,cuda]` — on macOS arm64 the PyPI
`torch` wheel is CPU-only, which is precisely the surface above — and runs a dedicated
step with `MONICA_REQUIRE_BOTH_BACKENDS=1`. Under that flag the five cross-backend tests
carry **no skip marker**, so a missing backend raises ImportError and the step goes red;
`test_designated_job_has_both_backends` says the same thing up front with a message
naming the cause, and pytest's exit code 5 catches the file disappearing. Nothing outside
that job can police the flag, so `tests/test_ci_backend_matrix.py` parses `ci.yml` and
asserts exactly one job declares it, that it is `full-macos` on a `macos-*` runner
installing both extras — portable, so it runs on the Linux `portable` job.

Skipping remains the behaviour everywhere else, and that is correct: a Linux/CUDA box has
no mlx wheel, a Mac without torch has no second backend. In particular the Linux
`cuda-cpu` job runs `tests/test_backend_parity.py` and still skips 5 of its 6 tests —
only the torch-against-itself harness self-check executes there, so **`cuda-cpu` green is
not evidence of cross-backend parity**.

What the gate does *not* cover: it compares at **toy scale in fp32** (`config/toy.yaml`
and friends, `B=2, L=24`) against **torch on CPU**. Real-GPU CUDA kernels and poc-scale
shapes are compared by no job.

## Related

- [Architecture: the hardware seam](01-architecture-seam.md)
- [Model: two compute paths](02-model-ssm.md)
- [Smoke gate & eval](06-smoke-gate-and-eval.md)
