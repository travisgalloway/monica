# Training

[← Index](README.md)

Training is split between a **backend-free loop** ([`src/train/`](../../src/train/))
and a **backend-specific step**
([`src/model/mlx_train_step.py`](../../src/model/mlx_train_step.py)). The split is
what keeps the [seam](01-architecture-seam.md) intact while still doing real
backprop.

## A backend-free loop with an injected step

The loop drives the model only through `ModelInterface.forward` and the checkpoint
module. The thing that actually computes gradients is hardware-specific, so it is
*injected* rather than imported. From
[`src/train/loop.py`](../../src/train/loop.py):

> Drives the model only through `ModelInterface.forward` + the checkpoint module.
> Backend-free here; the backprop/optimizer primitive is backend-specific and is
> injected as `train_step` (e.g. MLX's `nn.value_and_grad` + optimizer.update on the
> Mac). This keeps the seam intact: the loop never imports MLX/CUDA.

The injected callable has a fixed contract:

```python
# (model, micro_batches, lr) -> dict(loss=, grad_norm=, [loss_scale=, skipped=])
TrainStepFn = Callable[[ModelInterface, list, float], dict]
```

`micro_batches` is a list of `(inputs, targets)` of length `cfg.grad_accum`; the step
averages gradients over them so an effective batch can exceed what fits in memory
(only one micro-batch is materialized at a time). The loop is pure orchestration:
schedule the LR, pull `grad_accum` micro-batches, call `train_step`, log, checkpoint,
and support resume via `start_step`. The "robust run" features it wires up: mixed
precision, warmup+cosine LR, gradient accumulation, gradient clipping, checkpointing,
and logging from step 1 (loss, val loss/perplexity, LR, grad norm, tokens/sec).

## LR schedule: warmup + cosine, never zero

From [`src/train/schedule.py`](../../src/train/schedule.py):

> Learning-rate schedule: linear warmup + cosine decay to a floor.
>
> Pure math, backend-free, fully testable in any environment. Decay stops at
> `min_lr_ratio * base_lr` (do NOT let LR hit zero).

Linear warmup from 0 to `base_lr`, then cosine decay to a floor of `min_lr_ratio *
base_lr` (default ratio 0.1). The floor matters — letting LR reach exactly zero stalls
learning at the tail. Being pure math, it is unit-tested anywhere (no MLX needed).

### WSD is the second schedule (#238)

`schedule.py` also provides `WSDSchedule` — linear warmup → **constant `base_lr`
plateau** → `1-sqrt` decay to a floor over the trailing `decay_steps`. A factory selects
between the two from a duck-typed config (`lr_schedule`, `decay_frac`, …).

WSD exists for the M12 shape, not as a general upgrade: a **stable trunk checkpoint can
be re-decayed independently downstream** (cheap re-decay per branch — e.g. per
sparse-upcycle branch, #223), which cosine cannot give you without re-running the trunk.
Both schedules are indexed by `step` over `total_steps` and WSD's decay window is a
*fraction* (`decay_frac`) of it, so nothing here depends on `total_steps` being fixed —
which is what keeps it compatible with the length curriculum (#216), where tokens/step
varies. `scripts/train.py --init <path>` (#214) is the missing half of this lever: it
loads foreign PORTABLE weights (via `check_weight_keys`, `strict=True`-equivalent) as a
fresh step-0 start with a NEW LR schedule/optimizer state, which is exactly what a WSD
re-decay branch needs — `--resume` (same-backend, same schedule, same optimizer state)
answers a different question and always wins if both are passed (see
[Sparse-upcycle init (#214)](#sparse-upcycle-init-214) below for `--init`'s other
consumer).

## The backend step: clipping + loss scaling

From `src/model/mlx_train_step.py`:

> MLX training primitive (Apple Silicon, below the seam — may import mlx).
>
> Provides the backend-specific `train_step` that `train.loop.train` injects, plus
> optimizer-state (de)serialization for within-backend exact resume.

`make_train_step(model, optimizer, grad_clip=1.0, scaler=None)` closes over the
optimizer so Adam moments persist across steps, and returns the `TrainStepFn`
closure. Three notable choices:

- **Gradient accumulation** — the step sums grads/loss over the micro-batch list and
  divides by the count, `mx.eval`-ing between micro-batches so peak memory stays at one
  micro-batch. A single micro-batch with no scaler is numerically identical to a plain
  unscaled step (this is what keeps the fp32 [smoke gate](06-smoke-gate-and-eval.md)
  bit-exact).
- **Gradient clipping** by global norm (`grad_clip=1.0`) — proven necessary even at
  toy scale to keep training stable.
- **Dynamic loss scaling** for the fp16 path — a
  [`DynamicLossScaler`](../../src/train/loss_scale.py) scales the loss before backprop
  and the grads are unscaled after. The *policy* (halve on a non-finite gradient, grow
  after N clean steps) is a pure-Python state machine kept **above the seam** so it is
  unit-testable without MLX; only the inf/nan detection on the gradient tensors lives in
  the backend step. On overflow the optimizer step is **skipped** and the scale backs
  off (the returned dict carries `loss_scale`/`skipped`). Pass `None` for fp32
  (toy/smoke). This is the runtime half of the
  [fp16 precision decision](07-configs-and-decisions.md).

## Checkpointing: two concerns, deliberately separate

From [`src/train/checkpoint.py`](../../src/train/checkpoint.py):

> Checkpointing: two DISTINCT concerns, deliberately not conflated.
>
> 1. Weights — PORTABLE format (safetensors). This is what lets an MLX checkpoint
>    seed a CUDA run and lets a CUDA-trained model run on the Mac. Backend-agnostic:
>    a flat dict of {param_name: numpy array} plus the config.
>
> 2. Optimizer state — needed ONLY for exact resume on the SAME backend after an
>    interruption. It does NOT need to be cross-backend portable (MLX and PyTorch
>    optimizer state differ internally; the migration trains fresh on CUDA anyway).
>    Saved via a backend-provided serializer, scoped to within-backend resume.

This separation is the crux of the migration story:

- **`save_weights`** writes safetensors + a `<path>.config.json` sidecar — portable,
  the bridge between backends.
- **`CheckpointStore`** writes the same-backend resume bundle (portable weights +
  optimizer state + step + fp16 loss-scale state + the dataloader position) via a
  backend-supplied serializer — optimizer-state layout *is* backend-specific, and you
  never need to resume a half-finished MLX run on CUDA. It is **double-buffered and
  crash-safe**: two slots plus an atomically-flipped `LATEST` pointer, so a checkpoint
  interrupted mid-write (a spot preemption on a multi-week run) leaves the *previous*
  checkpoint fully intact. No model RNG is persisted, because the model has no train-time
  randomness (dropout-free; weight init is overwritten by the loaded weights).
- **The data position used to be *reconstructed*, and since #216 it is *recorded*.** The
  old rule — "data order on resume is derived deterministically from `(seed, step,
  grad_accum)`" — holds only while `seq_len`, and therefore `len(loader)`, is constant for
  the whole run. The length curriculum below breaks that, so `save(data_state=...)`
  persists the stream's explicit position. It rides **inside the slot's existing
  `resume_meta.json`**, so it is committed by the same atomic `LATEST` flip as the weights
  and the step: a checkpoint can never ship weights that disagree with the data position
  they were trained to. `data_state` is optional and read with `meta.get("data_state")`,
  so pre-#216 bundles still resume via the old reconstruction.

`safetensors` is imported lazily so the module loads without the dependency present.

The [smoke gate](06-smoke-gate-and-eval.md) exercises exactly this: save a full
checkpoint, tear everything down, rebuild, load, and continue — then assert the
trajectory matches bit-for-bit.

## The scale run (M5)

[`scripts/train.py`](../../scripts/train.py) is the real run driver — it wires the
pieces above to the MLX backend for `config/poc.yaml`: it loads + validates the config,
builds the model and AdamW, turns on the `DynamicLossScaler` when `precision == fp16`,
opens train/val `PackedLoader`s, and runs the loop with a JSONL logger, periodic
checkpoints, and a held-out val-perplexity callback. It resumes from `<out>/resume`
automatically when present (or via `--resume`), restoring weights + optimizer + step +
loss-scale and appending to `metrics.jsonl`. Run params (steps/tokens, batch,
grad-accum, cadences) are **CLI flags**, not model config — they don't belong in
`MambaConfig`; the recommended invocation is recorded as comments in `config/poc.yaml`.

Success is read straight off `<out>/metrics.jsonl`: a smoothly decreasing
`val_perplexity` with a stable `grad_norm`. Note the toy model is a *correctness* model
on ~1M repetitive bytes — it is fine for short smoke/validation runs but will eventually
destabilize in fp32 if pushed far past that regime; the poc run (100M params, fp16 +
dynamic scaling, ~3B diverse tokens) is the regime M5 actually targets.

## Sparse-upcycle init (#214)

`scripts/train.py --init <path>` starts a FRESH step-0 run from a foreign checkpoint's
PORTABLE weights (safetensors + config sidecar), rather than resuming its optimizer
state/step/schedule the way `--resume` does. `--resume` **wins unconditionally** if both
are passed, and the ignore is printed, never silent — a mistyped `--init` on an
already-resumable run must not silently do the wrong thing. `check_weight_keys`
(`src/train/checkpoint.py`) enforces `strict=True`-equivalent key matching on load: CUDA's
`load_state_dict(strict=True)` already refuses a missing/extra/wrong-shaped key, but MLX's
`_load_portable` (`mlx_backend.py:1006`) ends in a plain `self.update(...)`, which is
*silently lenient* — a missing key leaves that param at random init and a wrong-shaped
tensor is silently rebound. `check_weight_keys` closes that gap so `--init` is equally safe
on both backends.

`src/train/upcycle.py` (portable, numpy-only) is the actual sparse-upcycle transform this
flag is built for: `upcycle_dense_to_moe(...)` takes a **dense** checkpoint — specifically
a degenerate `n_experts: 1, top_k: 1` "MoE" block, which validate() permits and which IS a
plain SwiGLU FFN (softmax over one logit is exactly 1.0) — and replicates its
`experts.0.{gate,up,down}` into every expert slot of a real MoE target, re-initializes the
router (`U(±1/sqrt(d_model))`, matching what both backends' `nn.Linear` already produces —
no new magic constant), and zeros the shared expert's `down` projection so the combined
block reproduces the dense forward exactly at step 0 regardless of the combination
formula. `check_upcycle_compatible(src_cfg, dst_cfg)` reports **every** dimension mismatch
at once (not one-at-a-time raises, which is how a run ends up mis-dimensioned) and hard-
`UpcycleError`s on any `d_model` mismatch — expert replication can copy a tensor, not widen
it; see [13-code-model-moe.md](13-code-model-moe.md#the-ssi-fold-structural-signal-integration--secondary)'s
"Open (#223)" note for why that is a live constraint, not a hypothetical one.
`scripts/upcycle.py` runs this offline as an explicit, one-shot CLI (never implicitly
inside `train.py` — a mistyped `--init` reshaping instead of erroring would burn a whole
run on a wrong init that still produces a plausible-looking loss curve), writing one
weights artifact plus a `.upcycle.json` manifest (source sha256, seed, both configs) next
to it.

## Length curriculum + dataloader-state resume (#216)

Two changes that the issue deliberately pairs, because **the first invalidates the
invariant the second replaces**.

**The curriculum.** `--curriculum "0.25:2048,0.5:4096,1.0:16384"` on `scripts/train.py`
ramps `seq_len` across the run — short contexts through the early, high-LR steps, long
context only in the second half. SSMs extend gracefully, and extending late is far
cheaper than training long throughout. The spec is `until_frac:seq_len[:batch_size]`,
comma separated, where **`until_frac` is cumulative** (the fraction of the run *ending* at
that stage) and the last must be `1.0`; it is validated hard, because misreading it as
"share of the run" would silently produce a plausible-but-wrong allocation. An omitted
`batch_size` is derived as `max(1, base_batch*base_seq // seq_len_i)` — a **floor**, so a
derived stage can only ever get cheaper than the reference, never surprise-OOM at 16k.
Like every other run param, it is a **CLI flag, not model config**.

Two things it deliberately does not do. **Val loss stays pinned at `cfg.seq_len`** — a
val curve measured at a changing context length is not comparable across the run, which
would make the POC's primary signal unreadable. And **holding tokens/step constant does
not hold *cost* constant**: attention is O(L²), so 8× the length at ⅛ the batch is ~8× the
attention FLOPs and activation memory for the ~12.5% attention layers. `grad_checkpoint:
true` stays mandatory, and the explicit per-stage `batch_size` field exists precisely so
an operator can shrink further at 16k.

**Why explicit dataloader state became necessary.** Chunking is `seq_len`-dependent
(`stride = seq_len + 1`, so `n_chunks` and `len(loader)` both move), which is exactly what
the old step→position reconstruction assumed was fixed. On an interruptible pod the
failure would have been *silent*: the run resumes, resamples data it has already seen, and
its epoch accounting is quietly wrong while every log line looks healthy. So
[`src/train/stream.py`](../../src/train/stream.py)'s `MicroBatchStream` carries the
position as five counters — `stage_idx`, `micro_in_stage`, `global_micro`, `epoch_idx`,
`batches_into_epoch` — and `state_dict()`/`load_state_dict()` round-trip them through the
checkpoint. **Every failure path raises**; none degrades to "start a fresh epoch".
`load_state_dict` checks a fingerprint (`seed`, `grad_accum`, per-stage shapes, corpus
path + size) and an RNG tripwire, and names `--ignore-data-state` as the escape hatch.
`steps` is deliberately *outside* the fingerprint, so extending a run with a bigger
`--total-tokens` stays legal.

Details worth carrying:

- **Stage lengths are in optimizer steps**, so a boundary always lands on a step boundary
  and no single step ever mixes shapes.
- **`epoch_idx` is global, not per-stage.** The per-epoch shuffle reseeds with
  `seed + epoch_idx`, so a global counter makes the one-stage case bit-identical to the
  pre-#216 stream — which is what keeps SFT/DPO and the smoke gate unchanged. `train()`
  synthesizes that one-stage curriculum when none is passed, so there is a single code
  path.
- **`torch.compile` recompiles at each boundary** (CUDA only). Bounded — at most
  `n_stages - 1` events, and inductor promotes to dynamic shapes after the second distinct
  `(B, L)` — so the mitigation is *visibility*, not a code change: the startup banner sizes
  the cost explicitly and the loop emits an `{"event": "stage", ...}` line before each
  boundary, so a stall there is attributable rather than mysterious. `dynamic=True`,
  `mark_dynamic`, and CUDA graphs are out of scope.
- **Acceptance is "kill-and-resume reproduces the exact data stream"**, checked at three
  levels: `tests/test_stream.py` (pure numpy, kill points mid-epoch / on an epoch boundary
  / on and after a stage boundary), `tests/test_train_loop.py::test_resume_across_curriculum_boundary`
  (through the real loop), and the smoke gate's **phase 7**, which drives the real
  `train()` + `MicroBatchStream` + `CheckpointStore` across a boundary and asserts both
  that the post-resume loss trajectory matches *and* that the `seq_len` sequence really
  changed — a curriculum that silently failed to engage must not pass.

The guarantee is exact stream *reproduction*, not "no token is seen twice": `PackedLoader`
cuts windows from one flat `train.bin` with no document structure, so chunking at a
different `seq_len` is a different partition of the same corpus and later stages
necessarily re-cover earlier tokens at a different alignment.

## Efficiency levers (M12, landed 2026-07-20/21)

Four levers were folded in from the efficiency-survey review, sequenced to land *before*
the #219 ablation sweep so the sweep and the #222/#223 runs all carry them. The repo was
already mature on the survey's biggest axes (dedup/filtering, fused AdamW, SDPA, the
mamba-ssm kernels, grad checkpointing), so these are the net-new ones.

| Lever | Issue / commit | Where | Status |
|---|---|---|---|
| Hybrid **Muon + AdamW** optimizer | #237 / `3b02e6b` | `src/model/cuda_muon.py`, selected at the `make_optimizer` seam in `src/model/backend.py` | **landed** |
| **WSD** LR schedule | #238 / `8fe62f7` | `src/train/schedule.py` (`WSDSchedule`) — see above | **landed** |
| **`torch.compile`** default-on for real CUDA runs | #239 / `7a71073` | `src/model/cuda_backend.py` | **landed** |
| **fp8** MoE-expert linears (Transformer Engine, Hopper) | #240, landed with #214 | `src/model/cuda_backend.py` (`_te_linear_cls`, `MoEBlock._fp8_ctx`, `_layer_forward`'s TE-checkpoint branch) | **wired**, hardware-unverified (no Hopper CI runner) |
| **8-bit AdamW moments** (bitsandbytes) | #214 | `src/model/backend.py` (`_cuda_backend._make_optimizer`'s `_adamw` helper) | **wired**, hardware-unverified (no CUDA CI runner) |

Two details worth carrying:

- **Muon lives behind `make_optimizer`, not in the loop.** `HybridOptimizer` deliberately
  does *not* subclass `torch.optim.Optimizer` — it composes an Adam and a Muon and
  forwards `param_groups`/`zero_grad`/`step`/`state_dict`/`load_state_dict`. The loop
  above the seam is unchanged; the optimizer is a backend concern by construction.
- **`config.torch_compile` is tri-state.** `None` = AUTO, which compiles **only on a real
  CUDA device**; `True`/`False` force it on any device. CPU is never auto-compiled, because
  CPU is the fp32 parity/conformance surface (`tests/test_cuda_parity.py`) and must stay
  eager. fp8's `te.Linear` graph-breaks `torch.compile` the same way `mamba-ssm` does —
  safe, but it is why the fp8 path is opaque to inductor.

The fp8 code is now live: `_Expert`'s linears are built by `_expert_linear` (a `te.Linear`
when `fp8_experts` resolves TE on a Hopper+ device, else the bf16/fp16 `nn.Linear`
fallback), and `MoEBlock._fp8_ctx` wraps the expert-compute region — dense/gather
dispatch plus the shared-expert sum, but deliberately NOT the router (a plain `nn.Linear`
that always routes in fp32) — in `transformer_engine.pytorch.fp8_autocast(...)`, since a
`te.Linear` called outside that context silently runs in bf16. Grad-checkpointed MoE
layers under `fp8_experts` route through `transformer_engine.pytorch.checkpoint` instead
of the plain `torch.utils.checkpoint` call every other layer uses — plain checkpoint's
recompute pass would double-update the fp8 amax history TE calibrates its scale from.
None of this is CI-verified (no Hopper runner); see `tests/test_cuda_fp8.py`'s two
Hopper-gated acceptance tests and docs/infrastructure.md's manual-verification checklist.

### 8-bit AdamW moments (#214)

`optimizer_8bit: bool` on `MambaConfig` is an axis **orthogonal to** `optimizer`
(`adamw`/`muon`), not a third enum value, so it composes with the Muon hybrid: Muon
already routes 2D hidden weight matrices (including MoE expert gate/up/down) through
Newton-Schulz orthogonalization with its own (non-Adam) state, so `optimizer_8bit` only
changes how the **AdamW-routed remainder** stores its moments — embeddings, norms,
biases, the router, `dt_proj` under `muon`, or literally everything under plain `adamw`.
`backend.py::_cuda_backend._make_optimizer`'s `_adamw(params)` helper is the ONE place
the 8-bit switch appears, shared by both branches: it builds `bnb.optim.AdamW8bit`
(`bitsandbytes`, imported lazily — a missing `[cuda-8bit]` install surfaces as a legible
`SystemExit` naming the extra, not a bare `ImportError`) and then forces the tied
embedding's optimizer state back to 32-bit via `bnb.optim.GlobalOptimManager`, since it
is the single largest AdamW-routed tensor and is touched every step regardless of batch
content. **`muon` + `optimizer_8bit` saves almost nothing** — Muon already owns the
memory-heavy expert matrices, and the embedding is forced back to 32-bit either way — so
the lever's real payoff is plain `adamw` + `optimizer_8bit`, not the stacked combination.
`bnb.optim.AdamW8bit` has no `fused` kwarg (a `TypeError` at run start if one is passed,
unlike stock `torch.optim.AdamW`). The MLX backend raises `NotImplementedError` on
`optimizer_8bit`, mirroring the existing Muon raise — bitsandbytes has no MLX/Metal port,
and a silent 32-bit fallback would make a Mac memory measurement quietly wrong.
Hardware-unverified like fp8, above; see `config/toy-moe-8bit.yaml` (a config-surface
fixture only) and docs/infrastructure.md's manual-verification checklist.

## Related

- [Architecture: the hardware seam](01-architecture-seam.md)
- [Smoke gate & eval](06-smoke-gate-and-eval.md) — checkpointing under test.
- [Configs & locked decisions](07-configs-and-decisions.md) — fp16 vs fp32.
- [M12 code model](13-code-model-moe.md) — the program these levers were sequenced for;
  MoE, hybrid attention, and the sparse-upcycle plan live there.
