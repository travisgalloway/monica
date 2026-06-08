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
- **`save_resume` / `load_resume`** write a same-backend bundle (step, RNG state,
  optimizer state) using a backend-supplied serializer — because optimizer-state
  layout *is* backend-specific, and you never need to resume a half-finished MLX run
  on CUDA.

`safetensors` is imported lazily so the module loads without the dependency present.

The [smoke gate](06-smoke-gate-and-eval.md) exercises exactly this: save portable
weights + a resume bundle, tear everything down, rebuild, load, and continue — then
assert the trajectory matches bit-for-bit.

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

## Related

- [Architecture: the hardware seam](01-architecture-seam.md)
- [Smoke gate & eval](06-smoke-gate-and-eval.md) — checkpointing under test.
- [Configs & locked decisions](07-configs-and-decisions.md) — fp16 vs fp32.
