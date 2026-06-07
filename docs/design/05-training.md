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
# (model, inputs, targets, lr) -> dict(loss=, grad_norm=)
TrainStepFn = Callable[[ModelInterface, object, object, float], dict]
```

The loop is pure orchestration: schedule the LR, call `train_step`, log, checkpoint,
and support resume via `start_step`. The "robust run" features it wires up
(documented in the loop docstring): mixed precision, warmup+cosine LR, gradient
accumulation, gradient clipping, checkpointing, and logging from step 1 (loss, val
loss/perplexity, LR, grad norm, tokens/sec).

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

`make_train_step(model, optimizer, grad_clip=1.0, loss_scale=None)` closes over the
optimizer so Adam moments persist across steps, and returns the `TrainStepFn`
closure. Two notable choices:

- **Gradient clipping** by global norm (`grad_clip=1.0`) — proven necessary even at
  toy scale to keep training stable.
- **Loss scaling** for the fp16 path — scales the loss before backprop and unscales
  the grads after; pass `None` for fp32 (toy/smoke). This is the runtime half of the
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

## Related

- [Architecture: the hardware seam](01-architecture-seam.md)
- [Smoke gate & eval](06-smoke-gate-and-eval.md) — checkpointing under test.
- [Configs & locked decisions](07-configs-and-decisions.md) — fp16 vs fp32.
