"""MLX training primitive (Apple Silicon, below the seam — may import mlx).

Provides the backend-specific `train_step` that `train.loop.train` injects, plus
optimizer-state (de)serialization for within-backend exact resume. The portable
loop never imports this; it receives `make_train_step(...)`'s closure as a callable
matching `TrainStepFn = (model, micro_batches, lr) -> {loss, grad_norm, ...}`.
"""

from __future__ import annotations

from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten


def _global_grad_norm(grads) -> mx.array:
    leaves = [v for _, v in tree_flatten(grads)]
    sq = mx.sum(mx.stack([mx.sum(g * g) for g in leaves]))
    return mx.sqrt(sq)


def make_train_step(model, optimizer, *, grad_clip: float = 1.0,
                    scaler=None) -> Callable:
    """Build a `train_step(model, micro_batches, lr) -> dict`.

    `micro_batches` is a list of `(inputs, targets)`; the step averages grads over
    them so an effective batch can exceed what fits in memory (only one micro-batch
    is live at a time). Closes over `optimizer` so Adam moments persist across steps.

    `scaler` (a `DynamicLossScaler`, fp16 path) scales the loss before backprop and
    unscales grads after. On a non-finite gradient the optimizer step is SKIPPED and
    the scale is backed off; the returned dict carries `loss_scale` and `skipped`.
    Pass None for fp32 (toy/smoke) — that path is numerically identical to a plain
    unscaled single-batch step.
    """
    def loss_fn(model, inputs, targets):
        logits = model.forward(inputs)                      # (B, L, V)
        V = logits.shape[-1]
        t = mx.array(targets).reshape(-1).astype(mx.int32)
        # Cross-entropy in fp32 (wide-vocab softmax stability). The MLX backend's
        # `_head` already returns fp32 logits, so this is a no-op there; the cast
        # keeps the contract explicit and backend-independent.
        ce = nn.losses.cross_entropy(logits.reshape(-1, V).astype(mx.float32),
                                     t, reduction="mean")
        return ce * scaler.scale if scaler else ce

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_step(model, micro_batches, lr: float) -> dict:
        # Accumulate grads/loss over the micro-batches, evaluating between them so
        # peak memory stays at one micro-batch.
        n = len(micro_batches)
        acc_grads = None
        acc_loss = mx.zeros(())
        for inputs, targets in micro_batches:
            loss, grads = loss_and_grad(model, inputs, targets)
            acc_grads = grads if acc_grads is None else _add(acc_grads, grads)
            acc_loss = acc_loss + loss
            mx.eval(acc_grads, acc_loss)
        grads = _unscale(acc_grads, 1.0 / n)
        loss = acc_loss / n

        if scaler:
            inv = 1.0 / scaler.scale
            grads = _unscale(grads, inv)
            loss = loss * inv
            norm = _global_grad_norm(grads)
            overflow = not bool(mx.isfinite(norm).item())
            scaler.update(overflow)
            if overflow:                                   # drop the step
                mx.eval(model.parameters())
                return {"loss": float(loss), "grad_norm": float("nan"),
                        "loss_scale": scaler.scale, "skipped": True}
        else:
            norm = _global_grad_norm(grads)

        if grad_clip:
            factor = mx.minimum(1.0, grad_clip / (norm + 1e-6))
            grads = _unscale(grads, factor)
        optimizer.learning_rate = lr
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        out = {"loss": float(loss), "grad_norm": float(norm)}
        if scaler:
            out["loss_scale"] = scaler.scale
            out["skipped"] = False
        return out

    return train_step


def _unscale(grads, factor):
    from mlx.utils import tree_map
    return tree_map(lambda g: g * factor, grads)


def _add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


# --- optimizer-state (de)serialization for within-backend resume ------------
def _st_path(path: str) -> str:
    path = str(path)
    return path if path.endswith(".safetensors") else path + ".safetensors"


def save_optimizer(optimizer, path: str) -> None:
    flat = {k: v for k, v in tree_flatten(optimizer.state) if isinstance(v, mx.array)}
    mx.save_safetensors(_st_path(path), flat)


def load_optimizer(optimizer, path: str) -> None:
    flat = mx.load(_st_path(path))
    optimizer.state = tree_unflatten(list(flat.items()))
    mx.eval(optimizer.state)
