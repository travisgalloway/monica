"""MLX training primitive (Apple Silicon, below the seam — may import mlx).

Provides the backend-specific `train_step` that `train.loop.train` injects, plus
optimizer-state (de)serialization for within-backend exact resume. The portable
loop never imports this; it receives `make_train_step(...)`'s closure as a callable
matching `TrainStepFn = (model, inputs, targets, lr) -> {loss, grad_norm}`.
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten


def _global_grad_norm(grads) -> mx.array:
    leaves = [v for _, v in tree_flatten(grads)]
    sq = mx.sum(mx.stack([mx.sum(g * g) for g in leaves]))
    return mx.sqrt(sq)


def make_train_step(model, optimizer, *, grad_clip: float = 1.0,
                    loss_scale: Optional[float] = None) -> Callable:
    """Build a `train_step(model, inputs, targets, lr) -> {loss, grad_norm}`.

    Closes over `optimizer` so Adam moments persist across steps. `loss_scale`
    (fp16 path) scales the loss before backprop and unscales grads after; pass
    None for fp32 (toy/smoke).
    """
    def loss_fn(model, inputs, targets):
        logits = model.forward(inputs)                      # (B, L, V)
        V = logits.shape[-1]
        t = mx.array(targets).reshape(-1).astype(mx.int32)
        ce = nn.losses.cross_entropy(logits.reshape(-1, V), t, reduction="mean")
        return ce * loss_scale if loss_scale else ce

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def train_step(model, inputs, targets, lr: float) -> dict:
        loss, grads = loss_and_grad(model, inputs, targets)
        if loss_scale:
            grads = _unscale(grads, 1.0 / loss_scale)
            loss = loss / loss_scale
        norm = _global_grad_norm(grads)
        if grad_clip:
            factor = mx.minimum(1.0, grad_clip / (norm + 1e-6))
            grads = _unscale(grads, factor)
        optimizer.learning_rate = lr
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        return {"loss": float(loss), "grad_norm": float(norm)}

    return train_step


def _unscale(grads, factor):
    from mlx.utils import tree_map
    return tree_map(lambda g: g * factor, grads)


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
