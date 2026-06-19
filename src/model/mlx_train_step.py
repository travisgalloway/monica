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


def _accumulate_and_step(model, optimizer, loss_and_grad, micro_batches, lr,
                         grad_clip, scaler) -> dict:
    """Shared accumulate -> (unscale) -> clip -> optimizer-step tail.

    `loss_and_grad(model, micro_batch) -> (loss, grads)` is the only objective-specific
    piece; everything below (grad accumulation/averaging, fp16 unscale + overflow-skip,
    clipping, the optimizer update, and the returned metrics dict) is identical for
    pretraining, SFT, and DPO, so they all funnel through here. Grads are averaged over
    the micro-batches so an effective batch can exceed what fits in memory (one
    micro-batch is live at a time).
    """
    n = len(micro_batches)
    acc_grads = None
    acc_loss = mx.zeros(())
    for mb in micro_batches:
        loss, grads = loss_and_grad(model, mb)
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
        if overflow:                                       # drop the step
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


def make_train_step(model, optimizer, *, grad_clip: float = 1.0,
                    scaler=None) -> Callable:
    """Build a `train_step(model, micro_batches, lr) -> dict` (pretraining CE).

    `micro_batches` is a list of `(inputs, targets)`. `scaler` (a `DynamicLossScaler`,
    fp16 path) scales the loss before backprop and unscales grads after; on a non-finite
    gradient the optimizer step is SKIPPED and the scale is backed off (see
    `_accumulate_and_step`). Pass None for fp32 (toy/smoke).
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

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def loss_and_grad(model, mb):
        inputs, targets = mb
        return value_and_grad(model, inputs, targets)

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(model, optimizer, loss_and_grad, micro_batches,
                                    lr, grad_clip, scaler)

    return train_step


def make_sft_train_step(model, optimizer, *, grad_clip: float = 1.0,
                        scaler=None) -> Callable:
    """Build an SFT `train_step(model, micro_batches, lr) -> dict` (masked CE).

    `micro_batches` is a list of `(inputs, targets, mask)` (the `SFTLoader` 3-tuple). The
    loss is the per-token cross-entropy averaged over the *response* tokens only:
    `sum(mask * CE) / sum(mask)`, so prompt/padding positions (mask 0) never contribute.
    Accumulation, fp16 scaling/overflow-skip, clipping, and the optimizer step are shared
    with pretraining via `_accumulate_and_step`.
    """
    def loss_fn(model, inputs, targets, mask):
        logits = model.forward(inputs)                      # (B, L, V)
        V = logits.shape[-1]
        t = mx.array(targets).reshape(-1).astype(mx.int32)
        ce = nn.losses.cross_entropy(logits.reshape(-1, V).astype(mx.float32),
                                     t, reduction="none")    # (B*L,)
        m = mx.array(mask).reshape(-1).astype(mx.float32)
        loss = (ce * m).sum() / mx.maximum(m.sum(), 1.0)     # response-token mean
        return loss * scaler.scale if scaler else loss

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def loss_and_grad(model, mb):
        inputs, targets, mask = mb
        return value_and_grad(model, inputs, targets, mask)

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(model, optimizer, loss_and_grad, micro_batches,
                                    lr, grad_clip, scaler)

    return train_step


def _masked_seq_logprob(model, inputs, targets, mask) -> mx.array:
    """Per-sequence SUMMED log-prob of `targets` over masked (response) positions. (B,).

    NOTE: this is the raw sum, not the length-normalized mean. The DPO margin is then
    proportional to response length, so chosen/rejected pairs of very different lengths
    bias the objective toward the shorter response. This matches the original DPO paper
    (sum); it is correct when chosen and rejected have comparable lengths. If you train on
    length-imbalanced pairs, switch to `/ mx.maximum(m.sum(-1), 1.0)` here AND in
    `cuda_train_step._masked_seq_logprob` to keep the two backends in parity.
    """
    logits = model.forward(inputs).astype(mx.float32)        # (B, L, V)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    t = mx.array(targets).astype(mx.int32)
    chosen = mx.take_along_axis(logp, t[..., None], axis=-1)[..., 0]   # (B, L)
    m = mx.array(mask).astype(mx.float32)
    return (chosen * m).sum(axis=-1)                         # (B,)


def _log_sigmoid(x: mx.array) -> mx.array:
    """Stable log(sigmoid(x)) = -softplus(-x)."""
    return -mx.logaddexp(mx.zeros_like(x), -x)


def make_dpo_train_step(policy_model, ref_model, optimizer, *, beta: float = 0.1,
                        grad_clip: float = 1.0, scaler=None) -> Callable:
    """Build a DPO `train_step(model, micro_batches, lr) -> dict`.

    `micro_batches` is a list of the `DPOLoader` 6-tuple `(c_in, c_tgt, c_mask, r_in,
    r_tgt, r_mask)`. The loss is `-mean(log sigmoid(beta * (pi_logratio - ref_logratio)))`
    where `pi_logratio = logpθ(chosen) - logpθ(rejected)` and `ref_logratio` is the same
    under the frozen reference. Gradients flow through `policy_model` only: the optimizer
    updates the policy, and the reference forward is wrapped in `mx.stop_gradient` (and
    `ref_model` is a distinct object never handed to the optimizer), so the reference
    stays frozen. Accumulation / fp16 scaling / clipping are shared via
    `_accumulate_and_step`.
    """
    def loss_fn(policy, c_in, c_tgt, c_mask, r_in, r_tgt, r_mask):
        lp_c = _masked_seq_logprob(policy, c_in, c_tgt, c_mask)
        lp_r = _masked_seq_logprob(policy, r_in, r_tgt, r_mask)
        lr_c = mx.stop_gradient(_masked_seq_logprob(ref_model, c_in, c_tgt, c_mask))
        lr_r = mx.stop_gradient(_masked_seq_logprob(ref_model, r_in, r_tgt, r_mask))
        margin = beta * ((lp_c - lr_c) - (lp_r - lr_r))     # (B,)
        loss = -mx.mean(_log_sigmoid(margin))
        return loss * scaler.scale if scaler else loss

    value_and_grad = nn.value_and_grad(policy_model, loss_fn)

    def loss_and_grad(model, mb):
        return value_and_grad(model, *mb)

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(model, optimizer, loss_and_grad, micro_batches,
                                    lr, grad_clip, scaler)

    return train_step


def make_grpo_train_step(model, optimizer, *, grad_clip: float = 1.0,
                         scaler=None) -> Callable:
    """Build a GRPO `train_step(model, micro_batches, lr) -> dict`.

    `micro_batches` is a list of `(inputs, targets, mask, advantages)`: a batch of sampled
    rollouts (mask = 1 on the generated/completion tokens) and their group-standardized
    advantages (one per sequence, precomputed in the driver via `train.grpo.group_advantages`
    from verifier rewards). The loss is `-mean(advantage * logpθ(completion))` — REINFORCE
    with the GRPO group baseline; gradients flow through the policy only. Accumulation /
    fp16 scaling / clipping are shared via `_accumulate_and_step`.
    """
    def loss_fn(model, inputs, targets, mask, advantages):
        logp = _masked_seq_logprob(model, inputs, targets, mask)     # (B,)
        adv = mx.array(advantages).astype(mx.float32).reshape(-1)    # (B,)
        loss = -mx.mean(adv * logp)
        return loss * scaler.scale if scaler else loss

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def loss_and_grad(model, mb):
        return value_and_grad(model, *mb)

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(model, optimizer, loss_and_grad, micro_batches,
                                    lr, grad_clip, scaler)

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
