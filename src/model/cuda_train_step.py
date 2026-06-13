"""CUDA / PyTorch train_step: grad accumulation + dynamic fp16 loss scaling.

The PyTorch counterpart of `mlx_train_step.py`, below the seam (imports torch). Provides
the backend-specific `train_step` that `train.loop.train` injects, plus optimizer-state
(de)serialization for within-backend exact resume. The portable loop never imports this;
it receives `make_train_step(...)`'s closure as a callable matching
`TrainStepFn = (model, micro_batches, lr) -> {loss, grad_norm, ...}`.

fp16 loss scaling reuses the PORTABLE policy in `src/train/loss_scale.py` (the same
`DynamicLossScaler` the MLX backend uses) rather than `torch.cuda.amp.GradScaler`, so the
fp16 skip/backoff behavior is identical across backends. The backend does only the
inf/nan grad check and skips overflowing steps.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def _global_grad_norm(params) -> torch.Tensor:
    leaves = [p.grad for p in params if p.grad is not None]
    sq = torch.stack([g.detach().float().pow(2).sum() for g in leaves]).sum()
    return torch.sqrt(sq)


def make_train_step(model, optimizer, *, grad_clip: float = 1.0,
                    scaler=None) -> Callable:
    """Build a `train_step(model, micro_batches, lr) -> dict`.

    `micro_batches` is a list of `(inputs, targets)` numpy pairs; the step averages
    grads over them so an effective batch can exceed what fits in memory (one micro-batch
    is live at a time). Closes over `optimizer` so Adam moments persist across steps.

    `scaler` (a portable `DynamicLossScaler`, fp16 path) scales the loss before backprop;
    grads are unscaled before the optimizer step. On a non-finite gradient the step is
    SKIPPED and the scale is backed off; the returned dict carries `loss_scale`/`skipped`.
    Pass None for fp32 (toy/smoke) — numerically identical to a plain unscaled step.
    """
    params = list(model.parameters())

    def _loss(inputs, targets) -> torch.Tensor:
        logits = model.forward(inputs)                       # (B, L, V)
        V = logits.shape[-1]
        t = torch.as_tensor(np.asarray(targets), dtype=torch.long,
                            device=logits.device).reshape(-1)
        # Cross-entropy in fp32 (wide-vocab softmax stability).
        return F.cross_entropy(logits.reshape(-1, V).float(), t, reduction="mean")

    def train_step(model, micro_batches, lr: float) -> dict:
        n = len(micro_batches)
        s = scaler.scale if scaler else 1.0
        optimizer.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for inputs, targets in micro_batches:
            ce = _loss(inputs, targets)
            # Scale for fp16 dynamic range; divide by n so accumulated .grad is the
            # average gradient (matches MLX's acc_grads / n).
            (ce * (s / n)).backward()
            acc_loss += float(ce.detach())
        loss = acc_loss / n

        if scaler:
            inv = 1.0 / s
            for p in params:                                 # unscale grads
                if p.grad is not None:
                    p.grad.mul_(inv)
            norm = _global_grad_norm(params)
            overflow = not bool(torch.isfinite(norm))
            scaler.update(overflow)
            if overflow:                                     # drop the step
                optimizer.zero_grad(set_to_none=True)
                return {"loss": loss, "grad_norm": float("nan"),
                        "loss_scale": scaler.scale, "skipped": True}
        else:
            norm = _global_grad_norm(params)

        if grad_clip:
            factor = min(1.0, grad_clip / (float(norm) + 1e-6))
            if factor < 1.0:
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(factor)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        out = {"loss": loss, "grad_norm": float(norm)}
        if scaler:
            out["loss_scale"] = scaler.scale
            out["skipped"] = False
        return out

    return train_step


# --- optimizer-state (de)serialization for within-backend resume ------------
def _pt_path(path: str) -> str:
    path = str(path)
    return path if path.endswith(".pt") else path + ".pt"


def save_optimizer(optimizer, path: str) -> None:
    torch.save(optimizer.state_dict(), _pt_path(path))


def load_optimizer(optimizer, path: str) -> None:
    # weights_only=False: this is our own trusted optimizer bundle (tensors + the
    # param_group hyperparameters), not untrusted input.
    optimizer.load_state_dict(torch.load(_pt_path(path), weights_only=False))
