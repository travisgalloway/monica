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


def _accumulate_and_step(optimizer, params, loss_fn, micro_batches, lr,
                         grad_clip, scaler) -> dict:
    """Shared accumulate -> (unscale) -> clip -> optimizer-step tail.

    `loss_fn(micro_batch) -> scalar torch loss` is the only objective-specific piece;
    everything below (scaled backward + grad accumulation, fp16 unscale + overflow-skip,
    clipping, the optimizer update, and the returned metrics dict) is identical for
    pretraining, SFT, DPO, and GRPO, so they all funnel through here — the torch mirror of
    `mlx_train_step._accumulate_and_step`.
    """
    n = len(micro_batches)
    s = scaler.scale if scaler else 1.0
    optimizer.zero_grad(set_to_none=True)
    acc_loss = 0.0
    for mb in micro_batches:
        loss = loss_fn(mb)
        # Scale for fp16 dynamic range; divide by n so accumulated .grad is the average
        # gradient (matches MLX's acc_grads / n).
        (loss * (s / n)).backward()
        acc_loss += float(loss.detach())
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


def make_train_step(model, optimizer, *, grad_clip: float = 1.0,
                    scaler=None) -> Callable:
    """Build a `train_step(model, micro_batches, lr) -> dict` (pretraining CE).

    `micro_batches` is a list of `(inputs, targets)` numpy pairs; the step averages
    grads over them so an effective batch can exceed what fits in memory (one micro-batch
    is live at a time). Closes over `optimizer` so Adam moments persist across steps.

    `scaler` (a portable `DynamicLossScaler`, fp16 path) scales the loss before backprop;
    grads are unscaled before the optimizer step. On a non-finite gradient the step is
    SKIPPED and the scale is backed off; the returned dict carries `loss_scale`/`skipped`.
    Pass None for fp32 (toy/smoke) — numerically identical to a plain unscaled step.
    """
    params = list(model.parameters())

    def _loss(mb) -> torch.Tensor:
        inputs, targets = mb
        logits = model.forward(inputs)                       # (B, L, V)
        V = logits.shape[-1]
        t = torch.as_tensor(np.asarray(targets), dtype=torch.long,
                            device=logits.device).reshape(-1)
        # Cross-entropy in fp32 (wide-vocab softmax stability).
        return F.cross_entropy(logits.reshape(-1, V).float(), t, reduction="mean")

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(optimizer, params, _loss, micro_batches, lr,
                                    grad_clip, scaler)

    return train_step


# --------------------------------------------------------------------------- #
# Post-training step factories (#110): SFT / DPO / GRPO — torch mirrors of the
# MLX factories in mlx_train_step.py. The objective-specific loss is computed in
# torch (the autodiff path); the portable loss math (src/train/dpo_math.py,
# src/train/grpo.py, src/eval/val_loss.py) is the numpy reference the tests check
# against. All four objectives share `_accumulate_and_step`.
# --------------------------------------------------------------------------- #
def make_sft_train_step(model, optimizer, *, grad_clip: float = 1.0,
                        scaler=None) -> Callable:
    """Build an SFT `train_step(model, micro_batches, lr) -> dict` (masked CE).

    `micro_batches` is a list of `(inputs, targets, mask)` (the `SFTLoader` 3-tuple). The
    loss is the per-token cross-entropy averaged over the *response* tokens only:
    `sum(mask * CE) / sum(mask)`, so prompt/padding positions (mask 0) never contribute.
    """
    params = list(model.parameters())

    def _loss(mb) -> torch.Tensor:
        inputs, targets, mask = mb
        logits = model.forward(inputs)                       # (B, L, V)
        V = logits.shape[-1]
        device = logits.device
        t = torch.as_tensor(np.asarray(targets), dtype=torch.long, device=device).reshape(-1)
        ce = F.cross_entropy(logits.reshape(-1, V).float(), t, reduction="none")   # (B*L,)
        m = torch.as_tensor(np.asarray(mask), dtype=torch.float32, device=device).reshape(-1)
        return (ce * m).sum() / torch.clamp(m.sum(), min=1.0)    # response-token mean

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(optimizer, params, _loss, micro_batches, lr,
                                    grad_clip, scaler)

    return train_step


def _masked_seq_logprob(model, inputs, targets, mask) -> torch.Tensor:
    """Per-sequence SUMMED log-prob of `targets` over masked (response) positions. (B,).

    Torch port of `mlx_train_step._masked_seq_logprob` — see its note on the summed (not
    length-normalized) convention and the length-bias it implies. Keep the two in parity:
    if you switch one to a mean (`/ m.sum(-1).clamp_min(1.0)`), switch both."""
    logits = model.forward(inputs).float()                   # (B, L, V)
    device = logits.device
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    t = torch.as_tensor(np.asarray(targets), dtype=torch.long, device=device)
    chosen = torch.gather(logp, -1, t.unsqueeze(-1)).squeeze(-1)   # (B, L)
    m = torch.as_tensor(np.asarray(mask), dtype=torch.float32, device=device)
    return (chosen * m).sum(dim=-1)                          # (B,)


def make_dpo_train_step(policy_model, ref_model, optimizer, *, beta: float = 0.1,
                        grad_clip: float = 1.0, scaler=None) -> Callable:
    """Build a DPO `train_step(model, micro_batches, lr) -> dict`.

    `micro_batches` is a list of the `DPOLoader` 6-tuple `(c_in, c_tgt, c_mask, r_in,
    r_tgt, r_mask)`. The loss is `-mean(log sigmoid(beta * (pi_logratio - ref_logratio)))`.
    Gradients flow through `policy_model` only: the optimizer holds the policy params and
    the reference forward runs under `torch.no_grad()` (and `ref_model` is a distinct
    object never handed to the optimizer), so the reference stays frozen.
    """
    params = list(policy_model.parameters())

    def _loss(mb) -> torch.Tensor:
        c_in, c_tgt, c_mask, r_in, r_tgt, r_mask = mb
        lp_c = _masked_seq_logprob(policy_model, c_in, c_tgt, c_mask)
        lp_r = _masked_seq_logprob(policy_model, r_in, r_tgt, r_mask)
        with torch.no_grad():
            lr_c = _masked_seq_logprob(ref_model, c_in, c_tgt, c_mask)
            lr_r = _masked_seq_logprob(ref_model, r_in, r_tgt, r_mask)
        margin = beta * ((lp_c - lr_c) - (lp_r - lr_r))     # (B,)
        return -F.logsigmoid(margin).mean()

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(optimizer, params, _loss, micro_batches, lr,
                                    grad_clip, scaler)

    return train_step


def make_grpo_train_step(model, optimizer, *, grad_clip: float = 1.0,
                         scaler=None) -> Callable:
    """Build a GRPO `train_step(model, micro_batches, lr) -> dict`.

    `micro_batches` is a list of `(inputs, targets, mask, advantages)`: sampled rollouts
    (mask = 1 on the completion tokens) and their group-standardized advantages (one per
    sequence, precomputed via `train.grpo.group_advantages`). The loss is
    `-mean(advantage * logpθ(completion))` — REINFORCE with the GRPO group baseline.
    """
    params = list(model.parameters())

    def _loss(mb) -> torch.Tensor:
        inputs, targets, mask, advantages = mb
        logp = _masked_seq_logprob(model, inputs, targets, mask)     # (B,)
        adv = torch.as_tensor(np.asarray(advantages), dtype=torch.float32,
                              device=logp.device).reshape(-1)        # (B,)
        return -(adv * logp).mean()

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(optimizer, params, _loss, micro_batches, lr,
                                    grad_clip, scaler)

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
