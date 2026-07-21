"""Muon optimizer + hybrid Muon/AdamW container (#237). Below the seam — imports torch.

Muon (MomentUm Orthogonalized by Newton-Schulz) orthogonalizes the momentum-accumulated
gradient of 2D hidden weight matrices via a 5-iteration quintic Newton-Schulz iteration
before applying the update, reaching target loss in fewer steps than AdamW on those params.
Everything else (embeddings, LM head, router, dt_proj, norms, biases, conv weights — see
`is_muon_param` in `src.model.blocks`) stays on AdamW. `HybridOptimizer` wraps one of each
so `cuda_train_step._accumulate_and_step` can drive them through the single
`optimizer.{param_groups,zero_grad,step,state_dict,load_state_dict}` surface it already uses.

The two-LR hazard: `_accumulate_and_step` writes `group["lr"] = lr` on EVERY param group
every step (a schedule value), so Muon cannot store its own learning rate in `group["lr"]`
— it would be clobbered. Instead Muon stores a constant `lr_scale = muon_lr / base_lr` in
its defaults and computes `group["lr"] * lr_scale` fresh inside `step()`.
"""

from __future__ import annotations

import torch


def _newton_schulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Orthogonalize `G` (2D) via `steps` quintic Newton-Schulz iterations.

    Runs in fp32 regardless of the param dtype so it stays deterministic under fp16/bf16
    training configs — required for the smoke gate's bit-exact resume assertion (the
    orthogonalized update must reproduce identically from a restored momentum buffer).
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """SGD-momentum + Newton-Schulz orthogonalization, for 2D hidden weight matrices only.

    `lr` seeds `group["lr"]` (so it starts sane before the first scheduled write); the
    effective per-step learning rate is always `group["lr"] * lr_scale` — see the module
    docstring for why the scale (not the raw lr) is what Muon owns.
    """

    def __init__(self, params, lr: float, lr_scale: float,
                momentum: float = 0.95, ns_steps: int = 5):
        defaults = dict(lr=lr, lr_scale=lr_scale, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            eff_lr = group["lr"] * group["lr_scale"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad)
                update = _newton_schulz5(buf, ns_steps)
                # Non-square rescale (standard Muon): keeps the update norm comparable
                # across matrix aspect ratios.
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(update, alpha=-eff_lr * scale)
        return loss


class HybridOptimizer:
    """Wraps an AdamW sub-optimizer (everything else) and a Muon sub-optimizer (2D hidden
    weight matrices), exposing the subset of `torch.optim.Optimizer`'s surface that
    `cuda_train_step.py` and `checkpoint.py` use: `param_groups`, `zero_grad`, `step`,
    `state_dict`, `load_state_dict`. Deliberately does NOT subclass `torch.optim.Optimizer`
    — that base class's `__init__` expects a single flat param-group list, which doesn't fit
    a container of two independent optimizers.

    Either sub-optimizer may be `None` (empty partition — e.g. a config with no 2D
    non-excluded param, not the real case but cheap to guard); its methods then no-op.
    """

    def __init__(self, adam, muon):
        self.adam = adam
        self.muon = muon

    @property
    def param_groups(self):
        # MUST return the sub-optimizers' live group dicts (not copies): this is a fresh
        # list each call, but its elements are the real dicts, so the scheduler's
        # `group["lr"] = lr` write (see `_accumulate_and_step`) reaches both real
        # optimizers — Muon then re-derives its effective lr via `lr_scale` in `step()`.
        groups = []
        if self.adam is not None:
            groups += self.adam.param_groups
        if self.muon is not None:
            groups += self.muon.param_groups
        return groups

    def zero_grad(self, set_to_none: bool = True):
        if self.adam is not None:
            self.adam.zero_grad(set_to_none=set_to_none)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if self.adam is not None:
            self.adam.step()
        if self.muon is not None:
            self.muon.step()
        return closure() if closure is not None else None

    def state_dict(self):
        return {
            "adam": self.adam.state_dict() if self.adam is not None else None,
            "muon": self.muon.state_dict() if self.muon is not None else None,
        }

    def load_state_dict(self, sd):
        if self.adam is not None and sd.get("adam") is not None:
            self.adam.load_state_dict(sd["adam"])
        if self.muon is not None and sd.get("muon") is not None:
            self.muon.load_state_dict(sd["muon"])
