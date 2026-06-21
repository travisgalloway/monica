"""CUDA / PyTorch distillation train step (#100) — torch port of `mlx_distill.py`.

The staged distillation matching: a student is trained against the frozen teacher through the
manifest's `stages` (`mixing-match -> hidden-align -> logit-distill`). Each stage is a different
`TrainStepFn` built by `make_distill_train_step(..., stage=...)`; the driver loops the stages
swapping the step. The objective-specific loss is computed in torch; accumulation / fp16 scaling /
clipping / the optimizer step are shared via `cuda_train_step._accumulate_and_step` (so the four
training objectives and distillation all funnel through one tail).

Stage micro-batch tuples (same as the MLX backend):
  * logit-distill : (inputs, targets, topk_vals, topk_idx)  — cached teacher top-k (#94).
  * hidden-align  : (inputs, teacher_hidden) cached, OR (inputs,) on-the-fly (teacher recompute).
  * mixing-match  : (inputs,) on-the-fly (teacher attention matrices recomputed).

This file imports `torch`; it lives below the seam and nothing portable imports it.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .cuda_train_step import _accumulate_and_step
from ..train.distill_manifest import DistillStage


def _align(i: int, n_src: int, n_dst: int) -> int:
    """Map index `i` in a depth-`n_src` stack onto a depth-`n_dst` one, endpoint-to-endpoint."""
    if n_src <= 1:
        return 0
    return min(n_dst - 1, max(0, int(round(i * (n_dst - 1) / (n_src - 1)))))


def _kl_topk(student_logits: torch.Tensor, topk_vals: torch.Tensor, topk_idx: torch.Tensor,
             temperature: float) -> torch.Tensor:
    """T^2-scaled KL(teacher || student) over the teacher's top-k support (Hinton scaling).

    `student_logits` (B,L,V) fp32; `topk_vals`/`topk_idx` (B,L,k). Teacher and student are each
    softmaxed over the SAME k indices at temperature T. Torch port of `mlx_distill._kl_topk`."""
    T = temperature
    p = F.softmax(topk_vals.float() / T, dim=-1)                # teacher over support
    sg = torch.gather(student_logits, -1, topk_idx.long()) / T  # student logits at the k indices
    logq = sg - torch.logsumexp(sg, dim=-1, keepdim=True)       # student log-softmax / k
    kl = (p * (torch.log(p + 1e-9) - logq)).sum(dim=-1)         # (B,L)
    return (T * T) * kl.mean()


def _ce(logits: torch.Tensor, targets) -> torch.Tensor:
    V = logits.shape[-1]
    t = torch.as_tensor(np.asarray(targets), dtype=torch.long, device=logits.device).reshape(-1)
    return F.cross_entropy(logits.reshape(-1, V), t, reduction="mean")


def _hidden_mse(student_hs, teacher_hs, layers: List[int], n_s: int, n_t: int) -> torch.Tensor:
    """Mean MSE between aligned student/teacher hidden states over the overlapping channels
    (widths differ; compare the shared `min(d)`). Torch port of `mlx_distill._hidden_mse`."""
    terms = []
    for s in layers:
        th_idx = 0 if s == 0 else 1 + _align(s - 1, n_s, n_t)
        th = teacher_hs[th_idx]
        sh = student_hs[s]
        m = min(sh.shape[-1], th.shape[-1])
        diff = sh[..., :m].float() - th[..., :m].float().to(sh.device)
        terms.append((diff * diff).mean())
    return torch.stack(terms).mean()


def _mixing_mse(student_mats, teacher_mats, n_s: int, n_t: int) -> torch.Tensor:
    """Mean MSE between each student Mamba layer's head-averaged mixing matrix and the
    depth-aligned teacher attention matrix (both (B,L,L)). Torch port."""
    terms = []
    for (i, sm) in student_mats:
        tm = teacher_mats[_align(i, n_s, n_t)]
        diff = sm.float() - tm.float().to(sm.device)
        terms.append((diff * diff).mean())
    return torch.stack(terms).mean()


def make_distill_train_step(model, optimizer, *, stage, teacher=None,
                            ce_weight: float = 0.1, kl_weight: float = 0.9,
                            temperature: float = 2.0, hidden_layers: Optional[List[int]] = None,
                            grad_clip: float = 1.0, scaler=None) -> Callable:
    """Build a distillation `train_step(model, micro_batches, lr) -> dict` for one `stage`.

    `stage` is a `DistillStage` (or its string). `teacher` (a frozen `ConversionTeacher`) is
    required for on-the-fly `hidden-align`/`mixing-match`; `logit-distill` and cached
    `hidden-align` need no teacher. Torch mirror of `mlx_distill.make_distill_train_step`.
    """
    if not isinstance(stage, DistillStage):
        stage = DistillStage.from_str(str(stage))
    n_s = model.config.n_layers
    params = list(model.parameters())
    dev = model._device

    if stage == DistillStage.LOGIT_DISTILL:
        def _loss(mb) -> torch.Tensor:
            inputs, targets, tv, ti = mb
            logits = model.forward(inputs).float()                      # (B,L,V)
            tv_t = torch.as_tensor(np.asarray(tv), dtype=torch.float32, device=logits.device)
            ti_t = torch.as_tensor(np.asarray(ti), dtype=torch.long, device=logits.device)
            return ce_weight * _ce(logits, targets) + kl_weight * _kl_topk(
                logits, tv_t, ti_t, temperature)

    elif stage == DistillStage.HIDDEN_ALIGN:
        n_t = teacher.n_layers if teacher is not None else None
        layers = hidden_layers if hidden_layers is not None else list(range(1, n_s + 1))
        if teacher is not None:                                          # on-the-fly recompute
            def _loss(mb) -> torch.Tensor:
                inputs = mb[0]
                with torch.no_grad():
                    t_hs = [h.detach().to(dev)
                            for h in teacher.forward(inputs, return_hidden=True).hidden_states]
                return _hidden_mse(model.hidden_states(inputs), t_hs, layers, n_s, n_t)
        else:                                                           # cached teacher hidden
            def _loss(mb) -> torch.Tensor:
                inputs, teacher_hidden = mb
                t_hs = [torch.as_tensor(np.asarray(h), device=dev) for h in teacher_hidden]
                nt = len(t_hs) - 1
                return _hidden_mse(model.hidden_states(inputs), t_hs, layers, n_s, nt)

    elif stage == DistillStage.MIXING_MATCH:
        if teacher is None:
            raise ValueError("mixing-match requires a teacher (on-the-fly attention matrices)")
        if all(model.config.is_attention_layer(i) for i in range(n_s)):
            raise ValueError("mixing-match has no Mamba layers to match in a pure-attention "
                             "layout (attn_every covers every layer); skip this stage")
        n_t = teacher.n_layers

        def _loss(mb) -> torch.Tensor:
            inputs = mb[0]
            with torch.no_grad():
                t_mats = [m.detach().to(dev) for m in teacher.attention_matrices(inputs)]
            return _mixing_mse(model.mixing_matrices(inputs), t_mats, n_s, n_t)
    else:                                                               # unreachable
        raise ValueError(f"unknown distill stage {stage!r}")

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(optimizer, params, _loss, micro_batches, lr,
                                    grad_clip, scaler)

    return train_step
