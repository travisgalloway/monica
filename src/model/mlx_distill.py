"""MLX distillation train steps (Apple Silicon, below the seam — may import mlx).

The staged distillation matching (#100): a student is trained against the frozen teacher (#93)
through the manifest's `stages` (`mixing-match -> hidden-align -> logit-distill`, #99). Each stage
is a different `TrainStepFn` built by `make_distill_train_step(..., stage=...)`; the driver loops
the stages (via `train.distill_manifest.distill_stages`) swapping the step. The backend-free loop
(`train.loop`) is unchanged — this mirrors SFT/DPO/GRPO and funnels through the shared
`_accumulate_and_step` (grad-accum, fp16 unscale + overflow-skip, clip, optimizer update) from
`mlx_train_step`.

The compound loss is a single scalar, so the existing `train.loss_scale.DynamicLossScaler` covers
it unchanged: the loss is scaled before backprop and the backend's inf/nan check skips overflowing
steps.

Stage micro-batch tuples:
  * logit-distill : (inputs, targets, topk_vals, topk_idx)  — cached teacher top-k (#94).
  * hidden-align  : (inputs, teacher_hidden) cached, OR (inputs,) on-the-fly (teacher recompute).
  * mixing-match  : (inputs,) on-the-fly (teacher attention matrices recomputed).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import mlx.core as mx
import mlx.nn as nn

from .mlx_train_step import _accumulate_and_step
from ..train.distill_manifest import DistillStage


def _align(i: int, n_src: int, n_dst: int) -> int:
    """Map index `i` in a depth-`n_src` stack onto the nearest index in a depth-`n_dst` one."""
    return min(n_dst - 1, max(0, int(round(i * n_dst / max(1, n_src)))))


def _kl_topk(student_logits: mx.array, topk_vals: mx.array, topk_idx: mx.array,
             temperature: float) -> mx.array:
    """T^2-scaled KL(teacher || student) over the teacher's top-k support (Hinton scaling).

    `student_logits` (B,L,V) fp32; `topk_vals`/`topk_idx` (B,L,k). Teacher and student are each
    softmaxed over the SAME k indices at temperature T, so the KL is computed on the shared
    support the cached signal (#94) provides."""
    T = temperature
    p = mx.softmax(topk_vals.astype(mx.float32) / T, axis=-1)            # teacher over support
    sg = mx.take_along_axis(student_logits, topk_idx.astype(mx.int32), axis=-1) / T
    logq = sg - mx.logsumexp(sg, axis=-1, keepdims=True)                 # student log-softmax/k
    kl = (p * (mx.log(p + 1e-9) - logq)).sum(axis=-1)                    # (B,L)
    return (T * T) * kl.mean()


def _ce(logits: mx.array, targets) -> mx.array:
    V = logits.shape[-1]
    t = mx.array(targets).reshape(-1).astype(mx.int32)
    return nn.losses.cross_entropy(logits.reshape(-1, V), t, reduction="mean")


def _hidden_mse(student_hs, teacher_hs, layers: List[int], n_s: int, n_t: int) -> mx.array:
    """Mean MSE between aligned student/teacher hidden states, over the overlapping channels
    (the teacher and student widths differ; compare the shared `min(d)` — a documented POC
    handling of the mismatch). `student_hs`/`teacher_hs` are (n+1)-length tuples (embedding +
    each layer); `layers` are student hidden indices to match."""
    terms = []
    for s in layers:
        th = teacher_hs[min(n_t, max(0, round(s * n_t / n_s)))]   # align embedding(0)+layers(1..n)
        sh = student_hs[s]
        m = min(sh.shape[-1], th.shape[-1])
        diff = sh[..., :m].astype(mx.float32) - th[..., :m].astype(mx.float32)
        terms.append(mx.mean(diff * diff))
    return mx.stack(terms).mean()


def _mixing_mse(student_mats, teacher_mats, n_s: int, n_t: int) -> mx.array:
    """Mean MSE between each student Mamba layer's head-averaged mixing matrix and the
    depth-aligned teacher attention matrix (both (B,L,L))."""
    terms = []
    for (i, sm) in student_mats:
        tm = teacher_mats[_align(i, n_s, n_t)]
        diff = sm.astype(mx.float32) - tm.astype(mx.float32)
        terms.append(mx.mean(diff * diff))
    return mx.stack(terms).mean()


def make_distill_train_step(model, optimizer, *, stage, teacher=None,
                            ce_weight: float = 0.1, kl_weight: float = 0.9,
                            temperature: float = 2.0, hidden_layers: Optional[List[int]] = None,
                            grad_clip: float = 1.0, scaler=None) -> Callable:
    """Build a distillation `train_step(model, micro_batches, lr) -> dict` for one `stage`.

    `stage` is a `DistillStage` (or its string). `teacher` (a frozen `ConversionTeacher`) is
    required for on-the-fly `hidden-align`/`mixing-match`; `logit-distill` and cached
    `hidden-align` need no teacher. Accumulation / fp16 scaling / clipping are shared via
    `_accumulate_and_step`.
    """
    if not isinstance(stage, DistillStage):
        stage = DistillStage.from_str(str(stage))
    n_s = model.config.n_layers

    if stage == DistillStage.LOGIT_DISTILL:
        def loss_fn(model, inputs, targets, tv, ti):
            logits = model.forward(inputs).astype(mx.float32)           # (B,L,V)
            loss = ce_weight * _ce(logits, targets) + kl_weight * _kl_topk(
                logits, mx.array(tv), mx.array(ti), temperature)
            return loss * scaler.scale if scaler else loss

    elif stage == DistillStage.HIDDEN_ALIGN:
        n_t = teacher.n_layers if teacher is not None else None
        layers = hidden_layers if hidden_layers is not None else list(range(1, n_s + 1))
        if teacher is not None:                                          # on-the-fly recompute
            def loss_fn(model, inputs):
                t_hs = [mx.stop_gradient(h)
                        for h in teacher.forward(inputs, return_hidden=True).hidden_states]
                loss = _hidden_mse(model.hidden_states(inputs), t_hs, layers, n_s, n_t)
                return loss * scaler.scale if scaler else loss
        else:                                                           # cached teacher hidden
            def loss_fn(model, inputs, teacher_hidden):
                t_hs = [mx.array(h) for h in teacher_hidden]
                nt = len(t_hs) - 1
                loss = _hidden_mse(model.hidden_states(inputs), t_hs, layers, n_s, nt)
                return loss * scaler.scale if scaler else loss

    elif stage == DistillStage.MIXING_MATCH:
        if teacher is None:
            raise ValueError("mixing-match requires a teacher (on-the-fly attention matrices)")
        n_t = teacher.n_layers

        def loss_fn(model, inputs):
            t_mats = [mx.stop_gradient(m) for m in teacher.attention_matrices(inputs)]
            loss = _mixing_mse(model.mixing_matrices(inputs), t_mats, n_s, n_t)
            return loss * scaler.scale if scaler else loss
    else:                                                               # unreachable
        raise ValueError(f"unknown distill stage {stage!r}")

    value_and_grad = nn.value_and_grad(model, loss_fn)

    def loss_and_grad(model, mb):
        return value_and_grad(model, *mb)

    def train_step(model, micro_batches, lr: float) -> dict:
        return _accumulate_and_step(model, optimizer, loss_and_grad, micro_batches,
                                    lr, grad_clip, scaler)

    return train_step
