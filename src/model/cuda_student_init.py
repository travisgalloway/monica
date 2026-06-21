"""CUDA / PyTorch student initialization from a frozen teacher (#99) — torch port of
`mlx_student_init.py`.

Turns the transformer teacher (#93) into the Mamba-2 hybrid student by one of two methods
(`docs/design/10-distillation.md`):

  * **Mamba-in-the-Llama** (`InitMethod.MAMBA_IN_THE_LLAMA`): init the Mamba layers from the
    teacher's attention projections (Q->C, K->B, V->input, O->output), copy the kept attention
    layers, and FREEZE them. Reference: arXiv:2408.15237.
  * **MOHAWK** (`InitMethod.MOHAWK`): copy attention layers where present, leave Mamba at default
    init, freeze nothing; the work is the progressive matching the distill loss runs (#100).

The mapping is adaptive (`_fit`): exact copy where dims align, else copy the overlapping block
and zero-pad/truncate. nn.Linear weight layout is `(out, in)` in BOTH torch and MLX, so the
per-projection shapes match the MLX port verbatim. Freezing uses `requires_grad_(False)`: the
optimizer holds `model.parameters()` but frozen params get no grad, so `cuda_train_step`'s grad
norm / AdamW step skip them (the torch analogue of MLX's `freeze` + `value_and_grad`).

This file imports `torch`; it lives below the seam and nothing portable imports it.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

from ..train.distill_manifest import InitMethod, InitReport


def _to_t(arr) -> torch.Tensor:
    """Teacher projections are torch tensors (CUDATeacher); accept numpy too for safety."""
    return arr if isinstance(arr, torch.Tensor) else torch.as_tensor(arr)


def _fit(src: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Adaptive copy of `src` into an array of `shape`: keep the overlapping region per axis,
    zero-pad/truncate the rest. Exact (returns `src`'s values) when shapes already match."""
    crop = src[tuple(slice(0, min(s, d)) for s, d in zip(src.shape, shape))]
    # F.pad consumes pad amounts from the LAST axis backward: [l_last, r_last, l_prev, r_prev, ...]
    pad: List[int] = []
    for d, c in zip(reversed(shape), reversed(crop.shape)):
        pad.extend([0, d - c])
    return F.pad(crop, pad)


def _expand_kv(w: torch.Tensor, n_heads: int, n_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Expand a GQA key/value projection (n_kv_heads*head_dim, d) to full MHA width
    (n_heads*head_dim, d) by repeating each kv head's block (matches inference-time repeat)."""
    if n_kv_heads == n_heads:
        return w
    rep = n_heads // n_kv_heads
    d = w.shape[-1]
    return w.reshape(n_kv_heads, head_dim, d).repeat_interleave(rep, dim=0).reshape(
        n_heads * head_dim, d)


def _copy(param: torch.nn.Parameter, value: torch.Tensor) -> None:
    """Copy `value` into `param` in place (shape must already match — `_fit` ensures it)."""
    with torch.no_grad():
        param.copy_(value.to(param.dtype))


def _teacher_layer_for(i: int, n_student: int, n_teacher: int) -> int:
    """Align student depth onto teacher depth, endpoint-to-endpoint (evenly spaced)."""
    if n_student <= 1:
        return 0
    return min(n_teacher - 1, int(round(i * (n_teacher - 1) / (n_student - 1))))


def _init_embeddings(student, teacher) -> None:
    """Copy the teacher's token embedding (and, untied, the lm_head) onto the student, cropping
    with `_fit` — making the student residual stream the teacher's first-`d_model` subspace."""
    cfg = student.config
    shape = (cfg.vocab_size, cfg.d_model)
    _copy(student.embedding.weight, _fit(_to_t(teacher.embedding_matrix()), shape))
    if not cfg.tie_embeddings:    # student.lm_head exists only when untied (cuda_backend)
        _copy(student.lm_head.weight, _fit(_to_t(teacher.lm_head_matrix()), shape))


def _init_attention_layer(layer, proj, tcfg) -> None:
    """Copy a teacher attention layer's Q/K/V/O onto a student attention block (`qkv_proj`,
    `o_proj`), expanding GQA to full heads and adaptively fitting to the student's width."""
    q = _to_t(proj.q)
    k = _expand_kv(_to_t(proj.k), tcfg.n_heads, tcfg.n_kv_heads, tcfg.head_dim)
    v = _expand_kv(_to_t(proj.v), tcfg.n_heads, tcfg.n_kv_heads, tcfg.head_dim)
    d_attn, d_model = layer.qkv_proj.weight.shape[0] // 3, layer.qkv_proj.weight.shape[1]
    blocks = [_fit(w, (d_attn, d_model)) for w in (q, k, v)]
    _copy(layer.qkv_proj.weight, torch.cat(blocks, dim=0))           # (3*d_attn, d_model)
    _copy(layer.o_proj.weight, _fit(_to_t(proj.o), tuple(layer.o_proj.weight.shape)))


def _init_mamba_layer(layer, proj, dt_rank: int, d_state: int) -> None:
    """Mamba-in-the-Llama mapping onto one Mamba block: Q->C, K->B (the d_state slices of
    `x_proj`), V->input (`in_proj` main half), O->`out_proj`. Untouched rows keep their init."""
    d_inner = layer.out_proj.weight.shape[1]
    # x_proj: rows are [dt(dt_rank) | B(d_state) | C(d_state)] -> set B from K, C from Q.
    xw = layer.ssm.x_proj.weight
    B_new = _fit(_to_t(proj.k), (d_state, d_inner))
    C_new = _fit(_to_t(proj.q), (d_state, d_inner))
    _copy(layer.ssm.x_proj.weight, torch.cat([xw[:dt_rank], B_new, C_new], dim=0))
    # in_proj: rows are [main(d_inner) | gate(d_inner)] -> set main from V, keep gate.
    iw = layer.in_proj.weight
    main_new = _fit(_to_t(proj.v), (d_inner, iw.shape[1]))
    _copy(layer.in_proj.weight, torch.cat([main_new, iw[d_inner:]], dim=0))
    # out_proj <- O.
    _copy(layer.out_proj.weight, _fit(_to_t(proj.o), tuple(layer.out_proj.weight.shape)))


def init_student(student, teacher, method: InitMethod) -> InitReport:
    """Initialize `student` (a `CUDAMambaModel`) from `teacher` (a `ConversionTeacher`).

    Returns an `InitReport`. For Mamba-in-the-Llama the kept attention layers are frozen
    (`requires_grad_(False)`); for MOHAWK nothing is frozen.
    """
    if not isinstance(method, InitMethod):
        method = InitMethod.from_str(str(method))
    cfg = student.config
    tcfg = teacher.config
    S, T = cfg.n_layers, teacher.n_layers
    dt_rank, d_state = cfg.dt_rank_resolved, cfg.d_state

    _init_embeddings(student, teacher)

    frozen_layers: List[int] = []
    n_mapped = 0
    for i in range(S):
        proj = teacher.attention_projection(_teacher_layer_for(i, S, T))
        is_attn = cfg.is_attention_layer(i)
        if is_attn:
            _init_attention_layer(student.layers[i], proj, tcfg)
            n_mapped += 1
            if method == InitMethod.MAMBA_IN_THE_LLAMA:
                student.layers[i].requires_grad_(False)
                frozen_layers.append(i)
        elif method == InitMethod.MAMBA_IN_THE_LLAMA:
            _init_mamba_layer(student.layers[i], proj, dt_rank, d_state)
            n_mapped += 1
        # MOHAWK leaves Mamba layers at their default init (refined by progressive matching #100).

    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    return InitReport(
        method=method.value, n_layers_mapped=n_mapped,
        n_frozen_params=total - trainable, n_trainable_params=trainable,
        frozen_layers=frozen_layers,
    )
