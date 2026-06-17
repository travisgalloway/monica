"""MLX student initialization from a frozen teacher (Apple Silicon, below the seam).

Phase 2 of the distillation (#99): turn the transformer teacher (#93) into the Mamba-2 hybrid
student by one of two methods (`docs/design/10-distillation.md`):

  * **Mamba-in-the-Llama** (`InitMethod.MAMBA_IN_THE_LLAMA`): initialize the Mamba layers from the
    teacher's attention projections — Q -> C, K -> B (the two d_state slices of `x_proj`),
    V -> input (`in_proj` main half), O -> `out_proj` — copy the kept attention layers from the
    teacher, and FREEZE them. The student is a pure Mamba+attention hybrid with no MLP blocks, so
    the retained attention layers play the role the paper's frozen MLPs play (the trainable set is
    the new Mamba layers). Reference: arXiv:2408.15237.
  * **MOHAWK** (`InitMethod.MOHAWK`): a lighter init (copy attention layers where present, leave
    Mamba at its default init, freeze nothing); the real work is the progressive matching the
    distill loss runs per `stages` (#100). Reference: arXiv:2408.10189.

Because the teacher (e.g. d_model 1536) and the swept student (e.g. 2048) differ in width, the
mapping is **adaptive**: exact copy where dims align, otherwise copy the overlapping block and
zero-pad/truncate to the student's shape (`_fit`). Init quality is validated downstream by the
distillation curve, not by exactness.

Freezing uses MLX's native `nn.Module.freeze`, which `nn.value_and_grad` already honors (it only
differentiates `trainable_parameters()`), so the #100 train step needs no change. This file
imports `mlx`; it lives below the seam and nothing portable imports it.
"""

from __future__ import annotations

from typing import List, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten

from ..train.distill_manifest import InitMethod, InitReport


def _fit(src: mx.array, shape: Tuple[int, ...]) -> mx.array:
    """Adaptive copy of `src` into an array of `shape`: keep the overlapping region per axis,
    zero-pad/truncate the rest. Exact (returns `src`'s values) when shapes already match."""
    crop = src[tuple(slice(0, min(s, d)) for s, d in zip(src.shape, shape))]
    pad = [(0, d - c) for d, c in zip(shape, crop.shape)]
    return mx.pad(crop, pad)


def _expand_kv(w: mx.array, n_heads: int, n_kv_heads: int, head_dim: int) -> mx.array:
    """Expand a GQA key/value projection (n_kv_heads*head_dim, d) to full MHA width
    (n_heads*head_dim, d) by repeating each kv head's block, matching inference-time repeat."""
    if n_kv_heads == n_heads:
        return w
    rep = n_heads // n_kv_heads
    d = w.shape[-1]
    return mx.repeat(w.reshape(n_kv_heads, head_dim, d), rep, axis=0).reshape(n_heads * head_dim, d)


def _set(param_owner, name: str, value: mx.array) -> None:
    setattr(param_owner, name, value)


def _teacher_layer_for(i: int, n_student: int, n_teacher: int) -> int:
    """Align student depth onto teacher depth, endpoint-to-endpoint (evenly spaced): student
    layer 0 -> teacher 0 and the last student layer -> the last teacher layer."""
    if n_student <= 1:
        return 0
    return min(n_teacher - 1, int(round(i * (n_teacher - 1) / (n_student - 1))))


def _init_attention_layer(layer, proj, tcfg) -> None:
    """Copy a teacher attention layer's Q/K/V/O onto a student attention block (`qkv_proj`,
    `o_proj`), expanding GQA to full heads and adaptively fitting to the student's width."""
    q = proj.q
    k = _expand_kv(proj.k, tcfg.n_heads, tcfg.n_kv_heads, tcfg.head_dim)
    v = _expand_kv(proj.v, tcfg.n_heads, tcfg.n_kv_heads, tcfg.head_dim)
    d_attn, d_model = layer.qkv_proj.weight.shape[0] // 3, layer.qkv_proj.weight.shape[1]
    blocks = [_fit(w, (d_attn, d_model)) for w in (q, k, v)]
    _set(layer.qkv_proj, "weight", mx.concatenate(blocks, axis=0))   # (3*d_attn, d_model)
    _set(layer.o_proj, "weight", _fit(proj.o, layer.o_proj.weight.shape))


def _init_mamba_layer(layer, proj, dt_rank: int, d_state: int) -> None:
    """Mamba-in-the-Llama mapping onto one Mamba block: Q->C, K->B (the d_state slices of
    `x_proj`), V->input (`in_proj` main half), O->`out_proj`. Untouched rows keep their init."""
    d_inner = layer.out_proj.weight.shape[1]
    # x_proj: rows are [dt(dt_rank) | B(d_state) | C(d_state)] -> set B from K, C from Q.
    xw = layer.ssm.x_proj.weight
    B_new = _fit(proj.k, (d_state, d_inner))
    C_new = _fit(proj.q, (d_state, d_inner))
    _set(layer.ssm.x_proj, "weight",
         mx.concatenate([xw[:dt_rank], B_new, C_new], axis=0))
    # in_proj: rows are [main(d_inner) | gate(d_inner)] -> set main from V, keep gate.
    iw = layer.in_proj.weight
    main_new = _fit(proj.v, (d_inner, iw.shape[1]))
    _set(layer.in_proj, "weight", mx.concatenate([main_new, iw[d_inner:]], axis=0))
    # out_proj <- O.
    _set(layer.out_proj, "weight", _fit(proj.o, layer.out_proj.weight.shape))


def init_student(student, teacher, method: InitMethod) -> InitReport:
    """Initialize `student` (an `MLXMambaModel`) from `teacher` (a `ConversionTeacher`).

    Returns an `InitReport`. For Mamba-in-the-Llama the kept attention layers are frozen; for
    MOHAWK nothing is frozen.
    """
    cfg = student.config
    tcfg = teacher.config
    S, T = cfg.n_layers, teacher.n_layers
    dt_rank, d_state = cfg.dt_rank_resolved, cfg.d_state

    frozen_layers: List[int] = []
    n_mapped = 0
    for i in range(S):
        proj = teacher.attention_projection(_teacher_layer_for(i, S, T))
        is_attn = cfg.is_attention_layer(i)
        if is_attn:
            _init_attention_layer(student.layers[i], proj, tcfg)
            n_mapped += 1
            if method == InitMethod.MAMBA_IN_THE_LLAMA:
                student.layers[i].freeze()
                frozen_layers.append(i)
        elif method == InitMethod.MAMBA_IN_THE_LLAMA:
            _init_mamba_layer(student.layers[i], proj, dt_rank, d_state)
            n_mapped += 1
        # MOHAWK leaves Mamba layers at their default init (refined by progressive matching #100).

    mx.eval(student.parameters())
    total = sum(v.size for _, v in tree_flatten(student.parameters()))
    trainable = sum(v.size for _, v in tree_flatten(student.trainable_parameters()))
    return InitReport(
        method=method.value, n_layers_mapped=n_mapped,
        n_frozen_params=total - trainable, n_trainable_params=trainable,
        frozen_layers=frozen_layers,
    )
