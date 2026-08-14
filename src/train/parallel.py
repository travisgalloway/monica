"""FSDP/expert-parallel policy layer (#271) — portable, never imports a backend.

The number/policy math for parallel training: how many ranks exist, how experts split
across an expert-parallel (EP) group, and the injection point for reducing per-expert
load counts across ranks BEFORE `MoEBalancer.update` sees them (the sharpest
correctness risk #271 names — see `src/train/moe_balance.py`). Mirrors the
`loss_scale.py` / `moe_balance.py` precedent: hardware-free policy objects/functions,
unit-testable with no backend, ready for `src/model/cuda_distributed.py` (below the
seam) to supply the real collectives via injected callables.

`world_size == 1` (and `ep_size == 1`) is a FIRST-CLASS case here, not an afterthought:
every function degenerates to the identity so single-process training (today's entire
test suite) stays byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence


@dataclass(frozen=True)
class ParallelConfig:
    """`world_size` ranks split into `dp_size = world_size // ep_size` data-parallel
    replicas, each an `ep_size`-way expert-parallel shard. `n_experts`, if given, is
    validated against `ep_size` up front (fail at config time, not deep in a forward
    pass) — optional because callers that only need the dp/ep arithmetic (e.g. the
    FSDP mesh shape) may not have a model config in hand yet.
    """

    world_size: int
    ep_size: int = 1
    n_experts: Optional[int] = None

    def __post_init__(self):
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if self.ep_size < 1:
            raise ValueError(f"ep_size must be >= 1, got {self.ep_size}")
        if self.world_size % self.ep_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by ep_size "
                f"({self.ep_size}) — every data-parallel replica needs an identical "
                "expert-parallel shard layout, or some ranks would get a phantom "
                "expert-parallel role."
            )
        if self.n_experts is not None and self.n_experts % self.ep_size != 0:
            raise ValueError(
                f"n_experts ({self.n_experts}) must be divisible by ep_size "
                f"({self.ep_size}) — a non-divisible split would give some EP ranks "
                "an extra expert."
            )

    @property
    def dp_size(self) -> int:
        return self.world_size // self.ep_size


def expert_partition(n_experts: int, ep_size: int) -> List[List[int]]:
    """Global expert ids per EP rank, contiguous blocks: `[[0..n/ep), [n/ep..2n/ep), ...]`.

    `ep_size == 1` returns `[[0, 1, ..., n_experts - 1]]` — the degenerate, everything-
    local case that keeps a non-distributed `MoEBlock` unchanged.
    """
    if n_experts < 1:
        raise ValueError(f"n_experts must be >= 1, got {n_experts}")
    if ep_size < 1:
        raise ValueError(f"ep_size must be >= 1, got {ep_size}")
    if n_experts % ep_size != 0:
        raise ValueError(
            f"n_experts ({n_experts}) must be divisible by ep_size ({ep_size}) — a "
            "non-divisible split would give some EP ranks an extra expert."
        )
    per_rank = n_experts // ep_size
    return [list(range(r * per_rank, (r + 1) * per_rank)) for r in range(ep_size)]


def owner_rank(expert_id: int, n_experts: int, ep_size: int) -> int:
    """Which EP rank owns global expert `expert_id`, under the contiguous-block split
    `expert_partition` uses. Raises on an out-of-range id or a non-divisible split —
    same validation as `expert_partition`, so the two never silently disagree."""
    if n_experts % ep_size != 0:
        raise ValueError(
            f"n_experts ({n_experts}) must be divisible by ep_size ({ep_size})."
        )
    if not (0 <= expert_id < n_experts):
        raise ValueError(f"expert_id {expert_id} out of range [0, {n_experts})")
    per_rank = n_experts // ep_size
    return expert_id // per_rank


def local_index(expert_id: int, n_experts: int, ep_size: int) -> int:
    """`expert_id`'s position WITHIN its owning rank's shard (0-indexed)."""
    if n_experts % ep_size != 0:
        raise ValueError(
            f"n_experts ({n_experts}) must be divisible by ep_size ({ep_size})."
        )
    if not (0 <= expert_id < n_experts):
        raise ValueError(f"expert_id {expert_id} out of range [0, {n_experts})")
    per_rank = n_experts // ep_size
    return expert_id % per_rank


def reduce_loads(loads: Sequence[Sequence[float]],
                 all_reduce_fn: Optional[Callable[[List[List[float]]], List[List[float]]]] = None
                 ) -> List[List[float]]:
    """Sum per-expert load counts across every rank BEFORE `MoEBalancer.update` sees
    them — the fix for #271's sharpest correctness risk (see `src/train/moe_balance.py`
    module docstring and `src/model/cuda_train_step.py`'s `load_reduce` hook).

    `loads` is `[n_moe_layers][n_experts]`, THIS rank's counts. `all_reduce_fn` is the
    injected collective (`cuda_distributed.all_reduce_loads`, below the seam); `None` —
    the non-distributed / `world_size == 1` default — is the identity, so single-process
    training stays byte-identical to before #271. This function is intentionally a thin
    seam/call-site, not the reduction itself: the real collective sums the WHOLE
    `[n_layers][n_experts]` tensor in one all_reduce, not a Python-level loop.
    """
    rows = [list(row) for row in loads]
    if all_reduce_fn is None:
        return rows
    return all_reduce_fn(rows)
