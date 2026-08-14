"""FSDP2 + in-node expert-parallel plumbing for the CUDA backend (#271). Below the
seam — imports torch/torch.distributed. This is the SEVENTH module allowed to touch a
hardware library (see the seam list in `CLAUDE.md`, updated alongside this file).

Everything here is process-group lifecycle, mesh construction, and collectives — the
*mechanism*. The policy math (partition sizes, who owns which expert) lives in the
portable `src/train/parallel.py`; this module supplies the real `torch.distributed`
collectives that policy's injected callables (`reduce_loads`'s `all_reduce_fn`) need.

Host note (see `.claude/plans/issue-271.md`): `fully_shard` MUST always be called with
an explicit `mesh=` — the no-mesh form resolves the default accelerator and crashes on
macOS (`torch.mps` has no `is_initialized`), which would take `full-macos` CI red. Every
function below that builds an FSDP unit takes `mesh` explicitly for that reason.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard


# --------------------------------------------------------------------------- #
# Process-group lifecycle
# --------------------------------------------------------------------------- #
def is_distributed() -> bool:
    """True once `init_distributed()` has set up a live default process group."""
    return dist.is_available() and dist.is_initialized()


def init_distributed(*, backend: Optional[str] = None) -> None:
    """Initialize the default process group from `torchrun`'s environment
    (`RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT`).

    `backend` defaults to `"nccl"` on a real CUDA device, `"gloo"` otherwise (the CPU
    path this repo's gloo-sim tests and `scripts/train_dist.py --nproc_per_node=N` on a
    CPU host both use). A no-op if already initialized — safe to call from both
    `scripts/train_dist.py` and a test's own setup without double-init errors.
    """
    if is_distributed():
        return
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def shutdown() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_primary() -> bool:
    """True on rank 0 (or when not distributed at all) — the rank allowed to write the
    committed checkpoint slot (see `src/train/checkpoint.py`'s `is_primary` param)."""
    return get_rank() == 0


# --------------------------------------------------------------------------- #
# Device mesh: a (dp_size, ep_size) 2D mesh. `mesh["dp"]` is what FSDP2 shards params
# over; `mesh["ep"]` (via `.get_group()`) is the process group `MoEBlock`'s all-to-all
# dispatch uses. ep_size == 1 degenerates to a pure-dp 1D-equivalent mesh.
# --------------------------------------------------------------------------- #
def build_mesh(dp_size: int, ep_size: int) -> DeviceMesh:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if dp_size * ep_size != get_world_size():
        raise ValueError(
            f"dp_size ({dp_size}) * ep_size ({ep_size}) = {dp_size * ep_size} does not "
            f"match world_size ({get_world_size()})."
        )
    return init_device_mesh(device_type, (dp_size, ep_size), mesh_dim_names=("dp", "ep"))


def ep_group(mesh: DeviceMesh):
    """The process group spanning the EP dimension of `mesh` (`None` at ep_size == 1's
    trivial single-rank groups, which is still a valid, degenerate group to pass
    through — `dist.all_to_all_single` on a size-1 group is a no-op copy)."""
    return mesh["ep"].get_group()


def dp_mesh(mesh: DeviceMesh) -> DeviceMesh:
    return mesh["dp"]


# --------------------------------------------------------------------------- #
# FSDP2 wrapping
# --------------------------------------------------------------------------- #
def wrap_backbone(model, mesh: DeviceMesh, *, reshard_after_forward: bool = True) -> Any:
    """Wrap `model`'s layers + remaining top-level leaves with FSDP2 (`fully_shard`) —
    ZeRO-2 (`reshard_after_forward=False`, keep params materialized after forward, less
    comm — the small-rung default) or ZeRO-3-equivalent (`True` — Large A). `mesh` is
    ALWAYS passed explicitly (see module docstring).

    Wraps `model.layers[i]` individually, then `model.embedding` (and `model.lm_head`
    when untied) — NOT a final root `fully_shard(model, ...)` call. This differs from
    the textbook "per-layer + root" FSDP2 recipe; the deviation is a REAL finding from
    this host, not a simplification:

    `fully_shard` implements itself by swapping `module.__class__` to a dynamically
    injected `FSDP<Cls>` subclass. `CUDAMambaModel(ModelInterface, nn.Module)` inherits
    from `ModelInterface(ABC)`; the ABC metaclass gives its instances a different
    internal layout (`_abc_impl`) than a plain `nn.Module`, and that class-swap raises
    `TypeError: __class__ assignment: ... object layout differs` — reproduced in-session
    with a minimal `class M(ABC, nn.Module)` wrapped in `fully_shard`, independent of
    anything else in this model. Per-layer blocks (`MambaBlock`/`AttentionBlock`/
    `MoEBlock`) are plain `nn.Module` subclasses and wrap fine; only the ABC-rooted
    top-level model cannot itself be a `fully_shard` target.

    Wrapping `model.embedding` directly (rather than via a root call) still gives the
    tied embedding (`_head` reads `self.embedding.weight.t()`) shard-consistency: both
    reads hit the SAME sharded parameter object, not two independently-resharded
    copies — the property the textbook root-wrap would have given for free.
    `model.norm_f` (a handful of scalars) is deliberately left unsharded — there is
    nothing to save there.
    """
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=reshard_after_forward)
    fully_shard(model.embedding, mesh=mesh, reshard_after_forward=reshard_after_forward)
    if not model._tie_embeddings:
        fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=reshard_after_forward)
    return model


# --------------------------------------------------------------------------- #
# Loss-Free-Balancing: all-reduce load counts BEFORE MoEBalancer.update (#271's
# sharpest correctness risk — see src/train/moe_balance.py and cuda_train_step.py).
# --------------------------------------------------------------------------- #
_MAX_EXACT_FP32_SUM = 2 ** 24   # exact integer range in fp32; see the docstring below


def all_reduce_loads(loads: List[List[float]], *, group=None) -> List[List[float]]:
    """Sum per-expert token counts across every rank in `group` (default: WORLD — every
    rank sees the SAME shard's disagreement, and only a world-wide reduction fixes it,
    not just a dp or ep sub-group).

    Flattens `[n_layers][n_experts]` into one fp32 tensor and does ONE
    `dist.all_reduce(SUM)`, not one per layer. Counts are integers carried in fp32; a
    SUM of integers is exact below `2**24` (~16.7M) — asserted here rather than assumed,
    because a silently-inexact sum would make `MoEBalancer.update`'s `sign()` diverge
    between ranks again, exactly the bug this function exists to prevent.
    """
    if not loads:
        return []
    n_layers = len(loads)
    n_experts = len(loads[0]) if loads[0] else 0
    flat = torch.tensor([v for row in loads for v in row], dtype=torch.float32)
    if flat.numel() and float(flat.abs().max()) >= _MAX_EXACT_FP32_SUM:
        raise ValueError(
            f"a per-expert load count already exceeds the fp32-exact-sum bound "
            f"({_MAX_EXACT_FP32_SUM}) BEFORE reducing — the all_reduce below can no "
            "longer guarantee a bit-identical sum across ranks."
        )
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    if float(flat.abs().max()) >= _MAX_EXACT_FP32_SUM:
        raise ValueError(
            f"reduced per-expert load count exceeds the fp32-exact-sum bound "
            f"({_MAX_EXACT_FP32_SUM}) — every rank's bias update is no longer "
            "guaranteed to agree bit-for-bit; lower the log/diagnostic cadence or "
            "shrink the effective batch."
        )
    out = flat.tolist()
    return [out[i * n_experts:(i + 1) * n_experts] for i in range(n_layers)]


def make_load_reduce_fn(group=None) -> Callable[[List[List[float]]], List[List[float]]]:
    """Bind `all_reduce_loads` to `group`, matching the `load_reduce` signature
    `cuda_train_step._accumulate_and_step` expects (`loads -> loads`, no group arg)."""
    return lambda loads: all_reduce_loads(loads, group=group)


# --------------------------------------------------------------------------- #
# Portable state-dict gather: DTensor -> full tensor (FSDP2) then, if the model is
# expert-parallel, merge every rank's LOCAL expert shard into one dict (EP). Both
# stages are collectives — EVERY rank must call this, even though only the primary
# rank writes the result to disk (a rank-0-only call deadlocks the others).
# --------------------------------------------------------------------------- #
def gather_portable_state_dict(model, *, ep_process_group=None) -> dict:
    sd = model._portable_state_dict()   # DTensor entries already .full_tensor()'d inside
    if ep_process_group is not None and dist.get_world_size(ep_process_group) > 1:
        gathered = [None] * dist.get_world_size(ep_process_group)
        dist.all_gather_object(gathered, sd, group=ep_process_group)
        merged: dict = {}
        for part in gathered:
            merged.update(part)
        sd = merged
    return sd


# --------------------------------------------------------------------------- #
# Sharded-optimizer resume via torch.distributed.checkpoint (DCP): reshards DTensor-
# backed (FSDP) optimizer/model state across a CHANGED world size for free. EP-sharded
# expert state is the documented exception (see module docstring / plan step 7): each
# rank's local experts are distinct parameters, not shards of one tensor, so a changed
# `ep_size` cannot be resharded automatically and must raise instead of silently
# dropping or duplicating expert state.
# --------------------------------------------------------------------------- #
def _real_optimizers(optimizer) -> List[torch.optim.Optimizer]:
    # `HybridOptimizer.real_optimizers()` (#271) unwraps to the underlying real
    # `torch.optim.Optimizer`s DCP's helpers need; a plain optimizer (the `adamw`-only
    # config path) already IS one.
    real = getattr(optimizer, "real_optimizers", None)
    return real() if real is not None else [optimizer]


def _meta_path(path: str) -> str:
    return os.path.join(str(path), "dcp_meta.json")


def save_resume_dcp(model, optimizer, path: str, *, step: int, world_size: int,
                    ep_size: int, extra: Optional[dict] = None) -> None:
    """Save model + optimizer state via DCP (world-size-resharding-capable), plus a
    small plain-JSON sidecar (`dcp_meta.json`) recording `step`/`world_size`/`ep_size`
    so a resuming run can check EP-degree compatibility BEFORE attempting the (for EP,
    unsupported) reshard. Every rank must call this — DCP's save is collective."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    optims = _real_optimizers(optimizer)
    model_sd, optim_sd = get_state_dict(model, optims)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(path))
    if is_primary():
        meta = {"step": int(step), "world_size": int(world_size), "ep_size": int(ep_size)}
        if extra:
            meta.update(extra)
        os.makedirs(str(path), exist_ok=True)
        with open(_meta_path(path), "w") as f:
            json.dump(meta, f)


def load_resume_dcp(model, optimizer, path: str, *, ep_size: int) -> dict:
    """Load a DCP resume bundle written by `save_resume_dcp`, resharding model/optimizer
    DTensor state to THIS run's (possibly different) world size for free. Raises before
    touching any collective if the bundle's `ep_size` differs from this run's — EP
    reshard is unsupported (see module docstring), and DCP would otherwise silently
    load whatever FQNs happen to match, which is exactly the silent-drop/duplicate
    failure #271 forbids."""
    with open(_meta_path(path)) as f:
        meta = json.load(f)
    saved_ep = int(meta.get("ep_size", 1))
    if saved_ep != ep_size:
        raise ValueError(
            f"resume bundle at {path!r} was saved at ep_size={saved_ep}, this run has "
            f"ep_size={ep_size}. Expert-parallel state is sharded by RANK (distinct "
            "parameters per rank, not shards of one tensor), so it cannot be "
            "automatically resharded across a changed EP degree — resume with the same "
            "--ep-size, or start a fresh run."
        )
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    optims = _real_optimizers(optimizer)
    model_sd, optim_sd = get_state_dict(model, optims)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, optims, model_state_dict=state["model"], optim_state_dict=state["optim"])
    return meta


def has_resume_dcp(path: str) -> bool:
    return os.path.exists(_meta_path(path))
