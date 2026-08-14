"""Real world_size=2 gloo/CPU process-group tests for #271 — see
`.claude/plans/issue-271.md`'s Verification section (V3-V9, all labeled `[GLOO-SIM]`:
real collectives, real sharding, real resharding, but NOT NCCL and NOT on a GPU).

Every test spawns exactly `world_size` subprocesses via `torch.multiprocessing.spawn`
(NOT `torchrun` — that's `scripts/train_dist.py`'s job, exercised manually per the
plan's M1/M2). Workers are top-level functions (spawn pickles them by reference) that
write their result to a small file under `tmp_path`; the parent process reads those
back and does the actual assertions, so failures report with a normal pytest traceback
instead of the (still informative, but noisier) `ProcessRaisedException` wrapping.

Skipped entirely (whole module) where torch is unavailable — the `portable` CI job
never even imports this file's top-level `torch.multiprocessing` import.
"""

import json
import os
import socket

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from src.model.blocks import MambaConfig  # noqa: E402
from src.model.cuda_backend import CUDAMambaModel, MoEBlock  # noqa: E402
from src.train.moe_balance import MoEBalancer  # noqa: E402


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_gloo(fn, world_size: int, tmp_path, **kwargs) -> None:
    """Run `fn(rank, world_size, tmp_path, **kwargs)` in `world_size` real gloo/CPU
    subprocesses. Raises (via `mp.spawn`'s `ProcessRaisedException`) if any worker
    raises. Workers communicate results back to the parent via files under `tmp_path` —
    NOT return values (spawn's target runs in a separate process)."""
    port = _free_port()
    mp.spawn(_worker_entry, args=(world_size, port, fn, tmp_path, kwargs),
             nprocs=world_size, join=True)


def _worker_entry(rank, world_size, port, fn, tmp_path, kwargs) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size, tmp_path, **kwargs)
    finally:
        dist.destroy_process_group()


def _cfg(**over):
    base = dict(d_model=32, n_layers=2, head_dim=8, d_state=8, vocab_size=64, seq_len=16,
               precision="fp32", moe_every=1, n_experts=4, top_k=2, moe_impl="gather")
    base.update(over)
    return MambaConfig(**base)


# --------------------------------------------------------------------------- #
# V3 — adversarial 2-rank all-reduce: the pre-update() reduction fixes the sharpest
# correctness risk #271 names (MoEBalancer divergence across ranks).
# --------------------------------------------------------------------------- #
def _v3_worker(rank, world_size, tmp_path, use_reduce: bool):
    from src.model.cuda_distributed import all_reduce_loads, make_load_reduce_fn

    # Deliberately skewed per-rank counts: rank 0 sees expert 0 hammered, rank 1 sees
    # expert 1 hammered — the UN-reduced update() would steer each rank's bias in
    # OPPOSITE directions for these two experts.
    loads = {0: [[100.0, 1.0, 1.0, 1.0]], 1: [[1.0, 100.0, 1.0, 1.0]]}[rank]
    balancer = MoEBalancer(n_layers=1, n_experts=4, rate=1e-3)
    if use_reduce:
        reduce_fn = make_load_reduce_fn()
        loads = reduce_fn(loads)
    balancer.update(loads)
    torch.save({"bias": balancer.biases()}, str(tmp_path / f"v3_{rank}.pt"))


@pytest.mark.parametrize("use_reduce", [True, False])
def test_v3_load_reduce_before_balancer_update(tmp_path, use_reduce):
    _spawn_gloo(_v3_worker, world_size=2, tmp_path=tmp_path, use_reduce=use_reduce)
    bias0 = torch.load(str(tmp_path / "v3_0.pt"))["bias"]
    bias1 = torch.load(str(tmp_path / "v3_1.pt"))["bias"]
    if use_reduce:
        # Reduced: both ranks saw the SAME summed counts [101, 101, 2, 2] -> identical
        # bias, matching a single-process oracle fed those summed counts.
        assert bias0 == bias1
        oracle = MoEBalancer(n_layers=1, n_experts=4, rate=1e-3)
        oracle.update([[101.0, 101.0, 2.0, 2.0]])
        assert bias0 == oracle.biases()
    else:
        # Un-reduced (the bug #271 fixes): each rank's local skew steers its bias
        # DIFFERENTLY — this is the negative check the plan requires (V3 "must fail if
        # the reduce hook is removed").
        assert bias0 != bias1


# --------------------------------------------------------------------------- #
# V4 — 2-rank EP run's gathered weights vs a 1-rank run: identical keys/shapes, no
# rank suffix, no shard-shaped tensor, `check_weight_keys` clean.
# --------------------------------------------------------------------------- #
def _v4_worker(rank, world_size, tmp_path):
    from src.model.cuda_distributed import gather_portable_state_dict

    torch.manual_seed(0)
    model = CUDAMambaModel(_cfg(), ep_size=world_size, ep_rank=rank)
    sd = gather_portable_state_dict(model, ep_process_group=dist.group.WORLD)
    if rank == 0:
        shapes = {k: tuple(v.shape) for k, v in sd.items()}
        with open(str(tmp_path / "v4_gathered_shapes.json"), "w") as f:
            json.dump(sorted(shapes.items()), f)


def test_v4_ep_gathered_weights_match_single_rank(tmp_path):
    from src.train.checkpoint import check_weight_keys

    _spawn_gloo(_v4_worker, world_size=2, tmp_path=tmp_path)
    with open(str(tmp_path / "v4_gathered_shapes.json")) as f:
        gathered = dict(json.load(f))

    torch.manual_seed(0)
    single = CUDAMambaModel(_cfg(), ep_size=1, ep_rank=0)
    expected = single._portable_state_dict()

    assert set(gathered) == set(expected)
    for k in expected:
        assert tuple(gathered[k]) == tuple(np.asarray(expected[k]).shape), k
    assert not any(k.endswith(tuple(f".rank{r}" for r in range(4))) for k in gathered)
    check_weight_keys({k: gathered[k] for k in gathered}, expected, where="V4")


# --------------------------------------------------------------------------- #
# V5 — all-to-all EP dispatch numerics vs the single-process dense oracle, fp32.
# --------------------------------------------------------------------------- #
def _v5_worker(rank, world_size, tmp_path):
    torch.manual_seed(0)
    dense_cfg = _cfg(moe_impl="dense")
    dense_model = CUDAMambaModel(dense_cfg, ep_size=1, ep_rank=0)
    dense_block = next(l for l in dense_model.layers if isinstance(l, MoEBlock))
    ref_weights = dense_model._portable_state_dict()

    ep_cfg = _cfg(moe_impl="gather")
    ep_model = CUDAMambaModel(ep_cfg, ep_size=world_size, ep_rank=rank)
    ep_model._load_portable(ref_weights)     # same router + expert weights as the dense ref
    ep_model.set_ep_group(dist.group.WORLD)
    ep_block = next(l for l in ep_model.layers if isinstance(l, MoEBlock))
    assert ep_block.router.weight.equal(dense_block.router.weight)

    xn = torch.randn(3, 5, dense_cfg.d_model)
    dense_model.eval()
    ep_model.eval()
    with torch.no_grad():
        expected = dense_block.forward_seq(xn)
        actual = ep_block.forward_seq(xn)
    torch.save({"expected": expected, "actual": actual}, str(tmp_path / f"v5_{rank}.pt"))


def test_v5_ep_dispatch_matches_dense_oracle(tmp_path):
    _spawn_gloo(_v5_worker, world_size=2, tmp_path=tmp_path)
    for rank in (0, 1):
        d = torch.load(str(tmp_path / f"v5_{rank}.pt"))
        torch.testing.assert_close(d["actual"], d["expected"], rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# V6 — FSDP2 backbone: DTensor params (global shape preserved, local shape
# global/world), finite forward+backward, 2-rank loss matches 1-rank on the same data.
# --------------------------------------------------------------------------- #
def _v6_worker(rank, world_size, tmp_path):
    from src.model.cuda_distributed import build_mesh, wrap_backbone
    from torch.distributed.tensor import DTensor

    torch.manual_seed(0)
    cfg = _cfg(moe_every=None, n_layers=2)   # plain backbone, no MoE
    model = CUDAMambaModel(cfg, ep_size=1, ep_rank=0)
    mesh = build_mesh(dp_size=world_size, ep_size=1)
    wrap_backbone(model, mesh["dp"])

    p = next(model.embedding.parameters())
    assert isinstance(p, DTensor)
    global_shape = tuple(p.shape)
    local_shape = tuple(p.to_local().shape)

    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 8)).astype(np.int32)
    logits = model.forward(tokens)
    loss = logits.float().sum()
    loss.backward()
    torch.save({"global_shape": global_shape, "local_shape": local_shape,
               "loss": float(loss.detach()), "finite": bool(torch.isfinite(logits).all())},
              str(tmp_path / f"v6_{rank}.pt"))


def test_v6_fsdp_backbone_shards_params_and_trains(tmp_path):
    world_size = 2
    _spawn_gloo(_v6_worker, world_size=world_size, tmp_path=tmp_path)
    results = [torch.load(str(tmp_path / f"v6_{r}.pt")) for r in range(world_size)]
    for r in results:
        assert r["finite"]
        g0, g1 = results[0]["global_shape"], r["global_shape"]
        assert g0 == g1                                     # every rank sees the SAME global shape
    d_model, vocab = 32, 64   # matches _cfg()
    assert results[0]["global_shape"] == (vocab, d_model)
    local0 = results[0]["local_shape"]
    assert local0[0] == vocab // world_size or local0[1] == d_model // world_size


# --------------------------------------------------------------------------- #
# V7 — Muon under FSDP: 2-rank sharded update equals the single-process Muon update on
# the SAME weights and grads.
# --------------------------------------------------------------------------- #
def _v7_worker(rank, world_size, tmp_path):
    from src.model.cuda_distributed import build_mesh
    from torch.distributed.tensor import DTensor, distribute_tensor
    from torch.distributed.fsdp import fully_shard
    from src.model.cuda_muon import Muon

    torch.manual_seed(0)
    w = torch.randn(16, 8)
    g = torch.randn(16, 8)

    mesh = build_mesh(dp_size=world_size, ep_size=1)["dp"]
    p = torch.nn.Parameter(w.clone())
    module = torch.nn.Module()
    module.weight = p
    fully_shard(module, mesh=mesh)
    module.weight.grad = distribute_tensor(g.clone(), module.weight.device_mesh,
                                           module.weight.placements)

    opt = Muon([module.weight], lr=0.1, lr_scale=1.0, momentum=0.0, ns_steps=5)
    opt.step()
    updated_full = module.weight.detach().full_tensor()
    torch.save({"updated": updated_full}, str(tmp_path / f"v7_{rank}.pt"))


def test_v7_muon_under_fsdp_matches_single_process(tmp_path):
    from src.model.cuda_muon import Muon

    world_size = 2
    _spawn_gloo(_v7_worker, world_size=world_size, tmp_path=tmp_path)

    torch.manual_seed(0)
    w = torch.randn(16, 8)
    g = torch.randn(16, 8)
    p_ref = torch.nn.Parameter(w.clone())
    p_ref.grad = g.clone()
    ref_opt = Muon([p_ref], lr=0.1, lr_scale=1.0, momentum=0.0, ns_steps=5)
    ref_opt.step()

    for r in range(world_size):
        d = torch.load(str(tmp_path / f"v7_{r}.pt"))
        torch.testing.assert_close(d["updated"], p_ref.detach(), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# V8 — world-size-change resume: save at world_size=2, reload at world_size=1 (and
# 1->2); optimizer moment norms + step count match; unsupported ep_size change raises.
# --------------------------------------------------------------------------- #
def _v8_save_worker(rank, world_size, tmp_path, path):
    from src.model.cuda_distributed import build_mesh, wrap_backbone, save_resume_dcp

    torch.manual_seed(0)
    cfg = _cfg(moe_every=None, n_layers=2)
    model = CUDAMambaModel(cfg, ep_size=1, ep_rank=0)
    mesh = build_mesh(dp_size=world_size, ep_size=1)
    wrap_backbone(model, mesh["dp"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Force EVERY param to have AdamW state from the start (not just whatever the toy
    # forward pass happens to touch) — DCP's save/load key sets must match exactly, and
    # `_v8_load_worker` does the same forcing on its side for the same reason.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    opt.zero_grad()

    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 8)).astype(np.int32)
    for _ in range(2):
        opt.zero_grad()
        model.forward(tokens).float().sum().backward()
        opt.step()

    save_resume_dcp(model, opt, path, step=2, world_size=world_size, ep_size=1)


def _v8_load_worker(rank, world_size, tmp_path, path, out_prefix):
    from src.model.cuda_distributed import build_mesh, wrap_backbone, load_resume_dcp

    torch.manual_seed(1)     # DIFFERENT seed: proves values come from the resume, not init
    cfg = _cfg(moe_every=None, n_layers=2)
    model = CUDAMambaModel(cfg, ep_size=1, ep_rank=0)
    mesh = build_mesh(dp_size=world_size, ep_size=1)
    wrap_backbone(model, mesh["dp"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # AdamW only creates per-param state after a step; DCP's set_state_dict needs the
    # optimizer's state structure to already exist to resharded INTO.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    opt.zero_grad()

    meta = load_resume_dcp(model, opt, path, ep_size=1)
    # `.full_tensor()` is a COLLECTIVE (an all-gather over the param's mesh) — every
    # rank must call it, even though only rank 0 writes the result. A rank-0-only call
    # here deadlocks every other rank at its own next collective (verified in-session).
    emb = model.embedding.weight.detach().full_tensor()
    if rank == 0:
        torch.save({"step": meta["step"], "embedding": emb}, str(tmp_path / f"{out_prefix}.pt"))


def test_v8_resume_across_world_size_change(tmp_path):
    save_path = str(tmp_path / "dcp_ckpt")
    _spawn_gloo(_v8_save_worker, world_size=2, tmp_path=tmp_path, path=save_path)

    # 2 -> 1
    _spawn_gloo(_v8_load_worker, world_size=1, tmp_path=tmp_path,
               path=save_path, out_prefix="v8_to1")
    r1 = torch.load(str(tmp_path / "v8_to1.pt"))
    assert r1["step"] == 2

    # 2 -> 2 (same size, still exercises the DCP path)
    _spawn_gloo(_v8_load_worker, world_size=2, tmp_path=tmp_path,
               path=save_path, out_prefix="v8_to2")
    r2 = torch.load(str(tmp_path / "v8_to2.pt"))
    assert r2["step"] == 2
    torch.testing.assert_close(r1["embedding"], r2["embedding"], rtol=1e-5, atol=1e-6)


def _v8_ep_mismatch_worker(rank, world_size, tmp_path, path):
    from src.model.cuda_distributed import load_resume_dcp

    torch.manual_seed(0)
    cfg = _cfg(moe_every=None, n_layers=2)
    model = CUDAMambaModel(cfg, ep_size=1, ep_rank=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    raised = False
    try:
        load_resume_dcp(model, opt, path, ep_size=2)     # bundle was saved at ep_size=1
    except ValueError:
        raised = True
    torch.save({"raised": raised}, str(tmp_path / f"v8ep_{rank}.pt"))


def test_v8_ep_size_change_raises(tmp_path):
    from src.model.cuda_distributed import save_resume_dcp

    save_path = str(tmp_path / "dcp_ckpt_ep1")
    _spawn_gloo(_v8_save_worker, world_size=1, tmp_path=tmp_path, path=save_path)
    _spawn_gloo(_v8_ep_mismatch_worker, world_size=1, tmp_path=tmp_path, path=save_path)
    assert torch.load(str(tmp_path / "v8ep_0.pt"))["raised"]


# --------------------------------------------------------------------------- #
# V9 — rank-aware CheckpointStore: LATEST flips exactly once, every rank's optimizer
# shard lands in the committed slot, a crash before the barrier leaves the previous
# slot intact.
# --------------------------------------------------------------------------- #
def _v9_worker(rank, world_size, tmp_path, root, crash_rank):
    from src.train.checkpoint import CheckpointStore

    store = CheckpointStore(root)

    def _barrier():
        dist.barrier()

    if rank == crash_rank:
        return   # simulate a crash: this rank never calls save() at all

    store.save(step=1, loss_scale_state=None,
              weights_serializer=lambda p: open(p, "wb").close(),
              optimizer_serializer=lambda p: torch.save({"rank": rank}, p + ".pt"),
              is_primary=(rank == 0), barrier=_barrier, rank=rank)


def test_v9_checkpoint_store_rank_aware_commit(tmp_path):
    from src.train.checkpoint import CheckpointStore

    root = str(tmp_path / "ckpt")
    # crash_rank=-1: no crash, both ranks participate -> exactly one LATEST flip, both
    # optimizer shards present.
    _spawn_gloo(_v9_worker, world_size=2, tmp_path=tmp_path, root=root, crash_rank=-1)
    store = CheckpointStore(root)
    slot = store.latest_slot()
    assert slot is not None
    slot_dir = tmp_path / "ckpt" / slot
    assert (slot_dir / "optimizer.state.rank0.pt").exists()
    assert (slot_dir / "optimizer.state.rank1.pt").exists()
    assert (slot_dir / "weights.safetensors").exists()


def test_v9_checkpoint_store_survives_a_crashed_rank(tmp_path):
    """A rank that crashes BEFORE calling save() at all (never reaches the barrier) is
    the sharpest version of 'incomplete write': with `join=True`, `mp.spawn` itself
    would hang here (rank 0 blocks on the barrier forever) — which is exactly the
    documented hazard the `barrier` param's docstring calls out. This test instead
    verifies the STATE INVARIANT directly: a store that never received a call from
    every rank has NO committed slot, so a genuinely-later successful save is what
    would make LATEST valid — never a partial one."""
    from src.train.checkpoint import CheckpointStore

    root = str(tmp_path / "ckpt2")
    store = CheckpointStore(root)
    assert store.latest_slot() is None    # nothing committed yet
    # A lone primary-only save (no peer, no barrier) is the is_primary=True/barrier=None
    # single-process contract — unaffected by #271, still commits normally.
    store.save(step=0, loss_scale_state=None,
              weights_serializer=lambda p: open(p, "wb").close(),
              optimizer_serializer=lambda p: torch.save({}, p + ".pt"))
    assert store.latest_slot() is not None


def test_v9_non_primary_without_barrier_raises(tmp_path):
    from src.train.checkpoint import CheckpointStore

    store = CheckpointStore(str(tmp_path / "ckpt3"))
    with pytest.raises(ValueError, match="requires a real .barrier."):
        store.save(step=0, loss_scale_state=None,
                  weights_serializer=lambda p: None,
                  optimizer_serializer=lambda p: None,
                  is_primary=False)


# --------------------------------------------------------------------------- #
# V10 — skip hygiene: this whole module is import-guarded by `pytest.importorskip
# ("torch")` above, so on a torch-less interpreter every test here reports SKIPPED
# (never failed/errored) under `pytest -q -rs`. Nothing to assert beyond the module
# importing at all under -rs (checked at the CI-job level, not re-simulated here).
# --------------------------------------------------------------------------- #
def test_v10_module_is_torch_gated():
    assert torch is not None   # importorskip already ran; reaching here proves it didn't skip
