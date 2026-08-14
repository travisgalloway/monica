"""Pure-Python tests for `src.train.parallel` (#271) — no torch, runs everywhere,
including the `portable` CI job. See `tests/test_cuda_distributed.py` for the real
`world_size=2` gloo tests that exercise these numbers under actual collectives."""

import pytest

from src.train.parallel import (ParallelConfig, expert_partition, owner_rank,
                                local_index, reduce_loads)


# --------------------------------------------------------------------------- #
# ParallelConfig
# --------------------------------------------------------------------------- #
def test_parallel_config_dp_size():
    cfg = ParallelConfig(world_size=8, ep_size=2)
    assert cfg.dp_size == 4


def test_parallel_config_world_size_one_is_identity():
    cfg = ParallelConfig(world_size=1)
    assert cfg.ep_size == 1
    assert cfg.dp_size == 1


def test_parallel_config_rejects_non_divisible_world_size():
    with pytest.raises(ValueError, match="world_size.*8.*ep_size.*3"):
        ParallelConfig(world_size=8, ep_size=3)


def test_parallel_config_rejects_non_divisible_n_experts():
    with pytest.raises(ValueError, match="n_experts.*6.*ep_size.*4"):
        ParallelConfig(world_size=4, ep_size=4, n_experts=6)


def test_parallel_config_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        ParallelConfig(world_size=0)
    with pytest.raises(ValueError):
        ParallelConfig(world_size=4, ep_size=0)


# --------------------------------------------------------------------------- #
# expert_partition / owner_rank / local_index
# --------------------------------------------------------------------------- #
def test_expert_partition_contiguous_blocks():
    assert expert_partition(8, 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert expert_partition(8, 4) == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_expert_partition_ep_size_one_is_everything_local():
    assert expert_partition(8, 1) == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_expert_partition_rejects_non_divisible():
    with pytest.raises(ValueError, match="n_experts.*7.*ep_size.*2"):
        expert_partition(7, 2)


def test_expert_partition_covers_every_expert_exactly_once():
    for n_experts, ep_size in [(8, 2), (8, 4), (16, 8), (4, 1)]:
        parts = expert_partition(n_experts, ep_size)
        assert len(parts) == ep_size
        flat = sorted(e for part in parts for e in part)
        assert flat == list(range(n_experts))


def test_owner_rank_and_local_index_round_trip():
    n_experts, ep_size = 8, 4
    parts = expert_partition(n_experts, ep_size)
    for rank, ids in enumerate(parts):
        for local_i, gid in enumerate(ids):
            assert owner_rank(gid, n_experts, ep_size) == rank
            assert local_index(gid, n_experts, ep_size) == local_i


def test_owner_rank_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        owner_rank(8, 8, 2)
    with pytest.raises(ValueError, match="out of range"):
        owner_rank(-1, 8, 2)


def test_local_index_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        local_index(8, 8, 2)


# --------------------------------------------------------------------------- #
# reduce_loads
# --------------------------------------------------------------------------- #
def test_reduce_loads_identity_when_no_reduce_fn():
    loads = [[1.0, 2.0], [3.0, 4.0]]
    out = reduce_loads(loads, all_reduce_fn=None)
    assert out == loads
    assert out is not loads   # a copy, not the same list object


def test_reduce_loads_calls_injected_fn():
    loads = [[1.0, 2.0]]
    seen = {}

    def fake_all_reduce(rows):
        seen["rows"] = rows
        return [[10.0, 20.0]]

    out = reduce_loads(loads, all_reduce_fn=fake_all_reduce)
    assert out == [[10.0, 20.0]]
    assert seen["rows"] == [[1.0, 2.0]]
