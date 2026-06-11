"""Serving-layer tests (Milestone 7): SessionStore + RewindTree.

Both modules sit above the seam and are exercised offline with a deterministic,
duck-typed FakeModel — no backend needed. The FakeModel's state is a running token
sum, so session isolation, snapshot independence, and rewind-branch determinism are
all checkable with plain integer arithmetic. `step` returns a FRESH array (proving the
functional contract the store relies on) and `clone_state` copies, mirroring the MLX
backend's immutable-snapshot guarantee.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.model.blocks import MambaConfig
from src.serve.rewind import RewindTree
from src.serve.sessions import (
    SessionStore,
    per_session_state_bytes,
    per_session_state_floats,
)


class FakeModel:
    """Minimal ModelInterface stand-in. State = running sum of tokens fed to a session."""

    def __init__(self, vocab_size: int = 8):
        # d_inner/n_heads are properties on the real config; here they're plain attrs.
        self.config = SimpleNamespace(
            n_layers=2, d_conv=4, d_inner=128, n_heads=8, head_dim=16, d_state=16,
            precision="fp32", vocab_size=vocab_size,
        )

    def init_state(self, batch_size: int):
        return np.zeros((batch_size,), dtype=np.int64)

    def step(self, token, state):
        new_state = state + np.asarray(token)  # fresh array -> functional, no mutation
        logits = np.eye(self.config.vocab_size)[np.asarray(token) % self.config.vocab_size]
        return logits, new_state

    def clone_state(self, state):
        return np.array(state, copy=True)


# --- byte helper math ---------------------------------------------------------------

def test_per_session_state_floats_matches_formula():
    cfg = MambaConfig(d_model=64, n_layers=2, head_dim=16)  # d_inner=128, n_heads=8
    # 2 * ((4-1)*128 + 8*16*16) = 2 * (384 + 2048) = 4864
    assert per_session_state_floats(cfg) == 4864


def test_per_session_state_bytes_conservative_vs_accurate():
    cfg = MambaConfig(d_model=64, n_layers=2, head_dim=16, precision="fp16")
    conservative = per_session_state_bytes(cfg)                       # 4 bytes/elem
    accurate = per_session_state_bytes(cfg, conservative_fp32=False)  # 2 bytes/elem
    assert conservative == 4864 * 4
    assert accurate == 4864 * 2
    # Conservative must never under-count (over-budget is the safe failure direction).
    assert conservative > accurate


# --- admission / budget math --------------------------------------------------------

def test_max_concurrent_from_memory_budget():
    model = FakeModel()
    one = per_session_state_bytes(model.config)
    store = SessionStore(model, memory_budget_bytes=3 * one)
    assert store.max_concurrent == 3


def test_budget_too_small_for_one_session_raises():
    model = FakeModel()
    one = per_session_state_bytes(model.config)
    with pytest.raises(ValueError):
        SessionStore(model, memory_budget_bytes=one - 1)


def test_explicit_max_concurrent_below_one_raises():
    with pytest.raises(ValueError):
        SessionStore(FakeModel(), max_concurrent=0)


def test_explicit_max_concurrent_overrides_budget():
    model = FakeModel()
    store = SessionStore(model, memory_budget_bytes=10**12, max_concurrent=2)
    assert store.max_concurrent == 2


# --- session isolation & lifecycle --------------------------------------------------

def test_session_isolation():
    store = SessionStore(FakeModel())
    store.create("a")
    store.create("b")
    for t in (1, 2, 3):
        store.step("a", t)
    store.step("b", 5)
    assert int(store.get_state("a")[0]) == 6
    assert int(store.get_state("b")[0]) == 5


def test_step_returns_logits_shape():
    store = SessionStore(FakeModel(vocab_size=8))
    store.create("a")
    logits = store.step("a", 3)
    assert logits.shape == (1, 8)


def test_create_duplicate_raises():
    store = SessionStore(FakeModel())
    store.create("a")
    with pytest.raises(ValueError):
        store.create("a")


def test_remove_then_step_raises():
    store = SessionStore(FakeModel())
    store.create("a")
    store.remove("a")
    assert "a" not in store
    with pytest.raises(KeyError):
        store.step("a", 1)


# --- LRU eviction -------------------------------------------------------------------

def test_lru_eviction_drops_least_recently_stepped():
    store = SessionStore(FakeModel(), max_concurrent=2)
    store.create("a")
    store.create("b")
    store.step("a", 1)              # a is now most-recently-used; b is the LRU
    evicted = store.create("c")    # admitting c must evict b
    assert evicted == ["b"]
    assert "b" not in store
    assert "a" in store and "c" in store
    assert len(store) == 2


# --- snapshot independence ----------------------------------------------------------

def test_snapshot_is_independent_and_restorable():
    store = SessionStore(FakeModel())
    store.create("a")
    store.step("a", 4)
    snap = store.get_state("a")        # captures sum=4
    store.step("a", 10)                # advance to sum=14
    assert int(snap[0]) == 4           # snapshot unaffected by later steps
    assert int(store.get_state("a")[0]) == 14
    store.set_state("a", snap)         # restore
    assert int(store.get_state("a")[0]) == 4


# --- rewind tree --------------------------------------------------------------------

def _sum_state(v: int):
    return np.array([v], dtype=np.int64)


def test_rewind_branch_creates_two_children():
    tree = RewindTree()
    n0 = tree.commit(_sum_state(0))
    n1 = tree.commit(_sum_state(3))
    n2 = tree.commit(_sum_state(9))   # n2 is child of n1
    assert tree.parent(n2) == n1
    tree.rewind(n1)                   # branch point back at n1
    n3 = tree.commit(_sum_state(5))   # new branch off n1
    assert tree.parent(n3) == n1
    assert set(tree.children(n1)) == {n2, n3}
    assert tree.parent(n1) == n0


def test_rewind_returns_exact_snapshot():
    tree = RewindTree()
    tree.commit(_sum_state(0))
    n1 = tree.commit(_sum_state(7))
    tree.commit(_sum_state(99))
    restored = tree.rewind(n1)
    assert int(restored[0]) == 7
    assert tree.current() == n1


def test_rewind_unknown_node_raises():
    tree = RewindTree()
    tree.commit(_sum_state(0))
    with pytest.raises(KeyError):
        tree.rewind(999)


def test_max_depth_cap_holds_and_keeps_current():
    tree = RewindTree(max_depth=3)
    ids = [tree.commit(_sum_state(i)) for i in range(5)]
    assert len(tree) == 3
    # The two oldest linear nodes are evicted; the current (last) node survives.
    assert ids[0] not in tree and ids[1] not in tree
    assert tree.current() == ids[-1] and ids[-1] in tree


def test_eviction_reparents_children_onto_grandparent():
    # Chain n0 -> n1 -> n2. Touch n0 then n2 (via rewind) so n1 becomes the LRU front
    # while n2 is current. Committing n3 overflows max_depth=3 and evicts the *interior*
    # node n1 — its child n2 must reparent onto n1's parent n0 (the grandparent).
    tree = RewindTree(max_depth=3)
    n0 = tree.commit(_sum_state(0))
    n1 = tree.commit(_sum_state(1))
    n2 = tree.commit(_sum_state(2))
    tree.rewind(n0)                   # touch order: n1, n2, n0
    tree.rewind(n2)                   # touch order: n1, n0, n2 ; current = n2
    n3 = tree.commit(_sum_state(3))   # len 4 > 3 -> evict LRU front n1
    assert n1 not in tree
    assert tree.parent(n2) == n0      # n2 reparented onto the grandparent
    assert n2 in tree.children(n0)
    assert tree.parent(n3) == n2 and tree.current() == n3
