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
    SessionHistory,
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

    def prefill(self, token_batch, seg_ids=None, *, last_only=False):
        """Seam `prefill` stand-in (#165): an internal step loop, so `SessionStore.prefill`
        is checked against the same running-sum arithmetic as `step`."""
        assert seg_ids is None
        token_batch = np.asarray(token_batch)
        state = self.init_state(token_batch.shape[0])
        rows = []
        for t in range(token_batch.shape[1]):
            logits, state = self.step(token_batch[:, t], state)
            rows.append(logits)
        return (rows[-1] if last_only else np.stack(rows, axis=1)), state

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


# --- prefill (#165) -----------------------------------------------------------------

def test_prefill_matches_step_loop():
    """One `prefill` call must leave the session exactly where a `step` loop would."""
    prompt = [1, 2, 3, 4]
    a = SessionStore(FakeModel())
    a.create("a")
    logits_pre = a.prefill("a", prompt)

    b = SessionStore(FakeModel())
    b.create("b")
    for t in prompt:
        logits_step = b.step("b", t)

    assert int(a.get_state("a")[0]) == int(b.get_state("b")[0]) == sum(prompt)
    assert np.array_equal(logits_pre, logits_step)      # last position's logits


def test_prefill_marks_session_most_recently_used():
    store = SessionStore(FakeModel(), max_concurrent=2)
    store.create("a")
    store.create("b")
    store.prefill("a", [1, 2])     # a is now MRU; b is the LRU
    assert store.create("c") == ["b"]


def test_prefill_on_stepped_session_raises():
    # Fresh-session only: the seam seeds attention RoPE from position 0, so prefilling a
    # session that already consumed tokens would mis-position the prompt.
    store = SessionStore(FakeModel())
    store.create("a")
    store.step("a", 1)
    with pytest.raises(ValueError):
        store.prefill("a", [2, 3])


def test_prefill_after_prefill_raises():
    store = SessionStore(FakeModel())
    store.create("a")
    store.prefill("a", [1, 2])
    with pytest.raises(ValueError):
        store.prefill("a", [3, 4])


def test_prefill_on_restored_session_raises():
    # A snapshot's token position is unknowable above the seam; unknown must not be
    # treated as fresh.
    store = SessionStore(FakeModel())
    store.create("a")
    snap = store.get_state("a")
    store.set_state("a", snap)
    with pytest.raises(ValueError):
        store.prefill("a", [1, 2])


def test_prefill_empty_prompt_raises():
    store = SessionStore(FakeModel())
    store.create("a")
    with pytest.raises(ValueError):
        store.prefill("a", [])


def test_prefill_unknown_session_raises():
    store = SessionStore(FakeModel())
    with pytest.raises(KeyError):
        store.prefill("missing", [1])


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


# --- SessionHistory: the SessionStore x RewindTree composition (#305) ----------------

def _history(max_depth: int = 32, vocab_size: int = 8):
    store = SessionStore(FakeModel(vocab_size=vocab_size), max_concurrent=1)
    store.create("s1")
    return store, SessionHistory(store, "s1", max_depth=max_depth)


def test_session_history_requires_an_existing_session():
    store = SessionStore(FakeModel())
    with pytest.raises(KeyError, match="does not exist"):
        SessionHistory(store, "nope")


def test_session_history_rejects_a_zero_max_depth():
    store = SessionStore(FakeModel())
    store.create("s1")
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        SessionHistory(store, "s1", max_depth=0)


def test_commit_turn_snapshots_the_live_state_and_depth_tracks_it():
    store, history = _history()
    assert history.depth() == 0 and history.current() is None
    root = history.commit_turn()
    store.step("s1", 5)
    second = history.commit_turn()
    assert history.depth() == 2
    assert history.retained_ids() == [root, second]
    assert history.nodes() == [(root, None, [second]), (second, root, [])]


def test_rewind_turns_restores_the_earlier_snapshot():
    store, history = _history()
    store.step("s1", 4)
    boundary = history.commit_turn()          # state = 4
    store.step("s1", 7)
    history.commit_turn()                     # state = 11
    assert int(store.get_state("s1")[0]) == 11
    assert history.rewind_turns(1) == boundary
    assert int(store.get_state("s1")[0]) == 4


def test_commit_after_rewind_forks_history():
    store, history = _history()
    store.step("s1", 4)
    boundary = history.commit_turn()
    store.step("s1", 7)
    first_branch = history.commit_turn()
    history.rewind_turns(1)
    store.step("s1", 2)
    second_branch = history.commit_turn()
    children = dict((nid, kids) for nid, _, kids in history.nodes())
    assert sorted(children[boundary]) == sorted([first_branch, second_branch])


def test_rewind_turns_rejects_non_positive_n():
    _, history = _history()
    history.commit_turn()
    with pytest.raises(ValueError, match="n must be >= 1"):
        history.rewind_turns(0)


def test_rewind_turns_rejects_an_uncommitted_session():
    _, history = _history()
    with pytest.raises(ValueError, match="no committed turn boundaries"):
        history.rewind_turns(1)


def test_rewind_turns_rejects_going_deeper_than_the_retained_history():
    store, history = _history()
    history.commit_turn()
    store.step("s1", 3)
    history.commit_turn()
    before = int(store.get_state("s1")[0])
    with pytest.raises(ValueError, match=r"cannot rewind 4 turn\(s\).*at most 1 rewind"):
        history.rewind_turns(4)
    assert int(store.get_state("s1")[0]) == before   # a rejected rewind changes nothing


def test_rewind_to_unknown_node_raises_lookup_error_naming_the_retained_ids():
    _, history = _history()
    history.commit_turn()
    with pytest.raises(LookupError, match=r"node 42 is not retained.*retained ids: \[0\]"):
        history.rewind_to(42)


def test_retained_ids_and_depth_follow_eviction():
    store, history = _history(max_depth=3)
    for token in range(5):
        store.step("s1", token + 1)
        history.commit_turn()
    assert len(history) == 3 == len(history.tree)
    assert len(history.retained_ids()) == 3
    # Eviction reparents onto the grandparent, so the retained chain is 3 deep and
    # `rewind_turns` counts RETAINED boundaries, not the five turns actually taken.
    assert history.depth() == 3
    with pytest.raises(ValueError, match="at most 2 rewind"):
        history.rewind_turns(3)


def test_budget_bytes_is_max_depth_times_the_per_session_state():
    store, history = _history(max_depth=6)
    assert history.per_node_bytes() == per_session_state_bytes(store.model.config)
    assert history.budget_bytes() == 6 * history.per_node_bytes()
    assert history.max_depth == 6


def test_rewound_snapshot_is_independent_of_later_steps():
    """`get_state`/`set_state` both clone, so a retained node cannot be aliased."""
    store, history = _history()
    store.step("s1", 9)
    boundary = history.commit_turn()
    store.step("s1", 1)
    history.commit_turn()
    history.rewind_to(boundary)
    store.step("s1", 100)                     # mutate the live session hard
    history.rewind_to(boundary)               # the retained node must be untouched
    assert int(store.get_state("s1")[0]) == 9


# --- the real opaque state, on the real backend (mlx-gated, toy scale) ---------------

def test_mlx_rewind_restores_the_opaque_state_and_the_continuation():
    """Toy-scale MLX check that rewind restores the *actual* opaque `State`.

    Everything above is numpy arithmetic through a FakeModel; this is the one place the
    real per-layer `(conv_state, ssm_state)` tuples and the real `clone_state` semantics
    are exercised. Greedy (`temperature=0`) so the continuation identity needs no RNG
    bookkeeping.
    """
    mx = pytest.importorskip("mlx.core")
    from functools import partial

    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel
    from src.serve import sampling
    from src.serve.generate import generate

    cfg = load_config("config/toy.yaml")
    mx.random.seed(0)
    model = MLXMambaModel(cfg)
    store = SessionStore(model, max_concurrent=1)
    store.create("s")
    history = SessionHistory(store, "s", max_depth=8)

    greedy = partial(sampling.sample, temperature=0.0)
    decode = dict(sampler=greedy, to_numpy=lambda a: np.array(a), max_new_tokens=6)

    generate(store, "s", [3, 14, 15, 92], **decode)          # turn A (fresh -> prefill)
    boundary = history.commit_turn()
    snapshot = store.get_state("s")

    out_b = generate(store, "s", [65, 35, 89], prefill=False, **decode)
    history.commit_turn()

    history.rewind_to(boundary)
    restored = store.get_state("s")
    assert len(restored) == cfg.n_layers
    for layer, ((snap_a, snap_b), (rest_a, rest_b)) in enumerate(zip(snapshot, restored)):
        assert np.array_equal(np.array(snap_a), np.array(rest_a)), f"layer {layer} conv"
        assert np.array_equal(np.array(snap_b), np.array(rest_b)), f"layer {layer} ssm"

    out_x = generate(store, "s", [7, 7, 7], prefill=False, **decode)   # discarded branch
    history.commit_turn()
    history.rewind_to(boundary)
    out_b_again = generate(store, "s", [65, 35, 89], prefill=False, **decode)

    assert out_b_again == out_b        # state restored -> byte-identical continuation
    assert out_b_again != out_x        # ... and a real branch, not a replay
