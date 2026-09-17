"""Tests for Issue #201: AR harness ablation (fast / slow / both loops).

Tests:
1. Four ablation cells on ScriptedFakeLM (pure unit, no GPU, no network):
   - Baseline: unconstrained, no diagnostic checks
   - Fast-loop: completion-list / grammar masking restricts next-token draws
   - Slow-loop: statement-boundary diagnostic check rolls back and bans offending token
   - Both loops: completion-mask constrained decode active during generation, and
     slow loop rolls back on any surviving diagnostics.
2. SessionStoreLMAdapter on FakeModel:
   - Implements LMAdapter protocol
   - Implements SnapshotCapable protocol
   - Rollback via RewindTree vs re-prefill
3. create_repair_sampler in src/serve/generate.py:
   - Wraps injected sampler with pass_context=True
   - Banned logits set to -inf
   - Fast-loop masker allowed_ids restriction
4. sample(..., banned_ids=...) in src/serve/sampling.py:
   - Banning specific token ids
   - Combined allowed_ids and banned_ids
   - Graceful fallback when all tokens are banned
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from src.lsp.completion_mask import CompletionMasker
from src.lsp.diagnostics import Diagnostic
from src.lsp.harness import GenResult, generate_slow_loop
from src.lsp.lm import LMAdapter, SnapshotCapable
from src.serve.generate import create_repair_sampler, generate
from src.serve.sampling import sample
from src.serve.sessions import SessionHistory, SessionStore, SessionStoreLMAdapter


# --------------------------------------------------------------------------- #
# Fixtures and doubles
# --------------------------------------------------------------------------- #

class ScriptedFakeLM:
    def __init__(self, script: Dict[Tuple[int, ...], List[int]], vocab_size: int = 256):
        self.script = script
        self.vocab_size = vocab_size
        self.n_forward_tokens = 0
        self.n_forward_tokens_nocache = 0
        self._gen_ids: List[int] = []

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in token_ids)

    def _logits_for(self, history: Tuple[int, ...]) -> np.ndarray:
        logits = np.full(self.vocab_size, -10.0, dtype=np.float64)
        prefs = self.script.get(history, [ord(";")])
        for rank, tok in enumerate(prefs):
            logits[tok] = float(len(prefs) - rank) * 10.0
        self.n_forward_tokens += 1
        self.n_forward_tokens_nocache += 1
        return logits

    def reset(self, context: str) -> np.ndarray:
        self._gen_ids = []
        return self._logits_for(())

    def step(self, token_id: int) -> np.ndarray:
        self._gen_ids.append(token_id)
        return self._logits_for(tuple(self._gen_ids))

    def rollback(self, n_tokens: int) -> None:
        if n_tokens <= 0:
            return
        self._gen_ids = self._gen_ids[: len(self._gen_ids) - n_tokens]


class FixedLabelSource:
    def __init__(self, labels: Sequence[str]):
        self.labels = list(labels)
        self.n_queries = 0

    def query(self, path: str, text: str, anchor_offset: int) -> List[str]:
        self.n_queries += 1
        return list(self.labels)


from types import SimpleNamespace

class FakeModel:
    def __init__(self, vocab_size: int = 256):
        self.config = SimpleNamespace(
            n_layers=2, d_conv=4, d_inner=128, n_heads=8, head_dim=16, d_state=16,
            precision="fp32", vocab_size=vocab_size,
        )

    def init_state(self, batch_size: int):
        return np.zeros((batch_size,), dtype=np.int64)

    def step(self, token, state):
        new_state = state + np.asarray(token)
        logits = np.eye(self.config.vocab_size)[np.asarray(token) % self.config.vocab_size]
        return logits, new_state

    def prefill(self, token_batch, seg_ids=None, *, last_only=False):
        token_batch = np.asarray(token_batch)
        state = self.init_state(token_batch.shape[0])
        rows = []
        for t in range(token_batch.shape[1]):
            logits, state = self.step(token_batch[:, t], state)
            rows.append(logits)
        return (rows[-1] if last_only else np.stack(rows, axis=1)), state

    def clone_state(self, state):
        return np.array(state, copy=True)

    def state_size(self) -> int:
        return 64


# --------------------------------------------------------------------------- #
# 1. Four ablation cells
# --------------------------------------------------------------------------- #

def test_ar_ablation_cell_1_baseline():
    """Cell 1 (Baseline): free running, no masking, no diagnosis or rollback."""
    # Script produces "bad;" greedily, or "ok;" if banned
    script = {
        (): [ord("b")],
        (ord("b"),): [ord("a")],
        (ord("b"), ord("a")): [ord("d")],
        (ord("b"), ord("a"), ord("d")): [ord(";")],
    }
    lm = ScriptedFakeLM(script)
    diagnose_called = []

    def diagnose(src: str):
        diagnose_called.append(src)
        return [Diagnostic("TS2339", "Property 'bad' does not exist", 2, 5)]

    res = generate_slow_loop(lm, diagnose, "u.", repair="none", masker=None, budget="stmt")
    assert res.strategy == "baseline"
    assert res.completion == "bad;"
    assert len(diagnose_called) == 0  # no slow loop diagnostic check
    assert res.n_rollbacks == 0
    assert res.n_mask_steps == 0


def test_ar_ablation_cell_2_fast_loop():
    """Cell 2 (Fast loop): completion mask restricts next-token draws at decode time."""
    # Model prefers 'b' but label source permits only "good"
    script = {
        (): [ord("b"), ord("g")],
        (ord("g"),): [ord("o")],
        (ord("g"), ord("o")): [ord("o")],
        (ord("g"), ord("o"), ord("o")): [ord("d")],
        (ord("g"), ord("o"), ord("o"), ord("d")): [ord(";")],
    }
    lm = ScriptedFakeLM(script)
    labels = FixedLabelSource(["good"])
    masker = CompletionMasker(labels, "rec.ts", lm.decode, mask_scope="member", encode=lm.encode)

    res = generate_slow_loop(lm, lambda s: [], "u.", repair="none", masker=masker, budget="stmt")
    assert res.strategy == "fast"
    assert res.completion == "good;"
    assert res.n_mask_steps > 0
    assert res.n_rollbacks == 0


def test_ar_ablation_cell_3_slow_loop():
    """Cell 3 (Slow loop): statement check flags diagnostic, rolls back and bans token."""
    # Model prefers "bad;" initially. If 'b' is banned, falls back to 'g' -> "good;".
    script = {
        (): [ord("b"), ord("g")],
        (ord("b"),): [ord("a")],
        (ord("b"), ord("a")): [ord("d")],
        (ord("b"), ord("a"), ord("d")): [ord(";")],
        (ord("g"),): [ord("o")],
        (ord("g"), ord("o")): [ord("o")],
        (ord("g"), ord("o"), ord("o")): [ord("d")],
        (ord("g"), ord("o"), ord("o"), ord("d")): [ord(";")],
    }
    lm = ScriptedFakeLM(script)

    def diagnose(src: str):
        if "bad" in src:
            idx = src.find("bad")
            return [Diagnostic("TS2339", 1, idx + 1, "Property 'bad' does not exist", idx)]
        return []

    res = generate_slow_loop(lm, diagnose, "u.", repair="hard", masker=None, budget="stmt")
    assert res.strategy == "slow-hard"
    assert res.completion == "good;"
    assert res.n_rollbacks == 1
    assert res.n_mask_steps == 0


def test_ar_ablation_cell_4_both_loops():
    """Cell 4 (Both loops): completion mask active and slow loop handles remaining defects."""
    # Model prefers "bad;" initially. Mask restricts to {"bad", "good"}.
    # Diagnostic flags "bad", triggering slow-loop rollback; retry emits "good;".
    script = {
        (): [ord("b"), ord("g")],
        (ord("b"),): [ord("a")],
        (ord("b"), ord("a")): [ord("d")],
        (ord("b"), ord("a"), ord("d")): [ord(";")],
        (ord("g"),): [ord("o")],
        (ord("g"), ord("o")): [ord("o")],
        (ord("g"), ord("o"), ord("o")): [ord("d")],
        (ord("g"), ord("o"), ord("o"), ord("d")): [ord(";")],
    }
    lm = ScriptedFakeLM(script)
    labels = FixedLabelSource(["bad", "good"])
    masker = CompletionMasker(labels, "rec.ts", lm.decode, mask_scope="member", encode=lm.encode)

    def diagnose(src: str):
        if "bad" in src:
            idx = src.find("bad")
            return [Diagnostic("TS2339", 1, idx + 1, "Property 'bad' does not exist", idx)]
        return []

    res = generate_slow_loop(lm, diagnose, "u.", repair="hard", masker=masker, budget="stmt")
    assert res.strategy == "both-hard"
    assert res.completion == "good;"
    assert res.n_rollbacks == 1
    assert res.n_mask_steps > 0


# --------------------------------------------------------------------------- #
# 2. SessionStoreLMAdapter
# --------------------------------------------------------------------------- #

def test_session_store_lm_adapter_protocols():
    store = SessionStore(FakeModel())
    adapter = SessionStoreLMAdapter(store)
    assert isinstance(adapter, LMAdapter)
    assert isinstance(adapter, SnapshotCapable)


def test_session_store_lm_adapter_step_and_rollback_rewind_tree():
    store = SessionStore(FakeModel())
    adapter = SessionStoreLMAdapter(store, use_rewind_tree=True)

    logits0 = adapter.reset("prefix")
    assert logits0.shape == (256,)
    assert adapter.n_forward_tokens == len("prefix")

    logits1 = adapter.step(ord("a"))
    logits2 = adapter.step(ord("b"))
    assert adapter.n_forward_tokens == len("prefix") + 2

    # Roll back 1 token to 'a'
    adapter.rollback(1)
    assert adapter.n_snapshot_rollbacks == 1
    assert adapter._gen_ids == [ord("a")]

    # Next step from rewound point
    logits3 = adapter.step(ord("c"))
    assert adapter._gen_ids == [ord("a"), ord("c")]


def test_session_store_lm_adapter_rollback_reprefill():
    store = SessionStore(FakeModel())
    adapter = SessionStoreLMAdapter(store, rollback_strategy="reprefill", use_rewind_tree=False)

    adapter.reset("prefix")
    adapter.step(ord("x"))
    adapter.step(ord("y"))

    adapter.rollback(1)
    assert adapter.n_reprefill_rollbacks == 1
    assert adapter._gen_ids == [ord("x")]


# --------------------------------------------------------------------------- #
# 3. create_repair_sampler
# --------------------------------------------------------------------------- #

def test_create_repair_sampler_enforces_banned_tokens():
    ban_table = {
        (10, 20): {42},  # at prefix (10, 20), token 42 is banned
    }
    sampler = create_repair_sampler(ban_table)

    logits = np.zeros(100, dtype=np.float32)
    logits[42] = 50.0  # normally greedy winner
    logits[99] = 20.0  # runner up

    tok = sampler(logits, previous_tokens=[10, 20])
    assert tok == 99  # 42 was banned!


def test_create_repair_sampler_with_masker():
    class DummyMasker:
        def mask_for(self, text, vocab_size):
            return [7, 8]  # only tokens 7 and 8 allowed

    sampler = create_repair_sampler(
        masker=DummyMasker(),
        decode_fn=lambda ids: "".join(chr(i) for i in ids),
        prompt_text="",
    )
    logits = np.zeros(100, dtype=np.float32)
    logits[50] = 100.0  # would win unconstrained
    logits[7] = 10.0

    tok = sampler(logits, previous_tokens=[ord("a")])
    assert tok == 7


# --------------------------------------------------------------------------- #
# 4. sample(..., banned_ids=...)
# --------------------------------------------------------------------------- #

def test_sample_with_banned_ids():
    logits = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    # Greedy argmax is index 2. Ban index 2 -> index 1 wins
    tok = sample(logits, temperature=0.0, banned_ids=[2])
    assert tok == 1


def test_sample_with_both_allowed_and_banned():
    logits = np.array([1.0, 5.0, 10.0, 15.0], dtype=np.float32)
    # allowed: {1, 3}, but 3 is banned -> 1 must win
    tok = sample(logits, temperature=0.0, allowed_ids=[1, 3], banned_ids=[3])
    assert tok == 1


def test_sample_all_tokens_banned_fallback():
    logits = np.array([10.0, 20.0], dtype=np.float32)
    rng = np.random.default_rng(42)
    tok = sample(logits, temperature=0.0, banned_ids=[0, 1], rng=rng)
    assert tok in [0, 1]
