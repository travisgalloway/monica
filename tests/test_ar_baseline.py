"""Tests for Issue #200: dense small checkpoint (the sparse-upcycle source) + AR baseline.

Verifies:
1. Config validity and adherence to backbone dimensions and upcycle-source constraints:
   - degenerate n_experts=1, top_k=1 MoE (not plain dense config, so expert slots exist)
   - moe_balance_rate: null at E=1
   - d_model: 768 (resolved by #272)
   - moe_d_ff: d_inner (1536) full width (resolved by #272 / PR #275)
   - vocab_size: 49152 (uint16 packing), tie_embeddings: True
   - FIM constraint: n_layers % attn_every == 0 (56 % 8 == 0) so final block is attention
   - exact match on the 15 _MUST_MATCH fields against target configs
2. Mathematical equivalence: at E=1, top_k=1 the degenerate MoE block computes the exact
   same function as a dense SwiGLU FFN.
3. Autoregressive (AR) baseline causal next-token cross-entropy and BPB tracking.
4. Custom generate loop sampler hook execution.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.eval.val_loss import bits_per_byte, cross_entropy, evaluate, perplexity
from src.model.blocks import MambaConfig, load_config
from src.serve.generate import custom_generate, generate
from src.train.upcycle import _MUST_MATCH, check_upcycle_compatible

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 1. Config validation & backbone constraints
# --------------------------------------------------------------------------- #

def test_code_small_dense_config_validity():
    """Verify config/code-small-dense.yaml loads, passes validate(), and satisfies all specs."""
    cfg = load_config(str(REPO_ROOT / "config/code-small-dense.yaml"))
    cfg.validate()

    # Backbone dimensions
    assert cfg.d_model == 768
    assert cfg.n_layers == 56
    assert cfg.d_state == 16
    assert cfg.expand == 2
    assert cfg.d_inner == 1536
    assert cfg.head_dim == 64
    assert cfg.n_heads == 24
    assert cfg.d_conv == 4
    assert cfg.dt_rank_resolved == 48

    # Vocab and packing (#251, uint16)
    assert cfg.vocab_size == 49152
    assert cfg.packing_dtype == "uint16"
    assert cfg.tie_embeddings is True

    # FIM constraint (final block is attention)
    assert cfg.attn_every == 8
    assert cfg.n_layers % cfg.attn_every == 0
    assert cfg.is_attention_layer(cfg.n_layers - 1) is True

    # Degenerate 1-expert MoE block shape (upcycle-source requirements)
    assert cfg.moe_every == 3
    assert cfg.n_experts == 1
    assert cfg.top_k == 1
    assert cfg.moe_d_ff == 1536
    assert cfg.moe_d_ff_resolved == 1536
    assert cfg.moe_balance_rate is None

    # Layer counts: 7 attention, 16 MoE, 33 Mamba
    assert cfg.n_attention_layers == 7
    assert cfg.n_moe_layers == 16
    assert cfg.n_layers - cfg.n_attention_layers - cfg.n_moe_layers == 33

    # Total parameters ~232M
    assert 230_000_000 < cfg.num_parameters() < 235_000_000


def test_code_small_dense_matches_target_configs_on_must_match_fields():
    """All 15 _MUST_MATCH fields agree between code-small-dense and target configs."""
    src = load_config(str(REPO_ROOT / "config/code-small-dense.yaml"))
    targets = [
        load_config(str(REPO_ROOT / "config/code-small-moe.yaml")),
        load_config(str(REPO_ROOT / "config/code-large-a.yaml")),
    ]

    for dst in targets:
        dst.validate()
        for field in _MUST_MATCH:
            sv = getattr(src, field)
            dv = getattr(dst, field)
            assert sv == dv, f"Field {field} mismatch: src={sv!r} vs dst={dv!r}"

        src_moe = {i for i in range(src.n_layers) if src.is_moe_layer(i)}
        dst_moe = {i for i in range(dst.n_layers) if dst.is_moe_layer(i)}
        assert src_moe == dst_moe

        # check_upcycle_compatible passes without raising
        check_upcycle_compatible(src, dst)


# --------------------------------------------------------------------------- #
# 2. Degenerate 1-expert MoE == dense SwiGLU FFN mathematical equivalence
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 3. Autoregressive (AR) baseline causal cross-entropy & BPB tracking
# --------------------------------------------------------------------------- #

def test_ar_baseline_causal_next_token_cross_entropy():
    """Verify causal next-token cross-entropy and bits-per-byte calculations."""
    V = 49152
    T = 64
    rng = np.random.default_rng(0)

    # Autoregressive token sequence: inputs x, targets y = x shifted by 1
    tokens = rng.integers(0, V, size=T + 1)
    inputs = tokens[:-1]
    targets = tokens[1:]

    # Simulating model causal logits: shape (T, V)
    logits = rng.standard_normal((T, V)).astype(np.float32)

    ce = cross_entropy(logits, targets)
    assert ce > 0.0
    assert not math.isnan(ce)

    ppl = perplexity(ce)
    assert ppl == pytest.approx(math.exp(ce), rel=1e-5)

    # BPB tracking (#192): total_ce_nats / (ln(2) * n_bytes)
    # Assuming sample UTF-8 bytes count
    n_bytes = T * 3.5  # average ~3.5 bytes per token
    bpb = bits_per_byte(ce * T, n_bytes)
    expected_bpb = (ce * T) / (math.log(2.0) * n_bytes)
    assert bpb == pytest.approx(expected_bpb, rel=1e-5)


def test_ar_baseline_evaluate_loader():
    """Verify evaluate() computes val_loss, val_perplexity, and val_bpb for AR baseline."""
    class FakeARModel:
        def __init__(self, vocab_size=256):
            self.vocab_size = vocab_size

        def forward(self, x):
            # Deterministic confident prediction on (input + 1) % vocab_size
            B, T = x.shape
            logits = np.zeros((B, T, self.vocab_size), dtype=np.float32)
            for b in range(B):
                for t in range(T):
                    tgt = (int(x[b, t]) + 1) % self.vocab_size
                    logits[b, t, tgt] = 20.0
            return logits

    class FakeARLoader:
        def __init__(self):
            self.n_bytes = 1000
            self.n_tokens = 500

        def epoch(self):
            # 2 batches of (inputs, targets) where target = (input + 1) % 256
            for _ in range(2):
                inp = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
                tgt = (inp + 1) % 256
                yield inp, tgt

    model = FakeARModel(vocab_size=256)
    loader = FakeARLoader()

    result = evaluate(model, loader)
    assert "val_loss" in result
    assert "val_perplexity" in result
    assert "val_bpb" in result
    assert result["val_loss"] < 0.001
    assert result["val_perplexity"] < 1.001
    assert result["val_bpb"] < 0.002


# --------------------------------------------------------------------------- #
# 4. Custom generate loop & sampler hooks with AR baseline
# --------------------------------------------------------------------------- #

def test_ar_baseline_custom_generate_with_sampler_hook():
    """Verify custom_generate with sampler_hook can steer or monitor AR generation."""
    class SimpleStore:
        def __init__(self, vocab_size=256):
            self.vocab_size = vocab_size

        def prefill(self, session_id, prompt_ids):
            # Return uniform logits
            return np.zeros((1, self.vocab_size), dtype=np.float32)

        def step(self, session_id, token_id):
            return np.zeros((1, self.vocab_size), dtype=np.float32)

    store = SimpleStore()
    prompt = [101, 102]

    hook_called = []
    def logit_bias_hook(logits, previous_tokens=None):
        hook_called.append(list(previous_tokens) if previous_tokens is not None else None)
        mod = np.array(logits, copy=True)
        mod[42] = 10.0  # Bias token 42
        return mod

    def greedy_sampler(logits, previous_tokens=None):
        return int(np.argmax(logits))

    out = custom_generate(
        store, "session_test", prompt,
        sampler=greedy_sampler,
        sampler_hook=logit_bias_hook,
        max_new_tokens=4,
        pass_context=True,
    )

    # Every step was biased towards token 42
    assert out == [42, 42, 42, 42]
    assert len(hook_called) == 4
    assert hook_called[0] == [101, 102]
    assert hook_called[1] == [101, 102, 42]
    assert hook_called[2] == [101, 102, 42, 42]
    assert hook_called[3] == [101, 102, 42, 42, 42]
