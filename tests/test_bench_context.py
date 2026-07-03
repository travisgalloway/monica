"""Toy-scale test for the #104 context-throughput harness (scripts/bench_context.py).

MLX-gated (skips cleanly where mlx is unavailable), mirroring tests/test_mlx_parity.py.
"""

import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.serve.sessions import per_session_state_bytes
from scripts.bench_context import analytic_state_bytes, arm_config, measure, run_sweep


CFG = "config/toy.yaml"
LENGTHS = [4, 8]
DECODE_TOKENS = 3


def test_arm_config_ssm_unchanged_attn_every_layer():
    cfg = load_config(CFG)
    ssm_cfg = arm_config(cfg, "ssm")
    assert ssm_cfg is cfg
    attn_cfg = arm_config(cfg, "attn")
    assert attn_cfg.attn_every == 1
    assert all(attn_cfg.is_attention_layer(i) for i in range(attn_cfg.n_layers))


def test_measure_both_arms_finite_tok_s():
    cfg = load_config(CFG)
    for arm in ("ssm", "attn"):
        acfg = arm_config(cfg, arm)
        model = MLXMambaModel(acfg)
        mx.eval(model.parameters())
        for length in LENGTHS:
            m = measure(model, mx, length, DECODE_TOKENS, seed=0, warmup_steps=2)
            assert m["prefill_tok_s"] > 0 and m["prefill_tok_s"] < float("inf")
            assert m["decode_tok_s"] > 0 and m["decode_tok_s"] < float("inf")
            assert m["peak_gb"] > 0


def test_ssm_state_bytes_constant_across_lengths():
    """The core claim: the ssm arm's analytic state size does not depend on context
    length, and matches per_session_state_bytes directly (not hardcoded by the harness)."""
    cfg = load_config(CFG)
    ssm_cfg = arm_config(cfg, "ssm")
    expected = per_session_state_bytes(ssm_cfg, conservative_fp32=False)
    sizes = {analytic_state_bytes(ssm_cfg, length) for length in LENGTHS}
    assert sizes == {expected}


def test_attn_state_bytes_grow_with_length():
    """The contrasting claim: the attn arm's KV cache is NOT constant — it scales
    linearly with context length. Asserted on the analytic formula (not measured peak
    memory) since measured peak-mem deltas at toy scale are too small to be a reliable
    assertion — the formula itself is what the harness reports, so this is the thing
    that actually needs to be correct."""
    cfg = load_config(CFG)
    attn_cfg = arm_config(cfg, "attn")
    sizes = [analytic_state_bytes(attn_cfg, length) for length in LENGTHS]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]
    # linear in length: bytes / length is constant (bytes-per-token-of-KV-cache)
    per_token = {b / l for b, l in zip(sizes, LENGTHS)}
    assert len(per_token) == 1


def test_run_sweep_produces_expected_rows():
    cfg = load_config(CFG)
    rows = run_sweep(MLXMambaModel, mx, cfg, LENGTHS, DECODE_TOKENS, seed=0, warmup_steps=2)
    assert len(rows) == len(LENGTHS) * 2
    for r in rows:
        assert r["arm"] in ("ssm", "attn")
        assert r["length"] in LENGTHS
        assert r["state_bytes"] > 0


def test_max_attn_length_skips_over_cap_lengths(capsys):
    cfg = load_config(CFG)
    rows = run_sweep(MLXMambaModel, mx, cfg, LENGTHS, DECODE_TOKENS,
                     max_attn_length=LENGTHS[0], seed=0, warmup_steps=2)
    attn_rows = [r for r in rows if r["arm"] == "attn"]
    assert [r["length"] for r in attn_rows] == [LENGTHS[0]]
    ssm_rows = [r for r in rows if r["arm"] == "ssm"]
    assert len(ssm_rows) == len(LENGTHS)
    out = capsys.readouterr().out
    assert "skip" in out and str(LENGTHS[1]) in out
