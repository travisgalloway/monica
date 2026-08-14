"""Toy-scale test for the #104 context-throughput harness (scripts/bench_context.py).

MLX-gated (skips cleanly where mlx is unavailable), mirroring tests/test_mlx_parity.py.
"""

import json
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.blocks import load_config
from src.model.mlx_backend import MLXMambaModel
from src.serve.sessions import per_session_state_bytes
from scripts.bench_context import (
    _parse_arms, _write_json, analytic_state_bytes, arm_config, measure, run_sweep,
)


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


# --- #170: --prefill-mode / --arms / --json ---

def test_parse_arms_valid_and_invalid():
    assert _parse_arms("ssm") == ("ssm",)
    assert _parse_arms("ssm,attn") == ("ssm", "attn")
    assert _parse_arms(" ssm , attn ") == ("ssm", "attn")
    with pytest.raises(ValueError):
        _parse_arms("")
    with pytest.raises(ValueError):
        _parse_arms("bogus")


def test_measure_prefill_mode_sequential_is_unchanged_default():
    """Default prefill_mode='sequential' must report the same key measure() always
    has — a regression here would silently break every existing caller of measure()."""
    cfg = load_config(CFG)
    model = MLXMambaModel(cfg)
    mx.eval(model.parameters())
    m = measure(model, mx, LENGTHS[0], DECODE_TOKENS, seed=0, warmup_steps=2)
    assert m["prefill_tok_s"] > 0
    assert m["sequential_prefill_tok_s"] == m["prefill_tok_s"]
    assert m.get("parallel_prefill_tok_s") is None
    assert m.get("prefill_speedup") is None


def test_measure_prefill_mode_parallel_uses_model_prefill():
    cfg = load_config(CFG)
    model = MLXMambaModel(cfg)
    mx.eval(model.parameters())
    m = measure(model, mx, LENGTHS[0], DECODE_TOKENS, seed=0, warmup_steps=2,
               prefill_mode="parallel")
    assert m["parallel_prefill_tok_s"] > 0
    assert m["prefill_tok_s"] == m["parallel_prefill_tok_s"]
    assert m.get("sequential_prefill_tok_s") is None


def test_measure_prefill_mode_both_reports_speedup():
    cfg = load_config(CFG)
    model = MLXMambaModel(cfg)
    mx.eval(model.parameters())
    m = measure(model, mx, LENGTHS[0], DECODE_TOKENS, seed=0, warmup_steps=2,
               prefill_mode="both")
    assert m["sequential_prefill_tok_s"] > 0
    assert m["parallel_prefill_tok_s"] > 0
    assert m["prefill_speedup"] == pytest.approx(
        m["parallel_prefill_tok_s"] / m["sequential_prefill_tok_s"])
    # Backward-compatible single figure stays the sequential arm's, matching the
    # pre-#170 default behavior when both ran.
    assert m["prefill_tok_s"] == m["sequential_prefill_tok_s"]


def test_measure_invalid_prefill_mode_raises():
    cfg = load_config(CFG)
    model = MLXMambaModel(cfg)
    mx.eval(model.parameters())
    with pytest.raises(ValueError):
        measure(model, mx, LENGTHS[0], DECODE_TOKENS, seed=0, warmup_steps=2,
               prefill_mode="bogus")


def test_run_sweep_arms_filter_restricts_to_ssm_only():
    cfg = load_config(CFG)
    rows = run_sweep(MLXMambaModel, mx, cfg, LENGTHS, DECODE_TOKENS, seed=0, warmup_steps=2,
                     arms=("ssm",))
    assert rows and all(r["arm"] == "ssm" for r in rows)
    assert len(rows) == len(LENGTHS)


def test_write_json_schema_matches_monica_bench_source_field(tmp_path):
    """Not a byte-identical Swift Codable match (different languages/ecosystems) — the
    'same record schema' the plan asks for is a `source` field distinguishing the two
    harnesses plus directly comparable per-row prefill/decode numbers, so the two
    outputs can be diffed by a human or a script."""
    cfg = load_config(CFG)
    rows = run_sweep(MLXMambaModel, mx, cfg, [LENGTHS[0]], DECODE_TOKENS, seed=0, warmup_steps=2,
                     arms=("ssm",), prefill_mode="both")
    out = tmp_path / "py-bench.json"
    _write_json(rows, out, config_path=CFG, mlx_version=mx.__version__)
    record = json.loads(out.read_text())
    assert record["source"] == "python-mlx"
    assert record["config"] == CFG
    assert len(record["rows"]) == 1
    assert record["rows"][0]["prefill_speedup"] > 0
