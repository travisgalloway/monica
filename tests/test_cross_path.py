"""Tests for Issue #204: Cross-path comparison + tool-call efficiency baseline.

Covers:
1. Discrete diffusion generation (src.lsp.diffusion):
   - Noise schedule calculations (linear & cosine)
   - Unguided diffusion baseline (no discriminator calls)
   - Guided diffusion sampling (LSP discriminator, remasking, logit penalty, rollback count)
   - Truncation and stop-string behavior
2. Cross-path comparison engine (src.eval.cross_path):
   - CellMetrics aggregation from summaries
   - Pairwise metric and efficiency delta calculations
   - Core hypothesis verification (distribution-level vs tool-call tokens)
   - Markdown table and report generation
   - JSON serialization / deserialization round-trip
3. Evaluation CLI script (scripts.eval_cross_path):
   - build_comparison with ablation and toolcall data sources
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from src.eval.cross_path import CellMetrics, CrossPathComparison
from src.lsp.diagnostics import Diagnostic
from src.lsp.diffusion import DiffusionConfig, _mask_schedule, generate_diffusion
from scripts.eval_cross_path import build_comparison


# --------------------------------------------------------------------------- #
# Test Doubles
# --------------------------------------------------------------------------- #

class ScriptedFakeLM:
    """Deterministic LMAdapter test double for diffusion and AR generation."""
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
        # Context may be prompt or prompt + committed tokens
        suffix = context[2:] if context.startswith("u.") else context
        self._gen_ids = [ord(c) for c in suffix]
        return self._logits_for(tuple(self._gen_ids))

    def step(self, token_id: int) -> np.ndarray:
        self._gen_ids.append(token_id)
        return self._logits_for(tuple(self._gen_ids))

    def rollback(self, n_tokens: int) -> None:
        if n_tokens <= 0:
            return
        self._gen_ids = self._gen_ids[: len(self._gen_ids) - n_tokens]


# --------------------------------------------------------------------------- #
# 1. Diffusion Sampler Tests
# --------------------------------------------------------------------------- #

def test_mask_schedule_monotonicity():
    total_len = 16
    n_steps = 8

    linear_counts = [_mask_schedule(s, n_steps, total_len, "linear") for s in range(n_steps)]
    assert linear_counts[0] < total_len
    assert linear_counts[-1] == 0
    # Non-increasing
    for i in range(len(linear_counts) - 1):
        assert linear_counts[i] >= linear_counts[i + 1]

    cosine_counts = [_mask_schedule(s, n_steps, total_len, "cosine") for s in range(n_steps)]
    assert cosine_counts[0] < total_len
    assert cosine_counts[-1] == 0
    for i in range(len(cosine_counts) - 1):
        assert cosine_counts[i] >= cosine_counts[i + 1]


def test_diffusion_unguided_baseline():
    """Unguided diffusion baseline should never query the LSP discriminator."""
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
        return [Diagnostic("TS2339", 1, 3, "Property 'bad' does not exist", 2)]

    cfg = DiffusionConfig(n_steps=4, canvas_len=4)
    res = generate_diffusion(lm, diagnose, "u.", guided=False, config=cfg)

    assert res.strategy == "diffusion-baseline"
    assert res.completion == "bad;"
    assert len(diagnose_called) == 0
    assert res.n_tsc_calls == 0
    assert res.n_rollbacks == 0


def test_diffusion_guided_discriminator_remasking():
    """Guided diffusion: discriminator flags defect, remasks faulty token, and falls back."""
    # Model prefers 'b' over 'g'. When 'b' is penalized, chooses 'g' -> "good;".
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

    cfg = DiffusionConfig(n_steps=6, canvas_len=5, guidance_scale=50.0, check_interval=1)
    res = generate_diffusion(lm, diagnose, "u.", guided=True, config=cfg)

    assert res.strategy == "diffusion-guided"
    assert res.completion == "good;"
    assert res.n_tsc_calls > 0
    assert res.n_rollbacks > 0
    assert any(ev["kind"] == "guided_remask" for ev in res.events)


# --------------------------------------------------------------------------- #
# 2. Cross-Path Comparison Engine Tests
# --------------------------------------------------------------------------- #

def test_cell_metrics_creation_and_dict():
    raw_summary = {
        "n": 96,
        "diagnostic_clean_rate": 0.792,
        "error_avoidance_rate": 0.940,
        "exact_gold_rate": 0.0,
        "over_repair_rate": 0.083,
        "no_progress_rate": 0.0,
        "total_n_forward_tokens": 8608,
        "mean_n_forward_tokens": 89.7,
        "total_n_generated_tokens": 350,
        "mean_n_generated_tokens": 3.65,
        "total_n_tsc_calls": 160,
        "mean_n_tsc_calls": 1.67,
        "total_n_rollbacks": 64,
        "mean_n_rollbacks": 0.67,
        "total_wall_s": 86.06,
        "mean_wall_s": 0.896,
    }
    cell = CellMetrics.from_summary(
        cell_id="ar_both",
        paradigm="Autoregressive",
        strategy="Both Loops",
        description="Fast logit mask + slow rollback",
        summary=raw_summary,
    )
    assert cell.cell_id == "ar_both"
    assert cell.diagnostic_clean_rate == 0.792
    assert cell.mean_n_forward_tokens == 89.7

    d = cell.to_dict()
    assert d["cell_id"] == "ar_both"
    assert d["mean_n_rollbacks"] == 0.67


def test_cross_path_comparison_efficiency_evaluation():
    comp = CrossPathComparison(eval_set="eval_sets/ts_error_injection/eval.jsonl", n_records=96)

    # Add AR baseline
    comp.add_cell(CellMetrics(
        cell_id="ar_baseline", paradigm="Autoregressive", strategy="Baseline",
        description="AR baseline", n=96, diagnostic_clean_rate=0.625,
        error_avoidance_rate=0.810, mean_n_forward_tokens=52.1,
    ))

    # Add AR both
    comp.add_cell(CellMetrics(
        cell_id="ar_both", paradigm="Autoregressive", strategy="Both Loops",
        description="AR both", n=96, diagnostic_clean_rate=0.792,
        error_avoidance_rate=0.940, mean_n_forward_tokens=89.7,
        mean_n_rollbacks=0.67, mean_n_tsc_calls=1.67,
    ))

    # Add Tool-call baseline
    comp.add_cell(CellMetrics(
        cell_id="toolcall_baseline", paradigm="Tool-Call", strategy="Chat k=1",
        description="Toolcall chat", n=96, diagnostic_clean_rate=0.802,
        error_avoidance_rate=0.976, mean_n_forward_tokens=140.2,
        no_progress_rate=0.667,
    ))

    # Add Diffusion guided
    comp.add_cell(CellMetrics(
        cell_id="diffusion_guided", paradigm="Diffusion", strategy="Guided",
        description="Guided diffusion", n=96, diagnostic_clean_rate=0.677,
        error_avoidance_rate=0.845, mean_n_forward_tokens=268.0,
    ))

    eff = comp.efficiency_analysis
    assert eff["hypothesis_verified"] is True
    # AR both uses ~89.7 vs 140.2 forward tokens: ~36% savings
    assert eff["token_savings_vs_toolcall_pct"] > 30.0
    # Clean rate per token is significantly higher for AR both
    assert eff["efficiency_gain_vs_toolcall_pct"] > 50.0
    assert "Distribution-level feedback decisively beats tool-call tokens" in eff["verdict"]

    # Table generation
    md_table = comp.to_markdown_table()
    assert "| `ar_baseline` |" in md_table
    assert "| `ar_both` |" in md_table
    assert "| `toolcall_baseline` |" in md_table

    # Report generation
    report = comp.to_markdown_report()
    assert "# Cross-Path Comparison & Tool-Call Efficiency Baseline (#204)" in report
    assert "Verdict: VERIFIED." in report


def test_cross_path_comparison_json_roundtrip():
    comp = CrossPathComparison(eval_set="eval_sets/ts_error_injection/eval.jsonl", n_records=96)
    comp.add_cell(CellMetrics(
        cell_id="ar_baseline", paradigm="Autoregressive", strategy="Baseline",
        description="AR baseline", n=96, diagnostic_clean_rate=0.625,
        error_avoidance_rate=0.810, mean_n_forward_tokens=52.1,
    ))
    comp.add_cell(CellMetrics(
        cell_id="ar_both", paradigm="Autoregressive", strategy="Both Loops",
        description="AR both", n=96, diagnostic_clean_rate=0.792,
        error_avoidance_rate=0.940, mean_n_forward_tokens=89.7,
    ))

    blob = comp.to_dict()
    loaded = CrossPathComparison.from_dict(blob)
    assert loaded.eval_set == comp.eval_set
    assert loaded.n_records == comp.n_records
    assert "ar_baseline" in loaded.cells
    assert "ar_both" in loaded.cells
    assert loaded.cells["ar_both"].diagnostic_clean_rate == 0.792


# --------------------------------------------------------------------------- #
# 3. CLI Script Tests
# --------------------------------------------------------------------------- #

def test_build_comparison_full_pipeline(tmp_path):
    ar_mock = {
        "n_records": 96,
        "summaries": {
            "baseline": {"n": 96, "diagnostic_clean_rate": 0.625, "error_avoidance_rate": 0.810, "mean_n_forward_tokens": 52.1},
            "fast": {"n": 96, "diagnostic_clean_rate": 0.688, "error_avoidance_rate": 0.869, "mean_n_forward_tokens": 52.1},
            "slow": {"n": 96, "diagnostic_clean_rate": 0.781, "error_avoidance_rate": 0.929, "mean_n_forward_tokens": 99.9},
            "both": {"n": 96, "diagnostic_clean_rate": 0.792, "error_avoidance_rate": 0.940, "mean_n_forward_tokens": 89.7},
        }
    }
    toolcall_mock = {
        "summaries": {
            "toolcall-chat-k1": {"n": 96, "diagnostic_clean_rate": 0.802, "error_avoidance_rate": 0.976, "mean_n_forward_tokens": 140.2, "no_progress_rate": 0.667}
        }
    }

    comp = build_comparison(ar_data=ar_mock, toolcall_data=toolcall_mock)
    assert len(comp.cells) == 7
    assert set(comp.cells.keys()) == {
        "ar_baseline",
        "ar_fast",
        "ar_slow",
        "ar_both",
        "diffusion_baseline",
        "diffusion_guided",
        "toolcall_baseline",
    }
    assert comp.efficiency_analysis["hypothesis_verified"] is True
