"""Manifest-driven student sweep harness (#98). Portable — runs without a backend.

One test per acceptance criterion of #98:
  * a manifest resolves to a runnable student config + its frozen-artifact paths;
  * two sibling manifests differ only in `layout` and share one teacher signal;
  * changing a `layout` field invalidates nothing upstream (frozen signal unchanged).
"""

import dataclasses

import pytest

from src.train.distill_manifest import load_manifest
from src.train.sweep import (FrozenSignal, ResolvedTrial, Sweep, format_sweep_table,
                             frozen_signal, load_sweep, load_sweep_dir, resolve_trial,
                             shared_signal)

ATTN8 = "config/manifests/student-1b-attn8pct.yaml"
ATTN12 = "config/manifests/student-1b-attn12pct.yaml"
MANIFESTS = [ATTN8, ATTN12]
MANIFEST_DIR = "config/manifests"


# --- acceptance #1: a manifest resolves to a runnable config + frozen-artifact paths --------

@pytest.mark.parametrize("path", MANIFESTS)
def test_trial_resolves_to_config_and_artifacts(path):
    trial = resolve_trial(load_manifest(path))
    assert isinstance(trial, ResolvedTrial)
    trial.config.validate()                       # a runnable student config
    # the frozen artifacts it trains against are named and non-empty
    assert trial.signal.corpus and trial.signal.teacher_outputs
    assert trial.signal.conversion_teacher == "open-r1/OpenR1-Distill-7B"
    # a param/memory estimate accompanies the layout (#66 sizing)
    assert trial.sizing_row["params"] > 0 and trial.sizing_row["train_gb"] > 0


# --- acceptance #2: two siblings share one teacher signal, differ only in layout ------------

def test_siblings_share_signal_differ_only_in_layout():
    m8, m12 = load_manifest(ATTN8), load_manifest(ATTN12)
    sig = shared_signal([m8, m12])                # collapses to one -> they share it
    assert isinstance(sig, FrozenSignal)
    assert sig == frozen_signal(m8) == frozen_signal(m12)
    # the only thing that differs is the layout (the swept variables)
    assert m8.layout != m12.layout
    assert m8.layout["attention_every"] != m12.layout["attention_every"]
    assert m8.layout["state_size"] != m12.layout["state_size"]


# --- acceptance #3: changing a layout field invalidates nothing upstream -------------------

def test_layout_change_does_not_invalidate_upstream():
    m = load_manifest(ATTN12)
    before = frozen_signal(m)
    # walk each swept variable; the frozen teacher signal must not move
    mutated = dataclasses.replace(m, layout={**m.layout, "state_size": 256,
                                             "attention_every": 6, "n_layers": 36})
    assert frozen_signal(mutated) == before
    # and the mutated layout still resolves to a runnable config
    resolve_trial(mutated).config.validate()


def test_divergent_teacher_signal_raises():
    m = load_manifest(ATTN8)
    other = dataclasses.replace(m, corpus="some/other/corpus")
    with pytest.raises(ValueError, match="corpus"):
        shared_signal([m, other])


def test_shared_signal_empty_raises():
    with pytest.raises(ValueError):
        shared_signal([])


# --- the Sweep view: one row per trial, distinct layouts -> distinct sizing ----------------

def test_load_sweep_dir_one_row_per_manifest_distinct_params():
    sweep = load_sweep_dir(MANIFEST_DIR)
    assert isinstance(sweep, Sweep)
    rows = sweep.table()
    assert len(rows) == len(sweep.trials) >= 2
    # the two layouts produce distinct param counts and attention fractions
    params = {r["params"] for r in rows}
    attn = {round(r["attn_pct"], 4) for r in rows}
    assert len(params) == len(rows)
    assert len(attn) == len(rows)
    # attn8pct (attn_every 12) has a *lower* attention fraction than attn12pct (attn_every 8)
    by_name = {r["tier"]: r for r in rows}
    assert by_name["1b-attn8pct"]["attn_pct"] < by_name["1b-attn12pct"]["attn_pct"]


def test_load_sweep_explicit_paths_matches_dir():
    assert load_sweep(MANIFESTS).signal == load_sweep_dir(MANIFEST_DIR).signal


def test_format_sweep_table_renders_signal_and_rows():
    out = format_sweep_table(load_sweep_dir(MANIFEST_DIR))
    assert "Shared frozen teacher signal" in out
    assert "open-r1/OpenR1-Distill-7B" in out
    assert "1b-attn8pct" in out and "1b-attn12pct" in out
    assert "attn%" in out
