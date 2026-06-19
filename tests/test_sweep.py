"""Manifest-driven student sweep harness (#98). Portable — runs without a backend.

One test per acceptance criterion of #98:
  * a manifest resolves to a runnable student config + its frozen-artifact paths;
  * two sibling manifests differ only in `layout` and share one teacher signal;
  * changing a `layout` field invalidates nothing upstream (frozen signal unchanged).
"""

import dataclasses

import pytest

from src.train.distill_manifest import InitMethod, load_manifest
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
    sig = shared_signal([m8, m12])                # passes only if they are true siblings
    assert isinstance(sig, FrozenSignal)
    assert sig == frozen_signal(m8) == frozen_signal(m12)
    # the swept variables differ...
    assert m8.layout != m12.layout
    assert m8.layout["attention_every"] != m12.layout["attention_every"]
    assert m8.layout["state_size"] != m12.layout["state_size"]
    # ...and EVERYTHING else (teacher signal + recipe) is identical -> only `layout` differs
    non_layout = {f.name for f in dataclasses.fields(m8)} - {"student", "layout"}
    for name in non_layout:
        assert getattr(m8, name) == getattr(m12, name), f"siblings diverge on {name!r}"


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


def test_divergent_recipe_raises():
    # The student-side recipe (init/stages/schedule) is part of the sweep invariant too —
    # a sweep over architecture must hold it constant, else the trials aren't comparable.
    m = load_manifest(ATTN8)
    diff_stages = dataclasses.replace(m, stages=m.stages[:-1])         # drop a stage
    with pytest.raises(ValueError, match="stages"):
        shared_signal([m, diff_stages])
    diff_init = dataclasses.replace(m, init=InitMethod.MOHAWK)         # m defaults to MiL
    with pytest.raises(ValueError, match="init"):
        shared_signal([m, diff_init])


def test_resolve_rejects_nonpositive_n_layers():
    # A layout typo (n_layers: 0) must be caught by config validation at resolve time, not
    # crash later in attn_pct with a bare ZeroDivisionError.
    m = dataclasses.replace(load_manifest(ATTN8), layout={**load_manifest(ATTN8).layout,
                                                          "n_layers": 0})
    with pytest.raises(ValueError, match="n_layers"):
        resolve_trial(m)


def test_shared_signal_empty_raises():
    with pytest.raises(ValueError):
        shared_signal([])


def test_load_sweep_empty_inputs_raise(tmp_path):
    with pytest.raises(ValueError):
        load_sweep([])
    with pytest.raises(ValueError, match="no .*manifests"):
        load_sweep_dir(tmp_path)        # an empty directory has no *.yaml manifests


# --- the Sweep view: one row per trial, distinct layouts -> distinct sizing ----------------

def test_load_sweep_dir_one_row_per_manifest_distinct_params():
    sweep = load_sweep_dir(MANIFEST_DIR)
    assert isinstance(sweep, Sweep)
    rows = sweep.table()
    assert len(rows) == len(sweep.trials) >= 2
    # different layouts produce meaningfully different sizing (the sweep saw real variation)
    assert len({r["params"] for r in rows}) >= 2
    assert len({round(r["attn_pct"], 4) for r in rows}) >= 2
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
