"""Manifest-driven student sweep harness (#98). Portable — NO backend import.

The distillation POC is an **architecture search** over the student's attention fraction,
layer placement, and state size (`docs/design/10-distillation.md`). Each trial is a
lightweight manifest (`config/manifests/*.yaml`) naming the FROZEN teacher artifacts plus
the student `layout`; a sweep is a set of *sibling* manifests pointing at the **same** teacher
signal, where only `layout` varies.

This module composes the two pieces that already exist — the manifest resolver
(`distill_manifest`, which turns a manifest into a `MambaConfig`) and the sizing tooling
(`model.sizing`, #66, which turns a config into a param/memory estimate) — into the sweep
view #98 asks for:

  * `resolve_trial` — a manifest -> a runnable student `MambaConfig` + its frozen-artifact
    paths + a sizing row.
  * `shared_signal` — assert a set of manifests share one frozen teacher signal (the guard
    that makes a sweep a sweep: change `layout`, reuse corpus + teacher outputs unchanged).
  * `Sweep` / `load_sweep[_dir]` / `format_sweep_table` — load a set of siblings and render
    the per-trial sizing+layout table under the one shared teacher signal.

Nothing here imports a backend (only `distill_manifest`, `blocks`, `sizing`), so it stays
above the seam.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, List, Union

from ..model.blocks import MambaConfig
from ..model.sizing import family_row
from .distill_manifest import DistillManifest, load_manifest, manifest_to_config


@dataclass(frozen=True)
class FrozenSignal:
    """The upstream artifacts a student `layout` change must NOT invalidate.

    These are exactly the fields shared across a sweep: the conversion teacher + tokenizer
    (which fix the vocab and the matching target), the sequence length, and the precomputed
    corpus / teacher-output / SFT / RL paths. Frozen + hashable, so a set of these collapses
    to a single element iff the manifests truly share one teacher signal (`shared_signal`).
    """

    conversion_teacher: str
    tokenizer: str
    seq_len: int
    corpus: str
    teacher_outputs: str
    sft: str
    rl: str


def frozen_signal(manifest: DistillManifest) -> FrozenSignal:
    """The frozen teacher signal of a manifest (everything upstream of the student layout)."""
    return FrozenSignal(
        conversion_teacher=manifest.conversion_teacher,
        tokenizer=manifest.tokenizer,
        seq_len=manifest.seq_len,
        corpus=manifest.corpus,
        teacher_outputs=manifest.teacher_outputs,
        sft=manifest.sft,
        rl=manifest.rl,
    )


@dataclass
class ResolvedTrial:
    """A manifest resolved to a runnable student config + its frozen-artifact paths.

    This is acceptance criterion #1: a manifest resolves to (a) a `MambaConfig` the backend
    can build (`config`, already validated by `manifest_to_config`) and (b) the frozen
    artifacts it trains against (`signal`). `sizing_row` is the #66 param/memory estimate for
    the layout.
    """

    name: str
    manifest: DistillManifest
    config: MambaConfig

    @property
    def signal(self) -> FrozenSignal:
        return frozen_signal(self.manifest)

    @property
    def sizing_row(self) -> dict:
        """Param/memory estimate for this trial's layout (via `model.sizing.family_row`)."""
        return family_row(self.name, self.config)

    @property
    def attn_pct(self) -> float:
        """Fraction of blocks that are attention (one of the three swept variables)."""
        return self.config.n_attention_layers / self.config.n_layers


def resolve_trial(manifest: DistillManifest) -> ResolvedTrial:
    """Resolve a manifest to a `ResolvedTrial` (config + frozen-artifact paths + sizing)."""
    return ResolvedTrial(
        name=manifest.student,
        manifest=manifest,
        config=manifest_to_config(manifest),
    )


def shared_signal(manifests: Iterable[DistillManifest]) -> FrozenSignal:
    """The single frozen teacher signal shared by `manifests`; raise if they diverge.

    This is the guard that makes a set of manifests a valid *sweep*: every sibling must point
    at the same teacher signal so that varying `layout` reuses the (expensive) corpus +
    teacher outputs unchanged. On divergence the error names the offending field(s) and their
    distinct values, so a mis-pointed manifest is easy to spot.
    """
    manifests = list(manifests)
    if not manifests:
        raise ValueError("shared_signal requires at least one manifest")
    signals = [frozen_signal(m) for m in manifests]
    first = signals[0]
    diverging = {
        f.name: sorted({getattr(s, f.name) for s in signals})
        for f in fields(FrozenSignal)
        if any(getattr(s, f.name) != getattr(first, f.name) for s in signals)
    }
    if diverging:
        detail = "; ".join(f"{k}={v}" for k, v in diverging.items())
        raise ValueError(
            "manifests do not share one frozen teacher signal — a sweep may only vary "
            f"`layout`. Diverging field(s): {detail}"
        )
    return first


@dataclass
class Sweep:
    """A resolved set of sibling trials sharing one frozen teacher signal."""

    trials: List[ResolvedTrial]
    signal: FrozenSignal

    def table(self) -> List[dict]:
        """One row per trial: the sizing estimate plus the swept layout columns.

        Columns beyond `model.sizing.family_row`: `attn_pct`, `d_state`, `n_layers`,
        `d_model` — the architecture variables a layout sweep walks over.
        """
        rows = []
        for t in self.trials:
            row = dict(t.sizing_row)
            row.update(
                attn_pct=t.attn_pct,
                d_state=t.config.d_state,
                n_layers=t.config.n_layers,
                d_model=t.config.d_model,
            )
            rows.append(row)
        return rows


def load_sweep(paths: Iterable[Union[str, Path]]) -> Sweep:
    """Load + validate a set of sibling manifests into a `Sweep` (asserts shared signal)."""
    manifests = [load_manifest(p) for p in paths]
    if not manifests:
        raise ValueError("load_sweep requires at least one manifest path")
    signal = shared_signal(manifests)
    return Sweep(trials=[resolve_trial(m) for m in manifests], signal=signal)


def load_sweep_dir(directory: Union[str, Path]) -> Sweep:
    """Load every `*.yaml` manifest in `directory` (sorted) into a `Sweep`."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise ValueError(f"no *.yaml manifests found in {directory}")
    return load_sweep(paths)


def format_sweep_table(sweep: Sweep) -> str:
    """Render a sweep: the one shared teacher signal, then the per-trial sizing+layout table."""
    sig = sweep.signal
    head = [
        "Shared frozen teacher signal (reused by every trial):",
        f"  conversion_teacher : {sig.conversion_teacher}",
        f"  tokenizer          : {sig.tokenizer}",
        f"  seq_len            : {sig.seq_len}",
        f"  corpus             : {sig.corpus}",
        f"  teacher_outputs    : {sig.teacher_outputs}",
        f"  sft                : {sig.sft}",
        f"  rl                 : {sig.rl}",
        "",
    ]
    header = (f"{'student':<14} {'params':>10} {'bf16 wt':>9} {'train':>8} "
              f"{'attn%':>6} {'d_state':>8} {'layers':>7} {'d_model':>8}")
    lines = [header, "-" * len(header)]
    for r in sweep.table():
        lines.append(
            f"{r['tier']:<14} {r['params'] / 1e6:>9.1f}M {r['weights_gb']:>8.2f}G "
            f"{r['train_gb']:>7.1f}G {r['attn_pct'] * 100:>5.1f}% {r['d_state']:>8} "
            f"{r['n_layers']:>7} {r['d_model']:>8}"
        )
    return "\n".join(head + lines)
