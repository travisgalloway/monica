"""Distillation manifest resolver (portable — NO backend import).

A student trial (#65/#98) is a lightweight **manifest** naming the frozen teacher artifacts
plus the student layout and the conversion method (`docs/design/10-distillation.md`). A sweep is
a set of sibling manifests pointing at the *same* teacher signal; only `layout` changes.
`config/manifests/*.yaml` are the seed manifests.

This module parses a manifest into a `DistillManifest`, validates the `init:` method and the
`stages:` list, and resolves the `layout` sweep-schema onto a `MambaConfig` (the model's single
source of truth). The student-init step (#99, `model.mlx_student_init`) consumes `init`; the
distill loss (#100) consumes `stages` to run mixing-match -> hidden-align -> logit-distill in
order. Nothing here imports a backend, so it stays above the seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from ..model.blocks import MambaConfig


class InitMethod(Enum):
    """Teacher -> student conversion method, selected per manifest by `init:`."""

    MAMBA_IN_THE_LLAMA = "mamba-in-the-llama"
    MOHAWK = "mohawk"

    @classmethod
    def from_str(cls, value: str) -> "InitMethod":
        try:
            return cls(value)
        except ValueError:
            allowed = ", ".join(m.value for m in cls)
            raise ValueError(f"unknown init method {value!r}; expected one of: {allowed}")


# Canonical distillation + post-training stages, in their natural run order. The manifest's
# `stages:` is a subset/ordering of these; #100 runs the distill stages, post-training (#11)
# runs the SFT/RL ones.
CANONICAL_STAGES = (
    "mixing-match", "hidden-align", "logit-distill",     # distillation matching (#100)
    "instruct-sft", "reasoning-sft", "tool-sft",         # supervised fine-tuning (#11)
    "grpo",                                              # verifiable RL (#78)
)

# Tokenizer name -> vocab size. Qwen2.5 is fixed by the conversion teacher (#90); the value
# matches config/student-1b.yaml (151646, the tokenizer vocab, < the teacher's padded 151936).
_TOKENIZER_VOCAB = {"qwen25": 151646}


@dataclass
class DistillManifest:
    """A parsed student-trial manifest. `init` is an `InitMethod`; `stages` is validated."""

    student: str
    conversion_teacher: str
    tokenizer: str
    seq_len: int
    init: InitMethod
    stages: List[str]
    layout: Dict[str, Any]
    corpus: str = ""
    teacher_outputs: str = ""
    sft: str = ""
    rl: str = ""
    schedule: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.tokenizer not in _TOKENIZER_VOCAB:
            raise ValueError(
                f"unknown tokenizer {self.tokenizer!r}; known: {sorted(_TOKENIZER_VOCAB)}")
        unknown = [s for s in self.stages if s not in CANONICAL_STAGES]
        if unknown:
            raise ValueError(
                f"unknown stage(s) {unknown}; expected from: {list(CANONICAL_STAGES)}")
        if not self.stages:
            raise ValueError("manifest `stages` must be non-empty")

    @property
    def vocab_size(self) -> int:
        return _TOKENIZER_VOCAB[self.tokenizer]


@dataclass
class InitReport:
    """Summary of a student-init run (#99), returned by the backend `init_student`.

    Portable (plain ints/strings) so a driver can log it without importing a backend.
    `frozen_layers` lists the student layer indices whose params were frozen (the kept
    attention layers under Mamba-in-the-Llama); `n_frozen_params + n_trainable_params`
    equals the student's total parameter count.
    """

    method: str
    n_layers_mapped: int
    n_frozen_params: int
    n_trainable_params: int
    frozen_layers: List[int] = field(default_factory=list)


def load_manifest(path: Union[str, Path]) -> DistillManifest:
    """Load + validate a manifest YAML (`config/manifests/*.yaml`)."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    m = DistillManifest(
        student=str(raw["student"]),
        conversion_teacher=str(raw["conversion_teacher"]),
        tokenizer=str(raw["tokenizer"]),
        seq_len=int(raw["seq_len"]),
        init=InitMethod.from_str(str(raw["init"])),
        stages=list(raw.get("stages", [])),
        layout=dict(raw.get("layout", {})),
        corpus=str(raw.get("corpus", "")),
        teacher_outputs=str(raw.get("teacher_outputs", "")),
        sft=str(raw.get("sft", "")),
        rl=str(raw.get("rl", "")),
        schedule=dict(raw.get("schedule", {})),
    )
    m.validate()
    return m


def manifest_to_config(manifest: DistillManifest) -> MambaConfig:
    """Resolve a manifest's `layout` sweep-schema onto a `MambaConfig`.

    The manifest's layout keys are sweep-schema names; map them onto model fields:
      `d_model`, `n_layers`, `attention_every` -> `attn_every`, `state_size` -> `d_state`.
    `vocab_size`/`seq_len` come from the manifest; `precision`/`grad_checkpoint`/`head_dim`
    mirror `config/student-1b.yaml` (bf16 CUDA training at this depth). Optional layout keys
    `head_dim`, `expand`, `n_attn_heads` override the student defaults when present.
    """
    layout = manifest.layout
    cfg = MambaConfig(
        d_model=int(layout["d_model"]),
        n_layers=int(layout["n_layers"]),
        d_state=int(layout.get("state_size", layout.get("d_state", 128))),
        expand=int(layout.get("expand", 2)),
        head_dim=int(layout.get("head_dim", 64)),
        attn_every=_opt_int(layout.get("attention_every", layout.get("attn_every"))),
        n_attn_heads=_opt_int(layout.get("n_attn_heads")),
        vocab_size=manifest.vocab_size,
        seq_len=manifest.seq_len,
        tie_embeddings=True,
        precision="bf16",
        grad_checkpoint=True,
    )
    cfg.validate()
    return cfg


def _opt_int(value: Any) -> Any:
    return None if value is None else int(value)
