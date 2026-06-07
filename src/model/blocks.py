"""Portable architecture description for the Mamba POC.

This module is the single source of truth for model dimensions. It contains NO
backend imports (no MLX, no CUDA/torch) so it can be loaded anywhere and shared
unchanged across backends.

The Mamba block (described here textually; implemented in a backend):

    input projection
      -> split into `main` and `gate`
      -> short causal depthwise conv on `main` (width `d_conv`)
      -> SiLU
      -> selective SSM (diagonal A; input-dependent B, C, delta; parallel scan)
      -> multiply by SiLU(gate)
      -> output projection

Each block is wrapped pre-norm (RMSNorm) with a residual connection. The full
model is: token embedding -> N residual blocks -> final RMSNorm -> tied LM head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class MambaConfig:
    """Architecture + run parameters. Loaded from `config/*.yaml`.

    The tied token embedding (vocab_size x d_model) is a large fraction of the
    parameter budget at POC scale (~38M of ~100M at d_model=768, vocab~50k), so
    `tie_embeddings` is mandatory there, not optional.
    """

    # --- core dimensions ---
    d_model: int
    n_layers: int
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4
    # dt projection rank; "auto" -> ceil(d_model / 16)
    dt_rank: Union[int, str] = "auto"

    # --- vocab / sequence ---
    # OLMo tokenizer vocab. MUST stay < 65536 to pack token ids as uint16.
    vocab_size: int = 50280
    seq_len: int = 1024
    tie_embeddings: bool = True

    # --- precision / numerics ---
    # fp32 | fp16 | bf16. Default fp32 for toy/smoke (correctness + exact resume).
    # For poc.yaml the choice is CONFIRMED ON MLX in milestone 1 (do not assume bf16;
    # fp16 + loss scaling is the likely Metal-friendly choice).
    precision: str = "fp32"

    # Chunked scan working-set bound. None => the backend's default chunk size (the
    # MLX backend uses 32); fine for seq_len up to ~2k. Set an int to tune the chunk
    # for long-context (keeps the per-chunk decay bounded so exp stays finite).
    chunk_size: Optional[int] = None

    # --- dt-projection bias init (LOAD-BEARING) ---
    # Inverse-softplus of a sample in [dt_min, dt_max] initializes the dt bias.
    # Without this the model fails to learn recall. Carry these into every backend.
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def dt_rank_resolved(self) -> int:
        if self.dt_rank == "auto":
            return math.ceil(self.d_model / 16)
        return int(self.dt_rank)

    def validate(self) -> None:
        if self.vocab_size >= 65536:
            raise ValueError(
                f"vocab_size={self.vocab_size} does not fit uint16 packing (<65536). "
                "Either confirm the tokenizer vocab or change the packed dtype."
            )
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown precision {self.precision!r}")
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive or None")
        if self.d_conv < 1:
            raise ValueError("d_conv must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Union[str, Path]) -> MambaConfig:
    """Load a `MambaConfig` from a YAML file and validate it."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = MambaConfig(**raw)
    cfg.validate()
    return cfg
