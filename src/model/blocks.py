"""Portable architecture description for the Mamba POC.

This module is the single source of truth for model dimensions. It contains NO
backend imports (no MLX, no CUDA/torch) so it can be loaded anywhere and shared
unchanged across backends.

The Mamba block (described here textually; implemented in a backend):

    input projection
      -> split into `main` and `gate`
      -> short causal depthwise conv on `main` (width `d_conv`)
      -> SiLU
      -> selective SSM (Mamba-2 / SSD: scalar A per head; input-dependent B, C,
         delta; chunked-matmul parallel scan)
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
    d_state: int = 16          # SSM state width N (per head, shared B/C group)
    expand: int = 2
    d_conv: int = 4
    # Mamba-2 / SSD head dimension P. d_inner is split into n_heads = d_inner//head_dim
    # heads, each with a SCALAR decay A (the SSD restriction that makes the scan a
    # matmul). Must divide d_inner. The configs override this: poc.yaml uses 64
    # (d_inner 1536 -> 24 heads); toy.yaml uses 16 (d_inner 128 -> 8 heads).
    head_dim: int = 64
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

    # SSD chunk length Q. None => the backend default (64). The chunked-matmul scan
    # processes the sequence in chunks of Q; the sequence is padded up to a multiple
    # of Q (padded steps carry zero input, trimmed from the output).
    chunk_size: Optional[int] = None

    # Recompute each layer's forward in the backward pass instead of retaining its
    # activations (mlx.nn.utils.checkpoint). Trades ~one extra forward for a large
    # memory cut — required at poc scale (without it the 24-layer backward exceeds
    # 32GB and swaps). Off for toy/smoke (tiny; keep exact-resume cheap).
    grad_checkpoint: bool = False

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
    def n_heads(self) -> int:
        return self.d_inner // self.head_dim

    @property
    def dt_rank_resolved(self) -> int:
        if self.dt_rank == "auto":
            return math.ceil(self.d_model / 16)
        return int(self.dt_rank)

    def parameter_breakdown(self) -> dict:
        """Closed-form trainable-parameter count, broken out by named term.

        Returned as an ordered dict {term: count} so callers can inspect where the
        budget goes and so the hybrid-attention work (#67) can add an "attention"
        term without disturbing the existing keys. Verified exactly against the
        built model's `_portable_state_dict()` (tests/test_sizing.py MLX safety-net):
        every weight/bias/buffer registered as a parameter is accounted for here,
        including the per-head SSM `A_log` and `D` (the `ssm_A_D` term).

        Mirrors the layer construction in the backends:
          in_proj  (d_model -> 2*d_inner)      conv (depthwise, width d_conv, + bias)
          x_proj   (d_inner -> dt_rank+2*N)     dt_proj (dt_rank -> n_heads, + bias)
          A_log,D  (n_heads each)               out_proj (d_inner -> d_model)
        plus a pre-block RMSNorm per layer, a final RMSNorm, and the (tied) embedding.
        """
        d_model = self.d_model
        d_inner = self.d_inner
        n_heads = self.n_heads
        dt_rank = self.dt_rank_resolved
        N = self.d_state

        per_layer = (
            d_model                                  # pre-block RMSNorm weight
            + 2 * d_inner * d_model                  # in_proj
            + d_inner * (self.d_conv + 1)            # depthwise conv weight + bias
            + d_inner * (dt_rank + 2 * N)            # x_proj (delta, B, C)
            + dt_rank * n_heads + n_heads            # dt_proj weight + bias
            + 2 * n_heads                            # A_log + D (scalar per head)
            + d_inner * d_model                      # out_proj
        )

        bd = {
            "embedding": self.vocab_size * d_model,
            "layers": self.n_layers * per_layer,
            "final_norm": d_model,
        }
        # Tied embedding reuses the input matrix as the LM head -> no extra params.
        if not self.tie_embeddings:
            bd["lm_head"] = self.vocab_size * d_model
        return bd

    def num_parameters(self) -> int:
        """Total trainable parameters (sum of `parameter_breakdown()`)."""
        return sum(self.parameter_breakdown().values())

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
        if self.head_dim <= 0 or self.d_inner % self.head_dim != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must divide d_inner={self.d_inner} "
                "(d_inner = expand*d_model)."
            )

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Union[str, Path]) -> MambaConfig:
    """Load a `MambaConfig` from a YAML file and validate it."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = MambaConfig(**raw)
    cfg.validate()
    return cfg
