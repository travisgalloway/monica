"""Portable parameter/memory sizing for the Mamba config family.

Turns a `MambaConfig` into the numbers the #65 epic's sizing table needs: exact
parameter count (via `MambaConfig.num_parameters()`), inference footprint, a
documented training-memory estimate, and a formatted family table mapping each
tier (100M -> 1B -> 2B -> 4B) to a GPU/RAM class.

ABOVE THE SEAM — imports no backend (`mlx`/`torch`). Memory figures are closed-form
estimates from the parameter count and a few documented per-param/activation
constants, NOT measurements; treat them as planning aids (the real peak is measured
by `scripts/bench_train_step.py`). Mamba has **no KV cache**, so for inference the
weights ARE essentially the whole footprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from .blocks import MambaConfig, load_config

GIB = 1024 ** 3

# Bytes per element for the weight/activation dtype.
BYTES_PER_DTYPE = {"fp32": 4, "fp16": 2, "bf16": 2}

# Training memory ~ (weights + grads + optimizer state) per parameter. Both options
# below are the epic's "VRAM-tight" levers vs the classic ~16 B/param fp32 AdamW:
#   adamw    ~8 B/param : lean all-bf16 Adam — bf16 weight(2) + bf16 grad(2)
#                         + bf16 Adam m,v(2+2), no fp32 master copy.
#   adam8bit ~10 B/param: 8-bit Adam moments + an fp32 master weight copy for
#                         stability — fp32 master(4) + bf16 weight(2) + bf16 grad(2)
#                         + 8-bit m,v(1+1). The VRAM-tight lever called out in #65.
OPTIMIZER_BYTES_PER_PARAM = {"adamw": 8, "adam8bit": 10}

# Crude per-token activation footprint: a handful of (batch, seq, d_model) tensors
# live per layer in the block (in_proj output ~2*d_inner, conv, ssm y, ...). We fold
# that into a single multiplier on d_model. With gradient checkpointing only one
# layer's activations are held live in the backward (the rest are recomputed), plus
# a per-layer residual save at each checkpoint boundary.
ACT_BYTES_MULTIPLIER = 16  # ~tensors-worth of d_model-wide activations per layer


def _bytes_per(dtype: str) -> int:
    if dtype not in BYTES_PER_DTYPE:
        raise ValueError(f"unknown dtype {dtype!r} (expected one of {list(BYTES_PER_DTYPE)})")
    return BYTES_PER_DTYPE[dtype]


def inference_bytes(cfg: MambaConfig, dtype: str = "bf16") -> int:
    """Bytes to hold the model for inference. No KV cache -> weights are the footprint."""
    return cfg.num_parameters() * _bytes_per(dtype)


def activation_bytes(cfg: MambaConfig, batch_size: int, seq_len: int,
                     dtype: str = "bf16") -> int:
    """Rough peak activation bytes for one training step (documented estimate)."""
    per_layer_token = cfg.d_model * ACT_BYTES_MULTIPLIER * _bytes_per(dtype)
    tokens = batch_size * seq_len
    if cfg.grad_checkpoint:
        # One live layer being recomputed + a cheap residual save per checkpoint boundary.
        return tokens * (per_layer_token + cfg.n_layers * cfg.d_model * _bytes_per(dtype))
    return tokens * per_layer_token * cfg.n_layers


def training_bytes(cfg: MambaConfig, optimizer: str = "adam8bit",
                   batch_size: int = 8, seq_len: int | None = None,
                   dtype: str = "bf16") -> dict:
    """Estimated peak training memory, broken into model+optimizer and activations.

    Returns {model_opt, activations, total} in bytes. `optimizer` selects the
    per-param constant (see OPTIMIZER_BYTES_PER_PARAM). Defaults to a modest
    micro-batch shape; the real run tunes batch/grad-accum to the GPU.
    """
    if optimizer not in OPTIMIZER_BYTES_PER_PARAM:
        raise ValueError(
            f"unknown optimizer {optimizer!r} (expected one of {list(OPTIMIZER_BYTES_PER_PARAM)})"
        )
    seq_len = cfg.seq_len if seq_len is None else seq_len
    model_opt = cfg.num_parameters() * OPTIMIZER_BYTES_PER_PARAM[optimizer]
    acts = activation_bytes(cfg, batch_size, seq_len, dtype)
    return {"model_opt": model_opt, "activations": acts, "total": model_opt + acts}


# GPU/RAM tiers keyed on parameter count, matching the #65 sizing table. Bucketed on
# params (not on the activation-sensitive train total) so the column tracks model size
# the way the epic's table does. (lo_exclusive, hi_inclusive, gpu_train, ram_infer)
_TIERS = [
    (0,        0.3e9, "T4 16GB / L4 24GB",       "any (<1GB)"),
    (0.3e9,    1.3e9, "L4 / A10 24GB",            "16GB Mac"),
    (1.3e9,    2.5e9, "A100 40GB / L40S 48GB",    "16GB Mac"),
    (2.5e9,    4.5e9, "A100 80GB / H100 80GB",    "32GB Mac"),
]


def _tier(num_params: int) -> tuple:
    for lo, hi, gpu, ram in _TIERS:
        if lo < num_params <= hi:
            return gpu, ram
    return ">80GB (multi-GPU)", "64GB+ Mac"


def family_row(name: str, cfg: MambaConfig) -> dict:
    """One sizing-table row for a named config."""
    n = cfg.num_parameters()
    gpu, ram = _tier(n)
    return {
        "tier": name,
        "params": n,
        "weights_gb": inference_bytes(cfg, "bf16") / GIB,   # bf16 weights == inference footprint
        "train_gb": training_bytes(cfg)["total"] / GIB,
        "gpu_train": gpu,
        "ram_infer": ram,
    }


def family_table(configs: Iterable[tuple]) -> list:
    """Rows for a list of (name, cfg) pairs, in order."""
    return [family_row(name, cfg) for name, cfg in configs]


def format_family_table(configs: Iterable[tuple]) -> str:
    """Render the family table as fixed-width text (matches the #65 sizing table)."""
    rows = family_table(configs)
    header = f"{'tier':<6} {'params':>10} {'bf16 wt':>9} {'train':>8}  {'GPU (train)':<24} {'RAM (infer)':<10}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['tier']:<6} {r['params'] / 1e6:>9.1f}M {r['weights_gb']:>8.2f}G "
            f"{r['train_gb']:>7.1f}G  {r['gpu_train']:<24} {r['ram_infer']:<10}"
        )
    return "\n".join(lines)


def load_family(config_dir: Union[str, Path] = "config",
                names: Iterable[str] = ("poc", "1b")) -> list:
    """Load (name, cfg) pairs for the config family from `config/<name>.yaml`.

    The 100M `poc` is the cheap architecture-validation rung; `1b` is the single
    target model. (The earlier 2B/4B scale tiers were dropped — see epic #65.)
    """
    config_dir = Path(config_dir)
    return [(name, load_config(config_dir / f"{name}.yaml")) for name in names]
