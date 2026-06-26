"""Portable training-time estimator: param size × hardware -> wall-clock.

Turns a parameter count and a token budget into an estimated training wall-clock
time on a few reference machines (M1 Pro, single H100, 8×H100). It complements
`src/model/sizing.py` (which answers "does it fit?") by answering "how long?".

ABOVE THE SEAM — imports no backend (`mlx`/`torch`). Everything here is closed-form
arithmetic over two well-worn approximations; treat the outputs as PLANNING
ESTIMATES, not measurements (the only measured throughput is the M1 Pro anchor
below — see `scripts/bench_train_step.py` for the real per-step cost).

Two approximations:

  * Training compute  ~  6 · N_params · N_tokens   (the standard Kaplan/Chinchilla
    forward+backward FLOPs estimate: ~2·N per token forward, ~4·N backward).
  * Achieved throughput  =  peak_flops · MFU · n_devices · scaling_efficiency
    where MFU (model-FLOPs utilization) and the multi-GPU scaling efficiency are
    the tunable assumptions.

The **M1 Pro** entry is not assumed — it is CALIBRATED from the one measured point
in the repo: ~99 s/step at 131,072 tokens/step for the ~127M `poc` config
(CLAUDE.md, issue #30). That fixes M1 Pro's achieved throughput at ~1.0 TFLOP/s
and makes the M1 Pro column reproduce the documented "3B-token run ≈ 26 days".
The H100 entries use H100 bf16 dense peak × an assumed MFU; there is no in-repo
H100 benchmark yet, so they are clearly labelled estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from .sizing import load_family

TFLOP = 1e12

# --- 6·N·D training-FLOPs constant -------------------------------------------
FLOPS_PER_PARAM_PER_TOKEN = 6  # ~2N fwd + ~4N bwd per token

# --- Measured calibration anchor (CLAUDE.md / issue #30) ----------------------
# config/poc.yaml at the standard protocol: batch 32 × grad_accum 4 × seq 1024.
ANCHOR_PARAMS = 126_731_712       # == load_config("config/poc.yaml").num_parameters()
ANCHOR_TOKENS_PER_STEP = 131_072  # 32 × 4 × 1024
ANCHOR_SECONDS_PER_STEP = 99.0

# --- Chinchilla compute-optimal token budget ---------------------------------
CHINCHILLA_TOKENS_PER_PARAM = 20

# --- H100 reference peak (bf16, dense, no sparsity) ---------------------------
H100_PEAK_TFLOPS = 990.0
DEFAULT_MFU = 0.40            # plausible bf16 MFU for a tuned SSM/attention hybrid
DEFAULT_SCALING = 0.85        # 8-GPU near-linear scaling efficiency assumption


def training_flops(params: int, tokens: float) -> float:
    """Estimated total training FLOPs for `tokens` tokens (6·N·D)."""
    return FLOPS_PER_PARAM_PER_TOKEN * params * tokens


def chinchilla_tokens(params: int) -> int:
    """Compute-optimal token budget: 20 tokens per parameter."""
    return CHINCHILLA_TOKENS_PER_PARAM * params


@dataclass(frozen=True)
class Hardware:
    """A reference machine and its achieved (effective) training throughput.

    `effective_flops` is FLOP/s actually delivered to training — for GPUs it is
    `peak_tflops·TFLOP · mfu · n_devices · scaling`; for the calibrated M1 Pro it
    is derived from the measured bench point.
    """

    name: str
    effective_flops: float
    note: str
    peak_tflops: Optional[float] = None
    mfu: Optional[float] = None
    n_devices: int = 1
    scaling: float = 1.0
    calibrated: bool = False


def _m1pro_effective_flops() -> float:
    """Achieved FLOP/s on M1 Pro, calibrated from the measured 99 s/step anchor."""
    tokens_per_sec = ANCHOR_TOKENS_PER_STEP / ANCHOR_SECONDS_PER_STEP
    return training_flops(ANCHOR_PARAMS, tokens_per_sec)


def gpu_hardware(name: str, peak_tflops: float, mfu: float, n_devices: int,
                 scaling: float, note: str) -> Hardware:
    """Build a GPU Hardware from peak × MFU × devices × scaling."""
    eff = peak_tflops * TFLOP * mfu * n_devices * scaling
    return Hardware(name, eff, note, peak_tflops=peak_tflops, mfu=mfu,
                    n_devices=n_devices, scaling=scaling, calibrated=False)


def default_registry(mfu: float = DEFAULT_MFU,
                     scaling: float = DEFAULT_SCALING) -> dict:
    """The three reference machines, keyed by name (insertion order preserved)."""
    m1pro = Hardware(
        "m1pro", _m1pro_effective_flops(),
        "poc 99 s/step @ 131K tok/step (CLAUDE.md)",
        peak_tflops=10.4, mfu=None, n_devices=1, scaling=1.0, calibrated=True,
    )
    h100 = gpu_hardware(
        "h100", H100_PEAK_TFLOPS, mfu, 1, 1.0,
        f"H100 bf16 peak × {mfu:.0%} MFU (no in-repo bench)",
    )
    cluster = gpu_hardware(
        "8xh100", H100_PEAK_TFLOPS, mfu, 8, scaling,
        f"8×H100 bf16 peak × {mfu:.0%} MFU × {scaling:.0%} scaling",
    )
    return {hw.name: hw for hw in (m1pro, h100, cluster)}


def train_seconds(params: int, tokens: float, hw: Hardware) -> float:
    """Estimated wall-clock seconds to train `params` over `tokens` on `hw`."""
    return training_flops(params, tokens) / hw.effective_flops


def parse_count(text: str) -> int:
    """Parse a human count with a K/M/B/G/T suffix: '270M' -> 270_000_000."""
    s = text.strip().upper()
    mult = 1.0
    suffixes = {"K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}
    if s and s[-1] in suffixes:
        mult = suffixes[s[-1]]
        s = s[:-1]
    return int(float(s) * mult)


def format_count(params: int) -> str:
    """Render a param count compactly: 126731712 -> '127M'."""
    if params >= 1e9:
        return f"{params / 1e9:.2f}B"
    if params >= 1e6:
        return f"{params / 1e6:.0f}M"
    return str(params)


def format_time(seconds: float) -> str:
    """Render seconds in the largest sensible unit: s / m / h / d / y."""
    minute, hour, day, year = 60.0, 3600.0, 86400.0, 86400.0 * 365.0
    if seconds < minute:
        return f"{seconds:.0f} s"
    if seconds < hour:
        return f"{seconds / minute:.1f} m"
    if seconds < day:
        return f"{seconds / hour:.1f} h"
    if seconds < year:
        return f"{seconds / day:.1f} d"
    return f"{seconds / year:.1f} y"


def default_sizes(config_dir: Union[str, Path] = "config") -> list:
    """(label, params) ladder: real configs (poc, 1b) + a generic round ladder."""
    sizes = [(name, cfg.num_parameters()) for name, cfg in load_family(config_dir)]
    for label in ("270M", "3B", "7B"):
        sizes.append((label, parse_count(label)))
    return sorted(sizes, key=lambda lp: lp[1])


def format_report(sizes: Iterable[tuple], hardware: Iterable[Hardware],
                  fixed_tokens: Optional[int] = None) -> str:
    """Render the full estimate: an assumptions block + a size × hardware table.

    `sizes` is an iterable of (label, params). Token budget per row is
    `fixed_tokens` if given, else Chinchilla 20×params.
    """
    sizes = list(sizes)
    hardware = list(hardware)

    budget_desc = (f"{format_count(fixed_tokens)} tokens (fixed, --tokens)"
                   if fixed_tokens is not None
                   else f"{CHINCHILLA_TOKENS_PER_PARAM}× params (Chinchilla compute-optimal)")

    lines = [
        "Training-time estimate  (PLANNING ESTIMATE — not a benchmark)",
        f"  compute model : {FLOPS_PER_PARAM_PER_TOKEN}·N·D FLOPs  (fwd+bwd)",
        f"  token budget  : {budget_desc}",
        "  throughput    :",
    ]
    for hw in hardware:
        tag = "measured-calibrated" if hw.calibrated else "estimate"
        lines.append(f"    {hw.name:<8} {hw.effective_flops / TFLOP:>8.1f} TFLOP/s  "
                     f"({tag}: {hw.note})")
    lines.append("")

    # Table: size | params | tokens | one time column per hardware.
    name_w = max(5, max(len(label) for label, _ in sizes))
    col_w = max(8, max(len(hw.name) for hw in hardware))
    header = (f"{'size':<{name_w}} {'params':>8} {'tokens':>9}  "
              + "  ".join(f"{hw.name:>{col_w}}" for hw in hardware))
    lines.append(header)
    lines.append("-" * len(header))

    for label, params in sizes:
        tokens = fixed_tokens if fixed_tokens is not None else chinchilla_tokens(params)
        cells = []
        for hw in hardware:
            cells.append(f"{format_time(train_seconds(params, tokens, hw)):>{col_w}}")
        lines.append(
            f"{label:<{name_w}} {format_count(params):>8} {format_count(tokens):>9}  "
            + "  ".join(cells)
        )
    return "\n".join(lines)
