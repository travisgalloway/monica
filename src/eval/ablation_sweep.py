"""Small-model ablation sweep harness (#219).

Defines the ablation grid across:
  - attention ratio (8%, 12%, 16%) with arbitrary-depth attention layer placement
  - d_state (128 vs 256)
  - mixture architecture variants (Jamba-style with 1 shared expert vs Routing-Mamba with 0 shared experts)

Evaluates candidates on recall (multi-query associative recall / cross-file recall),
applies the mid-training routing kill-criterion (#217 / src.eval.moe_routing), and
ranks the candidates to select the winning configuration for the full run.

ABOVE THE SEAM: pure Python/numpy + stdlib. Never imports MLX or torch directly.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..model.blocks import MambaConfig, load_config
from .moe_routing import kill_check, specialization_report


DEFAULT_ATTN_RATIOS = (0.08, 0.12, 0.16)
DEFAULT_D_STATES = (128, 256)
DEFAULT_VARIANTS = ("jamba", "routing_mamba")


def compute_attention_layers(n_layers: int, attn_ratio: float,
                             force_final_layer: bool = True) -> List[int]:
    """Compute explicit 0-indexed attention layer positions for an attention ratio.

    Uses arbitrary-depth layer placement. When `force_final_layer=True` (default),
    places the top attention block at `n_layers - 1` (the FIM constraint documented
    in docs/design/13-code-model-moe.md and src/eval/fim_eval.py).
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    k = max(1, round(n_layers * attn_ratio))
    if k > n_layers:
        k = n_layers

    if force_final_layer:
        indices = [round((i + 1) * n_layers / k) - 1 for i in range(k)]
    else:
        indices = [round((i + 0.5) * n_layers / k) for i in range(k)]

    unique = sorted(set(indices))
    if len(unique) < k or unique[-1] >= n_layers or unique[0] < 0:
        # Uniform linear fallback
        step = (n_layers - 1) / max(1, k - 1)
        unique = sorted({min(n_layers - 1, max(0, round(i * step))) for i in range(k)})
    return unique


@dataclass
class AblationCandidate:
    """A single configuration candidate in the ablation grid."""
    name: str
    variant: str
    attn_ratio: float
    d_state: int
    config: MambaConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AblationResult:
    """Evaluation result for an ablation candidate."""
    candidate: AblationCandidate
    recall_score: float
    routing_report: Optional[dict] = None
    kill_result: Optional[dict] = None
    killed: bool = False
    rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def default_small_base_config() -> MambaConfig:
    """Default base config for the small MoE ablation sweep (56 layers, d_model 768)."""
    cfg = MambaConfig(
        d_model=768,
        n_layers=56,
        d_state=128,
        expand=2,
        d_conv=4,
        head_dim=64,
        dt_rank="auto",
        attn_every=None,
        attn_layers=None,
        n_attn_heads=None,
        moe_every=3,
        n_experts=8,
        top_k=2,
        moe_d_ff=1536,
        moe_balance_rate=0.001,
        n_shared_experts=1,
        vocab_size=49152,
        seq_len=2048,
        tie_embeddings=True,
        precision="fp32",
        chunk_size=None,
        grad_checkpoint=True,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=0.0001,
    )
    cfg.validate()
    return cfg


def generate_ablation_grid(
    base_cfg: Optional[MambaConfig] = None,
    *,
    attn_ratios: Sequence[float] = DEFAULT_ATTN_RATIOS,
    d_states: Sequence[int] = DEFAULT_D_STATES,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    force_final_layer: bool = True,
) -> List[AblationCandidate]:
    """Generate the full ablation grid across attention ratio, d_state, and mixture variants.

    Grid combinations: len(attn_ratios) x len(d_states) x len(variants).
    Default 3 x 2 x 2 = 12 configurations.
    """
    if base_cfg is None:
        base_cfg = default_small_base_config()

    candidates: List[AblationCandidate] = []
    for var in variants:
        if var not in ("jamba", "routing_mamba"):
            raise ValueError(f"unknown variant {var!r} (expected 'jamba' or 'routing_mamba')")
        n_shared = 1 if var == "jamba" else 0

        for ratio in attn_ratios:
            attn_layers = compute_attention_layers(
                base_cfg.n_layers, ratio, force_final_layer=force_final_layer
            )
            pct = int(round(ratio * 100))

            for ds in d_states:
                name = f"{var}_attn{pct:02d}_dstate{ds}"
                cfg_dict = base_cfg.to_dict()
                updates = {
                    "attn_every": None,
                    "attn_layers": list(attn_layers),
                    "d_state": ds,
                    "n_shared_experts": n_shared,
                }
                # If seeded from dense E=1 checkpoint (#200), promote to small MoE rung defaults (E=8, top_k=2)
                if cfg_dict.get("n_experts", 0) <= 1:
                    updates["n_experts"] = 8
                    updates["top_k"] = 2
                    if cfg_dict.get("moe_balance_rate") is None:
                        updates["moe_balance_rate"] = 0.001
                cfg_dict.update(updates)
                cfg = MambaConfig(**cfg_dict)
                cfg.validate()

                meta = {
                    "n_attention_layers": cfg.n_attention_layers,
                    "attention_percentage": (cfg.n_attention_layers / cfg.n_layers) * 100.0,
                    "n_moe_layers": cfg.n_moe_layers,
                    "n_shared_experts": cfg.n_shared_experts,
                    "num_parameters": cfg.num_parameters(),
                    "active_num_parameters": cfg.active_num_parameters(),
                }
                candidates.append(
                    AblationCandidate(
                        name=name,
                        variant=var,
                        attn_ratio=ratio,
                        d_state=ds,
                        config=cfg,
                        metadata=meta,
                    )
                )

    return candidates


def evaluate_candidate(
    candidate: AblationCandidate,
    eval_fn: Callable[[AblationCandidate], Tuple[float, Optional[dict]]],
    *,
    kill_threshold: float = 0.90,
    kill_pair: Tuple[str, str] = ("typescript", "math"),
) -> AblationResult:
    """Evaluate one candidate with `eval_fn` and apply the routing kill-criterion."""
    recall_score, routing_report = eval_fn(candidate)
    kill_res = None
    killed = False

    if routing_report is not None:
        kill_res = kill_check(routing_report, pair=kill_pair, threshold=kill_threshold)
        killed = bool(kill_res.get("triggered", False))

    return AblationResult(
        candidate=candidate,
        recall_score=recall_score,
        routing_report=routing_report,
        kill_result=kill_res,
        killed=killed,
    )


def rank_candidates(results: Sequence[AblationResult]) -> List[AblationResult]:
    """Rank candidates: non-killed sorted by recall_score descending, killed at the end."""
    survived = [r for r in results if not r.killed]
    killed = [r for r in results if r.killed]

    survived_sorted = sorted(survived, key=lambda r: r.recall_score, reverse=True)
    killed_sorted = sorted(killed, key=lambda r: r.recall_score, reverse=True)

    ranked: List[AblationResult] = []
    current_rank = 1
    for r in survived_sorted:
        res = dataclasses.replace(r, rank=current_rank)
        ranked.append(res)
        current_rank += 1

    for r in killed_sorted:
        res = dataclasses.replace(r, rank=None)
        ranked.append(res)

    return ranked


def select_winner(ranked_results: Sequence[AblationResult]) -> AblationResult:
    """Return the top-ranking non-killed configuration."""
    survived = [r for r in ranked_results if not r.killed and r.rank is not None]
    if not survived:
        raise RuntimeError(
            "No candidate survived the routing kill-criterion! All ablation configurations were eliminated."
        )
    return min(survived, key=lambda r: r.rank if r.rank is not None else float("inf"))


def mock_evaluator(
    candidate: AblationCandidate,
    *,
    seed: int = 42,
    simulate_kill_for: Optional[Sequence[str]] = None,
) -> Tuple[float, dict]:
    """Deterministic offline evaluator for testing and ablation sweep verification.

    Models recall score as a function of attention ratio and d_state, and
    generates synthetic domain histograms for the routing kill-check.
    """
    rng = np.random.default_rng(seed + hash(candidate.name) % 10000)

    # Higher attention ratio and wider d_state improve recall
    base_recall = 0.55
    attn_bonus = candidate.attn_ratio * 1.5      # +0.12 for 8%, +0.18 for 12%, +0.24 for 16%
    dstate_bonus = 0.05 if candidate.d_state == 256 else 0.02
    variant_bonus = 0.03 if candidate.variant == "jamba" else 0.01  # shared expert stabilizes
    noise = float(rng.normal(0.0, 0.005))
    recall_score = float(np.clip(base_recall + attn_bonus + dstate_bonus + variant_bonus + noise, 0.0, 1.0))

    n_moe = candidate.config.n_moe_layers
    n_exp = candidate.config.n_experts

    should_kill = False
    if simulate_kill_for and candidate.name in simulate_kill_for:
        should_kill = True

    # Build synthetic routing histograms
    # If should_kill: overlap between typescript and math will exceed threshold
    hist_ts = []
    hist_math = []
    for _ in range(n_moe):
        if should_kill:
            # identical distributions -> overlap ~ 1.0
            base = rng.integers(10, 100, size=n_exp)
            hist_ts.append(base.tolist())
            hist_math.append((base + rng.integers(0, 2, size=n_exp)).tolist())
        else:
            # specialized distributions -> low overlap
            ts = rng.integers(10, 100, size=n_exp)
            ts[0:n_exp // 2] += 200  # TS prefers lower half experts
            math_counts = rng.integers(10, 100, size=n_exp)
            math_counts[n_exp // 2:] += 200  # Math prefers upper half
            hist_ts.append(ts.tolist())
            hist_math.append(math_counts.tolist())

    hists = {
        "typescript": hist_ts,
        "math": hist_math,
    }
    routing_report = specialization_report(hists)

    return recall_score, routing_report


class AblationSweepRunner:
    """Runner for the small-model ablation sweep."""

    def __init__(
        self,
        base_cfg: Optional[MambaConfig] = None,
        *,
        attn_ratios: Sequence[float] = DEFAULT_ATTN_RATIOS,
        d_states: Sequence[int] = DEFAULT_D_STATES,
        variants: Sequence[str] = DEFAULT_VARIANTS,
        kill_threshold: float = 0.90,
        kill_pair: Tuple[str, str] = ("typescript", "math"),
    ):
        self.base_cfg = base_cfg or default_small_base_config()
        self.candidates = generate_ablation_grid(
            self.base_cfg,
            attn_ratios=attn_ratios,
            d_states=d_states,
            variants=variants,
        )
        self.kill_threshold = kill_threshold
        self.kill_pair = kill_pair

    def run(
        self,
        eval_fn: Callable[[AblationCandidate], Tuple[float, Optional[dict]]],
    ) -> Tuple[List[AblationResult], AblationResult]:
        """Run all candidates through evaluation, rank them, and return (results, winner)."""
        raw_results = [
            evaluate_candidate(
                cand,
                eval_fn,
                kill_threshold=self.kill_threshold,
                kill_pair=self.kill_pair,
            )
            for cand in self.candidates
        ]
        ranked = rank_candidates(raw_results)
        winner = select_winner(ranked)
        return ranked, winner


def format_sweep_table(results: Sequence[AblationResult], winner: Optional[AblationResult] = None) -> str:
    """Format the ablation sweep results as a human-readable table."""
    lines = [
        "| Rank | Candidate Name | Variant | Attn % (Layers) | d_state | Recall | Overlap | Kill Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        c = r.candidate
        rank_str = f"#{r.rank}" if r.rank is not None else "KILLED"
        pct_str = f"{c.attn_ratio * 100:.0f}% ({c.config.n_attention_layers}L)"
        overlap_str = f"{r.kill_result['overlap']:.4f}" if r.kill_result else "N/A"
        kill_status = "TRIGGERED" if r.killed else "PASSED"
        is_winner = " (WINNER)" if winner and winner.candidate.name == c.name else ""
        lines.append(
            f"| {rank_str} | `{c.name}`{is_winner} | {c.variant} | {pct_str} | {c.d_state} | "
            f"{r.recall_score:.4f} | {overlap_str} | {kill_status} |"
        )
    return "\n".join(lines)
