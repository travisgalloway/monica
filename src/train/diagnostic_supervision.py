"""Diagnostic supervision: rejection-sampled FT + contrastive hard negatives (#227).

Part of #198 (SSI-P3-T1/T2):
  - Rejection-sampled FT (SSI-P3-T1):
    Sample n candidate completions per prompt, keep zero-error survivors (zero diagnostics,
    no #225 M5 escape hatches, not degenerate), train with SFT for 2–3 iterative rounds.
    Compared against a random-filter control arm (where candidates are kept at random
    without diagnostic gating, producing a flat clean-rate curve).
    Tracks empirical entropy and distinct-n (distinct-1, distinct-2, distinct-3) to detect
    and monitor diversity collapse across rounds.

  - Contrastive hard negatives (SSI-P3-T2):
    In-scope-but-wrong identifiers from `completions(pos)` mined as free negatives.
    Auxiliary margin loss: L = L_sft + aux_weight * max(0, gamma - score(pos) + score(neg)).
    Three mining arms:
      - `random`: negative identifier chosen from outside identifiers / unrelated pool.
      - `in-scope`: negative identifiers chosen from candidates in `completions(pos)` (any
        candidate in scope with candidate.label != pos_label).
      - `typed`: hard negative identifiers chosen from `completions(pos)` that match the
        positive target's type / kind (e.g. property vs property, method vs method).
    Acceptance: typed negatives improve resolve-correct without hurting edit-sim.

SSI measurement contract (#225 M1–M5):
  Declared and validated via `src.eval.ssi_contract.ArmSpec` and `validate_arms`. Every
  treatment arm declares its variable, references its baseline, and carries an
  availability-vs-use null sibling (M4: signal_available=True, signal_used=False).

ABOVE THE SEAM — stdlib + numpy only. No hardware backend imports (mlx, torch,
bitsandbytes) at module level (guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from ..eval.lsp_eval import RESOLUTION_CODES
from ..eval.ssi_contract import ArmSpec, ContractViolation, validate_arms
from ..lsp.diagnostics import Diagnostic, find_escape_hatches
from .verifiers import _select_hatches, is_degenerate_output

# Tokenization regex for code n-grams: identifiers/keywords, numeric literals,
# multi-char operators, single punctuation. Whitespace is stripped.
_CODE_TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+(?:\.\d+)?|===|!==|==|!=|<=|>=|=>|\+\+|--|[^\s\w]")


# --------------------------------------------------------------------------- #
# Diversity metrics (distinct-n & entropy) to track diversity collapse
# --------------------------------------------------------------------------- #

def tokenize_code(text: str) -> List[str]:
    """Tokenize code text into syntactic tokens for n-gram and diversity metrics."""
    if not text:
        return []
    return _CODE_TOKEN_RE.findall(text)


def extract_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Extract sequence of n-grams from token sequence. Returns empty list if len < n."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Sequence[str], n: int = 1) -> float:
    """Distinct-n ratio: |unique n-grams| / |total n-grams| across all texts.

    Returns 0.0 if total n-grams is 0. A falling distinct-n across fine-tuning
    rounds indicates diversity collapse.
    """
    if not texts or n <= 0:
        return 0.0
    all_ngrams: List[Tuple[str, ...]] = []
    for t in texts:
        toks = tokenize_code(t)
        all_ngrams.extend(extract_ngrams(toks, n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / float(len(all_ngrams))


def distinct_n_metrics(texts: Sequence[str]) -> Dict[str, float]:
    """Calculate distinct-1, distinct-2, and distinct-3 metrics for a corpus."""
    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
    }


def ngram_entropy(texts: Sequence[str], n: int = 1) -> float:
    """Shannon entropy (in bits) over the empirical n-gram distribution.

    H = - sum_g p(g) * log2(p(g)).
    Returns 0.0 if no n-grams exist. A sharp decrease across rounds signals
    mode collapse onto a small subset of patterns.
    """
    if not texts or n <= 0:
        return 0.0
    counts: Dict[Tuple[str, ...], int] = {}
    total = 0
    for t in texts:
        toks = tokenize_code(t)
        ngrams = extract_ngrams(toks, n)
        for g in ngrams:
            counts[g] = counts.get(g, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    ent = 0.0
    for cnt in counts.values():
        p = cnt / float(total)
        ent -= p * math.log2(p)
    return ent


@dataclass(frozen=True)
class DiversityReport:
    """Diversity and survivor metrics for one round of rejection-sampled FT."""
    round_idx: int
    n_prompts: int
    n_samples: int
    n_survivors: int
    survivor_rate: float
    clean_rate: float
    distinct_1: float
    distinct_2: float
    distinct_3: float
    entropy: float


class DiversityTracker:
    """Tracks diversity collapse metrics (entropy, distinct-n) across SFT rounds."""

    def __init__(self) -> None:
        self.reports: List[DiversityReport] = []

    def record_round(
        self,
        round_idx: int,
        completions: Sequence[str],
        survivors: Sequence[str],
        *,
        n_prompts: int,
    ) -> DiversityReport:
        """Record metrics for a generation + filtering round."""
        n_samples = len(completions)
        n_surv = len(survivors)
        surv_rate = (n_surv / float(n_samples)) if n_samples > 0 else 0.0
        clean_rate = surv_rate

        metrics = distinct_n_metrics(completions)
        ent = ngram_entropy(completions, n=1)

        rep = DiversityReport(
            round_idx=round_idx,
            n_prompts=n_prompts,
            n_samples=n_samples,
            n_survivors=n_surv,
            survivor_rate=surv_rate,
            clean_rate=clean_rate,
            distinct_1=metrics["distinct_1"],
            distinct_2=metrics["distinct_2"],
            distinct_3=metrics["distinct_3"],
            entropy=ent,
        )
        self.reports.append(rep)
        return rep

    def detect_collapse(self, drop_threshold: float = 0.25) -> Dict[str, Any]:
        """Check whether entropy or distinct-1 dropped by more than `drop_threshold`
        relative to the first round.

        Returns {collapsed: bool, entropy_drop: float, distinct_1_drop: float}.
        """
        if len(self.reports) < 2:
            return {"collapsed": False, "entropy_drop": 0.0, "distinct_1_drop": 0.0}
        first = self.reports[0]
        latest = self.reports[-1]

        ent_drop = (first.entropy - latest.entropy) / first.entropy if first.entropy > 0 else 0.0
        d1_drop = (first.distinct_1 - latest.distinct_1) / first.distinct_1 if first.distinct_1 > 0 else 0.0

        collapsed = (ent_drop > drop_threshold) or (d1_drop > drop_threshold)
        return {
            "collapsed": collapsed,
            "entropy_drop": ent_drop,
            "distinct_1_drop": d1_drop,
        }


# --------------------------------------------------------------------------- #
# Rejection-sampled FT: Zero-error survivor filtering & Random-filter control
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CandidateEvaluation:
    prompt: str
    completion: str
    artifact: str
    diagnostics: Tuple[Diagnostic, ...] = ()
    escape_hatches: Tuple[str, ...] = ()
    degenerate_reason: Optional[str] = None
    is_clean: bool = True


class RejectionFilter:
    """Filters candidate completions, keeping only zero-error survivors.

    Guards enforced:
      1. Diagnostic count: zero diagnostics reported by `diagnose_fn`.
      2. Escape-hatch lint gate (#225 M5): zero escape hatches in completion
         (evaluated via `find_escape_hatches`).
      3. Anti-degenerate guard: completion is not empty, whitespace-only,
         comment-only, or too short (evaluated via `is_degenerate_output`).
    """

    def __init__(
        self,
        diagnose_fn: Callable[[str], Sequence[Diagnostic]],
        *,
        hatches: str = "superset",
        min_code_chars: int = 2,
    ) -> None:
        self.diagnose_fn = diagnose_fn
        self.hatches_mode = hatches
        self.min_code_chars = min_code_chars

    def evaluate_candidate(self, prompt: str, completion: str) -> CandidateEvaluation:
        """Evaluate a single candidate completion against all three gates."""
        artifact = prompt + completion

        # 1. Anti-degenerate guard (evaluated on completion alone)
        degen = is_degenerate_output(completion, min_code_chars=self.min_code_chars)

        # 2. Escape hatch gate (evaluated on completion alone)
        raw_hatches = find_escape_hatches(completion)
        filtered_hatches = tuple(_select_hatches(raw_hatches, self.hatches_mode))

        # 3. Diagnostic oracle (evaluated on full artifact)
        diags = tuple(self.diagnose_fn(artifact))

        is_clean = (len(diags) == 0) and (len(filtered_hatches) == 0) and (degen is None)

        return CandidateEvaluation(
            prompt=prompt,
            completion=completion,
            artifact=artifact,
            diagnostics=diags,
            escape_hatches=filtered_hatches,
            degenerate_reason=degen,
            is_clean=is_clean,
        )

    def filter_candidates(
        self,
        prompt: str,
        candidates: Sequence[str],
    ) -> List[CandidateEvaluation]:
        """Evaluate candidates and return only the zero-error survivors."""
        evals = [self.evaluate_candidate(prompt, c) for c in candidates]
        return [e for e in evals if e.is_clean]


class RandomFilter:
    """Control arm filter: selects candidate completions uniformly at random
    without diagnostic or escape-hatch filtering.

    Used to isolate whether improvements in clean-rate come from the diagnostic
    gate or simply from self-training on model-generated completions.
    Under this control, clean-rate remains flat across rounds.
    """

    def __init__(self, *, keep_fraction: float = 0.5, seed: Optional[int] = None) -> None:
        self.keep_fraction = keep_fraction
        self.rng = random.Random(seed)

    def select(
        self,
        candidates: Sequence[str],
        *,
        n_survivors_target: Optional[int] = None,
    ) -> List[str]:
        """Select candidates randomly. If `n_survivors_target` is provided,
        samples min(n_survivors_target, len(candidates)) items.
        """
        if not candidates:
            return []
        if n_survivors_target is not None:
            k = min(max(0, n_survivors_target), len(candidates))
        else:
            k = max(1, int(round(len(candidates) * self.keep_fraction)))
            k = min(k, len(candidates))
        return self.rng.sample(list(candidates), k)


# --------------------------------------------------------------------------- #
# Contrastive hard negatives: Mining arms (random / in-scope / typed)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CandidateItem:
    """LSP candidate item representation for negative mining."""
    label: str
    kind: Optional[int] = None
    detail: Optional[str] = None


def normalize_candidate_items(
    items: Sequence[Union[CandidateItem, Mapping[str, Any], str]],
) -> List[CandidateItem]:
    """Normalize mixed inputs (objects, dicts, strings) into CandidateItem list."""
    out: List[CandidateItem] = []
    for it in items:
        if isinstance(it, CandidateItem):
            out.append(it)
        elif isinstance(it, str):
            out.append(CandidateItem(label=it))
        elif isinstance(it, Mapping):
            out.append(CandidateItem(
                label=it.get("label", ""),
                kind=it.get("kind"),
                detail=it.get("detail"),
            ))
        elif hasattr(it, "label"):
            out.append(CandidateItem(
                label=getattr(it, "label", ""),
                kind=getattr(it, "kind", None),
                detail=getattr(it, "detail", None),
            ))
        else:
            out.append(CandidateItem(label=str(it)))
    return out


def mine_negatives(
    positive_label: str,
    candidates: Sequence[Union[CandidateItem, Mapping[str, Any], str]],
    arm: str = "typed",
    *,
    pool: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
    max_negatives: int = 1,
) -> List[str]:
    """Mine negative identifier labels for contrastive supervision.

    Arms:
      - `random`: Drawn from `pool` (external/global identifiers not present in `candidates`).
      - `in-scope`: Drawn from in-scope candidates returned by `completions(pos)` where
        candidate.label != positive_label, without type filtering.
      - `typed`: Hard negatives drawn from `completions(pos)` that match the positive
        target's `kind` (e.g. Method, Property) or `detail` (type signature). If no same-kind
        candidate exists, falls back gracefully to in-scope negatives.

    Returns up to `max_negatives` negative identifier strings.
    """
    if arm not in ("random", "in-scope", "typed"):
        raise ValueError(f"unknown arm {arm!r}, expected 'random', 'in-scope', or 'typed'")

    rng = random.Random(seed)
    norm_candidates = normalize_candidate_items(candidates)
    candidate_labels = {c.label for c in norm_candidates}

    if arm == "random":
        allowed_pool = [x for x in (pool or ()) if x not in candidate_labels and x != positive_label]
        if not allowed_pool:
            default_pool = ["randomIdentA", "dummyVarB", "fallbackValC", "externalPropD"]
            allowed_pool = [x for x in default_pool if x not in candidate_labels and x != positive_label]
        if not allowed_pool:
            return []
        k = min(max_negatives, len(allowed_pool))
        return rng.sample(allowed_pool, k)

    other_candidates = [c for c in norm_candidates if c.label and c.label != positive_label]
    if not other_candidates:
        return []

    if arm == "in-scope":
        k = min(max_negatives, len(other_candidates))
        selected = rng.sample(other_candidates, k)
        return [c.label for c in selected]

    # Arm == "typed"
    pos_item = next((c for c in norm_candidates if c.label == positive_label), None)
    pos_kind = pos_item.kind if pos_item is not None else None
    pos_detail = pos_item.detail.strip().lower() if (pos_item is not None and pos_item.detail) else None

    typed_matches: List[CandidateItem] = []
    if pos_kind is not None:
        typed_matches = [c for c in other_candidates if c.kind == pos_kind]

    if pos_detail and not typed_matches:
        typed_matches = [c for c in other_candidates if c.detail and c.detail.strip().lower() == pos_detail]

    selection_pool = typed_matches if typed_matches else other_candidates
    k = min(max_negatives, len(selection_pool))
    selected = rng.sample(selection_pool, k)
    return [c.label for c in selected]


# --------------------------------------------------------------------------- #
# Auxiliary Margin Loss (Pure NumPy / Math)
# --------------------------------------------------------------------------- #

def margin_loss(
    pos_score: float,
    neg_score: float,
    *,
    margin: float = 1.0,
) -> float:
    """Scalar margin ranking loss: max(0, margin - (pos_score - neg_score))."""
    return max(0.0, margin - (pos_score - neg_score))


def contrastive_margin_loss(
    pos_scores: Union[float, Sequence[float], np.ndarray],
    neg_scores: Union[float, Sequence[float], np.ndarray],
    *,
    margin: float = 1.0,
) -> np.ndarray:
    """Vectorized contrastive margin loss over batches of positive and negative scores.

    loss = np.maximum(0.0, margin - (pos_scores - neg_scores))
    """
    pos = np.asarray(pos_scores, dtype=np.float32)
    neg = np.asarray(neg_scores, dtype=np.float32)
    return np.maximum(0.0, margin - (pos - neg))


def auxiliary_contrastive_loss(
    sft_loss: float,
    pos_score: float,
    neg_scores: Sequence[float],
    *,
    margin: float = 1.0,
    aux_weight: float = 0.5,
) -> float:
    """Combined SFT + auxiliary contrastive margin loss:

    total_loss = sft_loss + aux_weight * mean(max(0, margin - (pos_score - neg_score)))
    When `aux_weight == 0.0` (the M4 null arm), returns `sft_loss` unchanged.
    """
    if aux_weight == 0.0 or not neg_scores:
        return float(sft_loss)
    margins = [margin_loss(pos_score, ns, margin=margin) for ns in neg_scores]
    mean_margin = sum(margins) / float(len(margins))
    return float(sft_loss + aux_weight * mean_margin)


# --------------------------------------------------------------------------- #
# Edit Similarity (Levenshtein) & Acceptance Evaluation
# --------------------------------------------------------------------------- #

def levenshtein_distance(s1: str, s2: str) -> int:
    """Dynamic programming Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(s2)]


def edit_similarity(pred: str, target: str) -> float:
    """Normalized string edit similarity in [0.0, 1.0].

    edit_sim = 1.0 - (LevenshteinDistance(pred, target) / max(len(pred), len(target))).
    Returns 1.0 if both strings are empty.
    """
    if pred == target:
        return 1.0
    max_len = max(len(pred), len(target))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(pred, target)
    return max(0.0, 1.0 - float(dist) / float(max_len))


def is_resolve_correct(diagnostics: Sequence[Diagnostic]) -> bool:
    """Check if completion resolved name correctly without any resolution-code
    diagnostics (`RESOLUTION_CODES`: TS2304, TS2339, TS2551, TS2552, TS2694, TS2724).
    """
    codes = {d.code for d in diagnostics}
    return not any(c in RESOLUTION_CODES for c in codes)


# --------------------------------------------------------------------------- #
# SSI Measurement Contract (#225 M1–M5) Arm Declarations
# --------------------------------------------------------------------------- #

def build_diagnostic_supervision_arms(
    seeds: Tuple[int, ...] = (0, 1, 2),
) -> List[ArmSpec]:
    """Declare and return the complete #225-compliant arm set for #227.

    Satisfies:
      - M1 (one variable per arm, unique variables against baseline).
      - M2 (>= 3 seeds).
      - M4 (availability-vs-use null arm for every signal_used=True arm).
    """
    declared_seeds = tuple(sorted(set(seeds)))
    if len(declared_seeds) < 3:
        raise ValueError(f"M2 requires >= 3 seeds, got {seeds}")

    arms = [
        # Rejection-sampled FT arms:
        ArmSpec(
            name="rs-ft-baseline",
            variable="baseline",
            baseline="rs-ft-baseline",
            signal_available=False,
            signal_used=False,
            seeds=declared_seeds,
            notes="Standard SFT baseline without rejection sampling",
        ),
        ArmSpec(
            name="rs-ft-control",
            variable="random_filter_control",
            baseline="rs-ft-baseline",
            signal_available=False,
            signal_used=False,
            seeds=declared_seeds,
            notes="Random-filter control arm (flat clean-rate over rounds)",
        ),
        ArmSpec(
            name="rs-ft-null",
            variable="rejection_filter",
            baseline="rs-ft-baseline",
            signal_available=True,
            signal_used=False,
            seeds=declared_seeds,
            notes="M4 null: diagnostic filter evaluated but not used to select survivors",
        ),
        ArmSpec(
            name="rs-ft-treatment",
            variable="rejection_filter",
            baseline="rs-ft-baseline",
            signal_available=True,
            signal_used=True,
            seeds=declared_seeds,
            notes="Rejection-sampled FT: sample n, keep zero-error survivors across 2-3 rounds",
        ),

        # Contrastive hard negatives arms:
        ArmSpec(
            name="contrastive-baseline",
            variable="baseline",
            baseline="contrastive-baseline",
            signal_available=False,
            signal_used=False,
            seeds=declared_seeds,
            notes="SFT baseline without contrastive margin loss",
        ),
        ArmSpec(
            name="contrastive-random-null",
            variable="contrastive_random",
            baseline="contrastive-baseline",
            signal_available=True,
            signal_used=False,
            seeds=declared_seeds,
            notes="M4 null: random negative mined, aux margin loss weight 0.0",
        ),
        ArmSpec(
            name="contrastive-random",
            variable="contrastive_random",
            baseline="contrastive-baseline",
            signal_available=True,
            signal_used=True,
            seeds=declared_seeds,
            notes="Random negatives from outside scope with aux margin loss",
        ),
        ArmSpec(
            name="contrastive-inscope-null",
            variable="contrastive_inscope",
            baseline="contrastive-baseline",
            signal_available=True,
            signal_used=False,
            seeds=declared_seeds,
            notes="M4 null: in-scope negatives mined from completions(pos), aux loss weight 0.0",
        ),
        ArmSpec(
            name="contrastive-inscope",
            variable="contrastive_inscope",
            baseline="contrastive-baseline",
            signal_available=True,
            signal_used=True,
            seeds=declared_seeds,
            notes="In-scope-but-wrong identifiers from completions(pos) with aux margin loss",
        ),
        ArmSpec(
            name="contrastive-typed-null",
            variable="contrastive_typed",
            baseline="contrastive-baseline",
            signal_available=True,
            signal_used=False,
            seeds=declared_seeds,
            notes="M4 null: typed negatives mined from completions(pos), aux loss weight 0.0",
        ),
        ArmSpec(
            name="contrastive-typed",
            variable="contrastive_typed",
            baseline="contrastive-baseline",
            signal_available=True,
            signal_used=True,
            seeds=declared_seeds,
            notes="Typed hard negatives matching target kind/type from completions(pos) with aux margin loss",
        ),
    ]

    validate_arms(arms)
    return arms
