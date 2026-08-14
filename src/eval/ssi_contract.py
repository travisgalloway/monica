"""#225 -- the SSI measurement contract (M1-M5): the shared module every SSI arm
(#226 completion-list logit masking, #227 diagnostic supervision, #230 RLVR/
opengrep verifier reward) imports, so "applied identically across arms" is a
property of the import surface, not of discipline. Full rule / why / mechanism
/ how-a-run-proves-compliance writeup: `docs/design/15-ssi-measurement-contract.md`.

Wraps, never duplicates: `src.eval.lsp_eval.compare`/`mcnemar_p_value` for the
per-seed paired statistic (M2), and re-exports `src.lsp.diagnostics`'s
escape-hatch gate (M5) so an arm has one import surface for the whole contract.

Portable, stdlib-only (`dataclasses`, `hashlib`, `json`, `math.comb`) -- mirrors
`lsp_eval.py`'s own constraint (no numpy), so nothing new for the import guard
to worry about beyond the `PORTABLE_MODULES` listing.

ABOVE THE SEAM -- stdlib only. No `mlx`/`torch` import anywhere in this module
(guarded by `tests/test_import_guard.py`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import comb
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..lsp.diagnostics import deletion_of_target, find_escape_hatches, has_escape_hatch
from .lsp_eval import compare

__all__ = [
    "ContractViolation",
    "ArmSpec", "validate_arms",
    "per_seed_compare", "pooled_compare", "sign_test_p", "wilcoxon_signed_rank_p",
    "summarize_arm",
    "repo_of", "RepoSplit", "split_by_repo", "split_manifest", "assert_disjoint",
    "find_escape_hatches", "has_escape_hatch", "deletion_of_target",
    "contract_report",
]


class ContractViolation(Exception):
    """Raised whenever an arm set, a paired comparison, or a repo split fails
    one of the #225 M1-M5 rules."""


# --------------------------------------------------------------------------- #
# M1 + M4 -- arm declaration and validation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ArmSpec:
    """One measured SSI arm. `variable` is the ONE thing this arm changes vs.
    `baseline` -- M1's "one variable per arm" rule made a field instead of a
    convention. A root/control arm self-references (`baseline == name`), which
    trivially satisfies "baseline resolves to a declared arm"."""
    name: str
    variable: str
    baseline: str
    signal_available: bool
    signal_used: bool
    seeds: Tuple[int, ...]
    notes: str = ""


def validate_arms(arms: Sequence[ArmSpec]) -> None:
    """Raise `ContractViolation` if `arms` fails M1 (one variable per arm), M2
    (>=3 seeds), or M4 (availability-vs-use null arms).

    Never returns a "looks fine" for an EMPTY arm set -- per the repo's
    standing rule that a check that cannot observe its target reports BLIND,
    never healthy (an empty set is not evidence of a compliant set).
    """
    if not arms:
        raise ContractViolation("empty arm set -- nothing was validated")

    by_name = {a.name: a for a in arms}

    for a in arms:
        if not a.variable:
            raise ContractViolation(f"arm {a.name!r} declares an empty variable")
        if a.baseline not in by_name:
            raise ContractViolation(
                f"arm {a.name!r} declares baseline {a.baseline!r}, which is not a "
                f"declared arm")
        if len(set(a.seeds)) < 3:
            raise ContractViolation(
                f"arm {a.name!r} declares {len(set(a.seeds))} distinct seed(s), "
                f"M2 requires >= 3")

    # M1: two arms compared against the SAME baseline must not share `variable`
    # -- that would be two confounded changes measured as if they were one.
    # Scoped by `signal_used` too: an M4 null/treatment PAIR is REQUIRED to
    # share (baseline, variable) and differ only in signal_used -- that is not
    # the confound M1 exists to catch, it is the deliberate pairing M4 demands.
    seen: Dict[Tuple[str, str, bool], str] = {}
    for a in arms:
        key = (a.baseline, a.variable, a.signal_used)
        if key in seen and seen[key] != a.name:
            raise ContractViolation(
                f"arms {seen[key]!r} and {a.name!r} both declare variable "
                f"{a.variable!r} against baseline {a.baseline!r} with the same "
                f"signal_used={a.signal_used}")
        seen[key] = a.name

    # M4: every signal_used arm needs a null sibling -- same variable/baseline,
    # signal_available=True, signal_used=False. Without it, "the signal helped"
    # is indistinguishable from "the arm got extra tokens / compute / a
    # different prompt shape".
    for a in arms:
        if not a.signal_used:
            continue
        has_null = any(
            o.name != a.name and o.variable == a.variable and o.baseline == a.baseline
            and o.signal_available and not o.signal_used
            for o in arms
        )
        if not has_null:
            raise ContractViolation(
                f"arm {a.name!r} has signal_used=True but no availability-vs-use "
                f"null sibling (same variable/baseline, signal_available=True, "
                f"signal_used=False)")


# --------------------------------------------------------------------------- #
# M2 -- multi-seed paired stats, built on lsp_eval.compare
# --------------------------------------------------------------------------- #

def per_seed_compare(baseline_by_seed: Mapping[int, Sequence[dict]],
                      other_by_seed: Mapping[int, Sequence[dict]],
                      *, key: str) -> Dict[int, dict]:
    """One `lsp_eval.compare(...)` per seed. Paired means paired: raises if the
    two arms' seed sets differ, or if any seed's baseline/other record `id`
    sequence differs between arms -- `compare` matches by position and its
    docstring already puts the sort/align burden on the caller; this wrapper
    enforces that the caller actually did it.
    """
    if set(baseline_by_seed) != set(other_by_seed):
        raise ContractViolation(
            f"seed sets differ: baseline has {sorted(baseline_by_seed)}, "
            f"other has {sorted(other_by_seed)}")

    out: Dict[int, dict] = {}
    for seed, baseline_recs in baseline_by_seed.items():
        other_recs = other_by_seed[seed]
        base_ids = [r["id"] for r in baseline_recs]
        other_ids = [r["id"] for r in other_recs]
        if base_ids != other_ids:
            raise ContractViolation(
                f"seed {seed}: record id order differs between baseline and "
                f"other ({base_ids} vs {other_ids})")
        out[seed] = compare(baseline_recs, other_recs, key=key)
    return out


def pooled_compare(baseline_by_seed: Mapping[int, Sequence[dict]],
                    other_by_seed: Mapping[int, Sequence[dict]],
                    *, key: str) -> dict:
    """Concatenate every seed's aligned records and run ONE `compare`. Reported
    alongside, never instead of, `per_seed_compare`'s per-seed table."""
    if set(baseline_by_seed) != set(other_by_seed):
        raise ContractViolation(
            f"seed sets differ: baseline has {sorted(baseline_by_seed)}, "
            f"other has {sorted(other_by_seed)}")

    baseline_all: List[dict] = []
    other_all: List[dict] = []
    for seed in sorted(baseline_by_seed):
        baseline_all.extend(baseline_by_seed[seed])
        other_all.extend(other_by_seed[seed])
    return compare(baseline_all, other_all, key=key)


def sign_test_p(n_positive: int, n_negative: int) -> float:
    """Exact two-sided sign test on the seed-level effect directions: does the
    delta have the same sign in every seed more than chance would predict?
    Same exact-enumeration style as `lsp_eval.mcnemar_p_value`, which it
    deliberately mirrors (`(3, 0) -> 0.25`, `(0, 0) -> 1.0`, symmetric in its
    two arguments).
    """
    n = n_positive + n_negative
    if n == 0:
        return 1.0
    k = min(n_positive, n_negative)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilcoxon_signed_rank_p(deltas: Sequence[float]) -> float:
    """Exact two-sided Wilcoxon signed-rank test p-value, for continuous
    per-record metrics (BPB, pass@k rates) where McNemar's binary
    discordant-pair table doesn't apply. Zero-valued deltas carry no
    directional information and are dropped before ranking (standard
    procedure) -- an all-zero delta vector is therefore `n == 0` and returns
    `1.0`.

    Exact by enumerating all `2**n` sign assignments over the ranks of the
    surviving `|delta|`s. `n > 20` raises `ValueError` rather than silently
    falling back to a normal approximation -- an approximation that is wrong
    at small `n` is the same failure mode `mcnemar_p_value`'s docstring
    already rejects.
    """
    nonzero = [d for d in deltas if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    if n > 20:
        raise ValueError(
            f"wilcoxon_signed_rank_p is exact only up to n=20 nonzero deltas, got {n}")

    abs_deltas = [abs(d) for d in nonzero]
    order = sorted(range(n), key=lambda i: abs_deltas[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_deltas[order[j + 1]] == abs_deltas[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    signs = [1 if d > 0 else -1 for d in nonzero]
    w_observed = sum(r for r, s in zip(ranks, signs) if s > 0)
    center = sum(ranks) / 2.0
    dist = abs(w_observed - center)

    extreme = 0
    for mask in range(2 ** n):
        w = sum(ranks[i] for i in range(n) if (mask >> i) & 1)
        if abs(w - center) >= dist - 1e-9:
            extreme += 1
    return min(1.0, extreme / (2 ** n))


def summarize_arm(per_seed: Mapping[int, dict], sign_p: float, pooled: dict) -> dict:
    """The reportable block for one arm's M2 comparison: per-seed rates, mean +-
    spread, the sign test across seeds, the pooled McNemar, and an explicit
    `"consistent_direction"` bool -- a caller sees directly whether every seed
    agreed on direction rather than having to infer it from a p-value.
    """
    deltas = [s["other_rate"] - s["baseline_rate"] for s in per_seed.values()]
    n = len(deltas)
    mean_delta = (sum(deltas) / n) if n else float("nan")
    delta_spread = (max(deltas) - min(deltas)) if n else float("nan")
    signs = {(1 if d > 0 else (-1 if d < 0 else 0)) for d in deltas}
    consistent_direction = bool(signs) and (signs - {0} in ({1}, {-1}, set()))

    return {
        "n_seeds": n,
        "per_seed": {
            seed: {"baseline_rate": s["baseline_rate"], "other_rate": s["other_rate"],
                   "mcnemar_p": s["mcnemar_p"]}
            for seed, s in per_seed.items()
        },
        "mean_delta": mean_delta,
        "delta_spread": delta_spread,
        "sign_test_p": sign_p,
        "pooled": pooled,
        "consistent_direction": consistent_direction,
    }


# --------------------------------------------------------------------------- #
# M3 -- repo-level contamination split + manifest
# --------------------------------------------------------------------------- #

def repo_of(record: Any) -> Optional[str]:
    """`record.meta["repo"]` (populated only by `src.data.stack_v2:93` today).
    Returns `None` when absent -- never invents an identity. Accepts either a
    `src.data.corpus.Record`-like object (`.meta` attribute) or a plain dict
    with a `"meta"` key, so callers aren't forced through one shape."""
    meta = getattr(record, "meta", None)
    if meta is None and isinstance(record, Mapping):
        meta = record.get("meta")
    return (meta or {}).get("repo")


@dataclass(frozen=True)
class RepoSplit:
    train: Tuple[Any, ...]
    eval: Tuple[Any, ...]
    quarantine: Tuple[Any, ...]
    eval_fraction: float
    seed: int


def split_by_repo(records: Sequence[Any], *, eval_fraction: float, seed: int,
                   on_unknown: str = "raise") -> RepoSplit:
    """Assign WHOLE repos to train/eval by a stable hash --
    `sha256(f"{seed}:{repo}")` compared against `eval_fraction` -- deliberately
    NOT Python's `hash()`, which is salted per interpreter (the same gotcha
    already called out at `scripts/eval_code_suite.py:404`) and so is not
    reproducible across runs.

    Records whose repo is `None` (`repo_of` returned nothing) are, per
    `on_unknown`: `"raise"` (default) -- an SSI arm silently repo-splitting a
    corpus with no repo identity would produce a *file-level* split wearing a
    repo-level label, exactly the defect M3 exists to prevent; or
    `"quarantine"` -- routed to a bucket that is in NEITHER side and IS
    counted in the manifest, never silently dropped into train.
    """
    if on_unknown not in ("raise", "quarantine"):
        raise ValueError(f"on_unknown must be 'raise' or 'quarantine', got {on_unknown!r}")

    train: List[Any] = []
    ev: List[Any] = []
    quarantine: List[Any] = []
    repo_side: Dict[str, str] = {}

    for rec in records:
        repo = repo_of(rec)
        if repo is None:
            if on_unknown == "raise":
                raise ContractViolation(
                    "record has no meta['repo'] -- cannot repo-split it (pass "
                    "on_unknown='quarantine' to bucket instead of raising)")
            quarantine.append(rec)
            continue
        side = repo_side.get(repo)
        if side is None:
            digest = hashlib.sha256(f"{seed}:{repo}".encode("utf-8")).digest()
            frac = int.from_bytes(digest[:8], "big") / float(1 << 64)
            side = "eval" if frac < eval_fraction else "train"
            repo_side[repo] = side
        (ev if side == "eval" else train).append(rec)

    return RepoSplit(train=tuple(train), eval=tuple(ev), quarantine=tuple(quarantine),
                      eval_fraction=eval_fraction, seed=seed)


def assert_disjoint(train: Sequence[Any], eval_: Sequence[Any]) -> None:
    """Raise `ContractViolation` if any repo appears on both sides -- the
    file-vs-repo bug M3 exists to catch (a near-duplicate sibling file from the
    same repo landing in eval)."""
    train_repos = {r for r in (repo_of(rec) for rec in train) if r is not None}
    eval_repos = {r for r in (repo_of(rec) for rec in eval_) if r is not None}
    overlap = train_repos & eval_repos
    if overlap:
        raise ContractViolation(f"repos present on both sides of the split: {sorted(overlap)}")


def split_manifest(split: RepoSplit) -> dict:
    """Byte-reproducible manifest for a `RepoSplit`, following
    `scripts/build_decontam_blocklist.py`'s provenance-manifest pattern.
    `eval_repos_sha256` is over the sorted, newline-joined eval repo list, so
    two runs claiming the same split can be CHECKED, not merely trusted.
    """
    train_repos = sorted({r for r in (repo_of(rec) for rec in split.train) if r is not None})
    eval_repos = sorted({r for r in (repo_of(rec) for rec in split.eval) if r is not None})
    eval_repos_sha256 = hashlib.sha256(
        ("\n".join(eval_repos) + "\n").encode("utf-8")).hexdigest()

    manifest = {
        "n_train": len(split.train),
        "n_eval": len(split.eval),
        "n_quarantine": len(split.quarantine),
        "n_train_repos": len(train_repos),
        "n_eval_repos": len(eval_repos),
        "eval_fraction": split.eval_fraction,
        "seed": split.seed,
        "eval_repos_sha256": eval_repos_sha256,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    return manifest


# --------------------------------------------------------------------------- #
# Glue -- the single reportable artifact
# --------------------------------------------------------------------------- #

def contract_report(arms: Sequence[ArmSpec], splits: Mapping[str, RepoSplit],
                     stats: Mapping[str, dict]) -> dict:
    """Assemble the JSON block a run writes next to its results -- the artifact
    that makes M1-M5 compliance auditable after the fact rather than merely
    asserted in a PR description."""
    return {
        "arms": [
            {"name": a.name, "variable": a.variable, "baseline": a.baseline,
             "signal_available": a.signal_available, "signal_used": a.signal_used,
             "seeds": list(a.seeds), "notes": a.notes}
            for a in arms
        ],
        "splits": {name: split_manifest(s) for name, s in splits.items()},
        "stats": dict(stats),
    }
