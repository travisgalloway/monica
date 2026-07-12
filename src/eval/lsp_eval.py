"""Scoring for the LSP-in-the-loop harness (#199) against the #194 eval set.

Pure scoring core — mirrors `bfcl_adapter.py`'s split from its wiring script
(`scripts/eval_bfcl.py`): `score_record` takes an already-generated result (a
`src.lsp.harness.GenResult`, or any object/mapping exposing the same fields) and an
injected `diagnose_fn`; nothing here calls a model or shells out to `tsc` itself.
`scripts/eval_lsp_harness.py` does the generation and wiring.

Metrics (defined in `docs/design/12-lsp-in-the-loop.md`, fixed before the
measurement so results can't be rationalized after the fact):

- `diagnostic_clean_rate` — raw `tsc` on the ARTIFACT, no filter (it's complete, so
  every diagnostic is real). `truncated` or a suppression hack (`@ts-ignore`,
  `as any`) forces "not clean" even if `tsc` itself is silent.
- `error_avoidance_rate` — **primary**. Over the 84 non-`clean_control` rows:
  `rec["expected_diagnostic"] not in codes(tsc(artifact))`.
- `exact_gold_rate` — secondary/informational only (see the design doc's risks:
  exact-match-to-gold would score an equally-correct alternative completion as a
  failure, so it is never the headline).
- `over_repair_rate` — fraction of the 12 `clean_control` rows with >= 1 rollback.
- `repair_rate` / `regression_rate` (via `compare`) — paired vs. baseline across
  all 96: broken -> clean / clean -> broken.
- `no_progress_rate`, `suggestion_leak_rate` — soft-repair diagnostics.
- cost — `n_forward_tokens` (cached) and `_nocache`, `n_generated_tokens`,
  `n_tsc_calls`, `n_rollbacks`, `wall_s`, `tsc_wall_s`, carried through from the
  `GenResult` for `summarize`'s aggregation.

ABOVE THE SEAM — stdlib only (+ `src.lsp.diagnostics`, itself portable). No
`mlx`/`torch` import anywhere in this module (guarded by
`tests/test_import_guard.py`).
"""

from __future__ import annotations

from math import comb
from typing import Any, Callable, Dict, List, Sequence

from ..lsp.diagnostics import SUPPRESSION_RE, Diagnostic, statement_boundary

DiagnoseFn = Callable[[str], List[Diagnostic]]

_COST_FIELDS = ("n_forward_tokens", "n_forward_tokens_nocache", "n_generated_tokens",
                "n_tsc_calls", "n_rollbacks", "n_soft_repairs", "n_retries", "wall_s", "tsc_wall_s")


def score_record(rec: dict, gen_result: Any, diagnose: DiagnoseFn) -> dict:
    """Score one generated completion against its #194 record.

    `gen_result` needs `.artifact`, `.completion`, `.n_rollbacks`, `.no_progress`,
    `.events` and the cost fields (`GenResult`'s shape). `diagnose` is called once
    here, on the complete ARTIFACT, unfiltered — unlike the harness's internal
    frontier-filtered checks on a possibly-incomplete generation-in-progress, the
    artifact is finished text, so every diagnostic `tsc` reports on it is real.
    """
    artifact = gen_result.artifact
    diags = diagnose(artifact)
    codes = [d.code for d in diags]

    # "Truncated": generation never reached a clean statement boundary (most
    # likely hit a safety cap before finishing) -- a truncated completion cannot
    # be fairly scored as "clean" even if the partial text happens to compile.
    truncated = statement_boundary(gen_result.completion) is None

    suppression_hack = bool(SUPPRESSION_RE.search(gen_result.completion))

    clean = (not codes) and not truncated and not suppression_hack

    error_class = rec["error_class"]
    is_error_row = error_class != "clean_control"
    avoided = (rec["expected_diagnostic"] not in codes) if is_error_row else None

    exact_gold = gen_result.completion == rec["gold_completion"]
    rolled_back = gen_result.n_rollbacks > 0

    # A "suggestion leak": a soft-repair round was exposed to tsc's own spelling
    # correction ("Did you mean 'x'?") -- an exposure signal, not proof the model
    # copied it; see the design doc's risks section on this ablation's purpose.
    suggestion_leak = any(
        "Did you mean" in ev.get("comment", "") for ev in getattr(gen_result, "events", [])
        if ev.get("kind") == "soft_repair"
    )

    out = {
        "id": rec["id"],
        "error_class": error_class,
        "is_error_row": is_error_row,
        "clean": clean,
        "avoided": avoided,
        "exact_gold": exact_gold,
        "rolled_back": rolled_back,
        "no_progress": bool(getattr(gen_result, "no_progress", False)),
        "suggestion_leak": suggestion_leak,
        "truncated": truncated,
        "suppression_hack": suppression_hack,
        "codes": codes,
    }
    for field in _COST_FIELDS:
        out[field] = getattr(gen_result, field, None)
    return out


def summarize(scored: Sequence[dict]) -> dict:
    """Aggregate a list of `score_record` outputs into the metrics table."""
    n = len(scored)
    if n == 0:
        return {"n": 0}

    error_rows = [s for s in scored if s["is_error_row"]]
    clean_control_rows = [s for s in scored if not s["is_error_row"]]
    soft_repair_rows = [s for s in scored if s.get("n_soft_repairs") is not None
                         and s["n_soft_repairs"] > 0]

    def _rate(rows: Sequence[dict], key: str) -> float:
        return sum(1 for r in rows if r[key]) / len(rows) if rows else float("nan")

    summary = {
        "n": n,
        "diagnostic_clean_rate": _rate(scored, "clean"),
        "error_avoidance_rate": _rate(error_rows, "avoided"),
        "exact_gold_rate": _rate(scored, "exact_gold"),
        "over_repair_rate": _rate(clean_control_rows, "rolled_back"),
        "no_progress_rate": _rate(soft_repair_rows, "no_progress"),
        "suggestion_leak_rate": _rate(soft_repair_rows, "suggestion_leak"),
        "n_error_rows": len(error_rows),
        "n_clean_control_rows": len(clean_control_rows),
    }
    for field in _COST_FIELDS:
        values = [s[field] for s in scored if s.get(field) is not None]
        summary[f"total_{field}"] = sum(values) if values else 0
        summary[f"mean_{field}"] = (sum(values) / len(values)) if values else float("nan")
    return summary


def mcnemar_p_value(b_true_o_false: int, b_false_o_true: int) -> float:
    """Exact two-sided McNemar test p-value: a binomial test on the `n =
    b_true_o_false + b_false_o_true` discordant pairs, `k = min(...)` successes,
    `p = 0.5`. Exact (not chi-square-approximated) because a 96-record eval set's
    discordant-pair counts are typically small enough that the chi-square
    approximation is unreliable.
    """
    n = b_true_o_false + b_false_o_true
    if n == 0:
        return 1.0
    k = min(b_true_o_false, b_false_o_true)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(baseline: Sequence[dict], other: Sequence[dict], *, key: str) -> dict:
    """Paired comparison of two scored runs over the SAME records (matched by
    position — callers should sort/align both lists by `id` beforehand), on the
    boolean field `key` (e.g. `"avoided"` for the primary go/no-go metric,
    `"clean"` for `repair_rate`/`regression_rate` across all 96).

    Returns the 2x2 contingency table, each run's marginal rate, the exact
    McNemar p-value, and the two directional flip rates: `flip_to_true_rate`
    (fraction of baseline-`False` records `other` made `True` -- "repair_rate"
    when `key="clean"`) and `flip_to_false_rate` (the reverse -- "regression_rate").
    """
    if len(baseline) != len(other):
        raise ValueError(f"baseline has {len(baseline)} records, other has {len(other)}")

    tt = tf = ft = ff = 0
    for b, o in zip(baseline, other):
        bv, ov = bool(b[key]), bool(o[key])
        if bv and ov:
            tt += 1
        elif bv and not ov:
            tf += 1
        elif not bv and ov:
            ft += 1
        else:
            ff += 1

    n = len(baseline)
    n_baseline_false = ft + ff
    n_baseline_true = tt + tf
    return {
        "n": n,
        "key": key,
        "table": {"both_true": tt, "baseline_true_other_false": tf,
                  "baseline_false_other_true": ft, "both_false": ff},
        "baseline_rate": (tt + tf) / n if n else float("nan"),
        "other_rate": (tt + ft) / n if n else float("nan"),
        "mcnemar_p": mcnemar_p_value(tf, ft),
        "flip_to_true_rate": (ft / n_baseline_false) if n_baseline_false else float("nan"),
        "flip_to_false_rate": (tf / n_baseline_true) if n_baseline_true else float("nan"),
    }
