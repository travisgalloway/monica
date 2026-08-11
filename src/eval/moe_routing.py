"""MoE routing diagnostics: per-domain expert histograms + a mid-training kill-criterion
(#217).

ABOVE THE SEAM. Pure numpy + stdlib; touches a model only through the duck-typed
`ModelInterface.forward` plus the MoE accessors `set_moe_load_counting` / `pop_moe_load`
(both backends already expose these for Loss-Free-Balancing, #213/#214) — no `mlx` or
`torch` import here, ever.

What this measures, and what it is NOT
---------------------------------------
The question this module answers is: does the router send different domains (say,
TypeScript vs math) to different experts, or has it collapsed onto one shared set? That
would be the earliest, cheapest signal that MoE capacity is buying nothing.

Packed training shards carry no domain tag — `src/data/shard.py`'s manifest is
`{seq_len, dtype, tokenizer, n_documents, n_sequences, n_tokens, shards}`, and the Swift
packer (`swift/Sources/MonicaTokenizer/Packing.swift`) writes the identical shape; the
per-record `source`/`lang` that exist at the cleaned-Parquet stage are dropped by the time
tokens are packed (documented at `src/eval/domain_bpb.py:15-31`). #252, which would add a
`.domains` sidecar carrying that tag through packing, is open and blocked on #251. So —
exactly like `domain_bpb.py` — this reads the purpose-built per-domain val sets built by
`scripts/build_domain_val_sets.py` (`domains.json`, the same input `load_domain_index`
already parses). **This measures expert usage on held-out samples drawn from each domain,
not over the live training mix.** No `.domains` sidecar, no re-pack, no Swift packer
change, no dependency on #252.

The BLIND rule
---------------
A check that cannot observe its target must report loudly, never healthily. An empty
histogram, an unmeasured domain, or a missing kill-criterion input must never silently
read as "the router is fine" / "the router is specializing" — every such case here raises
or is recorded as an explicit `None`, never a zero or a default verdict.

The kill-criterion REPORTS, it does not abort
-----------------------------------------------
`kill_check` below is a diagnostic, not a control-flow decision: it never raises a
training run's abort signal. Killing a multi-day run from a threshold is the operator's
call, not this module's. And **early in training, routing is near-uniform by
construction**, so a high overlap between any two domains is *expected*, not evidence of
a broken router — a threshold crossed at step 50 is not the same event as one crossed at
step 50,000. Callers must read the verdict in that light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..data.loader import PackedLoader


def expert_histograms(model, domains: Dict[str, str], *, batch_size: int = 4,
                      seq_len: int = 512, max_batches: Optional[int] = 8,
                      to_numpy=np.asarray) -> Dict[str, Optional[list]]:
    """Per-domain, per-MoE-layer expert token counts from a forward-only pass.

    `domains` maps a domain name to its packed `.bin` (exactly `load_domain_index`'s
    `{name: entry["packed"]}` projection). Each domain is read `shuffle=False,
    drop_last=False` so the histogram is a deterministic function of
    `(seq_len, batch_size, max_batches)`.

    Returns `{domain: [[count per expert] per MoE layer]}`, or `None` for a domain whose
    val file is too small for one chunk (the `domain_bpb`/`long_context` convention —
    explicitly unmeasured, never a zero histogram).

    Mutates model state: turns per-expert load counting on (if it was not already) and
    drains the accumulators, both before measuring the first domain and after each one.
    """
    if not domains:
        raise ValueError("expert_histograms(): no domains — an empty report is a failure, "
                         "not a perfect score")

    model.set_moe_load_counting(True)
    # The training step drains the counters every step, but an fp16-overflow-skipped step
    # deliberately leaves its counts behind (`cuda_train_step.py:95-97`); discarding them
    # here keeps training tokens out of a domain histogram. The cost is that one skipped
    # step's counts never reach the balancer, which is noise under a sign-based update.
    if not model.pop_moe_load():
        raise ValueError(
            "expert_histograms(): model has no MoE layers — an expert histogram cannot "
            "be measured, and an empty report must not read as 'the router is fine'"
        )

    out: Dict[str, Optional[list]] = {}
    for name in sorted(domains):
        packed_path = Path(domains[name])
        if not packed_path.exists():
            raise FileNotFoundError(f"domain {name!r}: packed val file {packed_path} "
                                    "does not exist")
        try:
            loader = PackedLoader(packed_path, seq_len=seq_len, batch_size=batch_size,
                                  shuffle=False, drop_last=False)
        except ValueError:
            # Too small for even one (seq_len + 1) chunk. Explicitly unmeasured, not zero.
            out[name] = None
            continue

        for i, (inputs, _targets) in enumerate(loader.epoch()):
            if max_batches is not None and i >= max_batches:
                break
            # The forward output is discarded — this is a routing probe, not a loss eval
            # — but `to_numpy` is still called so the MLX lazy graph is forced exactly as
            # `val_loss.evaluate` does it (otherwise the router calls would never run).
            to_numpy(model.forward(inputs))

        hist = model.pop_moe_load()
        if sum(sum(layer) for layer in hist) <= 0:
            raise ValueError(
                f"domain {name!r}: every expert count is zero — the router was not "
                "observed (load counting off, or top_k == n_experts). A zero histogram "
                "must not be reported as a routing measurement."
            )
        out[name] = hist

    if all(v is None for v in out.values()):
        raise ValueError(
            "expert_histograms(): every domain was too small to score — no routing was "
            "measured"
        )
    return out


def histogram_overlap(h_a: list, h_b: list) -> dict:
    """Per-layer routing overlap between two per-layer expert histograms.

    `1 - TV` between the two L1-normalized distributions, i.e.
    `sum_e min(p_e, q_e)`: **1.0 = identical routing (no specialization)**, 0.0 = disjoint.
    Returns `{"per_layer": [float], "mean": float}`.

    Raises on a layer-count or expert-count mismatch, and on a layer whose total is zero in
    either histogram — an unnormalizable layer has no overlap, and defaulting it to 0.0
    would fabricate perfect specialization.
    """
    if len(h_a) != len(h_b):
        raise ValueError(f"histogram_overlap(): {len(h_a)} layers vs {len(h_b)} layers")
    per_layer = []
    for i, (a, b) in enumerate(zip(h_a, h_b)):
        if len(a) != len(b):
            raise ValueError(
                f"histogram_overlap(): layer {i} has {len(a)} experts vs {len(b)}"
            )
        total_a, total_b = sum(a), sum(b)
        if total_a <= 0 or total_b <= 0:
            raise ValueError(
                f"histogram_overlap(): layer {i} has a zero-total histogram "
                f"(total_a={total_a}, total_b={total_b}) — unnormalizable, not 0.0 overlap"
            )
        pa = [x / total_a for x in a]
        pb = [x / total_b for x in b]
        per_layer.append(sum(min(x, y) for x, y in zip(pa, pb)))
    mean = sum(per_layer) / len(per_layer) if per_layer else 0.0
    return {"per_layer": per_layer, "mean": mean}


def specialization_report(hists: Dict[str, list]) -> dict:
    """Pairwise `histogram_overlap` across every measured domain pair.

    Returns
      {"domains": [sorted measured names],
       "by_pair": {"a|b": {"per_layer": [...], "mean": float}},   # a < b, sorted key
       "mean_overlap": float,          # mean over pairs
       "max_overlap": float, "max_pair": "a|b",
       "unmeasured_domains": [...]}    # the `None` entries from expert_histograms

    Raises when fewer than two domains were measured — one domain yields no pair, and an
    empty `by_pair` must not read as "well separated".
    """
    unmeasured = sorted(n for n, v in hists.items() if v is None)
    measured = sorted(n for n, v in hists.items() if v is not None)
    if len(measured) < 2:
        raise ValueError(
            f"specialization_report(): only {len(measured)} domain(s) measured "
            f"({measured}) — need at least 2 to report an overlap, an empty result must "
            "not read as 'well separated'"
        )

    by_pair: Dict[str, dict] = {}
    for i, a in enumerate(measured):
        for b in measured[i + 1:]:
            key = "|".join(sorted((a, b)))
            by_pair[key] = histogram_overlap(hists[a], hists[b])

    mean_overlap = sum(v["mean"] for v in by_pair.values()) / len(by_pair)
    max_pair = max(by_pair, key=lambda k: by_pair[k]["mean"])
    return {
        "domains": measured,
        "by_pair": by_pair,
        "mean_overlap": mean_overlap,
        "max_overlap": by_pair[max_pair]["mean"],
        "max_pair": max_pair,
        "unmeasured_domains": unmeasured,
    }


def kill_check(report: dict, *, pair=("typescript", "math"), threshold: float = 0.90) -> dict:
    """The #217 mid-training kill-criterion: do two domains hit the same experts?

    Returns `{"pair": "a|b", "overlap": float, "per_layer": [...], "threshold": float,
              "specializing": bool, "triggered": bool}` where
    `triggered = overlap >= threshold` (i.e. NOT specializing) and
    `specializing = not triggered`.

    REPORTS ONLY — nothing here aborts a run (see the module docstring).

    Raises when either named domain is absent from the report, or was unmeasured: a
    kill-criterion that cannot see its two domains has no verdict, and returning
    `specializing=True` there would be exactly the BLIND failure it exists to prevent.
    """
    a, b = pair
    key = "|".join(sorted((a, b)))
    domains = set(report.get("domains", []))
    unmeasured = set(report.get("unmeasured_domains", []))
    for name in (a, b):
        if name in unmeasured:
            raise ValueError(
                f"kill_check(): domain {name!r} was unmeasured (val set too small) — the "
                "kill-criterion cannot see it, and reporting a verdict anyway would be "
                "the BLIND failure it exists to prevent"
            )
        if name not in domains:
            raise ValueError(
                f"kill_check(): domain {name!r} is not in the report ({sorted(domains)}) "
                "— the kill-criterion cannot see it, and reporting a verdict anyway would "
                "be the BLIND failure it exists to prevent"
            )
    entry = report["by_pair"][key]
    triggered = entry["mean"] >= threshold
    return {
        "pair": key,
        "overlap": entry["mean"],
        "per_layer": entry["per_layer"],
        "threshold": threshold,
        "specializing": not triggered,
        "triggered": triggered,
    }


def format_routing_report(report: dict, kill: Optional[dict] = None) -> str:
    """Human table — one line per domain pair + the kill verdict, in
    `domain_bpb.format_domain_bpb_table`'s style."""
    lines = ["MoE routing overlap by domain pair (1.0 = identical routing, 0.0 = disjoint):"]
    for key, entry in sorted(report["by_pair"].items()):
        lines.append(f"  {key:<32} overlap={entry['mean']:.4f}")
    lines.append(f"  {'mean':<32} overlap={report['mean_overlap']:.4f}")
    lines.append(f"  {'max (' + report['max_pair'] + ')':<32} overlap={report['max_overlap']:.4f}")
    if report["unmeasured_domains"]:
        lines.append(f"  unmeasured: {', '.join(report['unmeasured_domains'])}")
    if kill is not None:
        verdict = "NOT specializing (>= threshold)" if kill["triggered"] else "specializing"
        lines.append(f"  kill-check {kill['pair']}: overlap={kill['overlap']:.4f} "
                     f"threshold={kill['threshold']:.4f} -> {verdict}")
    return "\n".join(lines)
