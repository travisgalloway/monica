#!/usr/bin/env python3
"""Why did batch `tsc` move #199 F1 pass@1 (0.491->0.560, p=0.001) while the
persistent TS-LSP did not (0.491->0.503, ns)? This joins the two per-record F1
transcripts and pins the mechanism.

Inputs (both produced by scripts/eval_lsp_humaneval.py, 159 records, block 256,
greedy): `results/f1_base.jsonl` (batch tsc) and `results/f1_ts.jsonl`
(persistent LSP). Each carries baseline + slow-hard rows with per-record
`functional_pass`, `clean`, `codes`, `n_rollbacks`, `events`.

Finding: the win is NOT extra code coverage (both oracles flag the same codes on
the crux records). It is oracle PERSISTENCE. The open-document LSP clears a
diagnostic after one rollback and declares the candidate `clean` prematurely,
stopping the loop on still-functionally-wrong code; batch tsc's whole-program
compile keeps the same error alive across regenerations, driving ~3.4x more
repair iterations that converge to correct code. Stdlib only.

Usage: .venv/bin/python scripts/analyze_tsc_lsp_divergence.py \
           [--tsc results/f1_base.jsonl] [--lsp results/f1_ts.jsonl] \
           [--out results/tsc_lsp_divergence.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str):
    by = {"baseline": {}, "slow-hard": {}}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["strategy"] in by:
            by[r["strategy"]][r["id"]] = r
    return by


def _event_codes(rec):
    return [e.get("code") for e in rec.get("events", [])
            if isinstance(e, dict) and e.get("code")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsc", default="results/f1_base.jsonl")
    ap.add_argument("--lsp", default="results/f1_ts.jsonl")
    ap.add_argument("--out", default="results/tsc_lsp_divergence.json")
    args = ap.parse_args()

    tsc, lsp = _load(args.tsc), _load(args.lsp)
    ids = sorted(tsc["slow-hard"])

    base_mismatch = [i for i in ids
                     if tsc["baseline"][i]["functional_pass"] != lsp["baseline"][i]["functional_pass"]]
    base_pass = sum(tsc["baseline"][i]["functional_pass"] for i in ids)
    tsc_pass = sum(tsc["slow-hard"][i]["functional_pass"] for i in ids)
    lsp_pass = sum(lsp["slow-hard"][i]["functional_pass"] for i in ids)

    tsc_adv, lsp_adv = [], []
    for i in ids:
        t = tsc["slow-hard"][i]["functional_pass"]
        l = lsp["slow-hard"][i]["functional_pass"]
        if t and not l:
            tsc_adv.append(i)
        if l and not t:
            lsp_adv.append(i)

    def premature_clean(d):  # clean per oracle but functionally wrong
        sh = d["slow-hard"]
        return sum(1 for i in sh if sh[i].get("clean") and not sh[i]["functional_pass"])

    def total_rb(d):
        return sum(d["slow-hard"][i]["n_rollbacks"] for i in ids)

    crux = []
    for i in tsc_adv:
        tr, lr = tsc["slow-hard"][i], lsp["slow-hard"][i]
        crux.append({
            "id": i,
            "baseline_pass": tsc["baseline"][i]["functional_pass"],
            "tsc": {"pass": tr["functional_pass"], "clean": tr.get("clean"),
                    "n_rollbacks": tr["n_rollbacks"], "codes": _event_codes(tr)},
            "lsp": {"pass": lr["functional_pass"], "clean": lr.get("clean"),
                    "n_rollbacks": lr["n_rollbacks"], "codes": _event_codes(lr)},
            "lsp_premature_clean": bool(lr.get("clean") and not lr["functional_pass"]),
        })

    out = {
        "n_records": len(ids),
        "baseline_pass_at_1": base_pass / len(ids),
        "baseline_mismatch": len(base_mismatch),
        "tsc_slow_hard_pass_at_1": tsc_pass / len(ids),
        "lsp_slow_hard_pass_at_1": lsp_pass / len(ids),
        "tsc_advantage_records": len(tsc_adv),
        "lsp_advantage_records": len(lsp_adv),
        "tsc_advantage_won_via_rollback": sum(1 for c in crux if c["tsc"]["n_rollbacks"] > 0),
        "lsp_premature_clean_on_crux": sum(1 for c in crux if c["lsp_premature_clean"]),
        "clean_but_wrong_tsc": premature_clean(tsc),
        "clean_but_wrong_lsp": premature_clean(lsp),
        "total_rollbacks_tsc": total_rb(tsc),
        "total_rollbacks_lsp": total_rb(lsp),
        "crux": crux,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 68)
    print("tsc vs persistent-LSP F1 divergence (#199 / #198)")
    print("=" * 68)
    print(f"baseline pass@1:        {base_pass}/{len(ids)} = {out['baseline_pass_at_1']:.3f}"
          f"  (mismatch: {len(base_mismatch)})")
    print(f"tsc slow-hard pass@1:   {tsc_pass}/{len(ids)} = {out['tsc_slow_hard_pass_at_1']:.3f}")
    print(f"lsp slow-hard pass@1:   {lsp_pass}/{len(ids)} = {out['lsp_slow_hard_pass_at_1']:.3f}")
    print(f"tsc-advantage records:  {len(tsc_adv)}  (all via rollback: "
          f"{out['tsc_advantage_won_via_rollback']}/{len(tsc_adv)})")
    print(f"lsp-advantage records:  {len(lsp_adv)}")
    print(f"LSP premature-clean on crux: {out['lsp_premature_clean_on_crux']}/{len(tsc_adv)}")
    print(f"clean-but-WRONG (slow-hard): tsc {out['clean_but_wrong_tsc']} | "
          f"lsp {out['clean_but_wrong_lsp']}  (delta = the functional gap)")
    print(f"total rollbacks:        tsc {out['total_rollbacks_tsc']} | lsp {out['total_rollbacks_lsp']}"
          f"  (tsc = {out['total_rollbacks_tsc']/max(1,out['total_rollbacks_lsp']):.1f}x)")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
