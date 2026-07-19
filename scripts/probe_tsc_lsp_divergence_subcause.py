#!/usr/bin/env python3
"""Pin the sub-cause of the tsc-vs-LSP F1 pass@1 divergence (#198 / #199 follow-up).

Batch `tsc` moved #199 F1 pass@1 0.491->0.560 (p=0.001) where the persistent
open-document LSP did not (0.503). `scripts/analyze_tsc_lsp_divergence.py` localizes
the gap to 9 records LSP finishes "clean-but-functionally-wrong". This probe asks WHY.

It drives the WINNING batch-tsc slow-loop trajectory on the crux records and, at every
diagnose call, asks BOTH oracles for their verdict on the IDENTICAL candidate text. A
"fork" is a step where tsc flags but LSP is clean on the same text -- which would mean
tsc is a stricter detector (H1). The finding: ZERO forks (LSP is if anything MORE
sensitive on incomplete code), and separately batch tsc CLEARS every one of LSP's
clean-but-wrong final artifacts -- so the wrong code is type-clean to both oracles. The
divergence is exploration/trajectory variance among type-clean endpoints (H2), not
detection. Neither type oracle can rank correctness.

MLX-touching (loads the model); imports kept local to main() so the seam stays clean.

Usage: .venv/bin/python scripts/probe_tsc_lsp_divergence_subcause.py \
           [--crux results/tsc_lsp_divergence.json] \
           [--out results/tsc_lsp_divergence_subcause.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_HUMANEVAL = _REPO / "eval_sets/humaneval_ts/humaneval_ts.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crux", default="results/tsc_lsp_divergence.json",
                    help="output of analyze_tsc_lsp_divergence.py (supplies the crux ids)")
    ap.add_argument("--out", default="results/tsc_lsp_divergence_subcause.json")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-1.5B-bf16")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--max-retries", type=int, default=8)
    args = ap.parse_args()

    # Local (below-the-seam) imports.
    from src.lsp.tsc import TscRunner, resolve_tsc
    from src.lsp.ts_lsp import TsLspOracle, resolve_ts_lsp
    from src.lsp.harness import generate_slow_loop
    if resolve_tsc() is None or resolve_ts_lsp() is None:
        print("SKIP: need both tsc and typescript-language-server toolchains "
              "(run `npm install` in eval_sets/ts_error_injection).")
        return 0
    from src.model.mlx_lm_adapter import MLXLMAdapter

    crux = [c["id"] for c in json.loads(Path(args.crux).read_text())["crux"]]
    records = {}
    for line in _HUMANEVAL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["id"]] = r

    print("loading model + oracles...", flush=True)
    lm = MLXLMAdapter(args.model)
    tsc, lsp = TscRunner(), TsLspOracle()

    def make_dual(steps):
        def dual(source):
            t = tsc.diagnostics(source)
            l = lsp.diagnostics(source)
            steps.append({"len": len(source), "tsc": sorted({d.code for d in t}),
                          "lsp": sorted({d.code for d in l}), "fork": bool(t) and not l})
            return t   # drive the tsc trajectory
        return dual

    out = []
    for rid in crux:
        rec = records[rid]
        steps = []
        res = generate_slow_loop(lm, make_dual(steps), rec["prompt"], repair="hard",
                                 budget="block", block_size=args.block_size,
                                 max_retries=args.max_retries, temperature=0.0, rng=None,
                                 stop_strings=rec.get("stop_tokens") or None)
        forks = [k for k, s in enumerate(steps) if s["fork"]]
        out.append({"id": rid, "n_steps": len(steps), "n_rollbacks": res.n_rollbacks,
                    "n_fork_steps": len(forks), "first_fork_step": forks[0] if forks else None,
                    "steps": steps})
        print(f"{rid[:40]:40} steps={len(steps):2} rb={res.n_rollbacks:2} forks={len(forks)}", flush=True)

    tsc.close(); lsp.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    n_with_fork = sum(1 for r in out if r["n_fork_steps"] > 0)
    tot = sum(r["n_steps"] for r in out)
    lsp_more = sum(1 for r in out for s in r["steps"] if set(s["lsp"]) - set(s["tsc"]))
    tsc_more = sum(1 for r in out for s in r["steps"] if set(s["tsc"]) - set(s["lsp"]))
    print(f"\n{n_with_fork}/{len(out)} records have a fork (tsc flags, lsp clean on IDENTICAL text)")
    print(f"steps where LSP emits MORE codes than tsc: {lsp_more} | tsc more: {tsc_more} (of {tot})")
    print("VERDICT:", "SAME-TEXT DIFFERENT-VERDICT (H1 strictness)" if n_with_fork >= 6
          else "TRAJECTORY/EXPLORATION VARIANCE (H2) — type oracle can't rank correctness")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
