"""#199 LSP-in-the-loop no-training validation: baseline vs. hard/soft repair vs.
tool-call, on the #194 TS error-injection eval set.

Wires config -> MLXLMAdapter -> TscRunner -> src/lsp/harness's generation
strategies -> src/eval/lsp_eval's scoring, mirroring scripts/eval_bfcl.py's split
(mlx-only imports kept local to main() so the seam stays clean). One `MLXLMAdapter`
and one `TscRunner` are created per script run and reused across every record and
strategy — reloading the model or re-`npm install`-ing a scratch dir per record
would dominate the wall clock for no benefit.

Runs every requested `--strategies` entry over the same record set, scores each
against `--set` via `lsp_eval.score_record`/`summarize`, and reports each non
-baseline strategy's paired `compare(...)` against `baseline` (the McNemar go/no-go
evidence). Emits:
  - `--output` — one JSON blob: per-strategy summaries + pairwise comparisons.
  - `--transcript` — one JSONL file, one line per (strategy, record): context,
    artifact, diagnostics, repair events. This is what gets read when the numbers
    surprise us.
  - a markdown table on stdout.

    .venv/bin/python scripts/eval_lsp_harness.py \\
        --model mlx-community/Qwen2.5-Coder-1.5B-bf16 --budget stmt --temperature 0 \\
        --strategies baseline,slow-hard,slow-soft,slow-both,toolcall-k1,toolcall-k2 \\
        --output results/table_a.json --transcript results/table_a.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional


def _parse_strategy(name: str):
    """Map a `--strategies` token to `(kind, kwargs)` for dispatch in `main()`."""
    if name == "baseline":
        return "baseline", {}
    if name == "slow-hard":
        return "slow", {"repair": "hard"}
    if name == "slow-soft":
        return "slow", {"repair": "soft"}
    if name in ("slow-both", "slow-soft+hard"):
        return "slow", {"repair": "both"}
    if name.startswith("toolcall-k"):
        try:
            k = int(name[len("toolcall-k"):])
        except ValueError:
            raise SystemExit(f"bad toolcall strategy {name!r} (want toolcall-k<N>)")
        return "toolcall", {"k": k}
    raise SystemExit(f"unknown strategy {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-1.5B-bf16",
                    help="mlx-lm model path or HF repo (default: the pinned 1.5B base coder)")
    ap.add_argument("--dtype", default=None,
                    help="upcast dtype (e.g. float32); default None = model's native bf16")
    ap.add_argument("--set", type=Path, default=None,
                    help="path to eval.jsonl (default: the #194 eval set)")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of records")
    ap.add_argument("--budget", choices=("stmt", "block"), default="stmt")
    ap.add_argument("--block-size", type=int, default=96)
    ap.add_argument("--max-gen-tokens", type=int, default=200,
                    help="safety cap for stmt budget / baseline")
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--strategies", default="baseline,slow-hard,slow-soft,slow-both,toolcall-k1,toolcall-k2",
                    help="comma-separated: baseline, slow-hard, slow-soft, slow-both, toolcall-k<N>")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strip-suggestions", action="store_true",
                    help="strip tsc's 'Did you mean X?' before it reaches the model "
                         "(the suggestion-leak ablation)")
    ap.add_argument("--output", type=Path, default=None, help="write the results JSON here")
    ap.add_argument("--transcript", type=Path, default=None,
                    help="write the per-record JSONL transcript here")
    args = ap.parse_args()

    # mlx-only imports kept local so the seam stays clean for portable hosts.
    try:
        import mlx.core as mx  # noqa: F401 -- presence probe
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/eval_lsp_harness.py ...\n"
            "(mlx installs only on Apple Silicon via the '[mlx]' extra.)"
        ) from e
    import numpy as np

    from src.eval.lsp_eval import compare, score_record, summarize
    from src.eval.ts_error_eval import load_ts_error_set, DEFAULT_SET_PATH
    from src.lsp.harness import generate_baseline, generate_slow_loop, generate_toolcall
    from src.lsp.tsc import TscRunner, resolve_tsc
    from src.model.mlx_lm_adapter import MLXLMAdapter

    if resolve_tsc() is None:
        raise SystemExit("no node/tsc toolchain found — run `npm install` in "
                          "eval_sets/ts_error_injection first.")

    set_path = args.set or DEFAULT_SET_PATH
    records = load_ts_error_set(set_path)
    if args.limit is not None:
        records = records[: args.limit]

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if "baseline" not in strategies:
        strategies = ["baseline"] + strategies  # always need it for pairwise comparisons

    print(f"model={args.model} dtype={args.dtype} budget={args.budget} "
          f"temperature={args.temperature} records={len(records)} strategies={strategies}")

    t_load = time.monotonic()
    lm = MLXLMAdapter(args.model, dtype=args.dtype)
    runner = TscRunner()
    print(f"model + tsc scratch dir ready in {time.monotonic() - t_load:.1f}s")

    rng = np.random.default_rng(args.seed) if args.temperature > 0 else None

    transcript_f = None
    if args.transcript:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript_f = open(args.transcript, "w", encoding="utf-8")

    gen_kwargs = dict(budget=args.budget, block_size=args.block_size,
                       max_gen_tokens=args.max_gen_tokens, temperature=args.temperature, rng=rng)

    scored_by_strategy = {}
    t_run = time.monotonic()
    for strategy_name in strategies:
        kind, kwargs = _parse_strategy(strategy_name)
        scored: List[dict] = []
        for i, rec in enumerate(records):
            if kind == "baseline":
                result = generate_baseline(lm, rec["prompt"], **gen_kwargs)
            elif kind == "slow":
                result = generate_slow_loop(
                    lm, runner.diagnostics, rec["prompt"], max_retries=args.max_retries,
                    strip_suggestions=args.strip_suggestions, **kwargs, **gen_kwargs)
            else:  # toolcall
                result = generate_toolcall(
                    lm, runner.diagnostics, rec["prompt"],
                    strip_suggestions=args.strip_suggestions, **kwargs, **gen_kwargs)

            s = score_record(rec, result, runner.diagnostics)
            scored.append(s)

            if transcript_f is not None:
                transcript_f.write(json.dumps({
                    "strategy": strategy_name, "id": rec["id"], "error_class": rec["error_class"],
                    "prompt": rec["prompt"], "context": result.context, "artifact": result.artifact,
                    "codes": s["codes"], "clean": s["clean"], "avoided": s["avoided"],
                    "n_rollbacks": result.n_rollbacks, "n_soft_repairs": result.n_soft_repairs,
                    "no_progress": result.no_progress, "unrepaired": result.unrepaired,
                    "events": result.events,
                }) + "\n")

            if (i + 1) % 16 == 0:
                print(f"  {strategy_name}: {i + 1}/{len(records)}")

        scored_by_strategy[strategy_name] = scored
        summary = summarize(scored)
        print(f"[{strategy_name}] clean={summary['diagnostic_clean_rate']:.3f} "
              f"avoid={summary['error_avoidance_rate']:.3f} "
              f"exact_gold={summary['exact_gold_rate']:.3f} "
              f"over_repair={summary['over_repair_rate']:.3f} "
              f"no_progress={summary['no_progress_rate']:.3f} "
              f"mean_fwd_tok={summary['mean_n_forward_tokens']:.1f}")

    if transcript_f is not None:
        transcript_f.close()
        print(f"transcript -> {args.transcript}")

    baseline_scored = scored_by_strategy["baseline"]
    comparisons = {}
    for strategy_name, scored in scored_by_strategy.items():
        if strategy_name == "baseline":
            continue
        error_baseline = [s for s in baseline_scored if s["is_error_row"]]
        error_other = [s for s in scored if s["is_error_row"]]
        comparisons[strategy_name] = {
            "avoided_vs_baseline": compare(error_baseline, error_other, key="avoided"),
            "clean_vs_baseline": compare(baseline_scored, scored, key="clean"),
        }

    results = {
        "model": args.model, "dtype": args.dtype, "set": str(set_path), "n_records": len(records),
        "budget": args.budget, "block_size": args.block_size, "temperature": args.temperature,
        "seed": args.seed, "strip_suggestions": args.strip_suggestions,
        "summaries": {name: summarize(scored) for name, scored in scored_by_strategy.items()},
        "comparisons": comparisons,
        "tsc_n_calls_total": runner.n_calls, "tsc_wall_s_total": runner.wall_s,
        "wall_s_total": time.monotonic() - t_run,
    }

    print("\n| strategy | clean | avoid | exact_gold | over_repair | no_progress | "
          "mean_fwd_tok | mean_fwd_tok_nocache | mcnemar_p (avoid) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, summary in results["summaries"].items():
        p = (comparisons[name]["avoided_vs_baseline"]["mcnemar_p"]
             if name in comparisons else float("nan"))
        print(f"| {name} | {summary['diagnostic_clean_rate']:.3f} | "
              f"{summary['error_avoidance_rate']:.3f} | {summary['exact_gold_rate']:.3f} | "
              f"{summary['over_repair_rate']:.3f} | {summary['no_progress_rate']:.3f} | "
              f"{summary['mean_n_forward_tokens']:.1f} | "
              f"{summary['mean_n_forward_tokens_nocache']:.1f} | {p:.4f} |")

    runner.close()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"results -> {args.output}")


if __name__ == "__main__":
    main()
