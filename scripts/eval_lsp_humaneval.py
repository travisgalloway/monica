"""#199 F1 — does the hard-ban lift survive on REAL multi-line code generation?

Runs the LSP repair harness on MultiPL-E HumanEval-TS (159 real function-generation
problems) instead of the #194 single-statement error-injection toy. Two axes per strategy:

  - `diagnostic_clean_rate` (primary): does prompt+body type-check? (McNemar vs baseline)
  - functional `pass@1` (guard): does prompt+body pass its HumanEval tests? Executed via
    `src/lsp/execute.py`. Reported next to clean-rate so a HOLLOW gain -- clean-rate up
    while pass@1 falls, i.e. the loop made code compile without making it correct -- is
    caught, not hidden. This is the E4 lesson as a first-class metric.

Separate from `eval_lsp_harness.py`, which is bound to the #194 error-injection schema and
its `error_avoidance_rate` metric. Reuses the harness strategies, `lsp_eval` scoring, and
`compare()` (McNemar) unchanged.

Pre-registered decision rule (1.5B base, completion mode): GENERALIZES iff slow-hard's
clean-rate beats baseline by >= 5 points AND McNemar p < 0.05 AND pass@1 does not drop.

    .venv/bin/python scripts/eval_lsp_humaneval.py \
        --model mlx-community/Qwen2.5-Coder-1.5B-bf16 \
        --strategies baseline,slow-hard --block-size 320 \
        --output results/f1_base.json --transcript results/f1_base.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional


def _load_humaneval(path: Path) -> List[dict]:
    """Bypass `load_ts_error_set` (which enforces + strips to the 7-key schema): F1 records
    carry `tests`/`stop_tokens`/`name` that the strict loader would not preserve."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-1.5B-bf16")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--set", type=Path, default=Path("eval_sets/humaneval_ts/humaneval_ts.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=320,
                    help="max body tokens; stop_tokens end it earlier at the next construct")
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--strategies", default="baseline,slow-hard",
                    help="baseline, slow-hard, slow-both, toolcall-chat-k<N>")
    ap.add_argument("--oracle", choices=("ts", "opengrep", "both"), default="ts",
                    help="diagnostic oracle: persistent TS-LSP (default), opengrep "
                         "(experimental -- see src/lsp/opengrep.py), or both merged")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-functional", action="store_true",
                    help="skip pass@1 execution (clean-rate only)")
    ap.add_argument("--exec-timeout", type=float, default=15.0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--transcript", type=Path, default=None)
    args = ap.parse_args()

    try:
        import mlx.core as mx  # noqa: F401 -- presence probe
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit("mlx not found — run with the project venv on Apple Silicon.") from e
    import numpy as np

    from src.eval.lsp_eval import compare, score_record, summarize
    from src.lsp.execute import Executor
    from src.lsp.harness import (generate_baseline, generate_slow_loop,
                                  generate_toolcall_chat)
    from src.lsp.oracle import CompositeOracle, resolve_oracle

    if not resolve_oracle(args.oracle):
        raise SystemExit(
            f"no toolchain for --oracle {args.oracle} — for 'ts'/'both', run `npm install` "
            "in eval_sets/ts_error_injection and `npm i -D typescript-language-server`; "
            "for 'opengrep', install opengrep and put it on PATH "
            "(see eval_sets/opengrep_rules/README.md).")

    records = _load_humaneval(args.set)
    if args.limit is not None:
        records = records[: args.limit]

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if "baseline" not in strategies:
        strategies = ["baseline"] + strategies

    print(f"model={args.model} set={args.set.name} records={len(records)} "
          f"strategies={strategies} functional={not args.no_functional}")

    from src.model.mlx_lm_adapter import MLXLMAdapter
    t_load = time.monotonic()
    lm = MLXLMAdapter(args.model, dtype=args.dtype)
    oracle = CompositeOracle(args.oracle)
    executor = None if args.no_functional else Executor(timeout_s=args.exec_timeout)
    print(f"model + oracle({args.oracle}, sources_active={oracle.sources_active}) "
          f"ready in {time.monotonic() - t_load:.1f}s")

    rng = np.random.default_rng(args.seed) if args.temperature > 0 else None
    transcript_f = open(args.transcript, "w", encoding="utf-8") if args.transcript else None
    if transcript_f:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)

    scored_by_strategy, summaries = {}, {}
    t_run = time.monotonic()
    for name in strategies:
        scored: List[dict] = []
        n_pass = 0
        for i, rec in enumerate(records):
            stops = rec.get("stop_tokens") or None
            if name == "baseline":
                result = generate_baseline(lm, rec["prompt"], budget="block",
                                           block_size=args.block_size, temperature=args.temperature,
                                           rng=rng, stop_strings=stops)
            elif name.startswith("slow-"):
                result = generate_slow_loop(lm, oracle.diagnostics, rec["prompt"],
                                            repair=name[len("slow-"):], budget="block",
                                            block_size=args.block_size, max_retries=args.max_retries,
                                            temperature=args.temperature, rng=rng, stop_strings=stops)
            elif name.startswith("toolcall-chat-k"):
                k = int(name[len("toolcall-chat-k"):])
                result = generate_toolcall_chat(lm, oracle.diagnostics, rec["prompt"], k=k,
                                                budget="block", temperature=args.temperature, rng=rng)
            else:
                raise SystemExit(f"unknown strategy {name!r}")

            s = score_record(rec, result, oracle.diagnostics)
            # Functional pass@1 — the independent correctness guard.
            if executor is not None:
                ex = executor.run_tests(rec["prompt"], result.completion, rec["tests"])
                s["functional_pass"] = ex.passed
                s["functional_outcome"] = ex.outcome
                n_pass += int(ex.passed)
            scored.append(s)

            if transcript_f:
                transcript_f.write(json.dumps({
                    "strategy": name, "id": rec["id"], "prompt": rec["prompt"],
                    "completion": result.completion, "artifact": result.artifact,
                    "codes": s["codes"], "clean": s["clean"],
                    "functional_pass": s.get("functional_pass"),
                    "functional_outcome": s.get("functional_outcome"),
                    "n_rollbacks": result.n_rollbacks, "events": result.events,
                }) + "\n")
                transcript_f.flush()   # per-record, so progress is visible during a long run
            if (i + 1) % 20 == 0:
                print(f"  {name}: {i + 1}/{len(records)}")

        scored_by_strategy[name] = scored
        summary = summarize(scored)
        if executor is not None:
            summary["functional_pass_at_1"] = n_pass / len(scored) if scored else 0.0
        summaries[name] = summary
        pass_str = (f" pass@1={summary['functional_pass_at_1']:.3f}"
                    if executor is not None else "")
        print(f"[{name}] clean={summary['diagnostic_clean_rate']:.3f}{pass_str} "
              f"rollbacks={summary['mean_n_rollbacks']:.2f}")

    if transcript_f:
        transcript_f.close()

    # Paired McNemar vs baseline, on BOTH axes.
    baseline = scored_by_strategy["baseline"]
    comparisons = {}
    for name, scored in scored_by_strategy.items():
        if name == "baseline":
            continue
        comp = {"clean_vs_baseline": compare(baseline, scored, key="clean")}
        if executor is not None:
            comp["functional_vs_baseline"] = compare(baseline, scored, key="functional_pass")
        comparisons[name] = comp

    results = {
        "model": args.model, "set": str(args.set), "n_records": len(records),
        "block_size": args.block_size, "temperature": args.temperature, "seed": args.seed,
        "functional": not args.no_functional,
        "oracle": args.oracle, "sources_active": oracle.sources_active,
        "summaries": summaries, "comparisons": comparisons,
        # Keep the pre-Stage-A key names (mapped from oracle.wall_s/.n_calls) so
        # anything reading past results JSONs doesn't need to change.
        "tsc_wall_s_total": oracle.wall_s, "wall_s_total": time.monotonic() - t_run,
    }
    # When the opengrep arm ran, record its reliability counters so the run is
    # self-describing about completeness (n_timeouts = silently-dropped scans).
    if oracle.opengrep_stats is not None:
        results["opengrep"] = oracle.opengrep_stats
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"results -> {args.output}")

    oracle.close()
    if executor is not None:
        executor.close()


if __name__ == "__main__":
    main()
