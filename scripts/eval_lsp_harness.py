"""#199 LSP-in-the-loop no-training validation: baseline vs. hard/soft repair vs.
tool-call, on the #194 TS error-injection eval set.

Wires config -> MLXLMAdapter -> CompositeOracle (`--oracle`: persistent TS-LSP by
default, optionally opengrep or both -- #199 Stage A) -> src/lsp/harness's generation
strategies -> src/eval/lsp_eval's scoring, mirroring scripts/eval_bfcl.py's split
(mlx-only imports kept local to main() so the seam stays clean). One `MLXLMAdapter`
and one `CompositeOracle` are created per script run and reused across every record
and strategy — reloading the model or restarting the oracle's language server(s)
per record would dominate the wall clock for no benefit.

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
    if name.startswith("toolcall-chat-k"):
        # The FAIR opponent (#199 follow-up): a real chat turn against an instruct
        # model, not a comment injected into a base model's context.
        try:
            k = int(name[len("toolcall-chat-k"):])
        except ValueError:
            raise SystemExit(f"bad strategy {name!r} (want toolcall-chat-k<N>)")
        return "toolcall-chat", {"k": k}
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
    ap.add_argument("--oracle", choices=("ts", "opengrep", "both"), default="ts",
                    help="diagnostic oracle: persistent TS-LSP (default), opengrep "
                         "(experimental -- see src/lsp/opengrep.py), or both merged")
    ap.add_argument("--rollback-strategy", choices=("auto", "trim", "reprefill", "snapshot"),
                     default="auto",
                     help="how rollback physically unwinds state (#202). 'auto' trims when the "
                          "cache allows (transformer) and re-prefills when it can't (SSM); "
                          "'reprefill' forces the re-prefill path even on a trimmable cache, so a "
                          "transformer can be compared like-for-like against an SSM.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strip-suggestions", action="store_true",
                    help="strip tsc's 'Did you mean X?' before it reaches the model "
                         "(the suggestion-leak ablation)")
    ap.add_argument("--ignore-module-resolution", action="store_true",
                    help="drop the module-resolution diagnostic family (TS2307 etc.) from "
                         "the in-loop diagnose AND scoring — for the #201 real-code "
                         "over-repair probe, whose clean_control files are selected to "
                         "compile clean ignoring unresolved imports")
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
    from src.lsp.harness import (generate_baseline, generate_slow_loop, generate_toolcall,
                                  generate_toolcall_chat)
    from src.lsp.diagnostics import MODULE_RESOLUTION_CODES, drop_codes
    from src.lsp.oracle import CompositeOracle, resolve_oracle
    from src.model.mlx_lm_adapter import MLXLMAdapter

    if not resolve_oracle(args.oracle):
        raise SystemExit(
            f"no toolchain for --oracle {args.oracle} — for 'ts'/'both', run `npm install` "
            "in eval_sets/ts_error_injection and `npm i -D typescript-language-server`; "
            "for 'opengrep', install opengrep and put it on PATH "
            "(see eval_sets/opengrep_rules/README.md).")

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
    lm = MLXLMAdapter(args.model, dtype=args.dtype, rollback_strategy=args.rollback_strategy)
    oracle = CompositeOracle(args.oracle)
    # For the #201 over-repair probe, make the module-resolution family invisible to BOTH
    # the in-loop diagnose and scoring (not just file selection) so an unresolved import
    # can't masquerade as a real diagnostic → rollback → false over-repair.
    diagnose = oracle.diagnostics
    if args.ignore_module_resolution:
        diagnose = drop_codes(oracle.diagnostics, MODULE_RESOLUTION_CODES)
    print(f"model + oracle({args.oracle}, sources_active={oracle.sources_active}) "
          f"ready in {time.monotonic() - t_load:.1f}s"
          + (" [ignoring module-resolution codes]" if args.ignore_module_resolution else ""))

    rng = np.random.default_rng(args.seed) if args.temperature > 0 else None

    transcript_f = None
    if args.transcript:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript_f = open(args.transcript, "w", encoding="utf-8")

    gen_kwargs = dict(budget=args.budget, block_size=args.block_size,
                       max_gen_tokens=args.max_gen_tokens, temperature=args.temperature, rng=rng)

    scored_by_strategy = {}
    summaries = {}
    t_run = time.monotonic()
    for strategy_name in strategies:
        kind, kwargs = _parse_strategy(strategy_name)
        scored: List[dict] = []
        for i, rec in enumerate(records):
            if kind == "baseline":
                result = generate_baseline(lm, rec["prompt"], **gen_kwargs)
            elif kind == "slow":
                result = generate_slow_loop(
                    lm, diagnose, rec["prompt"], max_retries=args.max_retries,
                    strip_suggestions=args.strip_suggestions, **kwargs, **gen_kwargs)
            elif kind == "toolcall-chat":
                # `budget` selects the system prompt, NOT a token cap: an instruct model
                # stops at <|im_end|>, so under budget="block" it must be *asked* for a
                # multi-statement continuation or it answers in ~10 characters while every
                # other strategy free-runs ~340 — and then wins on clean-rate for writing
                # almost nothing. Length parity is what makes this about feedback.
                result = generate_toolcall_chat(
                    lm, diagnose, rec["prompt"], budget=args.budget,
                    max_gen_tokens=args.max_gen_tokens, temperature=args.temperature,
                    rng=rng, strip_suggestions=args.strip_suggestions, **kwargs)
            else:  # toolcall
                result = generate_toolcall(
                    lm, diagnose, rec["prompt"],
                    strip_suggestions=args.strip_suggestions, **kwargs, **gen_kwargs)

            s = score_record(rec, result, diagnose)
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
        summaries[strategy_name] = summary
        # Extraction failure is a first-class outcome of the chat tool-call path, not a
        # glitch to hide: a model that answers with prose has failed the task, and a
        # clean-rate computed over only the parseable answers would flatter it.
        n_extract_fail = sum(1 for s in scored if s.get("extraction_failed"))
        summary["extraction_failure_rate"] = n_extract_fail / len(scored) if scored else 0.0
        # Which physical rollback path ran (#202) — the SSM cost story.
        summary["rollback_paths"] = {
            "trim": lm.n_trim_rollbacks,
            "reprefill": lm.n_reprefill_rollbacks,
            "snapshot": lm.n_snapshot_rollbacks,
            "reprefill_tokens": lm.n_reprefill_tokens,
        }
        summary["snapshot_bytes"] = lm.snapshot_bytes()
        print(f"[{strategy_name}] clean={summary['diagnostic_clean_rate']:.3f} "
              f"avoid={summary['error_avoidance_rate']:.3f} "
              f"exact_gold={summary['exact_gold_rate']:.3f} "
              f"over_repair={summary['over_repair_rate']:.3f} "
              f"no_progress={summary['no_progress_rate']:.3f} "
              f"extract_fail={summary['extraction_failure_rate']:.3f} "
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
        "oracle": args.oracle, "sources_active": oracle.sources_active,
        # Reuse the summaries built in the loop above — recomputing summarize() here would
        # silently discard the fields attached to them there (extraction_failure_rate,
        # rollback_paths, snapshot_bytes), which is how the SSM cost data went missing
        # from the first E3 run's JSON while still printing correctly to stdout.
        "summaries": summaries,
        "comparisons": comparisons,
        # Keep the pre-Stage-A key names (mapped from oracle.n_calls/.wall_s) so
        # anything reading past results JSONs doesn't need to change.
        "tsc_n_calls_total": oracle.n_calls, "tsc_wall_s_total": oracle.wall_s,
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

    oracle.close()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"results -> {args.output}")


if __name__ == "__main__":
    main()
