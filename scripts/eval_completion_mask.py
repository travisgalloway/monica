"""#226 — completion-list logit masking (constrained decode).

Five arms, forced by the #225 M4 null-sibling rule (`src.eval.ssi_contract.validate_arms`
rejects the naive three): `unconstrained`, `masked-null`, `masked`, `masked-oracle-null`,
`masked-oracle`. The headline is the paired `masked` -> `masked-oracle` gap: `masked`
says "a valid completion list was available and the model still couldn't pick"
(can't-name); `masked-oracle` restricts to the reference's own names, so the residual
gap isolates names-wrong from can't-name.

Primary set: `eval_sets/ts_error_injection/eval.jsonl` (96 records) — every prompt ends
at a member-access `.`, which is exactly what `masked-oracle` needs a real gold name for
(HumanEval-TS ships `gold_completion == "\\n"`, a placeholder, and is not usable for the
ceiling arm without an explicit `--oracle-reference` file).

    .venv/bin/python scripts/eval_completion_mask.py \\
        --model mlx-community/Qwen2.5-Coder-1.5B-bf16 \\
        --set eval_sets/ts_error_injection/eval.jsonl \\
        --seeds 0,1,2 --temperature 0.2 --mask-scope member \\
        --output results/ssi_226_mask.json --transcript results/ssi_226_mask.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


def _load_records(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-Coder-1.5B-bf16")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--set", type=Path, default=Path("eval_sets/ts_error_injection/eval.jsonl"))
    ap.add_argument("--oracle-reference", type=Path, default=None,
                    help="required for --set humaneval_ts.jsonl -- a JSONL of "
                         "{id, reference} the masked-oracle* arms read gold names from; "
                         "refused (not silently degraded) without it on that set")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--mask-scope", choices=("member", "identifier"), default="member")
    ap.add_argument("--max-gen-tokens", type=int, default=32)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--transcript", type=Path, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.temperature <= 0 and len(seeds) > 1:
        raise SystemExit("greedy (temperature<=0) with multiple seeds would report "
                         "identical 'seeds' -- pass --temperature > 0")
    if args.limit is None and len(set(seeds)) < 3:
        raise SystemExit(f"--seeds must give >= 3 distinct seeds (M2) for a real "
                         f"(non---limit) run, got {seeds}")
    # The #225 M1/M2 contract declares each arm's DESIGN as >= 3 seeds regardless of
    # how many this particular invocation actually runs -- a `--limit` smoke pass
    # (the plan's own example is `--seeds 0`) proves the wiring without claiming to
    # satisfy M2, which is about the measurement run, not the smoke pass.
    declared_seeds = tuple(sorted(set(seeds))) if len(set(seeds)) >= 3 else (0, 1, 2)

    is_humaneval = "humaneval" in args.set.name
    if is_humaneval and args.oracle_reference is None:
        raise SystemExit("--set is a HumanEval-style file (no real gold_completion) -- "
                         "the masked-oracle* arms need an explicit --oracle-reference "
                         "JSONL of {id, reference}; refusing rather than silently "
                         "degrading the ceiling arm")

    try:
        import mlx.core as mx  # noqa: F401 -- presence probe
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit("mlx not found -- run with the project venv on Apple Silicon.") from e
    import numpy as np

    from src.eval.lsp_eval import compare, score_record, summarize
    from src.eval.ssi_contract import (ArmSpec, ContractViolation, contract_report,
                                       pooled_compare, per_seed_compare, sign_test_p,
                                       summarize_arm, validate_arms)
    from src.lsp.completion_mask import (CompletionMasker, LspLabels, NullLabels,
                                          OracleLabels)
    from src.lsp.masked_decode import generate_masked
    from src.lsp.oracle import CompositeOracle
    from src.lsp.ts_service import TsLspService
    from src.model.mlx_lm_adapter import MLXLMAdapter

    records = _load_records(args.set)
    if args.limit is not None:
        records = records[: args.limit]

    oracle_refs: Dict[str, str] = {}
    if args.oracle_reference is not None:
        for rec in _load_records(args.oracle_reference):
            oracle_refs[rec["id"]] = rec.get("reference", "")
    else:
        for rec in records:
            oracle_refs[rec["id"]] = rec.get("reference") or rec.get("gold_completion", "")

    seeds_t = declared_seeds
    arms = [
        ArmSpec(name="unconstrained", variable="root", baseline="unconstrained",
                signal_available=False, signal_used=False, seeds=seeds_t,
                notes="free-running generation, no mask"),
        ArmSpec(name="masked-null", variable="mask", baseline="unconstrained",
                signal_available=True, signal_used=False, seeds=seeds_t,
                notes="#225 M4 null: queries the LSP per span, never applies the mask"),
        ArmSpec(name="masked", variable="mask", baseline="unconstrained",
                signal_available=True, signal_used=True, seeds=seeds_t,
                notes="constrained to the LSP's own completion list per span"),
        ArmSpec(name="masked-oracle-null", variable="mask-oracle", baseline="unconstrained",
                signal_available=True, signal_used=False, seeds=seeds_t,
                notes="#225 M4 null for the ceiling arm: reads the reference, applies no mask"),
        ArmSpec(name="masked-oracle", variable="mask-oracle", baseline="unconstrained",
                signal_available=True, signal_used=True, seeds=seeds_t,
                notes="ceiling: constrained to the reference's own identifier names"),
    ]
    validate_arms(arms)  # a contract violation must cost zero GPU minutes

    print(f"model={args.model} set={args.set.name} records={len(records)} "
          f"arms={[a.name for a in arms]} seeds={seeds}")

    t_load = time.monotonic()
    lm = MLXLMAdapter(args.model, dtype=args.dtype)
    oracle = CompositeOracle("ts")   # scoring stays bit-comparable with every prior #194/#199 run
    service = TsLspService()          # warm, multi-file -- masking only
    # One service instance == one project (`open_project` is a one-shot call) -- every
    # record's file must be registered up front, seeded with its PROMPT text, so
    # `update()` during generation (called by `LspLabels.query`) has something to
    # overwrite. `rec_{i}.ts` is the same path scheme used by the per-record loop below.
    service.open_project({f"rec_{i}.ts": rec["prompt"] for i, rec in enumerate(records)})
    print(f"model + oracle + mask service ready in {time.monotonic() - t_load:.1f}s")

    transcript_f = open(args.transcript, "w", encoding="utf-8") if args.transcript else None
    if transcript_f:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)

    n_oracle_unavailable = 0
    scored_by_arm_seed: Dict[str, Dict[int, List[dict]]] = {a.name: {} for a in arms}

    t_run = time.monotonic()
    for seed in seeds:
        rng = np.random.default_rng(seed) if args.temperature > 0 else None
        for arm in arms:
            scored: List[dict] = []
            for i, rec in enumerate(records):
                path = f"rec_{i}.ts"
                masker = None
                if arm.name != "unconstrained":
                    if arm.variable == "mask":
                        source = LspLabels(service, path)
                    else:  # mask-oracle
                        ref = oracle_refs.get(rec["id"], "")
                        if not ref.strip():
                            n_oracle_unavailable += 1
                            continue  # excluded from EVERY arm's paired comparison
                        source = OracleLabels(ref)
                    if not arm.signal_used:
                        source = NullLabels(source)
                    masker = CompletionMasker(source, path, lm.decode, mask_scope=args.mask_scope)

                result = generate_masked(lm, masker, rec["prompt"], budget="stmt",
                                         max_gen_tokens=args.max_gen_tokens,
                                         temperature=args.temperature, rng=rng,
                                         strategy=arm.name)
                s = score_record(rec, result, oracle.diagnostics)
                scored.append(s)

                if transcript_f:
                    transcript_f.write(json.dumps({
                        "arm": arm.name, "seed": seed, "id": rec["id"],
                        "completion": result.completion, "codes": s["codes"],
                        "clean": s["clean"], "resolved": s["resolved"],
                        "n_mask_bypass": result.n_mask_bypass,
                        "n_completion_calls": result.n_completion_calls,
                    }) + "\n")
                    transcript_f.flush()

            scored_by_arm_seed[arm.name][seed] = scored
            print(f"  seed={seed} arm={arm.name}: n={len(scored)}")

    if transcript_f:
        transcript_f.close()

    # #226's OracleLabels exclusion (R -- a blank reference) must apply IDENTICALLY
    # across every arm for `per_seed_compare` to stay paired; the driver skips the
    # SAME records in every arm above (the `continue` fires per-arm per-record, and
    # every arm walks the same `records` in the same order), so the resulting
    # per-seed lists are already aligned.

    summaries = {
        arm.name: {seed: summarize(recs) for seed, recs in by_seed.items()}
        for arm, by_seed in ((a, scored_by_arm_seed[a.name]) for a in arms)
    }

    def _pair(baseline_name: str, other_name: str, key: str) -> dict:
        base_by_seed = scored_by_arm_seed[baseline_name]
        other_by_seed = scored_by_arm_seed[other_name]
        per_seed = per_seed_compare(base_by_seed, other_by_seed, key=key)
        pooled = pooled_compare(base_by_seed, other_by_seed, key=key)
        deltas_sign = [1 if s["other_rate"] > s["baseline_rate"]
                       else (-1 if s["other_rate"] < s["baseline_rate"] else 0)
                       for s in per_seed.values()]
        n_pos = sum(1 for d in deltas_sign if d > 0)
        n_neg = sum(1 for d in deltas_sign if d < 0)
        return summarize_arm(per_seed, sign_test_p(n_pos, n_neg), pooled)

    stats = {
        "masked_vs_unconstrained": _pair("unconstrained", "masked", "resolved"),
        "masked_oracle_vs_unconstrained": _pair("unconstrained", "masked-oracle", "resolved"),
        # the headline
        "masked_oracle_vs_masked": _pair("masked", "masked-oracle", "resolved"),
    }

    latency = {}
    for arm in arms:
        by_seed = scored_by_arm_seed[arm.name]
        all_scored = [s for recs in by_seed.values() for s in recs]
        wall_s = sum(s.get("wall_s") or 0.0 for s in all_scored)
        n_tokens = sum(s.get("n_generated_tokens") or 0 for s in all_scored)
        latency[arm.name] = {
            "wall_s": wall_s,
            "ms_per_token": (1000.0 * wall_s / n_tokens) if n_tokens else float("nan"),
        }
    unconstrained_ms = latency["unconstrained"]["ms_per_token"]
    for name, block in latency.items():
        block["added_latency_ms_per_token"] = block["ms_per_token"] - unconstrained_ms

    results = {
        "model": args.model, "set": str(args.set), "n_records": len(records),
        "seeds": seeds, "temperature": args.temperature, "mask_scope": args.mask_scope,
        "n_oracle_unavailable": n_oracle_unavailable,
        "repo_split": None,  # ts_error_injection carries no meta["repo"] -- M3 doesn't apply here
        "repo_split_reason": "ts_error_injection records carry no meta['repo']; not repo-splittable",
        "summaries": summaries,
        "stats": stats,
        "latency": latency,
        "service": {"n_calls": service.n_calls, "wall_s": service.wall_s,
                    "op_counts": service.op_counts, "op_wall_s": service.op_wall_s,
                    "n_timeouts": service.n_timeouts, "n_restarts": service.n_restarts,
                    "n_no_publish": service.n_no_publish},
        "contract": contract_report(arms, {}, stats),
        "wall_s_total": time.monotonic() - t_run,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"results -> {args.output}")

    oracle.close()
    service.close()


if __name__ == "__main__":
    main()
