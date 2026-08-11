#!/usr/bin/env python3
"""The M12 code eval suite driver (#221) — deterministic, per-instance, no pass@1 gating.

Runs the custom probes (cross-file symbol recall, RULER-over-code needle, FIM by prefix
*and* recall distance, per-domain held-out BPB), the external-suite adapters, and the
existing type-aware `tsc` oracle, emitting **one shared per-instance record schema** to a
JSONL transcript plus a results JSON.

    # Offline, no backend, no checkpoint — the determinism gate:
    .venv/bin/python scripts/eval_code_suite.py --stub-model --byte-tokenizer \\
        --suites recall,needle,fim,external --seed 0 \\
        --output results/code_suite.json --transcript results/code_suite.jsonl

    # Against a real checkpoint:
    .venv/bin/python scripts/eval_code_suite.py --config config/toy-hybrid.yaml \\
        --checkpoint run/weights.safetensors --backend mlx \\
        --tokenizer-json artifacts/tokenizer.json \\
        --suites recall,needle,fim,domain-bpb --domains-json data/domains/domains.json

Design notes that are load-bearing
----------------------------------
* **Backend imports live inside `main()`** (the `scripts/eval_lsp_harness.py` /
  `scripts/smoke_test.py` pattern), so this module and everything it imports stay portable.
* **`--stub-model`** substitutes `code_suite.StubCausalModel`, a deterministic pure-numpy
  causal fake. There is no trained MHM checkpoint yet (#200/#222 are downstream), so the
  acceptance bar for this suite is fixtures + determinism, not quality numbers. **Stub
  numbers are meaningless as quality and must never be reported as measured recall.**
* **Determinism**: every scored suite emits records in a fixed order, the transcript is
  written with `sort_keys=True` canonical JSON, and all randomness flows through one
  `np.random.default_rng(--seed)`. Two runs at the same seed produce byte-identical JSONL.
  The results JSON is likewise deterministic **except** for its `timing` block, which is
  the only place wall-clock is recorded.
* **`--temperature`** exists for parity with the other eval drivers and defaults to `0.0`.
  Nothing in the teacher-forced suites generates, so it is unused by them; it applies only
  to `--suites tsc`. It is recorded in the config echo either way.
* A suite that cannot run **skips loudly** and is listed in `suites_skipped` with the
  reason. Silence is never treated as success.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

ALL_SUITES = ("recall", "needle", "fim", "domain-bpb", "external", "tsc")
DEFAULT_SUITES = "recall,needle,fim"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None, help="model config YAML")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="portable weights (safetensors) to load into the model")
    ap.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto")
    ap.add_argument("--stub-model", action="store_true",
                    help="use the deterministic pure-numpy causal fake instead of a backend "
                         "(offline CI / determinism gate; its numbers are NOT quality numbers)")
    ap.add_argument("--suites", default=DEFAULT_SUITES,
                    help=f"comma-separated subset of {','.join(ALL_SUITES)} "
                         f"(default: {DEFAULT_SUITES})")

    ap.add_argument("--byte-tokenizer", action="store_true",
                    help="encode with ByteTokenizer (vocab 256, offline fixture encoder)")
    ap.add_argument("--tokenizer-json", type=Path, default=None,
                    help="tokenizer.json for the native Swift tokenizer (the real encoder)")
    ap.add_argument("--tokenizer-bin", default="monica-tokenize")

    ap.add_argument("--fixture-repo", type=Path,
                    default=REPO_ROOT / "eval_sets/code_recall/fixture_repo.jsonl")
    ap.add_argument("--haystack", type=Path,
                    default=REPO_ROOT / "eval_sets/code_needle/haystack.jsonl")
    ap.add_argument("--domains-json", type=Path, default=None,
                    help="domains.json from scripts/build_domain_val_sets.py (--suites domain-bpb)")
    ap.add_argument("--shard-dir", type=Path, default=None,
                    help="Swift-packed SHARD dir for the FIM suite (not a split dir — split.py "
                         "drops the .bounds sidecars). Falls back to the fixture repo when absent.")
    ap.add_argument("--blocklist", type=Path,
                    default=REPO_ROOT / "eval_sets/decontam/blocklist.txt",
                    help="decontamination blocklist; its sha256 is echoed into the results JSON")

    ap.add_argument("--context-lens", default="512,1024",
                    help="needle grid context lengths, in tokens")
    ap.add_argument("--depths", default="0.0,0.25,0.5,0.75,1.0", help="needle grid depths")
    ap.add_argument("--needle-variants", default="single,multikey")
    ap.add_argument("--fim-per-doc", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the instances scored per suite")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches per domain for --suites domain-bpb")
    ap.add_argument("--seq-len", type=int, default=256, help="--suites domain-bpb chunk length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="unused by the teacher-forced suites (nothing here generates); "
                         "applies only to --suites tsc. Recorded in the config echo.")
    ap.add_argument("--tsc-set", type=Path,
                    default=REPO_ROOT / "eval_sets/ts_error_injection/eval.jsonl")

    ap.add_argument("--output", type=Path, default=None, help="write the results JSON here")
    ap.add_argument("--transcript", type=Path, default=None,
                    help="write the per-instance JSONL transcript here")
    args = ap.parse_args()

    unknown = [s for s in args.suites.split(",") if s.strip() and s.strip() not in ALL_SUITES]
    if unknown:
        ap.error(f"unknown suite(s) {unknown}; known: {list(ALL_SUITES)}")
    if not args.stub_model and args.config is None:
        ap.error("pass --config (and usually --checkpoint), or --stub-model for the offline run")
    if not args.byte_tokenizer and args.tokenizer_json is None:
        ap.error("pass --tokenizer-json <tokenizer.json>, or --byte-tokenizer for offline runs")
    return args


def _build_model(args):
    """Return `(model, to_numpy, identity_dict)`. Backend imports stay inside this function."""
    from src.eval.code_suite import StubCausalModel

    if args.stub_model:
        vocab = 256 if args.byte_tokenizer else 256
        return (StubCausalModel(vocab_size=vocab, seed=args.seed), None,
                {"kind": "stub", "vocab_size": vocab, "seed": args.seed,
                 "warning": "StubCausalModel is a deterministic fake — its scores are NOT "
                            "quality numbers"})

    from src.model.backend import get_backend
    from src.model.blocks import load_config

    backend = get_backend(args.backend)
    cfg = load_config(str(args.config))
    backend.seed(args.seed)
    model = backend.model_cls(cfg)
    if args.checkpoint:
        from src.train.checkpoint import load_weights
        load_weights(model, str(args.checkpoint))
    return model, backend.to_numpy, {
        "kind": backend.name, "config": str(args.config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "vocab_size": cfg.vocab_size,
    }


def _encoder(args):
    from src.eval.code_suite import make_byte_encoder, make_swift_encoder

    if args.byte_tokenizer:
        return make_byte_encoder(), {"kind": "byte", "vocab_size": 256}
    return (make_swift_encoder(args.tokenizer_json, binary=args.tokenizer_bin),
            {"kind": "swift", "tokenizer_json": str(args.tokenizer_json),
             "binary": args.tokenizer_bin})


# --------------------------------------------------------------------------------------- #
# Suites
# --------------------------------------------------------------------------------------- #

def _run_recall(args, model, to_numpy, encode, rng):
    from src.eval.code_recall import build_recall_instances, evaluate_code_recall
    from src.eval.code_suite import load_code_files

    files = load_code_files(args.fixture_repo)
    instances = build_recall_instances(files, encode, rng, max_instances=args.limit)
    if not instances:
        raise SystemExit(
            f"--suites recall: no instances from {args.fixture_repo}. The extractor is "
            "fail-closed, so this means no import resolved to an exported symbol with a "
            "locatable use site — check the fixture, do not lower the bar.")
    result = evaluate_code_recall(model, instances, batch_size=args.batch_size,
                                  to_numpy=to_numpy)
    return result, {"fixture_repo": str(args.fixture_repo), "n_files": len(files),
                    "n_instances": len(instances)}


def _run_needle(args, model, to_numpy, encode, rng):
    from src.eval.code_needle import build_needle_instances, evaluate_code_needle
    from src.eval.code_suite import load_code_files

    files = load_code_files(args.haystack)
    context_lens = tuple(int(v) for v in args.context_lens.split(",") if v.strip())
    depths = tuple(float(v) for v in args.depths.split(",") if v.strip())
    variants = tuple(v.strip() for v in args.needle_variants.split(",") if v.strip())
    instances = build_needle_instances(files, encode, rng, context_lens=context_lens,
                                       depths=depths, variants=variants)
    if args.limit is not None:
        instances = instances[:args.limit]
    if not instances:
        raise SystemExit(
            f"--suites needle: no instances — every requested context length is too small to "
            f"hold the needles plus the query ({args.context_lens}).")
    result = evaluate_code_needle(model, instances, batch_size=args.batch_size,
                                  to_numpy=to_numpy, context_lens=context_lens, depths=depths)
    return result, {"haystack": str(args.haystack), "context_lens": list(context_lens),
                    "depths": list(depths), "variants": list(variants),
                    "n_instances": len(instances)}


def _run_fim(args, model, to_numpy, encode, rng):
    """FIM by prefix distance AND recall distance from ONE forward pass."""
    import numpy as np

    from src.eval.code_suite import bucket_for_distance, load_code_files, make_record
    from src.eval.fim_eval import (DEFAULT_BUCKETS, build_fim_examples, documents_from_shards,
                                   evaluate_fim_multi_key)

    if args.shard_dir is not None:
        docs = documents_from_shards(args.shard_dir, min_len=8, max_docs=args.limit)
        source = {"shard_dir": str(args.shard_dir)}
    else:
        # Offline fallback: the fixture repo, encoded with the injected tokenizer. Real runs
        # should point at the packed corpus so the streams are the Swift-inserted ones.
        files = load_code_files(args.fixture_repo)
        docs = [np.asarray(list(encode(f["text"])), dtype=np.int64) for f in files]
        source = {"fixture_repo": str(args.fixture_repo),
                  "note": "no --shard-dir; FIM examples built from the fixture repo"}
    if not docs:
        raise SystemExit("--suites fim: no documents recovered — check --shard-dir")

    examples = build_fim_examples(docs, rng, n_per_doc=args.fim_per_doc, min_middle=2)
    if args.limit is not None:
        examples = examples[:args.limit]
    result = evaluate_fim_multi_key(model, examples, batch_size=args.batch_size,
                                    to_numpy=to_numpy)

    # Project onto the shared record schema, keyed on recall distance (the SSM-relevant one);
    # the prefix-length keying rides in the summary's `by_key` block.
    records = []
    for i, rec in enumerate(result["records"]):
        if not rec["n_middle_tokens"]:
            continue
        records.append(make_record(
            suite="fim", id=f"doc{rec['doc_index']:05d}_ex{i:05d}",
            bucket=bucket_for_distance(rec["recall_distance"], DEFAULT_BUCKETS),
            distance=rec["recall_distance"], n_scored_tokens=rec["n_middle_tokens"],
            ce_nats=rec["ce_nats"], token_accuracy=rec["token_accuracy"],
            exact_match=rec["exact_match"],
            meta={"prefix_len": rec["prefix_len"], "middle_len": rec["middle_len"],
                  "suffix_len": rec["suffix_len"], "doc_index": rec["doc_index"]}))
    summary = {"by_key": result["by_key"], "overall": result["overall"],
               "advisory": result["advisory"]}
    return {"records": records, **summary}, {**source, "n_examples": len(examples),
                                             "n_per_doc": args.fim_per_doc}


def _run_external(args, model, to_numpy, encode):
    """Teacher-forced scoring of the external-suite fixtures via their adapters.

    Only rows that ship a gold `answer` can be scored: MultiPL-E has none (it ships tests,
    not bodies), so those sets are recorded as **loaded but unscored** rather than given a
    fabricated score. Infill sets are laid out PSM-style with `FIMSentinels`' reserved ids,
    matching `src/eval/fim_eval.py`; under `--byte-tokenizer` those ids are ordinary control
    bytes, which is acceptable for a fixture-only adapter smoke run and is not how a real
    run is configured.
    """
    import numpy as np

    from src.eval.code_suite import (ScoreRow, bucket_for_distance, make_record, score_rows,
                                     summarize_bucketed)
    from src.eval.external_sets import EXTERNAL_SETS, external_sets_manifest, load_external
    from src.eval.fim_eval import DEFAULT_BUCKETS, FIMSentinels

    sentinels = FIMSentinels()
    records: List[dict] = []
    per_set: Dict[str, dict] = {}

    for name in sorted(EXTERNAL_SETS):
        rows = load_external(name, fixture_only=True, limit=args.limit)
        scored_rows: List[ScoreRow] = []
        ids: List[str] = []
        distances: List[int] = []
        for row in rows:
            if not row.get("answer"):
                continue
            ctx_parts: List[np.ndarray] = []
            if row["kind"] == "infill":
                ctx_parts = [
                    np.array([sentinels.prefix], dtype=np.int64),
                    np.asarray(list(encode(row["prompt"])), dtype=np.int64),
                    np.array([sentinels.suffix], dtype=np.int64),
                    np.asarray(list(encode(row["suffix"] or "")), dtype=np.int64),
                    np.array([sentinels.middle], dtype=np.int64),
                ]
            else:
                cross = (row.get("meta") or {}).get("crossfile_context")
                if cross:
                    ctx_parts.append(np.asarray(list(encode(cross)), dtype=np.int64))
                ctx_parts.append(np.asarray(list(encode(row["prompt"])), dtype=np.int64))
            prefix = np.concatenate([p.reshape(-1).astype(np.int64) for p in ctx_parts])
            answer = np.asarray(list(encode(row["answer"])), dtype=np.int64)
            if prefix.size < 1 or answer.size < 1:
                continue
            scored_rows.append(ScoreRow(tokens=np.concatenate([prefix, answer]),
                                        span_start=int(prefix.size), span_len=int(answer.size)))
            ids.append(row["id"])
            distances.append(int(prefix.size))

        if not scored_rows:
            per_set[name] = {"n_rows": len(rows), "n_scored": 0,
                             "reason": "no gold `answer` field — loaded and adapted, unscored"}
            continue

        scored = score_rows(model, scored_rows, batch_size=args.batch_size, to_numpy=to_numpy)
        for rid, dist, s in zip(ids, distances, scored):
            records.append(make_record(
                suite=f"external:{name}", id=rid,
                bucket=bucket_for_distance(dist, DEFAULT_BUCKETS), distance=dist,
                n_scored_tokens=s["n_scored_tokens"], ce_nats=s["ce_nats"],
                token_accuracy=s["token_accuracy"], exact_match=s["exact_match"],
                meta={"set": name}))
        per_set[name] = {"n_rows": len(rows), "n_scored": len(scored_rows)}

    if not records:
        raise SystemExit("--suites external: no external row carried a gold answer to score")
    summary = summarize_bucketed(records, [b[0] for b in DEFAULT_BUCKETS])
    return ({"records": records, **summary, "by_set": per_set},
            {"sets": external_sets_manifest(fixture_only=True)})


def _run_domain_bpb(args, model, to_numpy):
    from src.eval.domain_bpb import evaluate_domain_bpb, load_domain_index

    index = load_domain_index(args.domains_json)
    domains = {name: entry["packed"] for name, entry in index.items()}
    result = evaluate_domain_bpb(model, domains, batch_size=args.batch_size,
                                 seq_len=args.seq_len, max_batches=args.max_batches,
                                 to_numpy=to_numpy)
    return result, {"domains_json": str(args.domains_json), "n_domains": len(domains)}


def _run_tsc(args):
    """Type-aware completion — surfaced, NOT rebuilt.

    `TscRunner` / `CompositeOracle` (`src/lsp/`) and `scripts/eval_lsp_harness.py` already
    implement this; all that is new here is emitting the existing oracle's verdicts in the
    shared record schema. Records carry `n_scored_tokens=0` (nothing is teacher-forced), so
    this suite reports its own clean-rate rather than going through `summarize_records`.
    """
    from src.eval.code_suite import make_record, read_jsonl
    from src.lsp.oracle import CompositeOracle, resolve_oracle

    if not resolve_oracle("ts"):
        raise RuntimeError(
            "no TS-LSP toolchain — run `npm install` in eval_sets/ts_error_injection and "
            "`npm i -D typescript-language-server`")
    rows = read_jsonl(args.tsc_set)
    if args.limit is not None:
        rows = rows[:args.limit]

    oracle = CompositeOracle("ts")
    try:
        records = []
        n_clean = 0
        for row in rows:
            artifact = row["prompt"] + row.get("gold_completion", "")
            codes = [d.code for d in oracle.diagnostics(artifact)]
            clean = not codes
            n_clean += int(clean)
            records.append(make_record(
                suite="tsc", id=row["id"], bucket=row.get("error_class", "unknown"),
                distance=0, n_scored_tokens=0, exact_match=float(clean),
                meta={"codes": codes, "error_class": row.get("error_class"),
                      "expected_diagnostic": row.get("expected_diagnostic")}))
    finally:
        oracle.close()

    return ({"records": records,
             "summary": {"n": len(records),
                         "clean_rate": (n_clean / len(records)) if records else None}},
            {"tsc_set": str(args.tsc_set), "sources_active": oracle.sources_active})


# --------------------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------------------- #

def main() -> int:
    args = _parse_args()

    import numpy as np

    from src.eval.code_suite import format_bucket_table, sha256_file, write_jsonl

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    encode, tokenizer_id = _encoder(args)
    model, to_numpy, model_id = _build_model(args)
    if to_numpy is None:
        to_numpy = np.asarray

    t0 = time.monotonic()
    timings: Dict[str, float] = {}
    summaries: Dict[str, dict] = {}
    sources: Dict[str, dict] = {}
    skipped: Dict[str, str] = {}
    all_records: List[dict] = []

    for suite in suites:
        t_suite = time.monotonic()
        # One generator per suite, derived from --seed and the suite name, so adding or
        # dropping a suite never perturbs another suite's instances.
        # (`hash()` on a str is salted per interpreter, so the seed is derived from the
        # suite name's code points instead — reproducible across processes.)
        rng = np.random.default_rng([args.seed, sum(ord(c) for c in suite)])
        try:
            if suite == "recall":
                result, src = _run_recall(args, model, to_numpy, encode, rng)
            elif suite == "needle":
                result, src = _run_needle(args, model, to_numpy, encode, rng)
            elif suite == "fim":
                result, src = _run_fim(args, model, to_numpy, encode, rng)
            elif suite == "external":
                result, src = _run_external(args, model, to_numpy, encode)
            elif suite == "domain-bpb":
                if args.domains_json is None:
                    raise RuntimeError("--suites domain-bpb needs --domains-json (build it "
                                       "with scripts/build_domain_val_sets.py)")
                result, src = _run_domain_bpb(args, model, to_numpy)
            elif suite == "tsc":
                result, src = _run_tsc(args)
            else:                                        # unreachable: validated in _parse_args
                raise RuntimeError(f"unhandled suite {suite!r}")
        except RuntimeError as e:
            # A missing optional toolchain / artifact is a LOUD skip recorded in the results,
            # never a silent pass. Genuine failures (SystemExit, ValueError) still propagate.
            skipped[suite] = str(e)
            print(f"[{suite}] SKIPPED: {e}")
            continue

        records = result.pop("records", [])
        all_records.extend(records)
        summaries[suite] = result
        sources[suite] = src
        timings[suite] = time.monotonic() - t_suite

        if "by_bucket" in result:
            print(format_bucket_table(suite, result))
        elif suite == "fim":
            from src.eval.fim_eval import format_fim_multi_table
            print(format_fim_multi_table(result))
        elif suite == "domain-bpb":
            from src.eval.domain_bpb import format_domain_bpb_table
            print(format_domain_bpb_table(result))
        elif suite == "tsc":
            print(f"tsc: clean_rate={result['summary']['clean_rate']} "
                  f"over {result['summary']['n']} records")

    if not summaries:
        raise SystemExit("every requested suite was skipped — nothing was measured. "
                         f"Reasons: {skipped}")

    # Deterministic transcript order: (suite, id).
    all_records.sort(key=lambda r: (r["suite"], r["id"]))
    if args.transcript:
        n = write_jsonl(all_records, args.transcript)
        print(f"transcript -> {args.transcript} ({n} records)")

    blocklist_sha = None
    if args.blocklist and Path(args.blocklist).exists():
        blocklist_sha = sha256_file(args.blocklist)
    elif args.blocklist:
        print(f"[warn] blocklist {args.blocklist} does not exist — build it with "
              "scripts/build_decontam_blocklist.py; recording sha256=null")

    results = {
        "config": {
            "suites": suites, "seed": args.seed, "temperature": args.temperature,
            "batch_size": args.batch_size, "limit": args.limit, "seq_len": args.seq_len,
            "max_batches": args.max_batches,
            "tokenizer": tokenizer_id, "model": model_id,
            "fixture_repo": str(args.fixture_repo), "haystack": str(args.haystack),
            "domains_json": str(args.domains_json) if args.domains_json else None,
            "shard_dir": str(args.shard_dir) if args.shard_dir else None,
            "blocklist": str(args.blocklist) if args.blocklist else None,
            "blocklist_sha256": blocklist_sha,
        },
        "sources": sources,
        "summaries": summaries,
        "suites_skipped": skipped,
        "n_records": len(all_records),
        # Wall-clock is the ONLY non-deterministic part of this file and is quarantined
        # here, so a determinism check can diff `results` minus `timing`.
        "timing": {**timings, "total_s": time.monotonic() - t0},
    }

    print("\n| suite | instances | ce | tok_acc | exact | top1 | mrr |")
    print("|---|---|---|---|---|---|---|")
    for suite, summary in summaries.items():
        o = summary.get("overall")
        if not o:
            print(f"| {suite} | — | — | — | — | — | — |")
            continue

        def _f(v):
            return "—" if v is None else f"{v:.4f}"

        print(f"| {suite} | {o.get('n_instances', o.get('n_examples', '—'))} | "
              f"{_f(o.get('ce'))} | {_f(o.get('token_accuracy'))} | "
              f"{_f(o.get('exact_match_rate'))} | {_f(o.get('rank_top1_rate'))} | "
              f"{_f(o.get('mrr'))} |")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True, default=str)
        print(f"results -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
