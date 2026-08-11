#!/usr/bin/env python3
"""SLA harness for `src/lsp/ts_service.py`'s `TsLspService` (#220) -- the issue's
accept/kill criterion for whether a warm, multi-op `typescript-language-server`
process can sustain the throughput #226's decode-time completion-list masking
needs, and the latency the code eval suite's cross-file arm needs.

Generates a deterministic synthetic N-file TypeScript project (`_generate_project`,
`random.Random(seed)` only -- never eval transcripts, the `opengrep_soak.py`
contamination discipline), opens it once as a warm `TsLspService` program, runs a
BLIND-guarded sanity probe against KNOWN targets, then measures per-op
median/mean/min/p95 latency and calls/s for `diagnostics`, `definition`,
`references`, `completions`, `quickinfo`, `document_symbols`.

    # Fast smoke first, to shake out protocol shape before the 5k run:
    .venv/bin/python scripts/bench_ts_lsp.py --n-files 50 --warmup 3 --iters 20

    # The real run:
    .venv/bin/python scripts/bench_ts_lsp.py --n-files 5000 --warmup 20 --iters 200

**BLIND guard.** A project too large for the server, a bad tsconfig, or a
semantic-features-disabled tsserver makes every op return empty results very
FAST, which reads as a spectacular PASS. `_sanity_probe` confirms real,
known-answer results on every op BEFORE any timing starts; any failure prints
the failing checks, writes `{"verdict": "BLIND", ...}` to `--out`, and exits 2.
An empty measurement is never reported as a pass.

**Verdict**, keyed to the issue's two consumers:
  - `PASS`  -- median warm `diagnostics` (edit->re-check) < 50ms AND
               `completions` >= 200 calls/s.
  - `WARN`  -- between the two bars (either gate missed but completions >= 20/s).
  - `KILL`  -- completions < ~20 calls/s -> #226 decode-time masking rescopes
               to offline.
  - `BLIND` -- the sanity probe failed; nothing was measured.

If the bench lands in WARN or KILL, that is a real, reportable result -- record
the numbers and the #226 consequence in `docs/design/12-lsp-in-the-loop.md`.
Do NOT tune the harness or loosen a threshold until it passes.

Exits cleanly (0) with no traceback when no `typescript-language-server`
toolchain is resolvable (`opengrep_soak.py:80-84` style). Always tears down the
service and cleans its scratch dir in a `finally` unless `--keep-scratch`.

Stdlib + above-the-seam repo imports only (no mlx/torch).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Repo root on sys.path so `src.lsp.*` imports resolve when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lsp.tsc import SET_DIR  # noqa: E402
from src.lsp.ts_lsp import resolve_ts_lsp  # noqa: E402
from src.lsp.ts_service import TsLspService  # noqa: E402

_HUB_PATH = "src/hub.ts"
_IMPORT_RE = re.compile(r'import\s*\{[^}]*\}\s*from\s*"\./([A-Za-z0-9_]+)"')


# --------------------------------------------------------------------------- #
# Deterministic project generator
# --------------------------------------------------------------------------- #

def _mod_path(i: int) -> str:
    return f"src/mod_{i:04d}.ts"


def _module_index(path: str) -> int:
    """`src/hub.ts` -> 0; `src/mod_0007.ts` -> 7."""
    stem = Path(path).stem
    if stem == "hub":
        return 0
    return int(stem.split("_")[1])


def _resolve_import(path: str, target_rel: str) -> str:
    """`target_rel` is the bare module name from an `import ... from "./X"` --
    `"hub"` or `"mod_0007"`. `path` (the importer) is unused; kept for a
    stable, obvious call signature at the two call sites (bench + test)."""
    return f"src/{target_rel}.ts"


def _generate_project(n_files: int, seed: int) -> Dict[str, str]:
    """Deterministic synthetic TypeScript project: `src/hub.ts` (index 0,
    widely-imported -- the `references` fan-in target) plus
    `src/mod_0001.ts` .. `src/mod_{n_files-1:04d}.ts`. Every mod
    unconditionally imports `{ Vec, scale }` from hub, plus 1-3 seeded imports
    from LOWER-indexed mods only (acyclic by construction -- every import
    resolves, no TS2307 noise), and exports a function/interface that
    genuinely uses the imported symbols (mirrors
    `eval_sets/code_recall/fixture_repo.jsonl`'s import/export shape, scaled
    up). `random.Random(seed)` only -- no global RNG -- so the same
    `(n_files, seed)` is byte-identical every time."""
    if n_files < 2:
        raise ValueError("n_files must be >= 2")
    rng = random.Random(seed)
    files: Dict[str, str] = {
        _HUB_PATH: (
            "export interface Vec { x: number; y: number }\n\n"
            "export function scale(v: Vec, k: number): Vec {\n"
            "  return { x: v.x * k, y: v.y * k };\n"
            "}\n\n"
            "export const ORIGIN: Vec = { x: 0, y: 0 };\n"
        ),
    }

    for i in range(1, n_files):
        lower_mods = list(range(1, i))   # other mods only -- hub is handled separately below
        k = rng.randint(1, min(3, len(lower_mods))) if lower_mods else 0
        chosen = sorted(rng.sample(lower_mods, k)) if k else []

        import_lines = ['import { Vec, scale } from "./hub";']
        uses = ["scale(v, 2)"]
        for idx in chosen:
            import_lines.append(f'import {{ f{idx} }} from "./{Path(_mod_path(idx)).stem}";')
            uses.append(f"f{idx}(v)")

        body_uses = "\n  ".join(f"const u{j} = {u};" for j, u in enumerate(uses))
        files[_mod_path(i)] = (
            "\n".join(import_lines) + "\n\n"
            f"export function f{i}(v: Vec): Vec {{\n"
            f"  {body_uses}\n"
            f"  const vx = v.x;\n"
            f"  return scale(v, 1);\n"
            f"}}\n\n"
            f"export interface T{i} {{ v: Vec; tag: number }}\n"
        )

    return files


# --------------------------------------------------------------------------- #
# BLIND guard
# --------------------------------------------------------------------------- #

def _sanity_probe(service: TsLspService, files: Dict[str, str]) -> dict:
    """Confirm real, known-answer results on every op against a known target
    in the generated project, BEFORE any timing starts. Never trust an empty
    result as "the server has nothing to say" -- an empty-but-fast run is
    indistinguishable from a broken one unless something here fails loudly."""
    target = _mod_path(1)
    mod_text = files[target]
    hub_text = files[_HUB_PATH]
    checks: Dict[str, bool] = {}

    call_offset = mod_text.index("scale(")
    defs = service.definition(target, call_offset)
    checks["definition_resolves_to_hub"] = any(d.path == _HUB_PATH for d in defs)

    vec_offset = hub_text.index("interface Vec") + len("interface ")
    refs = service.references(_HUB_PATH, vec_offset, include_declaration=True)
    checks["references_fan_in"] = len(refs) > 1 and len({r.path for r in refs}) >= 2

    dot_offset = mod_text.index("v.x") + 2
    completions = service.completions(target, dot_offset)
    checks["completions_contains_x"] = any(c.label == "x" for c in completions)

    info = service.quickinfo(target, call_offset)
    checks["quickinfo_nonempty"] = bool(info)

    symbols = service.document_symbols(target)
    checks["document_symbols_nonempty"] = bool(symbols)

    broken = mod_text.replace("v.x", "v.gorblak", 1)
    service.update(target, broken)
    broken_diags = service.diagnostics(target)
    checks["diagnostics_detects_break"] = any(d.code == "TS2339" for d in broken_diags)
    service.update(target, mod_text)   # revert
    clean_diags = service.diagnostics(target)
    checks["diagnostics_clean_after_revert"] = clean_diags == []

    failed = sorted(k for k, v in checks.items() if not v)
    return {"ok": not failed, "checks": checks, "failed": failed}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def _is_nonempty(x) -> bool:
    if x is None:
        return False
    if isinstance(x, str):
        return bool(x)
    try:
        return len(x) > 0
    except TypeError:
        return bool(x)


def _summarize(op_name: str, samples_s: List[float], n_nonempty: int, iters: int) -> dict:
    ms = sorted(s * 1000.0 for s in samples_s)
    n = len(ms)
    mean_ms = sum(ms) / n if n else 0.0
    median_ms = ms[n // 2] if n else 0.0
    min_ms = ms[0] if n else 0.0
    p95_ms = ms[min(n - 1, int(round(0.95 * (n - 1))))] if n else 0.0
    total_s = sum(samples_s)
    calls_per_s = iters / total_s if total_s > 0 else 0.0
    return {
        "op": op_name, "iters": iters,
        "mean_ms": round(mean_ms, 3), "median_ms": round(median_ms, 3),
        "min_ms": round(min_ms, 3), "p95_ms": round(p95_ms, 3),
        "calls_per_s": round(calls_per_s, 2),
        "n_nonempty": n_nonempty,
        "nonempty_rate": round(n_nonempty / iters, 3) if iters else 0.0,
    }


def _measure_op(op_name: str, fn: Callable, targets: List[Tuple], *,
                warmup: int, iters: int) -> dict:
    n = len(targets)
    if n == 0:
        raise RuntimeError(f"{op_name}: no targets generated -- project too small?")
    for w in range(warmup):
        fn(*targets[w % n])
    samples_s: List[float] = []
    n_nonempty = 0
    for it in range(iters):
        args = targets[it % n]
        t0 = time.perf_counter()
        result = fn(*args)
        samples_s.append(time.perf_counter() - t0)
        if _is_nonempty(result):
            n_nonempty += 1
    return _summarize(op_name, samples_s, n_nonempty, iters)


def _run_measurements(service: TsLspService, files: Dict[str, str], args) -> Dict[str, dict]:
    mod_paths = [_mod_path(i) for i in range(1, args.n_files)]
    hub_text = files[_HUB_PATH]

    # Rotating position/file per op so the server can't serve one hot cache
    # entry forever.
    scale_call_targets = [(p, files[p].index("scale(")) for p in mod_paths]
    completion_targets = [(p, files[p].index("v.x") + 2) for p in mod_paths]
    symbol_targets = [(p,) for p in mod_paths]
    reference_targets = [
        (_HUB_PATH, hub_text.index("interface Vec") + len("interface ")),
        (_HUB_PATH, hub_text.index("function scale") + len("function ")),
    ]

    ops: Dict[str, dict] = {}
    ops["definition"] = _measure_op(
        "definition", lambda path, off: service.definition(path, off),
        scale_call_targets, warmup=args.warmup, iters=args.iters)
    ops["references"] = _measure_op(
        "references", lambda path, off: service.references(path, off, include_declaration=True),
        reference_targets, warmup=args.warmup, iters=args.iters)
    ops["completions"] = _measure_op(
        "completions", lambda path, off: service.completions(path, off),
        completion_targets, warmup=args.warmup, iters=args.iters)
    ops["quickinfo"] = _measure_op(
        "quickinfo", lambda path, off: service.quickinfo(path, off),
        scale_call_targets, warmup=args.warmup, iters=args.iters)
    ops["document_symbols"] = _measure_op(
        "document_symbols", lambda path: service.document_symbols(path),
        symbol_targets, warmup=args.warmup, iters=args.iters)

    # diagnostics: an EDIT -> RE-CHECK cycle, timed TOGETHER -- the shape
    # #226's consumer actually has (one edit per generated token). Timing a
    # cache hit and calling it a re-check would be the same BLIND failure in
    # a different coat.
    counter = [0]

    def _edit_and_check(path):
        counter[0] += 1
        service.update(path, files[path] + f"\n// bench edit {counter[0]}\n")
        return service.diagnostics(path)

    ops["diagnostics"] = _measure_op(
        "diagnostics", _edit_and_check, [(p,) for p in mod_paths],
        warmup=args.warmup, iters=args.iters)

    # diagnostics_cached: pure cache-hit path (unchanged document), reported
    # SEPARATELY -- explicitly NOT the SLA number.
    ops["diagnostics_cached"] = _measure_op(
        "diagnostics_cached", lambda path: service.diagnostics(path),
        [(p,) for p in mod_paths], warmup=args.warmup, iters=args.iters)

    return ops


def _verdict(diag_median_ms: float, completions_calls_per_s: float) -> str:
    if completions_calls_per_s < 20:
        return "KILL"
    if diag_median_ms < 50 and completions_calls_per_s >= 200:
        return "PASS"
    return "WARN"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _write_results(out_path: Path, result: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[bench_ts_lsp] wrote {out_path}")


def _print_report(summary: dict, ops: Dict[str, dict]) -> None:
    print("=" * 72)
    print("TS-LSP multi-op service bench (#220)")
    print("=" * 72)
    print(f"n_files={summary['n_files']}  cold_load_s={summary['cold_load_s']:.3f}  "
          f"server_sync_kind={summary['server_sync_kind']}  "
          f"diagnosticProvider={summary['diagnostic_provider']!r}")
    print("-" * 72)
    print(f"{'op':<20}{'median_ms':>12}{'p95_ms':>10}{'calls/s':>10}{'nonempty%':>11}")
    for name, s in ops.items():
        print(f"{name:<20}{s['median_ms']:>12.2f}{s['p95_ms']:>10.2f}"
              f"{s['calls_per_s']:>10.1f}{s['nonempty_rate'] * 100:>10.1f}%")
    print("-" * 72)
    print(f"n_restarts={summary['n_restarts']}  n_timeouts={summary['n_timeouts']}")
    print(f"VERDICT: {summary['verdict']}")
    if summary["verdict"] in ("WARN", "KILL"):
        print("  -> #226 decode-time completion-list masking rescopes to OFFLINE "
              "if this holds (throughput gate: completions >= 200 calls/s).")
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-files", type=int, default=5000, help="project size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20, help="untimed ops per kind")
    ap.add_argument("--iters", type=int, default=200, help="timed ops per kind")
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="results/ts_lsp_bench.json")
    ap.add_argument("--keep-scratch", action="store_true",
                    help="leave the generated project on disk for inspection")
    ap.add_argument("--max-server-memory", type=int, default=None,
                    help="initializationOptions.maxTsServerMemory (MB) -- raise this if a "
                         "large project silently disables semantic features (edge case: "
                         "tsserver project-size/memory limits)")
    args = ap.parse_args()
    if args.n_files < 2:
        ap.error("--n-files must be >= 2")
    if args.iters < 1:
        ap.error("--iters must be >= 1")
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    return args


def main() -> int:
    args = _parse_args()

    argv = resolve_ts_lsp()
    if argv is None:
        print("SKIP: no typescript-language-server toolchain resolvable "
              f"(need `npm i -D typescript-language-server` in {SET_DIR}).")
        return 0

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path

    print(f"[bench_ts_lsp] n_files={args.n_files} seed={args.seed} "
          f"warmup={args.warmup} iters={args.iters} timeout_s={args.timeout_s}")
    files = _generate_project(args.n_files, args.seed)

    init_options = {}
    if args.max_server_memory is not None:
        init_options["maxTsServerMemory"] = args.max_server_memory

    service = TsLspService(timeout_s=args.timeout_s, initialization_options=init_options)
    try:
        service.open_project(files)
        print(f"[bench_ts_lsp] cold load: {service.cold_load_s:.3f}s  "
              f"server_sync_kind={service.server_sync_kind}  "
              f"diagnosticProvider={service.server_capabilities.get('diagnosticProvider')!r}")

        probe = _sanity_probe(service, files)
        if not probe["ok"]:
            print("BLIND: sanity probe failed -- nothing was measured.")
            for name, ok in probe["checks"].items():
                print(f"  {name}: {'OK' if ok else 'FAILED'}")
            _write_results(out_path, {"summary": {"verdict": "BLIND", "sanity_probe": probe},
                                      "ops": {}})
            raise SystemExit(2)

        ops = _run_measurements(service, files, args)
        verdict = _verdict(ops["diagnostics"]["median_ms"], ops["completions"]["calls_per_s"])
        summary = {
            "verdict": verdict,
            "n_files": args.n_files, "seed": args.seed,
            "warmup": args.warmup, "iters": args.iters,
            "cold_load_s": service.cold_load_s,
            "server_sync_kind": service.server_sync_kind,
            "diagnostic_provider": service.server_capabilities.get("diagnosticProvider"),
            "sanity_probe": probe,
            "n_restarts": service.n_restarts, "n_timeouts": service.n_timeouts,
        }
        _write_results(out_path, {"summary": summary, "ops": ops})
        _print_report(summary, ops)
    finally:
        if args.keep_scratch:
            # Detach TemporaryDirectory's finalizer so it survives process exit.
            service._tmpdir_obj._finalizer.detach()
            service._teardown_process()
            print(f"[bench_ts_lsp] scratch dir kept at {service.scratch_dir}")
        else:
            service.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
