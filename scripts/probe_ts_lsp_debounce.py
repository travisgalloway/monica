#!/usr/bin/env python3
"""Decisive control for #278: is the `diagnostics` latency measured in #220 /
PR #277 (378 ms median against a 50 ms bar) a client-side DEBOUNCE inside the
pinned `typescript-language-server` 5.3.0, or whole-program type-checking
cost?

Deliberately bypasses `TsLspService` entirely -- raw JSON-RPC, no cache, no
seq-marker, no first-push-wins bookkeeping -- so neither the #278 seq-race fix
nor any other bookkeeping of ours can colour the number. Times exactly:

    didChange(version n+1)  ->  publishDiagnostics arrives

The debounce hypothesis says the wait is
`min(max(ceil(lineCount / 20), 300), 800) + 50` ms (`cli.mjs:17868` clamped by
`:17858`'s default plus `:20516`'s 50 ms `pDebounce`) and is roughly
INDEPENDENT of project size. Two predictions, both falsifiable:

  A. project-size sweep at fixed file length -> roughly FLAT ~350 ms
  B. file-LENGTH sweep                       -> 350 ms .. 850 ms, tracking
     the clamp's own formula

If instead the cost were whole-program semantic work, A would rise with
project size and B would be flat -- the two hypotheses predict opposite
things, so this separates them.

Each iteration alternates the edited file between a real type error and
clean, so the diagnostic SET genuinely changes every time -- that defeats
`cli.mjs:20524`'s "both old and new sets are empty -> suppress publish"
early-return, which would otherwise make some iterations produce no push at
all (indistinguishable from a hang).

Stdlib + above-the-seam repo imports only (no mlx/torch). Skips cleanly with
exit 0 if no `typescript-language-server` toolchain is resolvable.

Usage:
    .venv/bin/python scripts/probe_ts_lsp_debounce.py \
        --out results/ts_lsp_debounce_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

# Repo root on sys.path so `src.lsp.*` imports resolve when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lsp.jsonrpc import spawn  # noqa: E402
from src.lsp.tsc import DEFAULT_TSCONFIG_PATH, SET_DIR  # noqa: E402
from src.lsp.ts_lsp import resolve_ts_lsp  # noqa: E402

_CLEAN = "export const probe_value: number = {i};"
_ERROR = 'export const probe_value: number = "type-error-{i}";'

# Reference tables from the already-run control experiment (issue #278's
# writeup / the plan) -- printed alongside a fresh run so a re-run is
# immediately comparable without cross-referencing the issue.
_REFERENCE_A = [
    (10, 359.2, 9.2), (200, 360.5, 10.5), (2000, 368.1, 18.1), (5000, 381.1, 31.1),
]
_REFERENCE_B = [
    (12, 350, 361.1, 11.1), (6000, 350, 365.7, 15.7),
    (12000, 650, 668.3, 18.3), (20000, 850, 869.6, 19.6),
]


def _predicted_ms(lines: int) -> float:
    return min(max(math.ceil(lines / 20), 300), 800) + 50


def _pad(body: str, lines: int) -> str:
    """Pad to `lines` total with comment lines -- lineCount is the clamp's
    only input (`cli.mjs:17868`)."""
    filler = "\n".join(f"// pad {n}" for n in range(max(0, lines - 1)))
    return f"{filler}\n{body}\n" if filler else f"{body}\n"


def _run_point(n_files: int, edit_lines: int, iters: int, warmup: int,
               timeout_s: float) -> dict:
    """One (n_files, edit_lines) measurement point: `iters` timed
    `didChange` -> `publishDiagnostics` samples (plus `warmup` untimed ones),
    alternating error/clean so `cli.mjs:20524` can never suppress a push."""
    argv = resolve_ts_lsp()
    assert argv is not None  # caller already checked

    root = Path(tempfile.mkdtemp(dir=str(SET_DIR), prefix="lsp_debounce_"))
    try:
        (root / "tsconfig.json").write_text(
            DEFAULT_TSCONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        src = root / "src"
        src.mkdir()
        # Filler files so the PROGRAM is genuinely n_files big; each imports
        # its predecessor so tsserver cannot dismiss them as unreachable.
        for i in range(n_files - 1):
            prev = f'import {{ v{i - 1} }} from "./f{i - 1}";\n' if i else ""
            (src / f"f{i}.ts").write_text(
                f'{prev}export const v{i}: number = {i};\n', encoding="utf-8")
        target = src / "target.ts"
        target.write_text(_pad(_CLEAN.format(i=0), edit_lines), encoding="utf-8")
        uri = target.as_uri()

        pushes: List[tuple] = []
        got = threading.Event()

        def on_notification(msg: dict) -> None:
            if msg.get("method") != "textDocument/publishDiagnostics":
                return
            params = msg.get("params") or {}
            if params.get("uri") == uri:
                pushes.append((time.perf_counter(), len(params.get("diagnostics") or [])))
                got.set()

        proc, ep = spawn(argv + ["--stdio"], cwd=str(root), on_notification=on_notification)
        try:
            ep.request("initialize", {
                "processId": None,
                "rootUri": root.as_uri(),
                "capabilities": {"textDocument": {"publishDiagnostics": {}}},
                "initializationOptions": {},
            }, timeout=timeout_s)
            ep.notify("initialized", {})
            text = target.read_text(encoding="utf-8")
            ep.notify("textDocument/didOpen", {"textDocument": {
                "uri": uri, "languageId": "typescript", "version": 1, "text": text}})
            got.wait(min(30.0, timeout_s))  # let the program load

            samples: List[float] = []
            n_unmeasurable = 0
            for it in range(warmup + iters):
                body = (_ERROR if it % 2 == 0 else _CLEAN).format(i=it)
                text = _pad(body, edit_lines)
                got.clear()
                before = len(pushes)
                t0 = time.perf_counter()
                ep.notify("textDocument/didChange", {
                    "textDocument": {"uri": uri, "version": it + 2},
                    "contentChanges": [{"text": text}],
                })
                if not got.wait(timeout_s):
                    n_unmeasurable += 1
                    continue
                dt = (pushes[before][0] - t0) * 1000.0
                if it >= warmup:
                    samples.append(dt)

            pushes_with_errors = sum(1 for _, c in pushes if c)
            pushes_total = len(pushes)
            predicted = _predicted_ms(edit_lines)
            blind = len(samples) == 0 or pushes_with_errors == 0

            point = {
                "n_files": n_files, "edit_lines": edit_lines, "n": len(samples),
                "median_ms": round(statistics.median(samples), 1) if samples else None,
                "min_ms": round(min(samples), 1) if samples else None,
                "p95_ms": (round(sorted(samples)[int(len(samples) * 0.95)], 1)
                           if samples else None),
                "predicted_ms": predicted,
                "residual_ms": (round(statistics.median(samples) - predicted, 1)
                                if samples else None),
                "pushes_with_errors": pushes_with_errors, "pushes_total": pushes_total,
                "n_unmeasurable": n_unmeasurable,
                "blind": blind,
            }
            return point
        finally:
            ep.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def _verdict(sweep_a: List[dict], sweep_b: List[dict]) -> Dict[str, object]:
    """DEBOUNCE_CONFIRMED / PROJECT_SIZE_DOMINATES / INCONCLUSIVE, computed
    only from sweeps actually run. A `blind` point anywhere forces
    INCONCLUSIVE -- silence must never read as confirmation."""
    any_blind = any(p["blind"] for p in sweep_a + sweep_b)

    a_spread_ms = None
    a_monotonic_rising = None
    if len(sweep_a) >= 2 and all(p["median_ms"] is not None for p in sweep_a):
        by_size = sorted(sweep_a, key=lambda p: p["n_files"])
        a_spread_ms = round(by_size[-1]["median_ms"] - by_size[0]["median_ms"], 1)
        medians = [p["median_ms"] for p in by_size]
        a_monotonic_rising = all(b >= a for a, b in zip(medians, medians[1:]))

    b_max_abs_residual_ms = None
    b_ratio = None
    if len(sweep_b) >= 2 and all(p["median_ms"] is not None for p in sweep_b):
        residuals = [abs(p["residual_ms"]) for p in sweep_b]
        b_max_abs_residual_ms = round(max(residuals), 1)
        medians = [p["median_ms"] for p in sweep_b]
        b_ratio = round(max(medians) / min(medians), 2) if min(medians) else None

    debounce_confirmed = (
        not any_blind and a_spread_ms is not None and b_max_abs_residual_ms is not None
        and b_ratio is not None
        and a_spread_ms <= 100 and b_max_abs_residual_ms <= 60 and b_ratio >= 2.0
    )
    project_size_dominates = (
        not debounce_confirmed and not any_blind
        and a_spread_ms is not None and a_spread_ms > 100 and bool(a_monotonic_rising)
    )

    if any_blind:
        verdict = "INCONCLUSIVE"
    elif debounce_confirmed:
        verdict = "DEBOUNCE_CONFIRMED"
    elif project_size_dominates:
        verdict = "PROJECT_SIZE_DOMINATES"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "a_spread_ms": a_spread_ms,
        "a_monotonic_rising": a_monotonic_rising,
        "b_max_abs_residual_ms": b_max_abs_residual_ms,
        "b_ratio": b_ratio,
        "any_blind": any_blind,
    }


def _print_reference_tables() -> None:
    print("-" * 72)
    print("Reference (issue #278's already-run control experiment):")
    print("  A. project-size sweep (edit-file fixed at 12 lines):")
    print(f"     {'n_files':>8} {'median_ms':>10} {'minus_350':>10}")
    for n, median, minus in _REFERENCE_A:
        print(f"     {n:>8} {median:>10.1f} {minus:>10.1f}")
    print("  B. file-length sweep (project fixed at 200 files):")
    print(f"     {'lines':>8} {'predicted':>10} {'measured':>10} {'residual':>10}")
    for lines, pred, measured, resid in _REFERENCE_B:
        print(f"     {lines:>8} {pred:>10} {measured:>10.1f} {resid:>10.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", choices=["a", "b", "both"], default="both")
    ap.add_argument("--a-sizes", type=str, default="10,200,2000,5000",
                    help="comma-separated n_files values for sweep A")
    ap.add_argument("--b-lines", type=str, default="12,6000,12000,20000",
                    help="comma-separated edit-file line counts for sweep B")
    ap.add_argument("--b-n-files", type=int, default=200,
                    help="project size held fixed for sweep B")
    ap.add_argument("--a-lines", type=int, default=12,
                    help="edit-file line count held fixed for sweep A")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations for sweep A")
    ap.add_argument("--b-iters", type=int, default=12, help="timed iterations for sweep B")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timeout-s", type=float, default=60.0,
                    help="per-notification wait cap (s)")
    ap.add_argument("--out", type=str, default="results/ts_lsp_debounce_probe.json")
    args = ap.parse_args()

    argv = resolve_ts_lsp()
    if argv is None:
        print("SKIP: no typescript-language-server toolchain resolvable "
              f"(need `npm i -D typescript-language-server` in {SET_DIR}).")
        return 0

    a_sizes = [int(x) for x in args.a_sizes.split(",") if x]
    b_lines = [int(x) for x in args.b_lines.split(",") if x]

    print("clamp predicts: delay = min(max(ceil(lines/20), 300), 800) + 50ms debounce")
    _print_reference_tables()

    sweep_a: List[dict] = []
    if args.sweep in ("a", "both"):
        print("-" * 72)
        print("=== A. project-size sweep (file length fixed at "
              f"{args.a_lines} lines) ===")
        print("    debounce predicts FLAT ~350ms; whole-program work predicts RISING")
        for n in a_sizes:
            r = _run_point(n, args.a_lines, args.iters, args.warmup, args.timeout_s)
            sweep_a.append(r)
            _print_point_a(r)

    sweep_b: List[dict] = []
    if args.sweep in ("b", "both"):
        print("-" * 72)
        print(f"=== B. file-LENGTH sweep (project fixed at {args.b_n_files} files) ===")
        print("    predicted: min(max(ceil(lines/20),300),800)+50")
        for lines in b_lines:
            r = _run_point(args.b_n_files, lines, args.b_iters, args.warmup, args.timeout_s)
            sweep_b.append(r)
            _print_point_b(r)

    verdict = _verdict(sweep_a, sweep_b)

    summary = {
        "verdict": verdict["verdict"],
        "a_spread_ms": verdict["a_spread_ms"],
        "a_monotonic_rising": verdict["a_monotonic_rising"],
        "b_max_abs_residual_ms": verdict["b_max_abs_residual_ms"],
        "b_ratio": verdict["b_ratio"],
        "any_blind": verdict["any_blind"],
        "a_lines": args.a_lines, "b_n_files": args.b_n_files,
        "iters": args.iters, "b_iters": args.b_iters, "warmup": args.warmup,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"summary": summary, "sweep_a": sweep_a, "sweep_b": sweep_b}, indent=2),
        encoding="utf-8")

    print("=" * 72)
    print(f"VERDICT: {verdict['verdict']}")
    print(f"  a_spread_ms={verdict['a_spread_ms']}  "
          f"a_monotonic_rising={verdict['a_monotonic_rising']}")
    print(f"  b_max_abs_residual_ms={verdict['b_max_abs_residual_ms']}  "
          f"b_ratio={verdict['b_ratio']}")
    print(f"  any_blind={verdict['any_blind']}")
    print(f"wrote {out_path}")
    print("=" * 72)
    return 0


def _print_point_a(r: dict) -> None:
    if r["median_ms"] is None:
        print(f"  n_files={r['n_files']:>5}  !! nothing measured "
              f"(n_unmeasurable={r['n_unmeasurable']})")
        return
    print(f"  n_files={r['n_files']:>5}  median={r['median_ms']:>7.1f}ms  "
          f"min={r['min_ms']:>7.1f}  p95={r['p95_ms']:>7.1f}  "
          f"pushes_with_errors={r['pushes_with_errors']}/{r['pushes_total']}  "
          f"n_unmeasurable={r['n_unmeasurable']}"
          + ("  BLIND" if r["blind"] else ""))


def _print_point_b(r: dict) -> None:
    if r["median_ms"] is None:
        print(f"  lines={r['edit_lines']:>6}  !! nothing measured "
              f"(n_unmeasurable={r['n_unmeasurable']})")
        return
    print(f"  lines={r['edit_lines']:>6}  median={r['median_ms']:>7.1f}ms  "
          f"predicted={r['predicted_ms']:>4}ms  delta={r['residual_ms']:>+7.1f}"
          + ("  BLIND" if r["blind"] else ""))


if __name__ == "__main__":
    raise SystemExit(main())
