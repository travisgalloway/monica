#!/usr/bin/env python3
"""Probe the TS-LSP oracle's `publishDiagnostics` timing (#211).

`TsLspOracle.diagnostics()` arms a single `threading.Event` and returns on the
**first** `textDocument/publishDiagnostics` push for a candidate URI, with no
settle/quiescence window (unlike `OpengrepOracle._rescan`'s `_SETTLE_S`). If
`typescript-language-server` publishes twice for one document -- a fast syntactic
pass then a slower semantic pass -- the oracle would capture the early
syntactic-only push and MISS the semantic `TS2xxx` codes, biasing measured LSP
lift toward "finds nothing" and making the set timing-dependent.

This script does NOT change production code. It re-opens the same candidates the
oracle sees, but registers a list-append notification callback (not the one-shot
Event) that records EVERY push for the URI -- monotonic timestamp + codes -- so we
can measure: how many pushes arrive, their inter-arrival gaps, and whether a
semantic (`TS2xxx`) code ever appears ONLY in a later push. It also captures the
server's advertised `capabilities.diagnosticProvider` (present => LSP 3.17 pull
diagnostics available), which decides the fix path if a race is confirmed.

Stdlib + above-the-seam repo imports only (no mlx/torch). Skips cleanly with exit
0 if no `typescript-language-server` toolchain is resolvable.

Usage:
    .venv/bin/python scripts/probe_ts_lsp_multipush.py \
        --n 96 --out results/ts_lsp_multipush_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repo root on sys.path so `src.lsp.*` imports resolve when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lsp.jsonrpc import spawn  # noqa: E402
from src.lsp.tsc import DEFAULT_TSCONFIG_PATH, SET_DIR  # noqa: E402
from src.lsp.ts_lsp import resolve_ts_lsp  # noqa: E402

EVAL_JSONL = SET_DIR / "eval.jsonl"

# A crafted candidate carrying BOTH a syntactic defect (TS1xxx) and a distinct
# semantic type error (TS2xxx), so a syntactic-only early push is unambiguously
# distinguishable from the complete set. `const x: number = "s";` => TS2322
# (semantic); `const y = ;` => TS1109 "Expression expected" (syntactic).
CRAFTED_CANDIDATE = 'const x: number = "s";\nconst y = ;\n'


def _codes(diags: List[dict]) -> List[str]:
    """Format LSP integer codes the same way `_map_diagnostic` does (Trap A)."""
    return [f"TS{d['code']}" for d in diags if "code" in d]


def _is_semantic(code: str) -> bool:
    """A committed semantic code (the race's victim): anything but the TS1xxx
    syntax-incompleteness family."""
    return code.startswith("TS") and not code.startswith("TS1")


class _PushRecorder:
    """Records every `publishDiagnostics` push per URI, with relative timestamps."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0: Dict[str, float] = {}
        self.pushes: Dict[str, List[dict]] = {}   # uri -> [{t, codes, n, severities}]
        self._active: set = set()

    def arm(self, uri: str, t0: float) -> None:
        with self._lock:
            self._t0[uri] = t0
            self.pushes[uri] = []
            self._active.add(uri)

    def disarm(self, uri: str) -> None:
        with self._lock:
            self._active.discard(uri)

    def count(self, uri: str) -> int:
        with self._lock:
            return len(self.pushes.get(uri, []))

    def on_notification(self, msg: dict) -> None:
        if msg.get("method") != "textDocument/publishDiagnostics":
            return
        params = msg.get("params") or {}
        uri = params.get("uri")
        if uri is None:
            return
        with self._lock:
            if uri not in self._active:
                return
            diags = params.get("diagnostics", [])
            self.pushes[uri].append({
                "t": time.monotonic() - self._t0[uri],
                "codes": _codes(diags),
                "severities": [d.get("severity", 1) for d in diags],
                "n": len(diags),
            })


def _collect(recorder: _PushRecorder, uri: str, *, settle: float,
             max_wait: float) -> None:
    """Block until pushes for `uri` go quiet for `settle` seconds (after at least
    one push) or `max_wait` total elapses -- deliberately capturing LATE pushes the
    one-shot oracle would miss."""
    start = time.monotonic()
    last_n = 0
    last_change = start
    while True:
        now = time.monotonic()
        n = recorder.count(uri)
        if n != last_n:
            last_n = n
            last_change = now
        if n > 0 and (now - last_change) >= settle:
            return
        if (now - start) >= max_wait:
            return
        time.sleep(0.02)


def _analyze_pushes(pushes: List[dict]) -> dict:
    """Per-candidate verdict from its ordered push list."""
    n_pushes = len(pushes)
    first_codes = set(pushes[0]["codes"]) if pushes else set()
    all_codes: set = set()
    for p in pushes:
        all_codes |= set(p["codes"])
    later_only = sorted(all_codes - first_codes)
    semantic_later_only = sorted(c for c in later_only if _is_semantic(c))
    # Inter-arrival gaps between consecutive pushes.
    gaps = [round(pushes[i]["t"] - pushes[i - 1]["t"], 4) for i in range(1, n_pushes)]
    return {
        "n_pushes": n_pushes,
        "push_ts": [round(p["t"], 4) for p in pushes],
        "gaps": gaps,
        "first_push_codes": sorted(first_codes),
        "final_codes": sorted(all_codes),
        "later_only_codes": later_only,
        "semantic_appeared_only_later": semantic_later_only,
        "raced": bool(semantic_later_only),   # a semantic code the oracle would miss
        "multi_push": n_pushes > 1,
        "got_any": n_pushes > 0,
    }


def _open_and_collect(endpoint, recorder: _PushRecorder, scratch_dir: Path,
                      cand_id: str, idx: int, source: str, *,
                      settle: float, max_wait: float) -> dict:
    path = scratch_dir / f"cand_{idx}.ts"
    uri = path.as_uri()
    t0 = time.monotonic()
    recorder.arm(uri, t0)
    path.write_text(source, encoding="utf-8")
    endpoint.notify("textDocument/didOpen", {
        "textDocument": {"uri": uri, "languageId": "typescript", "version": 1, "text": source},
    })
    _collect(recorder, uri, settle=settle, max_wait=max_wait)
    endpoint.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
    recorder.disarm(uri)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    verdict = _analyze_pushes(recorder.pushes.get(uri, []))
    verdict["id"] = cand_id
    return verdict


def _load_candidates(n: Optional[int]) -> List[Tuple[str, str]]:
    """(id, source) pairs from the #194 set: prompt + error_completion (what the
    oracle checks in production)."""
    out: List[Tuple[str, str]] = []
    if EVAL_JSONL.exists():
        for line in EVAL_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append((str(rec.get("id", f"rec{len(out)}")),
                        rec["prompt"] + rec["error_completion"]))
    if n is not None:
        out = out[:n]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=None, help="max #194 candidates (default all)")
    ap.add_argument("--out", type=str, default="results/ts_lsp_multipush_probe.json")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="quiescence window (s) after the last push before moving on")
    ap.add_argument("--max-wait", type=float, default=8.0,
                    help="per-candidate wall cap (s)")
    ap.add_argument("--timeout", type=float, default=10.0, help="server init timeout (s)")
    args = ap.parse_args()

    argv = resolve_ts_lsp()
    if argv is None:
        print("SKIP: no typescript-language-server toolchain resolvable "
              f"(need `npm i -D typescript-language-server` in {SET_DIR}).")
        return 0

    recorder = _PushRecorder()
    tmp = tempfile.TemporaryDirectory(dir=str(SET_DIR), prefix="lsp_probe_")
    scratch_dir = Path(tmp.name)
    (scratch_dir / "tsconfig.json").write_text(
        DEFAULT_TSCONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    proc, endpoint = spawn(argv + ["--stdio"], cwd=str(scratch_dir),
                           on_notification=recorder.on_notification)
    # Advertise BOTH push and pull client capabilities so the server reveals
    # `diagnosticProvider` in its initialize result iff it supports pull.
    init_result = endpoint.request("initialize", {
        "processId": os.getpid(),
        "rootUri": scratch_dir.as_uri(),
        "capabilities": {"textDocument": {
            "publishDiagnostics": {},
            "diagnostic": {"dynamicRegistration": True, "relatedDocumentSupport": True},
        }},
        "initializationOptions": {},
    }, timeout=args.timeout)
    endpoint.notify("initialized", {})

    server_caps = (init_result or {}).get("capabilities", {})
    diagnostic_provider = server_caps.get("diagnosticProvider")
    server_info = (init_result or {}).get("serverInfo", {})

    results: List[dict] = []
    try:
        # The crafted both-defects candidate first (idx 0), clearly labelled.
        results.append(_open_and_collect(
            endpoint, recorder, scratch_dir, "crafted:syntax+semantic", 0,
            CRAFTED_CANDIDATE, settle=args.settle, max_wait=args.max_wait))

        for i, (cand_id, source) in enumerate(_load_candidates(args.n), start=1):
            v = _open_and_collect(endpoint, recorder, scratch_dir, cand_id, i,
                                  source, settle=args.settle, max_wait=args.max_wait)
            results.append(v)
    finally:
        endpoint.close()
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        tmp.cleanup()

    # Aggregate.
    n_total = len(results)
    n_multi = sum(1 for r in results if r["multi_push"])
    n_raced = sum(1 for r in results if r["raced"])
    n_no_push = sum(1 for r in results if not r["got_any"])
    max_pushes = max((r["n_pushes"] for r in results), default=0)
    push_hist: Dict[int, int] = {}
    for r in results:
        push_hist[r["n_pushes"]] = push_hist.get(r["n_pushes"], 0) + 1

    summary = {
        "server_info": server_info,
        "diagnostic_provider": diagnostic_provider,
        "pull_diagnostics_supported": diagnostic_provider is not None,
        "settle_s": args.settle,
        "max_wait_s": args.max_wait,
        "n_candidates": n_total,
        "n_multi_push": n_multi,
        "n_semantic_raced": n_raced,
        "n_no_push": n_no_push,
        "max_pushes_single_candidate": max_pushes,
        "push_count_histogram": {str(k): push_hist[k] for k in sorted(push_hist)},
        "verdict": (
            "MULTI_PUSH_RACE_CONFIRMED" if n_raced > 0 else
            "MULTI_PUSH_NO_SEMANTIC_LOSS" if n_multi > 0 else
            "SINGLE_COALESCED_PUSH"),
    }

    out_path = _REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"summary": summary, "candidates": results}, indent=2), encoding="utf-8")

    # Human-readable report.
    print("=" * 72)
    print("TS-LSP multi-push probe (#211)")
    print("=" * 72)
    print(f"server:              {server_info.get('name','?')} {server_info.get('version','?')}")
    print(f"diagnosticProvider:  {diagnostic_provider!r}")
    print(f"  -> pull diagnostics (textDocument/diagnostic) supported: "
          f"{summary['pull_diagnostics_supported']}")
    print(f"settle / max_wait:   {args.settle}s / {args.max_wait}s")
    print(f"candidates probed:   {n_total} (1 crafted + {n_total-1} from #194)")
    print(f"push-count histogram (n_pushes: count): {summary['push_count_histogram']}")
    print(f"max pushes for one candidate: {max_pushes}")
    print(f"multi-push candidates:        {n_multi}")
    print(f"SEMANTIC codes seen ONLY in a later push (the race): {n_raced}")
    print(f"no-push (would time out):     {n_no_push}")
    print(f"VERDICT: {summary['verdict']}")
    crafted = results[0]
    print("-" * 72)
    print(f"crafted candidate: n_pushes={crafted['n_pushes']} "
          f"first={crafted['first_push_codes']} final={crafted['final_codes']} "
          f"semantic_later_only={crafted['semantic_appeared_only_later']}")
    if n_raced > 0:
        print("-" * 72)
        print("Raced candidates (semantic code the one-shot oracle would MISS):")
        for r in results:
            if r["raced"]:
                print(f"  {r['id']}: pushes={r['n_pushes']} gaps={r['gaps']} "
                      f"missed={r['semantic_appeared_only_later']}")
    print("=" * 72)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
