"""Soak harness for `OpengrepOracle` -- validate the restart-on-stall + proactive
recycle mitigation against the measured ~10% full-timeout stall (see
`src/lsp/opengrep.py`'s module docstring).

Drives one long-lived oracle through N sequential rescans over a fixed rotating
set of **synthetic** TypeScript sources (rule fixtures / canary idioms authored
from the general bug taxonomy -- NEVER eval transcripts, so the contamination
discipline holds) and prints the reliability counters + stall rate.

Run it twice to demonstrate the drop:

    # pre-mitigation baseline (reproduce the stall):
    .venv/bin/python scripts/opengrep_soak.py --calls 160 --no-recycle --no-restart-on-stall
    # mitigation on (defaults):
    .venv/bin/python scripts/opengrep_soak.py --calls 160

Skips cleanly (exit 0) on a host without the `opengrep` binary. Commits no results
-- its printed numbers feed the design-doc write-up.
"""

from __future__ import annotations

import argparse
import sys
import time

# Allow running as a plain script (repo root on sys.path).
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from src.lsp.opengrep import (  # noqa: E402
    OpengrepOracle, resolve_opengrep, _DEFAULT_RECYCLE_EVERY,
)

# Synthetic rotating corpus: taxonomy-derived idioms (some fire a rule, some clean),
# NOT eval records. Repeatedly rescanning the same process is exactly the access
# pattern that surfaces the stall.
_SOURCES = [
    # loop-bound-off-by-one (fires)
    "function sumAll(a: number[]): number {\n"
    "  let t = 0;\n  for (let i = 0; i <= a.length; i++) { t += a[i]; }\n  return t;\n}\n",
    # clean counterpart
    "function sumAll(a: number[]): number {\n"
    "  let t = 0;\n  for (let i = 0; i < a.length; i++) { t += a[i]; }\n  return t;\n}\n",
    # index-at-length (fires)
    "function last(a: number[]): number {\n  return a[a.length];\n}\n",
    # parseint-no-radix (fires)
    "function toNum(s: string): number {\n  return parseInt(s);\n}\n",
    # clean
    "function toNum(s: string): number {\n  return parseInt(s, 10);\n}\n",
    # self-comparison (fires)
    "function bad(x: number): boolean {\n  return x == x;\n}\n",
    # typeof-array (fires)
    "function isArr(x: unknown): boolean {\n  return typeof x === \"array\";\n}\n",
    # clean, slightly larger body
    "function fib(n: number): number {\n"
    "  let a = 0, b = 1;\n  for (let i = 0; i < n; i++) { const c = a + b; a = b; b = c; }\n"
    "  return a;\n}\n",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calls", type=int, default=160, help="sequential rescans to run")
    p.add_argument("--timeout-s", type=float, default=10.0)
    p.add_argument("--recycle-every", type=int, default=_DEFAULT_RECYCLE_EVERY,
                   help="proactive recycle interval (0 disables)")
    p.add_argument("--no-recycle", action="store_true",
                   help="shorthand for --recycle-every 0")
    p.add_argument("--no-restart-on-stall", action="store_true",
                   help="disable reactive restart-and-retry (reproduce pre-fix behavior)")
    p.add_argument("--single-index", type=int, default=None,
                   help="pin one source (index into the corpus) and rescan it repeatedly -- "
                        "the 'sustained single-file rescan' pattern the stall was measured under")
    p.add_argument("--progress-every", type=int, default=20)
    args = p.parse_args()

    if resolve_opengrep() is None:
        print("opengrep not on PATH -- skipping soak (install per "
              "eval_sets/opengrep_rules/README.md)")
        return 0

    recycle_every = 0 if args.no_recycle else args.recycle_every
    restart_on_stall = not args.no_restart_on_stall

    print(f"soak: calls={args.calls} timeout_s={args.timeout_s} "
          f"recycle_every={recycle_every} restart_on_stall={restart_on_stall}")
    t0 = time.monotonic()
    oracle = OpengrepOracle(timeout_s=args.timeout_s, recycle_every=recycle_every,
                            restart_on_stall=restart_on_stall)
    try:
        for i in range(args.calls):
            idx = args.single_index if args.single_index is not None else i % len(_SOURCES)
            src = _SOURCES[idx % len(_SOURCES)]
            oracle.diagnostics(src)
            if args.progress_every and (i + 1) % args.progress_every == 0:
                print(f"  {i + 1}/{args.calls}  n_timeouts={oracle.n_timeouts} "
                      f"n_restarts={oracle.n_restarts} n_recycles={oracle.n_recycles} "
                      f"n_stall_recoveries={oracle.n_stall_recoveries}")
        wall = time.monotonic() - t0
        rate = oracle.n_timeouts / oracle.n_calls if oracle.n_calls else 0.0
        print("-" * 60)
        print(f"n_calls={oracle.n_calls} n_timeouts={oracle.n_timeouts} "
              f"stall_rate={rate:.1%}")
        print(f"n_restarts={oracle.n_restarts} n_recycles={oracle.n_recycles} "
              f"n_stall_recoveries={oracle.n_stall_recoveries}")
        print(f"wall_s={wall:.1f} (oracle.wall_s={oracle.wall_s:.1f})")
    finally:
        oracle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
