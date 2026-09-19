"""Go / No-Go Gate Evaluation CLI for Auxiliary Decision Critic Head.

Runs empirical verification across the 5 architectural gates:
  1. Parameter budget overhead (< 0.5% of M12-small active parameters).
  2. Forward evaluation latency (< 2.0ms vs 350ms LSP debounce floor).
  3. Calibration repair (ECE < 0.08 via temperature scaling).
  4. Discrimination gain (Brier score improvement >= 15% over base rate).
  5. Seam invariance (pure portable numpy execution without backend dependencies).

Usage:
  .venv/bin/python scripts/eval_critic_gate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model.critic import evaluate_critic_gate


def main():
    parser = argparse.ArgumentParser(description="Evaluate Go/No-Go gate for auxiliary critic heads in Monica")
    parser.add_argument("--d-model", type=int, default=768, help="Backbone model dimension (default: 768)")
    parser.add_argument("--active-params", type=int, default=120_000_000, help="Active parameter count (default: 120M)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of validation samples (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 72)
    print("MONICA ARCHITECTURE GATE: AUXILIARY DECISION CRITIC HEAD")
    print("=" * 72)

    res = evaluate_critic_gate(
        d_model=args.d_model,
        backbone_active_params=args.active_params,
        n_samples=args.samples,
        rng_seed=args.seed,
    )

    print("\nIndividual Gate Status:")
    for gate_name, passed in res.gate_details.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {gate_name}")

    print("\nSummary:")
    print(res.summary)
    print("=" * 72)
    print(f"FINAL VERDICT: {res.verdict}")
    print("=" * 72)

    sys.exit(0 if res.verdict == "GO" else 1)


if __name__ == "__main__":
    main()
