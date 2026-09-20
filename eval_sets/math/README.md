# Verifiable Math RLVR Prompt Sets (#103)

This directory holds verifiable math evaluation and training prompt sets for RLVR / GRPO.

## Structure
- `gsm8k.jsonl`: Multi-step grade school math word problems with numeric answers indicated by `#### <number>`.
- `math.jsonl`: High-school / competition level mathematics problems with exact answers indicated by `\boxed{...}`.
- `train.jsonl` / `val.jsonl`: Disjoint stratified splits generated via `scripts/build_math_rlvr_prompts.py`.
- `manifest.json`: Provenance manifest recording SHA-256 digests and row counts.

## Verifier Compatibility
Completions are verified using `src.train.verifiers.MathVerifier`:
1. Fast-path numeric extraction (`extract_final_number`) matching gold GSM8K answers.
2. Symbolic equivalence via SymPy (`sympy_symbolic_reward`) matching LaTeX expressions (`\frac`, radicals, polynomials, equations).
