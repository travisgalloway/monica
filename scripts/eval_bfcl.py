"""Tier-2 BFCL-style function-calling eval over the MLX backend (#102).

Wires config -> MLXMambaModel -> tokenizer -> src/eval/bfcl_adapter.evaluate_bfcl,
mirroring scripts/eval_olmes.py's structure (config -> model -> tokenizer -> adapter,
mlx imported locally so the seam stays clean). There is no BFCL dataset loader yet
(BFCL is eval-only and off the critical path per #102), so this runs the small
hand-authored `bfcl_adapter.BFCL_FIXTURE` (simple/parallel/abstention) by default;
point `examples` at a real loaded BFCL split later without touching the harness.

Generation stops at EOS (no forced "</tool_call>" stop string): truncating the
decoded text at that string would strip the closing tag itself, which
`bfcl_adapter.parse_tool_calls` needs to find a well-formed block. `--max-gen-toks`
bounds the completion instead; parsing tolerates trailing text after the block.

Judge by "runs end to end": with random-init or 100M-scale weights, accuracy will
sit near chance — that is expected and fine for the POC.

    .venv/bin/python scripts/eval_bfcl.py --config config/poc-qwen.yaml --limit 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc-qwen.yaml"))
    ap.add_argument("--weights", type=Path, default=None,
                    help="portable .safetensors checkpoint; RANDOM INIT if omitted")
    ap.add_argument("--tokenizer", choices=("qwen3", "qwen25", "olmo"), default="qwen25",
                    help="tokenizer matching the config's vocab (default: qwen25, the "
                         "from-scratch poc-qwen reserve; use qwen3 for the distillation "
                         "student config/student-1b.yaml, olmo for config/poc.yaml)")
    ap.add_argument("--byte-fallback", action="store_true",
                    help="offline ByteTokenizer (toy config only; not OLMo/Qwen-compatible)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of fixture examples (default: all)")
    ap.add_argument("--max-gen-toks", type=int, default=128,
                    help="generation budget per example")
    ap.add_argument("--output", type=Path, default=None, help="write results JSON here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # MLX-only imports kept local so the seam stays clean for portable hosts.
    try:
        import mlx.core as mx
    except ModuleNotFoundError as e:
        if e.name != "mlx":
            raise
        raise SystemExit(
            "mlx not found — run with the project venv on Apple Silicon:\n"
            "    .venv/bin/python scripts/eval_bfcl.py ...\n"
            "(mlx installs only on Apple Silicon via the '[mlx]' extra; a bare "
            "`python` likely points at a different interpreter.)"
        ) from e
    import numpy as np

    from src.data.chat_template import render
    from src.data.instruct_sft import _effective_vocab_size
    from src.data.tokenize import (
        ByteTokenizer,
        load_olmo_tokenizer,
        load_qwen25_tokenizer,
        load_qwen3_tokenizer,
    )
    from src.eval.bfcl_adapter import BFCL_FIXTURE, evaluate_bfcl
    from src.eval.olmes_adapter import generate_until_texts
    from src.model.blocks import load_config
    from src.model.mlx_backend import MLXMambaModel

    cfg = load_config(str(args.config))
    mx.random.seed(args.seed)
    model = MLXMambaModel(cfg)
    if args.weights:
        model.load(str(args.weights))
        print(f"loaded weights: {args.weights}")
    else:
        print("=" * 70)
        print("RANDOM INIT — no --weights given; scores will be chance level.")
        print("=" * 70)

    if args.byte_fallback:
        tok = ByteTokenizer()
    elif args.tokenizer == "qwen3":
        tok = load_qwen3_tokenizer()
    elif args.tokenizer == "qwen25":
        tok = load_qwen25_tokenizer()
    else:
        tok = load_olmo_tokenizer()
    # A tokenizer that can emit ids >= the model vocab would crash deep in the
    # embedding lookup; fail here with a clear message instead. Use the effective
    # vocab (len(tok)) rather than tok.vocab_size, since added special tokens
    # (e.g. Qwen3's <|im_start|>/<|im_end|>) can have ids past vocab_size.
    eff_vocab = _effective_vocab_size(tok)
    if eff_vocab is not None and eff_vocab > cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {eff_vocab} exceeds model vocab "
            f"{cfg.vocab_size} ({args.config}) — use a matching config, or "
            f"--byte-fallback only with toy-scale configs.")

    examples = list(BFCL_FIXTURE)
    if args.limit is not None:
        examples = examples[:args.limit]

    def generate_fn(prompt_or_messages) -> str:
        prompt = (prompt_or_messages if isinstance(prompt_or_messages, str)
                  else render(prompt_or_messages, add_generation_prompt=True))
        gen_kwargs = {"max_gen_toks": args.max_gen_toks}
        return generate_until_texts(
            model, tok, [(prompt, gen_kwargs)], max_length=cfg.seq_len,
            to_numpy=lambda a: np.array(a), seed=args.seed)[0]

    summary = evaluate_bfcl(examples, generate_fn)
    print(f"bfcl: {summary['n_examples']} examples, accuracy={summary['accuracy']:.3f}, "
          f"schema_valid_rate={summary['schema_valid_rate']:.3f}")
    print(f"per-category accuracy: {summary['per_category_accuracy']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"results -> {args.output}")


if __name__ == "__main__":
    main()
