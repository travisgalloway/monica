"""Tier-2 OLMES / lm-eval benchmark run over the MLX backend (issue #14).

Wires config -> MLXMambaModel -> tokenizer -> the lm-eval adapter
(src/eval/olmes_adapter.py) -> lm_eval.simple_evaluate. Judge by "runs end to
end": with random-init or 100M-scale weights the accuracies will sit near
chance — that is expected and fine for the POC.

Needs the eval extra (pip install -e ".[eval]") and network for the HF
datasets + OLMo tokenizer. If `datasets` refuses
piqa's script-based loader, set HF_DATASETS_TRUST_REMOTE_CODE=1 or drop piqa.

    .venv/bin/python scripts/eval_olmes.py --config config/poc.yaml --tasks piqa --limit 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--weights", type=Path, default=None,
                    help="portable .safetensors checkpoint; RANDOM INIT if omitted")
    ap.add_argument("--tasks", default="hellaswag,arc_easy,arc_challenge,piqa")
    ap.add_argument("--limit", type=int, default=None,
                    help="examples per task (small values for smoke runs)")
    ap.add_argument("--output", type=Path, default=None, help="write results JSON here")
    ap.add_argument("--byte-fallback", action="store_true",
                    help="offline ByteTokenizer (toy config only; not OLMo-compatible)")
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
            "    .venv/bin/python scripts/eval_olmes.py ...\n"
            "(mlx installs only on Apple Silicon via the '[mlx]' extra; a bare "
            "`python` likely points at a different interpreter.)"
        ) from e
    try:
        import lm_eval
    except ModuleNotFoundError as e:
        raise SystemExit(
            "lm-eval not found — install the eval extra:\n"
            "    .venv/bin/pip install -e \".[eval]\""
        ) from e
    from src.data.tokenize import ByteTokenizer, load_olmo_tokenizer
    from src.eval.olmes_adapter import make_lm_eval_adapter
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

    tok = ByteTokenizer() if args.byte_fallback else load_olmo_tokenizer()
    # A tokenizer that can emit ids >= the model vocab would crash deep in the
    # embedding lookup; fail here with a clear message instead.
    if tok.vocab_size > cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tok.vocab_size} exceeds model vocab "
            f"{cfg.vocab_size} ({args.config}) — use a matching config, or "
            f"--byte-fallback only with toy-scale configs.")

    lm = make_lm_eval_adapter(model, tok, to_numpy=lambda a: np.array(a))
    results = lm_eval.simple_evaluate(
        model=lm, tasks=args.tasks.split(","), limit=args.limit)

    print(lm_eval.utils.make_table(results))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results["results"], f, indent=2, default=str)
        print(f"results -> {args.output}")


if __name__ == "__main__":
    main()
