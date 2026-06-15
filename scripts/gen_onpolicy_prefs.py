"""On-policy DPO preference generation (#77).

Sample K responses per prompt from a checkpoint, score each, and emit
`{prompt, chosen, rejected}` pairs (best vs worst) as DPO JSONL. On-policy pairs are
inherently clean — the responses come from *our* model, not a commercial one — so they
need no external preference data, only a scorer.

  python scripts/gen_onpolicy_prefs.py --config config/poc.yaml \
      --init runs/sft/weights.safetensors --prompts prompts.txt --out data/dpo_onpolicy.jsonl

The default scorer is a placeholder (length + lexical diversity); swap in a real reward
model or verifier (#78) by editing `default_scorer` for production on-policy DPO. MLX-only
(uses the backend + serving recurrence), like scripts/sft.py / scripts/dpo.py.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.dpo_sources import pairs_from_scored


def default_scorer(text: str) -> float:
    """Placeholder reward: length (capped) + lexical diversity. NOT a real reward — swap
    in a reward model / verifier (#78) for production use."""
    toks = text.split()
    if not toks:
        return 0.0
    diversity = len(set(toks)) / len(toks)
    length = min(len(toks), 64) / 64.0
    return 0.5 * diversity + 0.5 * length


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--init", type=Path, required=True, help="checkpoint weights (SFT base)")
    ap.add_argument("--prompts", type=Path, required=True, help="one prompt per line")
    ap.add_argument("--out", type=Path, required=True, help="output DPO JSONL")
    ap.add_argument("--k", type=int, default=4, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.serve.generate import generate
    from src.serve.sessions import SessionStore
    from src.serve.sampling import sample
    from src.data.dpo_data import write_dpo_jsonl
    from src.data.tokenize import ByteTokenizer, load_olmo_tokenizer

    backend = get_backend()
    cfg = load_config(str(args.config))
    model = backend.model_cls(cfg)
    model.load(str(args.init))
    tok = ByteTokenizer() if args.byte_fallback else load_olmo_tokenizer(args.model_id)
    eos = getattr(tok, "eos_token_id", None)
    store = SessionStore(model)
    np_to = backend.to_numpy

    prompts = [ln.strip() for ln in args.prompts.read_text(encoding="utf-8").splitlines()
               if ln.strip()]
    rng = np.random.default_rng(args.seed)
    pairs = []
    for pi, prompt in enumerate(prompts):
        prompt_ids = list(tok.encode(prompt)) or [0]
        scored = []
        for k in range(args.k):
            sid = f"p{pi}-s{k}"
            store.create(sid)
            sampler = partial(sample, temperature=args.temperature, top_k=args.top_k,
                              rng=np.random.default_rng(int(rng.integers(1 << 30))))
            gen_ids = generate(store, sid, prompt_ids, sampler=sampler, to_numpy=np_to,
                               max_new_tokens=args.max_new_tokens, eos_id=eos)
            store.remove(sid)
            text = tok.decode(gen_ids)
            scored.append((text, default_scorer(text)))
        pref = pairs_from_scored(prompt, scored)
        if pref is not None:
            pairs.append(pref)

    n = write_dpo_jsonl(pairs, tok, args.out, max_seq_len=args.max_seq_len)
    print(f"on-policy: {len(prompts)} prompts -> {len(pairs)} pairs -> {n} DPO records")


if __name__ == "__main__":
    main()
