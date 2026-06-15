"""RLVR / GRPO with verifiable rewards (#78) — math first.

The cleanest post-training stage: sample K completions per problem from a checkpoint,
reward each with a **verifier** (math exact-match by default — no sandbox, the cheapest
clean reward loop), standardize the rewards within the group to advantages, and take a GRPO
step. The model generates the solutions; the verifier judges — so only problems + answers
are needed, no reference solutions (docs/design/08-corpus-pipeline.md lines 120-123).

  python scripts/rlvr.py --config config/poc.yaml --init runs/sft/weights.safetensors \
      --problems math.jsonl --steps 200

`--problems` is JSONL with `{"prompt": "...", "answer": "..."}` per line. `--reward math`
(default) uses the final-number exact-match; `--reward exact` uses normalized string match.
Code rewards need a sandbox (CodeVerifier is opt-in, off in CI) — out of scope for this
driver. MLX-only (backend + serving recurrence), like scripts/sft.py / scripts/dpo.py.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.train.grpo import group_advantages, reward_stats
from src.train.verifiers import exact_match_reward, math_reward


def collate_rollouts(rollouts, advantages, *, pad_id: int = 0):
    """Pad a group of (prompt_ids, gen_ids) rollouts into a GRPO micro-batch
    `(inputs, targets, mask, advantages)`; mask = 1 on the generated (completion) tokens
    so the GRPO loss only credits what the model produced."""
    fulls = [list(p) + list(g) for p, g in rollouts]
    glens = [len(g) for _, g in rollouts]
    L = max(len(f) for f in fulls)
    B = len(fulls)
    full = np.full((B, L), pad_id, dtype=np.int64)
    gen_mask = np.zeros((B, L), dtype=np.float32)
    for i, (f, gl) in enumerate(zip(fulls, glens)):
        full[i, :len(f)] = f
        gen_mask[i, len(f) - gl:len(f)] = 1.0          # the trailing gl tokens are generated
    inputs, targets = full[:, :-1], full[:, 1:]
    mask = gen_mask[:, 1:]                              # target j is a gen token?
    return inputs, targets, mask, np.asarray(advantages, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("config/poc.yaml"))
    ap.add_argument("--init", type=Path, required=True, help="checkpoint weights (SFT base)")
    ap.add_argument("--problems", type=Path, required=True, help="JSONL {prompt, answer}")
    ap.add_argument("--reward", choices=("math", "exact"), default="math")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--group-size", type=int, default=8, help="K completions per problem")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--byte-fallback", action="store_true", help="offline testing only")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.serve.generate import generate
    from src.serve.sessions import SessionStore
    from src.serve.sampling import sample
    from src.data.tokenize import ByteTokenizer, load_olmo_tokenizer

    backend = get_backend()
    cfg = load_config(str(args.config))
    model = backend.model_cls(cfg)
    model.load(str(args.init))
    tok = ByteTokenizer() if args.byte_fallback else load_olmo_tokenizer(args.model_id)
    eos = getattr(tok, "eos_token_id", None)
    store = SessionStore(model)
    np_to = backend.to_numpy
    opt = backend.make_optimizer(model, args.lr)
    grpo_step = backend.make_grpo_train_step(model, opt)
    reward_fn = math_reward if args.reward == "math" else exact_match_reward

    problems = [json.loads(ln) for ln in args.problems.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    if not problems:
        raise SystemExit(f"no problems in {args.problems}")
    rng = np.random.default_rng(args.seed)

    for step in range(args.steps):
        prob = problems[step % len(problems)]
        prompt_ids = list(tok.encode(prob["prompt"])) or [eos or 0]
        rollouts, rewards = [], []
        for k in range(args.group_size):
            sid = f"s{step}-{k}"
            store.create(sid)
            sampler = partial(sample, temperature=args.temperature, top_k=args.top_k,
                              rng=np.random.default_rng(int(rng.integers(1 << 30))))
            try:
                gen = generate(store, sid, prompt_ids, sampler=sampler, to_numpy=np_to,
                               max_new_tokens=args.max_new_tokens, eos_id=eos)
            finally:
                store.remove(sid)   # never leak the session if generate/sampling raises
            rollouts.append((prompt_ids, gen or [eos or 0]))
            rewards.append(reward_fn(tok.decode(gen), str(prob.get("answer", ""))))

        adv = group_advantages([rewards])[0]            # (K,)
        out = grpo_step(model, [collate_rollouts(rollouts, adv)], args.lr)
        if step % args.log_every == 0:
            stats = reward_stats(rewards)
            print(f"step {step:4d}  loss {out['loss']:.4f}  "
                  f"mean_reward {stats['mean_reward']:.3f}  solved {stats['frac_solved']:.3f}")

    save_dir = args.init.parent
    model.save(str(save_dir / "rlvr_weights.safetensors"))
    print(f"done — wrote {save_dir / 'rlvr_weights.safetensors'}")


if __name__ == "__main__":
    main()
