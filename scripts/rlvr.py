"""RLVR / GRPO with verifiable rewards (#78) — math first.

The cleanest post-training stage: sample K completions per problem from a checkpoint,
reward each with a **verifier** (math exact-match by default — no sandbox, the cheapest
clean reward loop), standardize the rewards within the group to advantages, and take a GRPO
step. The model generates the solutions; the verifier judges — so only problems + answers
are needed, no reference solutions (docs/design/08-corpus-pipeline.md lines 120-123).

  python scripts/rlvr.py --config config/poc.yaml --init runs/sft/weights.safetensors \
      --problems math.jsonl --steps 200 --out runs/rlvr --ckpt-every 50

`--problems` is JSONL with `{"prompt": "...", "answer": "..."}` per line. `--reward math`
(default) uses the final-number exact-match; `--reward exact` uses normalized string match.
`--reward lsp` (#230) is the diagnostic-cleanliness verifier reward for #198's SSI RLVR
arm — `src.train.verifiers.LspVerifier` over `src.lsp.oracle.CompositeOracle`. It is
**static analysis only** (never executes candidate code) and is therefore safe to enable
by default, unlike `CodeVerifier` below: real TS/Rust/SQL *execution* grading needs a
sandbox (CodeVerifier is opt-in, off in CI) — out of scope for this driver. MLX-only
(backend + serving recurrence), like scripts/sft.py / scripts/dpo.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.train.grpo import group_advantages, reward_stats
from src.train.verifiers import LspVerifier, exact_match_reward, math_reward


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
    ap.add_argument("--reward", choices=("math", "exact", "lsp"), default="math")
    ap.add_argument("--oracle", choices=("ts", "opengrep", "both"), default="ts",
                    help="--reward lsp only: diagnostic oracle (persistent TS-LSP by "
                         "default; #278's ~350ms didChange debounce makes 'both' costly "
                         "at K-completions-per-step scale, so 'ts' is the default)")
    ap.add_argument("--lsp-timeout-s", type=float, default=10.0,
                    help="--reward lsp only: per-call oracle timeout")
    ap.add_argument("--lsp-hatches", choices=("superset", "directives", "none"), default="superset",
                    help="--reward lsp only: escape-hatch set the hack floor enforces — "
                         "'superset' (default, the #225 M5 gate), 'directives' (the "
                         "narrower ts_ignore/ts_expect_error/ts_nocheck/as_any/"
                         "as_unknown_as ablation), or 'none' (control arm, hack floor off)")
    ap.add_argument("--lsp-ignore-module-resolution", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="--reward lsp only: drop module-resolution diagnostics (TS2307 "
                         "etc.) before scoring — an unresolved import only weakens "
                         "checking, it doesn't invent errors (default on)")
    ap.add_argument("--out", type=Path, default=Path("runs/rlvr"),
                    help="output dir for weights.safetensors (default runs/rlvr); kept "
                         "separate from --init so the base checkpoint is never clobbered")
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save intermediate weights every N steps (0 = only at the end), "
                         "so a long run that crashes does not lose all progress")
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

    # Fail fast BEFORE any model load — an --reward lsp run with no toolchain
    # should not pay the checkpoint-load cost only to die on step 0's first reward.
    if args.reward == "lsp":
        from src.lsp.oracle import resolve_oracle
        if not resolve_oracle(args.oracle):
            raise SystemExit(
                f"no toolchain for --oracle {args.oracle} — for 'ts'/'both', run `npm install` "
                "in eval_sets/ts_error_injection and `npm i -D typescript-language-server`; "
                "for 'opengrep', install opengrep and put it on PATH "
                "(see eval_sets/opengrep_rules/README.md).")

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.serve.generate import generate
    from src.serve.sessions import SessionStore
    from src.serve.sampling import sample
    from src.data.tokenize import ByteTokenizer, load_olmo_tokenizer
    from src.train.moe_balance import attach_balancer, balancer_for_config
    from src.train.checkpoint import check_weight_keys, load_weights_dict

    backend = get_backend()
    cfg = load_config(str(args.config))
    model = backend.model_cls(cfg)
    # #214: MLX's load is silently lenient (missing key -> stays at random init, wrong
    # shape -> silently rebound), so check explicitly before loading. Load the
    # safetensors once and reuse the dict for both the check and the load itself
    # (matches scripts/train.py's --init path) rather than reading it twice.
    init_weights = load_weights_dict(str(args.init))
    check_weight_keys(init_weights, model._portable_state_dict(),
                      where=f"--init {args.init}")
    model._load_portable(init_weights)
    tok = ByteTokenizer() if args.byte_fallback else load_olmo_tokenizer(args.model_id)
    eos = getattr(tok, "eos_token_id", None)
    store = SessionStore(model)
    np_to = backend.to_numpy
    opt = backend.make_optimizer(model, args.lr)
    # Loss-Free-Balancing (#213): see scripts/train.py for the off-switch. The bias
    # arrives with `--init`'s portable weights (D3); attach_balancer adopts it, pushes it
    # into the routers, and enables load counting. `balancer=None` is a no-op on either
    # backend's make_grpo_train_step (#214).
    balancer = balancer_for_config(cfg)
    grpo_step = backend.make_grpo_train_step(model, opt, balancer=balancer)
    attach_balancer(balancer, model)

    problems = [json.loads(ln) for ln in args.problems.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    if not problems:
        raise SystemExit(f"no problems in {args.problems}")
    rng = np.random.default_rng(args.seed)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    weights_path = str(out / "weights.safetensors")
    telemetry_path = out / "telemetry.json"

    with contextlib.ExitStack() as stack:
        if args.reward == "math":
            reward_fn, verifier = math_reward, None
        elif args.reward == "exact":
            reward_fn, verifier = exact_match_reward, None
        else:
            # Constructed ONCE, outside the step loop — it owns a persistent
            # language-server subprocess, so per-step construction would dominate
            # the wall clock. `stack` guarantees `.close()` on any exit path
            # (normal completion, SystemExit, or an unhandled exception).
            verifier = LspVerifier(kind=args.oracle, timeout_s=args.lsp_timeout_s,
                                   ignore_module_resolution=args.lsp_ignore_module_resolution,
                                   hatches=args.lsp_hatches)
            stack.enter_context(verifier)
            reward_fn = None  # bound per-step below with the current prompt

        run_start = time.monotonic()

        def write_telemetry() -> None:
            elapsed = time.monotonic() - run_start
            t = verifier.telemetry()
            t["elapsed_wall_s"] = elapsed
            t["oracle_wall_frac"] = (t["wall_s"] / elapsed) if elapsed > 0 else 0.0
            telemetry_path.write_text(json.dumps(t, indent=2), encoding="utf-8")

        for step in range(args.steps):
            prob = problems[step % len(problems)]
            prompt_ids = list(tok.encode(prob["prompt"])) or [eos or 0]
            step_reward_fn = (partial(verifier.reward, prompt=prob["prompt"])
                              if verifier is not None else reward_fn)
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
                rewards.append(step_reward_fn(tok.decode(gen), str(prob.get("answer", ""))))

            adv = group_advantages([rewards])[0]            # (K,)
            metrics = grpo_step(model, [collate_rollouts(rollouts, adv)], args.lr)
            if step % args.log_every == 0:
                stats = reward_stats(rewards)
                line = (f"step {step:4d}  loss {metrics['loss']:.4f}  "
                       f"mean_reward {stats['mean_reward']:.3f}  solved {stats['frac_solved']:.3f}")
                if verifier is not None:
                    t = verifier.telemetry()
                    elapsed = time.monotonic() - run_start
                    n = max(t["n_samples"], 1)
                    line += (f"  oracle_calls {t['n_calls']}  oracle_wall_s {t['wall_s']:.1f}  "
                            f"frac_hacked {t['n_hacked'] / n:.3f}  "
                            f"frac_degenerate {t['n_degenerate'] / n:.3f}  "
                            f"oracle_wall_frac {(t['wall_s'] / elapsed) if elapsed > 0 else 0.0:.3f}")
                print(line)
            if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
                model.save(weights_path)
                print(f"  [ckpt] step {step} -> {weights_path}")
                if verifier is not None:
                    write_telemetry()   # a crashed long run still leaves cost evidence

        model.save(weights_path)
        if verifier is not None:
            write_telemetry()
        print(f"done — wrote {weights_path}")


if __name__ == "__main__":
    main()
