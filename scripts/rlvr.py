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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.train.grpo import group_advantages, reward_stats
from src.train.verifiers import (CppVerifier, KotlinVerifier, LspVerifier,
                                 MemoizedVerifier, RustVerifier, SwiftVerifier,
                                 SympyVerifier, ToolSchemaVerifier,
                                 When2CallAbstentionVerifier, Z3Verifier,
                                 exact_match_reward, math_reward,
                                 score_rollouts)


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
    ap.add_argument("--reward", choices=("math", "exact", "lsp", "tool-schema", "when2call", "sympy", "z3", "rust-static", "cpp-static", "c-static", "swift-static", "kotlin-static", "sql", "data-pipeline", "pipeline", "openapi", "graphql", "protobuf", "clean-architecture", "architecture", "data-contracts", "ts-web", "typescript-web", "react", "python-web", "fastapi", "django", "go-web", "java-web", "spring", "csharp-web", "dotnet", "php-web", "laravel", "ruby-web", "rails", "web-backend", "web"), default="math")
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
    ap.add_argument("--verifier-workers", type=int, default=4,
                    help="concurrent worker threads for rollout scoring (0 or 1 = sequential)")
    ap.add_argument("--verifier-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="memoize verifier rewards across identical completions (default on)")
    ap.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True,
                    help="skip oracle/compiler when output is degenerate or hacked (default on)")
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
    elif args.reward == "rust-static":
        from src.train.verifiers.systems_mobile import resolve_rust_toolchain
        if not resolve_rust_toolchain():
            raise SystemExit("no toolchain for --reward rust-static (rustc / cargo required on PATH)")
    elif args.reward in ("cpp-static", "c-static"):
        from src.train.verifiers.systems_mobile import resolve_cpp_toolchain
        if not resolve_cpp_toolchain():
            raise SystemExit("no toolchain for --reward cpp-static (clang++ / g++ required on PATH)")
    elif args.reward == "swift-static":
        from src.train.verifiers.systems_mobile import resolve_swift_toolchain
        if not resolve_swift_toolchain():
            raise SystemExit("no toolchain for --reward swift-static (swiftc required on PATH)")
    elif args.reward == "kotlin-static":
        from src.train.verifiers.systems_mobile import resolve_kotlin_toolchain
        if not resolve_kotlin_toolchain():
            raise SystemExit("no toolchain for --reward kotlin-static (kotlinc required on PATH)")
    elif args.reward == "sql":
        from src.train.verifiers.data_contracts import resolve_sql_toolchain
        resolve_sql_toolchain()
    elif args.reward == "protobuf":
        from src.train.verifiers.data_contracts import resolve_protobuf_toolchain
        resolve_protobuf_toolchain()
    elif args.reward == "openapi":
        from src.train.verifiers.data_contracts import resolve_openapi_toolchain
        resolve_openapi_toolchain()
    elif args.reward == "graphql":
        from src.train.verifiers.data_contracts import resolve_graphql_toolchain
        resolve_graphql_toolchain()
    elif args.reward in ("data-pipeline", "pipeline", "clean-architecture", "architecture", "data-contracts"):
        pass
    elif args.reward in ("ts-web", "typescript-web", "react"):
        from src.train.verifiers.web_backend import resolve_ts_web_toolchain
        resolve_ts_web_toolchain()
    elif args.reward in ("python-web", "fastapi", "django"):
        from src.train.verifiers.web_backend import resolve_python_web_toolchain
        resolve_python_web_toolchain()
    elif args.reward == "go-web":
        from src.train.verifiers.web_backend import resolve_go_web_toolchain
        resolve_go_web_toolchain()
    elif args.reward in ("java-web", "spring"):
        from src.train.verifiers.web_backend import resolve_java_web_toolchain
        resolve_java_web_toolchain()
    elif args.reward in ("csharp-web", "dotnet"):
        from src.train.verifiers.web_backend import resolve_csharp_web_toolchain
        resolve_csharp_web_toolchain()
    elif args.reward in ("php-web", "laravel"):
        from src.train.verifiers.web_backend import resolve_php_web_toolchain
        resolve_php_web_toolchain()
    elif args.reward in ("ruby-web", "rails"):
        from src.train.verifiers.web_backend import resolve_ruby_web_toolchain
        resolve_ruby_web_toolchain()
    elif args.reward in ("web-backend", "web"):
        pass

    from src.model.backend import get_backend
    from src.model.blocks import load_config
    from src.serve.generate import generate
    from src.serve.sessions import SessionStore
    from src.serve.sampling import sample
    from src.data.tokenize import ByteTokenizer, load_olmo_tokenizer
    from src.train.moe_balance import attach_balancer, balancer_for_config
    from src.train.checkpoint import check_weight_keys, load_weights_dict
    import mlx.core as mx

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
        raw_verifier = None
        if args.reward == "math":
            raw_verifier, reward_fn = math_reward, None
        elif args.reward == "exact":
            raw_verifier, reward_fn = exact_match_reward, None
        elif args.reward == "lsp":
            raw_verifier = LspVerifier(kind=args.oracle, timeout_s=args.lsp_timeout_s,
                                       ignore_module_resolution=args.lsp_ignore_module_resolution,
                                       hatches=args.lsp_hatches, fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "tool-schema":
            raw_verifier = ToolSchemaVerifier()
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "when2call":
            raw_verifier = When2CallAbstentionVerifier()
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "sympy":
            raw_verifier = SympyVerifier()
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "z3":
            raw_verifier = Z3Verifier()
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "rust-static":
            raw_verifier = RustVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("cpp-static", "c-static"):
            raw_verifier = CppVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "swift-static":
            raw_verifier = SwiftVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "kotlin-static":
            raw_verifier = KotlinVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "sql":
            from src.train.verifiers.data_contracts import SqlVerifier
            raw_verifier = SqlVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("data-pipeline", "pipeline"):
            from src.train.verifiers.data_contracts import DataPipelineVerifier
            raw_verifier = DataPipelineVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "openapi":
            from src.train.verifiers.data_contracts import OpenApiVerifier
            raw_verifier = OpenApiVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "graphql":
            from src.train.verifiers.data_contracts import GraphQLVerifier
            raw_verifier = GraphQLVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "protobuf":
            from src.train.verifiers.data_contracts import ProtobufVerifier
            raw_verifier = ProtobufVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("clean-architecture", "architecture"):
            from src.train.verifiers.data_contracts import CleanArchitectureVerifier
            raw_verifier = CleanArchitectureVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "data-contracts":
            from src.train.verifiers.data_contracts import DataContractsVerifier
            raw_verifier = DataContractsVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("ts-web", "typescript-web", "react"):
            from src.train.verifiers.web_backend import TypeScriptWebVerifier
            raw_verifier = TypeScriptWebVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("python-web", "fastapi", "django"):
            from src.train.verifiers.web_backend import PythonWebVerifier
            raw_verifier = PythonWebVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward == "go-web":
            from src.train.verifiers.web_backend import GoWebVerifier
            raw_verifier = GoWebVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("java-web", "spring"):
            from src.train.verifiers.web_backend import JavaWebVerifier
            raw_verifier = JavaWebVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("csharp-web", "dotnet"):
            from src.train.verifiers.web_backend import CSharpWebVerifier
            raw_verifier = CSharpWebVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("php-web", "laravel"):
            from src.train.verifiers.web_backend import PhpWebVerifier
            raw_verifier = PhpWebVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("ruby-web", "rails"):
            from src.train.verifiers.web_backend import RubyWebVerifier
            raw_verifier = RubyWebVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        elif args.reward in ("web-backend", "web"):
            from src.train.verifiers.web_backend import WebBackendVerifier
            raw_verifier = WebBackendVerifier(fail_fast=args.fail_fast)
            stack.enter_context(raw_verifier)
            reward_fn = None
        else:
            raise ValueError(f"unknown reward {args.reward}")

        if args.verifier_cache and raw_verifier is not None:
            verifier = MemoizedVerifier(raw_verifier)
            stack.enter_context(verifier)
        else:
            verifier = raw_verifier if hasattr(raw_verifier, "reward") else None
            if verifier is None:
                reward_fn = raw_verifier

        executor = None
        if args.verifier_workers > 1:
            executor = ThreadPoolExecutor(max_workers=args.verifier_workers)
            stack.callback(executor.shutdown, wait=True)

        run_start = time.monotonic()

        def write_telemetry() -> None:
            elapsed = time.monotonic() - run_start
            t = verifier.telemetry() if verifier is not None else {}
            t["elapsed_wall_s"] = elapsed
            if "wall_s" in t:
                t["oracle_wall_frac"] = (t["wall_s"] / elapsed) if elapsed > 0 else 0.0
            telemetry_path.write_text(json.dumps(t, indent=2), encoding="utf-8")

        for step in range(args.steps):
            prob = problems[step % len(problems)]
            prompt_ids = list(tok.encode(prob["prompt"])) or [eos or 0]
            extra_kwargs = {}
            if "tools" in prob:
                extra_kwargs["tools"] = prob["tools"]
            if "category" in prob:
                extra_kwargs["category"] = prob["category"]
            if "abstain" in prob:
                extra_kwargs["abstain"] = prob["abstain"]
            if "constraints" in prob:
                extra_kwargs["constraints"] = prob["constraints"]
            step_reward_fn = (partial(verifier.reward, prompt=prob["prompt"], **extra_kwargs)
                              if verifier is not None else reward_fn)
            # Batched rollout generation: parallel prefill + concurrent decode across group_size
            p_batch = mx.repeat(mx.array([int(t) for t in prompt_ids])[None], args.group_size, axis=0)
            logits, state = model.prefill(p_batch, last_only=True)
            batched_gens = [[] for _ in range(args.group_size)]
            finished = [False] * args.group_size
            rngs = [np.random.default_rng(int(rng.integers(1 << 30))) for _ in range(args.group_size)]

            logits_np = np_to(logits)
            for _ in range(args.max_new_tokens):
                next_tokens = []
                all_done = True
                for k in range(args.group_size):
                    if finished[k]:
                        next_tokens.append(eos or 0)
                        continue
                    tk = sample(logits_np[k], temperature=args.temperature, top_k=args.top_k, rng=rngs[k])
                    batched_gens[k].append(tk)
                    if eos is not None and tk == eos:
                        finished[k] = True
                        next_tokens.append(eos)
                    else:
                        all_done = False
                        next_tokens.append(tk)
                if all_done:
                    break
                logits, state = model.step(mx.array(next_tokens), state)
                logits_np = np_to(logits)

            rollouts = [(prompt_ids, g or [eos or 0]) for g in batched_gens]
            ans_str = str(prob.get("answer", ""))
            decoded_completions = [tok.decode(g) for g in batched_gens]
            rewards = score_rollouts(step_reward_fn, decoded_completions, ans_str, executor=executor)

            adv = group_advantages([rewards])[0]            # (K,)
            metrics = grpo_step(model, [collate_rollouts(rollouts, adv)], args.lr)
            if step % args.log_every == 0:
                stats = reward_stats(rewards)
                line = (f"step {step:4d}  loss {metrics['loss']:.4f}  "
                       f"mean_reward {stats['mean_reward']:.3f}  solved {stats['frac_solved']:.3f}")
                if verifier is not None:
                    t = verifier.telemetry()
                    elapsed = time.monotonic() - run_start
                    if "n_samples" in t:
                        n = max(t.get("n_samples", 0), 1)
                        line += (f"  oracle_calls {t.get('n_calls', 0)}  oracle_wall_s {t.get('wall_s', 0.0):.1f}  "
                                f"frac_hacked {t.get('n_hacked', 0) / n:.3f}  "
                                f"frac_degenerate {t.get('n_degenerate', 0) / n:.3f}  "
                                f"oracle_wall_frac {(t.get('wall_s', 0.0) / elapsed) if elapsed > 0 else 0.0:.3f}")
                        if "n_valid" in t:
                            line += (f"  frac_valid {t.get('n_valid', 0) / n:.3f}  "
                                    f"frac_hallucinated {t.get('n_hallucinations', 0) / n:.3f}  "
                                    f"frac_syntax_err {t.get('n_syntax_errors', 0) / n:.3f}")
                        if "n_abstain_success" in t:
                            line += (f"  frac_abstain {t.get('n_abstain_success', 0) / n:.3f}  "
                                    f"frac_spurious {t.get('n_spurious_calls', 0) / n:.3f}")
                        if "n_equivalent" in t:
                            line += f"  frac_equiv {t.get('n_equivalent', 0) / n:.3f}"
                        if "n_satisfied" in t:
                            line += f"  frac_sat {t.get('n_satisfied', 0) / n:.3f}"
                    if "cache_hit_rate" in t and (t.get("cache_hits", 0) + t.get("cache_misses", 0)) > 0:
                        line += f"  cache_hit {t['cache_hit_rate']:.2f}"
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
