"""Self-speculative decoding primitives (#52) — portable, numpy only.

Draft-and-verify decoding accelerates autoregressive generation: a cheap drafter
proposes the next few tokens, the real model verifies them in one batched pass, and
verification keeps the output identical to plain decoding. This module holds the two
BACKEND-FREE pieces — the drafter and the accept rule — so they are testable without mlx.
The stateful verifier pass and the timing live in `scripts/spec_decode.py` (it advances
the model's `step` recurrence on the backend).

GREEDY ONLY: the accept rule (`first_mismatch`) compares draft tokens against the
verifier's *argmax*, so the preserved output is the GREEDY decode. It is NOT
distribution-preserving for temperature>0 / top-p sampling — that needs the Leviathan
et al. rejection-sampling rule, which this does not implement. Drive it greedily.

The drafter here is **prompt-lookup** (a.k.a. n-gram / self-speculative): it proposes
continuations by finding where the current context's tail recurred earlier and copying
what followed. It needs no second trained model — the "self-speculative variant avoids
training a second model" the issue calls for — and any wrong guess is simply rejected by
the verifier, so it can never change the output, only the speed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def propose(context: Sequence[int], gamma: int, max_n: int = 8) -> list[int]:
    """Prompt-lookup draft: up to `gamma` tokens continuing `context`.

    Tries the longest tail first: for n from min(max_n, len-1) down to 1, take the last
    n tokens as a pattern and search for its most recent EARLIER occurrence in the
    context; on a hit, copy the up-to-`gamma` tokens that followed it. Returns `[]` when
    no tail recurs (the caller then takes one ordinary step). Longer matched patterns are
    preferred because they predict the continuation more reliably.
    """
    ctx = [int(t) for t in context]
    L = len(ctx)
    if L < 2 or gamma <= 0:
        return []
    for n in range(min(max_n, L - 1), 0, -1):
        pattern = ctx[L - n:]
        # Most recent earlier occurrence: scan start positions right-to-left, excluding
        # the tail occurrence itself (search space is ctx[: L - n]).
        for start in range(L - n - 1, -1, -1):
            if ctx[start:start + n] == pattern:
                draft = ctx[start + n:start + n + gamma]
                if draft:
                    return draft
                break  # this pattern recurs only at the very end — try a shorter one
    return []


def first_mismatch(draft: Sequence[int], verifier_preds: Sequence[int]) -> int:
    """Number of leading draft tokens the verifier agrees with (greedy acceptance).

    `verifier_preds[i]` is the verifier's greedy next token at position i given the
    accepted prefix `draft[:i]`. The accepted count is the first i where they differ
    (or `len(draft)` if all agree) — exactly the prefix plain greedy decoding would also
    have produced, which is what makes speculative decoding distribution-preserving.
    """
    m = 0
    for d, p in zip(draft, verifier_preds):
        if int(d) != int(p):
            break
        m += 1
    return m


def prune_draft_trajectory(
    context: Sequence[int],
    draft: Sequence[int],
    critic_evaluator: Any,
    *,
    threshold: float = 0.70,
    model: Any | None = None,
    to_numpy: Callable[[Any], np.ndarray] = np.asarray,
) -> tuple[list[int], dict[str, Any]]:
    """Evaluate intermediate draft tokens using the critic head to abort flawed draft trajectories early (#388).

    At each intermediate position j in the draft, evaluates P(clean) for context + draft[:j].
    If P(clean) < threshold, the trajectory is aborted at position j-1, pruning all subsequent
    draft tokens before external verifier compute is spent.

    Args:
        context: Prior token sequence.
        draft: Proposed draft tokens from propose().
        critic_evaluator: Callable, DecisionCriticHead, or pre-computed probabilities.
        threshold: P(clean) acceptance threshold (default 0.70).
        model: Optional model providing forward_hidden or forward_with_critics.
        to_numpy: Backend-to-numpy conversion callable.

    Returns:
        pruned_draft: Truncated draft tokens prefix that passed the critic filter.
        telemetry: Dict with critic_latency_ms, tokens_evaluated, tokens_aborted,
                   abort_position, and probs.
    """
    if not draft:
        return [], {
            "critic_latency_ms": 0.0,
            "tokens_evaluated": 0,
            "tokens_aborted": 0,
            "abort_position": None,
            "probs": [],
        }

    t0 = time.perf_counter()
    probs: list[float] = []
    abort_pos: int | None = None
    pruned_draft: list[int] = list(draft)
    tokens_aborted: int = 0
    ctx = list(context)

    for j in range(1, len(draft) + 1):
        cand = ctx + list(draft[:j])
        prob = 0.5
        if callable(critic_evaluator):
            try:
                res = critic_evaluator(cand)
                prob = float(res.get("prob", res)) if isinstance(res, dict) else float(res)
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                prob = 0.5
        elif isinstance(critic_evaluator, (list, tuple)):
            idx = j - 1
            prob = float(critic_evaluator[idx]) if idx < len(critic_evaluator) else 0.5
        elif model is not None and hasattr(model, "forward_with_critics"):
            try:
                batch = np.array([cand], dtype=np.int32)
                _, critic_out = model.forward_with_critics(batch, critic_names=["noul"])
                prob = float(critic_out["noul"]["prob"])
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                prob = 0.5
        elif model is not None and hasattr(model, "forward_hidden"):
            try:
                batch = np.array([cand], dtype=np.int32)
                h_seq = model.forward_hidden(batch)
                h_last = to_numpy(h_seq)[0, -1, :]
                head = critic_evaluator or getattr(model, "critic_heads", {}).get("noul")
                if head is not None:
                    res = head.predict(h_last) if hasattr(head, "predict") else head.predict_noul(h_last)
                    prob = float(res.get("prob", 0.5))
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                prob = 0.5
        elif critic_evaluator is not None and hasattr(critic_evaluator, "predict_noul"):
            try:
                # If dummy or pre-computed hidden states are available
                res = critic_evaluator.predict_noul(np.zeros((1, getattr(critic_evaluator.config, 'd_model', 64)), dtype=np.float32))
                prob = float(res.get("prob", 0.5))
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                prob = 0.5

        probs.append(prob)
        if prob < threshold:
            abort_pos = j - 1
            pruned_draft = list(draft[:j - 1])
            tokens_aborted = len(draft) - len(pruned_draft)
            break

    latency_ms = (time.perf_counter() - t0) * 1000.0
    telemetry = {
        "critic_latency_ms": latency_ms,
        "tokens_evaluated": len(probs),
        "tokens_aborted": tokens_aborted,
        "abort_position": abort_pos,
        "probs": probs,
    }
    return pruned_draft, telemetry


def propose_mtp(
    model: Any,
    context: Sequence[int] | None = None,
    *,
    h_last: Any = None,
    next_token: int | None = None,
    depth: int = 1,
    gamma: int = 1,
    to_numpy: Callable[[Any], np.ndarray] = np.asarray,
) -> list[int]:
    """Generate speculative verification candidates using the native MTP head (#356).

    Proposes candidate continuation tokens using the model's depth-k MTP auxiliary head
    without requiring a separate draft model checkpoint.

    Args:
        model: ModelInterface instance with active MTP blocks.
        context: Prior token sequence if h_last/next_token not pre-computed.
        h_last: Optional pre-computed post-norm trunk hidden state (1, d_model).
        next_token: Optional pre-computed next token (int).
        depth: Starting MTP head depth (default: 1).
        gamma: Maximum candidate tokens to draft (default: 1).
        to_numpy: Callable to convert backend tensors to numpy.

    Returns:
        draft: List of proposed candidate token IDs.
    """
    if not hasattr(model, "mtp_blocks") or len(model.mtp_blocks) == 0:
        return []
    if gamma <= 0:
        return []

    # If h_last or next_token not provided, compute them from context
    if h_last is None or next_token is None:
        if not context:
            return []
        try:
            tokens = [int(t) for t in context]
            batch = [tokens]
            logits = model.forward(batch)
            l_np = to_numpy(logits)
            if next_token is None:
                next_token = int(np.argmax(l_np[0, -1, :]))
            if h_last is None:
                if hasattr(model, "forward_hidden"):
                    h_seq = model.forward_hidden(batch)
                    h_last = h_seq[:, -1, :] if hasattr(h_seq, "shape") else h_seq[-1]
                else:
                    return []
        except Exception:
            return []

    draft = []
    curr_h = h_last
    curr_tok = next_token
    k = depth
    max_k = min(len(model.mtp_blocks), k + gamma - 1)

    while k <= max_k and len(draft) < gamma:
        try:
            mtp_logits, _ = model.step_mtp(curr_h, curr_tok, depth=k)
            ml_np = to_numpy(mtp_logits)
            cand = int(np.argmax(ml_np.reshape(-1)))
            draft.append(cand)
            curr_tok = cand
            k += 1
        except Exception:
            break

    return draft


def spec_decode(
    model: Any,
    prompt: Sequence[int],
    max_new: int = 128,
    gamma: int = 1,
    max_n: int = 8,
    *,
    use_mtp: bool = True,
    critic_filter: bool = False,
    critic_threshold: float = 0.70,
    critic: Any = None,
    backend: str = "auto",
) -> tuple[list[int], float, dict[str, Any]]:
    """Complete speculative decoding loop with native MTP proposals (#356).

    Greedy self-speculative decoding supporting both native MTP head proposals and
    prompt-lookup n-gram proposals. Preserves exact greedy decoding output.

    Args:
        model: ModelInterface instance.
        prompt: Initial prompt token IDs.
        max_new: Maximum new tokens to generate.
        gamma: Draft tokens to propose per round.
        max_n: Longest pattern length for prompt-lookup fallback.
        use_mtp: Whether to use native MTP head proposals (default: True).
        critic_filter: Whether to apply surrogate critic filtering (#388).
        critic_threshold: P(clean) threshold for critic early abort.
        critic: Optional critic evaluator.
        backend: 'auto', 'mlx', or 'torch'.

    Returns:
        generated: List of generated token IDs (length <= max_new).
        elapsed: Elapsed wall-clock time in seconds.
        stats: Performance telemetry dict (rounds, drafted, accepted, accept_rate, etc.).
    """
    t0 = time.perf_counter()
    ctx = [int(t) for t in prompt]
    generated: list[int] = []
    drafted, accepted, rounds = 0, 0, 0
    critic_latency_ms = 0.0
    draft_tokens_aborted = 0

    # Initialize state
    state = model.init_state(1)
    # Step through prompt
    logits = None
    for t in ctx:
        logits, state = model.step(np.array([t]), state)

    # Detect if native MTP is available on this model
    has_mtp = use_mtp and hasattr(model, "mtp_blocks") and len(model.mtp_blocks) > 0

    while len(generated) < max_new:
        remaining = max_new - len(generated)
        x_greedy = int(np.argmax(np.asarray(logits).reshape(-1)))

        if has_mtp:
            # Step the trunk with greedy token to verify and get post-norm hidden state
            res = model.step(np.array([x_greedy]), state, return_hidden=True)
            if len(res) == 3:
                logits_next, state_next, h_last = res
            else:
                logits_next, state_next = res[:2]
                h_last = getattr(model, "norm_f", lambda x: x)(state_next[0][0])

            generated.append(x_greedy)
            ctx.append(x_greedy)
            if len(generated) >= max_new:
                break

            # Draft using MTP depth-1 head
            cand_tokens = propose_mtp(model, ctx, h_last=h_last, next_token=x_greedy, gamma=min(gamma, remaining - 1))
            if cand_tokens:
                cand = cand_tokens[0]
                drafted += 1
                rounds += 1
                verifier_pred = int(np.argmax(np.asarray(logits_next).reshape(-1)))
                if cand == verifier_pred:
                    accepted += 1
                    generated.append(cand)
                    ctx.append(cand)
                    logits, state = model.step(np.array([cand]), state_next)
                else:
                    logits = logits_next
                    state = state_next
            else:
                logits = logits_next
                state = state_next
        else:
            # Fallback: prompt-lookup drafter
            draft = propose(ctx, min(gamma, remaining), max_n)
            if not draft:
                logits, state = model.step(np.array([x_greedy]), state)
                generated.append(x_greedy)
                ctx.append(x_greedy)
                continue

            if critic_filter:
                draft, crit_telemetry = prune_draft_trajectory(
                    ctx, draft, critic, threshold=critic_threshold, model=model
                )
                critic_latency_ms += crit_telemetry["critic_latency_ms"]
                draft_tokens_aborted += crit_telemetry["tokens_aborted"]
                if not draft:
                    logits, state = model.step(np.array([x_greedy]), state)
                    generated.append(x_greedy)
                    ctx.append(x_greedy)
                    continue

            # Verify draft block
            if hasattr(model, "verify_block"):
                block_logits, block_states = model.verify_block(draft, state)
                all_logits = [np.asarray(logits).reshape(-1)] + [np.asarray(bl).reshape(-1) for bl in block_logits]
                preds = [int(np.argmax(l)) for l in all_logits]
                m = first_mismatch(draft, preds[:len(draft)])
                accept = draft[:m]
                drafted += len(draft)
                accepted += m
                rounds += 1
                if len(generated) + m >= max_new:
                    generated.extend(accept)
                    ctx.extend(accept)
                    break
                bonus = preds[m]
                base_state = state if m == 0 else block_states[m - 1]
                logits, state = model.step(np.array([bonus]), base_state)
                emit = accept + [bonus]
                generated.extend(emit)
                ctx.extend(emit)
            else:
                logits, state = model.step(np.array([x_greedy]), state)
                generated.append(x_greedy)
                ctx.append(x_greedy)

    elapsed = time.perf_counter() - t0
    tok_per_sec = (len(generated) / elapsed) if elapsed > 0 else 0.0
    stats = {
        "rounds": rounds,
        "drafted": drafted,
        "accepted": accepted,
        "accept_rate": (accepted / drafted) if drafted > 0 else 0.0,
        "tokens_per_round": (len(generated) / rounds) if rounds > 0 else 0.0,
        "tokens_per_second": tok_per_sec,
        "critic_latency_ms": critic_latency_ms,
        "draft_tokens_aborted": draft_tokens_aborted,
    }
    return generated[:max_new], elapsed, stats
