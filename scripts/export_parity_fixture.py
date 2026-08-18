"""Export a Swift/MLX logit-parity fixture (#166 / #167).

Writes a self-contained fixture directory that `swift/engine`'s `monica-parity` runner
consumes: the portable weights, the exact token batch, and the Python MLX backend's
reference logits for BOTH code paths (`forward` and stacked per-token `step`), plus each
layer's output so a Swift mismatch can be localized to a single block instead of a
whole-model logit diff. Also writes `generation.safetensors` (#167 AC1): a greedy-decode id
oracle (`prompt_ids`, `greedy_ids`, `margins`) that `monica-parity` compares its own greedy
decode against, exactly.

    python scripts/export_parity_fixture.py --config config/toy.yaml \\
        --out swift/engine/Fixtures/toy --batch 2 --seq 129 \\
        [--precision fp32] [--vocab-size N] [--gen-steps 16] [--moe-bias] [--train-steps K]

`--train-steps K` (#195) additionally writes `train.safetensors`: a K-step Swift/MLX
`train_step` trajectory oracle (per-step loss, grad-norm, the full gradient tree, and
post-step weights) that `monica-train` gates against. Only `toy`/`toy-hybrid`/`toy-moe`
carry one — see `swift/engine/Fixtures/README.md`'s "Which fixtures get a train oracle".

The checked-in fixtures under `swift/engine/Fixtures/` ARE the Swift contract, so this
script is also the regeneration command recorded there — and
`tests/test_parity_fixture_export.py` re-runs it against the checked-in `toy` reference so
a future change to `mlx_backend.py`'s math cannot silently leave the Swift gate testing a
stale oracle.

MLX-only (it builds the MLX backend), so it does not run on a Linux/CUDA host.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

from src.conformance.forward_step_parity import check_forward_step_parity
from src.conformance.prefill_decode_parity import check_prefill_decode_parity
from src.conformance.quant_parity import check_quant_parity
from src.conformance.tolerances import LOWP_MEAN_KL_MAX, band_for, route_margin_min
from src.model.blocks import load_config


def _tree_add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


def _tree_scale(a, factor):
    from mlx.utils import tree_map
    return tree_map(lambda x: x * factor, a)


def _export_train_oracle(cfg, weights_path: str, tokens, *, out: "Path",
                         train_steps: int, seed: int) -> dict:
    """#195: write `train.safetensors` — a K-step trajectory oracle for `monica-train`.

    Reuses the REAL production `make_train_step` (`src/model/mlx_train_step.py`) for the
    loss/grad_norm/skipped/weight-update trajectory a Swift port must reproduce. The only
    piece duplicated here is `loss_fn` + `nn.value_and_grad` — the "only objective-specific
    piece" per `_accumulate_and_step`'s own docstring — recomputed on the SAME pristine
    model state immediately before each production step call, so it reproduces (bit-for-
    bit, MLX being deterministic given identical inputs and no dropout/randomness in this
    graph) the pre-clip gradient tree `_accumulate_and_step` computes internally but never
    returns. This is what makes the per-parameter `grad.{k}.<param>` surface possible
    without modifying `mlx_train_step.py` itself.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from src.model.mlx_backend import MLXMambaModel
    from src.model.mlx_train_step import make_train_step
    from src.train.loss_scale import scaler_for_precision

    grad_accum = 2
    b = tokens.shape[0]
    if b < grad_accum or b % grad_accum != 0:
        raise SystemExit(
            f"--train-steps needs batch ({b}) divisible by grad_accum={grad_accum}")
    groups = np.array_split(tokens, grad_accum, axis=0)
    micro_batches = [(g[:, :-1], g[:, 1:]) for g in groups]

    lr_ladder = [1e-3, 5e-4, 2e-4]
    if train_steps > len(lr_ladder):
        raise SystemExit(
            f"--train-steps {train_steps} > {len(lr_ladder)} hard-coded LRs; extend the ladder")
    lrs = lr_ladder[:train_steps]

    def fresh_model() -> "MLXMambaModel":
        m = MLXMambaModel(cfg)
        m.load(weights_path)
        return m

    def run(grad_clip: float, want_grads: bool):
        # Reseed before every call (not just once at the top of `main`, #293 review): the
        # graph has no randomness today, but `run()` is invoked multiple times per export
        # (natural-norm dry run, real run, determinism re-run) and if a future objective
        # variant introduces one (e.g. dropout), the seed argument must actually govern
        # each of those calls rather than silently drifting with ambient RNG state — the
        # determinism check at step 3 below would then have a real seed to reproduce.
        mx.random.seed(seed)
        model = fresh_model()
        opt = optim.AdamW(learning_rate=lrs[0])
        scaler = scaler_for_precision(cfg.precision)
        step_fn = make_train_step(model, opt, grad_clip=grad_clip, scaler=scaler)

        def loss_fn(m, inputs, targets):
            logits = m.forward(inputs)
            v = logits.shape[-1]
            t = mx.array(targets).reshape(-1).astype(mx.int32)
            ce = nn.losses.cross_entropy(
                logits.reshape(-1, v).astype(mx.float32), t, reduction="mean")
            return ce * scaler.scale if scaler else ce

        value_and_grad = nn.value_and_grad(model, loss_fn)

        def recompute_avg_grads():
            n = len(micro_batches)
            acc_grads = None
            acc_loss = mx.zeros(())
            for inp, tgt in micro_batches:
                loss, grads = value_and_grad(model, inp, tgt)
                acc_grads = grads if acc_grads is None else _tree_add(acc_grads, grads)
                acc_loss = acc_loss + loss
                mx.eval(acc_grads, acc_loss)
            grads = _tree_scale(acc_grads, 1.0 / n)
            loss = acc_loss / n
            if scaler:
                inv = 1.0 / scaler.scale
                grads = _tree_scale(grads, inv)
                loss = loss * inv
            return loss, grads

        losses, norms, grads_by_step, weights_by_step = [], [], [], []
        for k in range(train_steps):
            pre_grads = None
            if want_grads:
                _, pre_grads = recompute_avg_grads()
                mx.eval(pre_grads)
            step_out = step_fn(model, micro_batches, lrs[k])
            if step_out.get("skipped"):
                raise SystemExit(
                    f"refusing to write {out}: training oracle step {k} was skipped "
                    "(fp16 overflow) — pick a different --seed")
            losses.append(float(step_out["loss"]))
            norms.append(float(step_out["grad_norm"]))
            if want_grads:
                grads_by_step.append(dict(tree_flatten(pre_grads)))
                weights_by_step.append(dict(tree_flatten(model.parameters())))
        return losses, norms, grads_by_step, weights_by_step

    # 1. dry run with clipping effectively off -> the "natural" per-step norm trajectory.
    natural_losses, natural_norms, _, _ = run(grad_clip=1e9, want_grads=False)
    grad_clip = float(np.median(natural_norms))
    if not (grad_clip > 0 and np.isfinite(grad_clip)):
        raise SystemExit(
            f"refusing to write {out}: bad grad_clip candidate {grad_clip} "
            f"from natural norms {natural_norms}")

    # 2. the real run, from a pristine reload, with grad_clip=median(natural norms).
    losses, norms, grads_by_step, weights_by_step = run(grad_clip=grad_clip, want_grads=True)
    clipped_steps = [i for i, n in enumerate(norms) if n > grad_clip]
    unclipped_steps = [i for i, n in enumerate(norms) if n <= grad_clip]
    if not clipped_steps or not unclipped_steps:
        raise SystemExit(
            f"refusing to write {out}: grad_clip={grad_clip} exercises no clip branch "
            f"(norms={norms}) — clipped={clipped_steps} unclipped={unclipped_steps}")

    # 3. determinism: re-running from the same seed must reproduce the same trajectory.
    losses2, norms2, _, _ = run(grad_clip=grad_clip, want_grads=False)
    if losses2 != losses or norms2 != norms:
        raise SystemExit(
            f"refusing to write {out}: training oracle is not deterministic across re-runs "
            f"(losses {losses} vs {losses2}, norms {norms} vs {norms2})")

    # 4. every step's grads must be finite and non-zero — a fixture must gate something.
    for k, gk in enumerate(grads_by_step):
        flat = list(gk.values())
        if not all(bool(mx.all(mx.isfinite(g)).item()) for g in flat):
            raise SystemExit(f"refusing to write {out}: step {k} grads contain non-finite values")
        if all(bool(mx.all(g == 0).item()) for g in flat):
            raise SystemExit(f"refusing to write {out}: step {k} grads are all-zero")

    train_out = {}
    for j, (inp, tgt) in enumerate(micro_batches):
        train_out[f"mb.{j}.inputs"] = np.asarray(inp, dtype=np.int32)
        train_out[f"mb.{j}.targets"] = np.asarray(tgt, dtype=np.int32)
    for k in range(train_steps):
        train_out[f"loss.{k}"] = np.array(losses[k], dtype=np.float32)
        train_out[f"grad_norm.{k}"] = np.array(norms[k], dtype=np.float32)
        for name, val in grads_by_step[k].items():
            train_out[f"grad.{k}.{name}"] = np.array(val, dtype=np.float32)
        for name, val in weights_by_step[k].items():
            train_out[f"weights_after.{k}.{name}"] = np.array(val, dtype=np.float32)
    from safetensors.numpy import save_file
    save_file(train_out, str(out / "train.safetensors"))

    return {
        "train_steps": train_steps,
        "train_grad_accum": grad_accum,
        "train_grad_clip": grad_clip,
        "train_lrs": lrs,
        "train_loss": losses,
        "train_grad_norm": norms,
        "train_clipped_steps": clipped_steps,
        # New keys, deliberately NOT `rtol`/`atol` — those are the fp32 LOGIT gate's keys
        # (and, for the two quantized fixtures, their own looser 2e-2/2e-2). A training
        # step is not a forward pass: gradients accumulate error through the whole
        # backward graph and AdamW moments compound it across steps, so it gets its own
        # band (see `.claude/plans/issue-195.md`'s tolerance table). `1e-5` atol would be
        # LOOSER than the grad values themselves on small toy leaves and gate nothing
        # there — hence the tighter atol traded against a modestly looser rtol.
        "train_rtol": 2e-4,
        "train_atol": 1e-6,
    }


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div)


def _eval_doc_length_expr(node: ast.expr, q: int) -> int:
    """Evaluate one AST node of a `--packed-doc-lengths` expression against a whitelist:
    integer literals, the single name `Q`, unary +/-, and +-*/ binary ops. Anything else
    (calls, attribute access, subscripts, comprehensions, ...) raises — this is arithmetic
    parsing, not a general expression evaluator, so there is no path to attacker-controlled
    code execution regardless of who supplies `spec`."""
    if isinstance(node, ast.Expression):
        return _eval_doc_length_expr(node.body, q)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name) and node.id == "Q":
        return q
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_doc_length_expr(node.operand, q)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left = _eval_doc_length_expr(node.left, q)
        right = _eval_doc_length_expr(node.right, q)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        return left // right if right and left % right == 0 else left / right  # ast.Div
    raise ValueError(f"unsupported expression in --packed-doc-lengths: {ast.dump(node)}")


def _resolve_packed_doc_lengths(spec: str, q: int) -> list[int]:
    """Parse a comma-separated list of chunk-length-relative expressions (e.g.
    `"Q,2*Q,5"`) into concrete doc lengths. `Q` is the only name in scope, so this is
    just enough to express "a chunk multiple, several chunks, and a short/ragged doc" —
    the exact shapes `tests/test_doc_boundary_parity.py` already gates, so the Swift P6
    gate and the Python doc-boundary gate stress the same geometry (#68/#263).

    Parsed via a small AST whitelist (`_eval_doc_length_expr`), not `eval` — `spec` is a
    `--packed-doc-lengths` CLI argument this maintainer-run export script's own caller
    supplies (never untrusted/network input), but a whitelisted parser is strictly safer
    than `eval` with stripped builtins and costs nothing here."""
    return [int(_eval_doc_length_expr(ast.parse(tok.strip(), mode="eval"), q))
            for tok in spec.split(",")]


def _save_finite(arrays: dict, path: str) -> None:
    """`safetensors.numpy.save_file`, but refuse a non-finite float array first (#266
    edge case). fp16's max finite value is 65504; attention logits pre-softmax and the
    SSD decay accumulation are the exposed overflow sites at low precision. Without this,
    an overflow silently becomes NaN/Inf that only surfaces later as a confusing
    tolerance-comparison failure — this reports it as what it is."""
    from safetensors.numpy import save_file

    non_finite = sorted(
        k for k, v in arrays.items()
        if np.issubdtype(v.dtype, np.floating) and not np.all(np.isfinite(v)))
    if non_finite:
        raise SystemExit(
            f"refusing to write {path}: non-finite values in {non_finite} — likely fp16 "
            "overflow (max finite 65504) or an underlying numerics bug; this is an "
            "overflow, not a tolerance failure")
    save_file(arrays, path)


def _np_f32(x) -> np.ndarray:
    """`np.array(x, dtype=np.float32)`, but for an MLX array cast to float32 IN MLX
    first. This local MLX build (0.32.0) cannot convert a `bfloat16` array to numpy via
    the buffer protocol directly (`RuntimeError: Item size 2 for PEP 3118 buffer format
    string B does not match the dtype B item size 1`) — final logits are already fp32 by
    the time they leave the model (the head computes in fp32, `mlx_backend.py`'s `_f32`
    convention), so this bug was invisible before #266 added bf16 fixtures; INTERMEDIATE
    arrays (hidden states, mixing matrices, per-layer state) stay in the model's native
    compute dtype and hit it. A no-op for anything already fp32/fp16."""
    import mlx.core as mx

    if isinstance(x, mx.array):
        x = x.astype(mx.float32)
    return np.array(x, dtype=np.float32)


def build_fixture(config_path: str, out_dir: str, *, batch: int, seq: int,
                  precision: str | None = None, moe_bias: bool = False,
                  seed: int = 0, vocab_size: int | None = None,
                  gen_steps: int = 16, quant_bits: int | None = None,
                  quant_group_size: int = 64, quant_head_bits: int | None = None,
                  train_steps: int = 0,
                  packed_doc_lengths: str | None = None,
                  allow_moe_load_omit: bool = False) -> dict:
    """Build and write one fixture directory. Returns the `meta.json` contents.

    `quant_bits` (#168), if given, makes this a QUANTIZED fixture: `weights.safetensors`
    holds the packed checkpoint (+ `quant` sidecar block) instead of plain fp32 weights,
    and `forward_logits`/`step_logits`/`hidden.*`/`generation.safetensors` are computed
    from the FAKE-QUANT reference model (dequantized weights loaded into the unmodified
    `MLXMambaModel`) rather than the fp model — the "Python quantized logits" Swift's
    true quantized kernel is compared against at a looser, fixture-carried tolerance
    (see `.claude/plans/issue-168.md`'s "validation problem" section). Two extra files
    ride along: `fp_forward_logits` (in `reference.safetensors`, the ORIGINAL fp model's
    logits — the top-1/KL statistical gate's reference) and `dequant_ref.safetensors`
    (the exact dequantized weight for each targeted module — the tight 1e-6 format-
    correctness gate).
    """
    import mlx.core as mx  # local import: this script is MLX-only, like scripts/smoke_test.py

    from src.eval.quantize import (
        dequantize_portable_state_dict, quant_targets, quantize_portable_state_dict)
    from src.model.mlx_backend import (
        _DTYPES, _f32, _linear, AttentionBlock, MambaBlock, MLXMambaModel, MoEBlock)
    from src.serve import sampling
    from src.serve.generate import generate as py_generate
    from src.serve.sessions import SessionStore
    from src.train.checkpoint import save_weights

    # #264: disable MLX's buffer-cache pool for the duration of this export. Diagnosed
    # while adding the mixing-matrix oracle — this local MLX build (0.32.0) has been
    # observed to silently corrupt a computation (deterministic-per-process, exact-shape
    # results, no exceptions) when many independently-constructed `MLXMambaModel`s
    # allocate/free/reuse buffers of the same size class in one process, which this
    # exporter does far more of than any other caller (repeated regeneration, the #166
    # staleness test re-invoking `build_fixture` inside an already-populated pytest
    # process). `mx.set_cache_limit(0)` (proven in isolation to take the observed failure
    # rate from ~50-100% down to 0/10 trials) trades allocator throughput — irrelevant for
    # a one-shot fixture export — for correctness; it must not be ported into a hot
    # training/inference path.
    mx.clear_cache()
    # #264 (review follow-up): capture the prior limit and restore it in `finally`
    # below so the disable is scoped to this export, not process-global — while still
    # covering the ENTIRE rest of the function (including every early `raise SystemExit`
    # refusal path), so the cache stays off for the full duration the mitigation above
    # depends on.
    _prev_cache_limit = mx.set_cache_limit(0)
    try:
        mx.random.seed(seed)
        cfg = load_config(config_path)
        if precision is not None:
            # The harness gates fp32 (poc.yaml is fp16). Override BEFORE `model.save`, so the
            # emitted sidecar is self-describing and the Swift loader needs no override of
            # its own.
            cfg.precision = precision
            cfg.validate()
        if vocab_size is not None:
            # Alongside --precision: override BEFORE `model.save` so the sidecar stays
            # self-describing and the Swift loader needs no override of its own. `toy-gen`
            # (#167) uses this — a `monica-tokenize`-trained tokenizer's vocab (256 base bytes +
            # specials) cannot fit `toy.yaml`'s 256-wide model, so the fixture widens the model
            # instead of shrinking the tokenizer.
            cfg.vocab_size = vocab_size
            cfg.validate()

        model = MLXMambaModel(cfg)

        if moe_bias:
            # One fixture carries `moe_route_bias.*` keys, so the biased-ranking branch AND the
            # loader's pop path are both exercised. A fixed, deliberately asymmetric vector:
            # a uniform bias would not change the ranking and would gate nothing.
            blocks = model.moe_blocks()
            if not blocks:
                raise SystemExit("--moe-bias needs a config with MoE layers")
            model.set_moe_biases([
                [0.5 * ((-1) ** e) * (e + 1) for e in range(cfg.n_experts)]
                for _ in blocks
            ])

        # The exact token construction used by tests/test_mlx_parity.py.
        tokens = np.random.default_rng(seed).integers(
            0, cfg.vocab_size, size=(batch, seq)).astype(np.int32)

        # --- #168: quantize, and point the rest of the export at the FAKE-QUANT reference ---
        quant_block = None
        fp_forward_logits = None
        dequant_ref = {}
        # `ref_model` is what forward_logits/step_logits/hidden.*/generation.safetensors get
        # computed from — the fp model normally, or the fake-quant (dequantized) reference
        # for a quantized fixture. `weights_sd`/`weights_quant` are what actually gets
        # written to weights.safetensors: the fp portable state dict normally, or the REAL
        # packed checkpoint for a quantized fixture (never the fake-quant float weights).
        ref_model = model
        weights_sd = None   # None => use ref_model.save() (fp path, unchanged from before)
        if quant_bits is not None:
            fp_forward_logits = _np_f32(model.forward(tokens))
            sd = model._portable_state_dict()
            targets = quant_targets(sd, group_size=quant_group_size, bits=quant_bits,
                                    head_bits=quant_head_bits)
            if not targets:
                raise SystemExit(
                    f"--quant-bits {quant_bits}: no quantizable tensors for "
                    f"group_size={quant_group_size} against config {config_path}")
            qsd, quant_block = quantize_portable_state_dict(sd, targets, group_size=quant_group_size)
            deq_sd = dequantize_portable_state_dict(qsd, quant_block)
            for path in targets:
                dequant_ref[f"{path}.weight"] = np.asarray(deq_sd[f"{path}.weight"], dtype=np.float32)
            qmodel = MLXMambaModel(cfg)
            qmodel._load_portable(deq_sd)
            ref_model = qmodel
            weights_sd = qsd

        # #266: the precision-derived elementwise band this fixture's own consistency
        # checks (and later the greedy-margin/generation checks) are measured against —
        # NOT the fp32 constants, which are too tight to be meaningful for fp16/bf16 (see
        # src/conformance/tolerances.py's module docstring). This is also the blocker fix:
        # before #266, both calls below ran with their fp32 DEFAULTS regardless of
        # `cfg.precision`, so a fp16/bf16 model always failed its own forward/step check
        # and no low-precision fixture could ever reach disk.
        rtol_fp, atol_fp = band_for(cfg.precision)

        # A fixture that fails PYTHON's own forward/step gate must never reach disk — it would
        # make the Swift gate compare against a reference that is itself internally inconsistent.
        verdict = check_forward_step_parity(ref_model, tokens, to_numpy=np.array,
                                            rtol=rtol_fp, atol=atol_fp)
        if not verdict["ok"]:
            raise SystemExit(
                f"refusing to write {out_dir}: the Python model fails its own forward/step "
                f"parity check (max_abs_diff={verdict['max_abs_diff']:.3e}, "
                f"rtol={rtol_fp}, atol={atol_fp}, precision={cfg.precision})")

        # Same rule for #169's state handoff: a fixture whose own Python prefill state
        # disagrees with its stepped state (or whose prefill logits disagree with forward, or
        # whose prefill-then-decode disagrees with pure step-by-step) can never reach disk —
        # it would make the Swift gate compare against an internally-inconsistent oracle.
        prefill_verdict = check_prefill_decode_parity(
            ref_model, tokens, to_numpy=np.array, n_decode=min(4, seq - 1),
            rtol=rtol_fp, atol=atol_fp)
        if not prefill_verdict["ok"]:
            raise SystemExit(
                f"refusing to write {out_dir}: the Python model fails its own prefill/decode "
                f"parity check ({prefill_verdict}, rtol={rtol_fp}, atol={atol_fp}, "
                f"precision={cfg.precision})")

        # #266 anti-vacuity refusals (low precision only — fp32's elementwise band IS the
        # load-bearing gate, see tolerances.py Step 3/4). A fixture whose OWN measured
        # noise floor is not comfortably inside its gate would make the whole low-precision
        # contract vacuous, so refuse before anything reaches disk rather than after.
        lowp_self_mean_kl = None
        if cfg.precision != "fp32":
            elementwise_ceiling = atol_fp / 4
            if verdict["max_abs_diff"] > elementwise_ceiling:
                raise SystemExit(
                    f"refusing to write {out_dir}: forward/step max_abs_diff="
                    f"{verdict['max_abs_diff']:.3e} exceeds the anti-vacuity ceiling "
                    f"atol/4={elementwise_ceiling:.3e} for precision={cfg.precision} — "
                    "the fixture's own noise floor sits too close to its gate")
            # A second, throwaway forward+step pass purely to measure the intra-Python
            # KL noise floor (the load-bearing low-precision tier, tolerances.py Step 4) —
            # this cannot reuse `verdict`, which only returns max_abs_diff/ok, not the
            # underlying logit arrays.
            probe_fwd = _np_f32(ref_model.forward(tokens))
            probe_state = ref_model.init_state(batch)
            probe_steps = []
            for t in range(seq):
                logits_t, probe_state = ref_model.step(tokens[:, t], probe_state)
                probe_steps.append(_np_f32(logits_t))
            probe_step = np.stack(probe_steps, axis=1)
            lowp_self_mean_kl = check_quant_parity(
                probe_fwd, probe_step, bits=0,
                thresholds={"top1": 0.0, "kl": float("inf")})["mean_kl"]
            kl_ceiling = LOWP_MEAN_KL_MAX[cfg.precision] / 4
            if lowp_self_mean_kl > kl_ceiling:
                raise SystemExit(
                    f"refusing to write {out_dir}: intra-Python forward-vs-step "
                    f"mean_kl={lowp_self_mean_kl:.3e} exceeds the anti-vacuity ceiling "
                    f"kl_max/4={kl_ceiling:.3e} for precision={cfg.precision}")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        weights_path = str(out / "weights.safetensors")
        if weights_sd is None:
            ref_model.save(weights_path)        # weights.safetensors + .config.json sidecar
        else:
            save_weights(weights_sd, weights_path, config=cfg, quant=quant_block)

        _save_finite({"tokens": tokens}, str(out / "inputs.safetensors"))

        # --- #265: MoE load counting + route-bias write-back oracle ------------------------
        # Load counting does not depend on the balancer (`moe_balance_rate` is null in every
        # config in the tree, #217) — `set_moe_load_counting` is an independent switch
        # (`mlx_backend.py:992-996`), so the harness turns it on explicitly around exactly the
        # forward pass whose logits are already pinned above. `top_k < n_experts` is required
        # because Python counts only in the `k < E` branch (at `k == E` no mask exists and
        # balancing is vacuous).
        moe_blocks_list = ref_model.moe_blocks()
        count_loads = quant_bits is None and bool(moe_blocks_list) and cfg.top_k < cfg.n_experts
        if count_loads:
            ref_model.set_moe_load_counting(True)
            # Drain first: `check_forward_step_parity`/`check_prefill_decode_parity` above
            # already ran forwards on this model and would otherwise contribute counts to
            # the pinned pass below.
            ref_model.pop_moe_load()
        forward_logits = _np_f32(ref_model.forward(tokens))
        moe_load_layers: list[int] = []
        moe_loads: list[list[float]] = []
        moe_route_margin_min = None
        moe_load_omitted_reason: str | None = None
        if count_loads:
            moe_loads = ref_model.pop_moe_load()
            ref_model.set_moe_load_counting(False)   # off before hidden_states/generation below
            moe_load_layers = [i for i, l in enumerate(ref_model.layers) if isinstance(l, MoEBlock)]
            expected_total = tokens.size * cfg.top_k
            length_mismatch = len(moe_loads) != len(moe_load_layers)
            all_zero = all(c == 0 for row in moe_loads for c in row)
            # `zip` (not indexed) so a length mismatch can never silently truncate the check —
            # `length_mismatch` above already refuses on that shape, independent of this loop.
            bad_totals = [] if length_mismatch else [
                (i, sum(row)) for i, row in zip(moe_load_layers, moe_loads)
                if abs(sum(row) - expected_total) > 1e-6]
            if not moe_load_layers or length_mismatch or all_zero or bad_totals:
                raise SystemExit(
                    f"refusing to write {out_dir}: the load-count oracle would be a silent "
                    f"no-op (layers={moe_load_layers}, length_mismatch={length_mismatch}, "
                    f"all_zero={all_zero}, bad_totals={bad_totals}, "
                    f"expected per-layer total={expected_total})")

            # The exactness hazard: load.{i} is compared without tolerance, exactly like
            # greedy_ids — and greedy_ids is only safe because the exporter records
            # greedy_margin_min. Compute the analogue: the gap between the k-th and
            # (k+1)-th largest SELECTION score (`logits + route_bias` when the bias is
            # active, else `logits`) per token, per MoE layer — reconstructed from the SAME
            # expression `_moe` ranks by, so this can never describe a different ranking
            # than the counts above.
            hs_for_margin = ref_model.hidden_states(tokens)   # hs_for_margin[i] is layer i's INPUT
            cd = _DTYPES[cfg.precision]
            margin_per_layer: dict[int, float] = {}
            for i in moe_load_layers:
                layer = ref_model.layers[i]
                xn = layer.norm(hs_for_margin[i])
                logits_i = _f32(_linear(layer.router, xn, cd))
                sel = logits_i + layer._route_bias if layer._bias_active else logits_i
                sorted_desc = mx.sort(sel, axis=-1)[..., ::-1]
                margin = sorted_desc[..., cfg.top_k - 1] - sorted_desc[..., cfg.top_k]
                margin_per_layer[i] = float(mx.min(margin).item())
            moe_route_margin_min = min(margin_per_layer.values())
            # #266: generalised from the hardcoded `1e-5` to `max(1e-5, 128 * u)` — that
            # recovers today's fp32 constant exactly (`128 * u_fp32 = 7.6e-6 < 1e-5`, so the
            # floor wins) and scales the hazard band for fp16/bf16 (see
            # src/conformance/tolerances.route_margin_min's docstring).
            route_margin_threshold = route_margin_min(cfg.precision)
            if moe_route_margin_min < route_margin_threshold:
                if allow_moe_load_omit:
                    # The declared-omission fallback (README's fallback ladder: bump
                    # --seed, then --moe-bias, then this) — a skipped load-count oracle
                    # must be DECLARED, never silent, so main.swift's anti-BLIND rule can
                    # accept it as a legitimate skip rather than a stale fixture.
                    moe_load_omitted_reason = (
                        f"moe_route_margin_min={moe_route_margin_min:.3e} < "
                        f"route_margin_min({cfg.precision})={route_margin_threshold:.3e} — "
                        "omitting the load-count oracle rather than writing a flaky exact "
                        "comparison (per-layer minima="
                        f"{margin_per_layer}, #266 fallback ladder)")
                    moe_load_layers = []
                    moe_loads = []
                else:
                    raise SystemExit(
                        f"refusing to write {out_dir}: moe_route_margin_min="
                        f"{moe_route_margin_min:.3e} is inside the exact-comparison hazard "
                        f"band (< {route_margin_threshold:.3e} for precision={cfg.precision}) "
                        "— a near-tie routing rank makes load.{i} equality flaky across "
                        f"implementations. Bump --seed and retry (per-layer minima="
                        f"{margin_per_layer})")
            for i, m in margin_per_layer.items():
                if m < 1e-3:
                    print(f"WARNING: {out_dir} layer {i} moe_route_margin_min={m:.3e} is "
                         "below 1e-3 — routing rank is close to a tie; watch this fixture for "
                         "future flakes.")

        state = ref_model.init_state(batch)
        steps = []
        for t in range(seq):
            logits_t, state = ref_model.step(tokens[:, t], state)
            steps.append(_np_f32(logits_t))
        step_logits = np.stack(steps, axis=1)

        ref = {"forward_logits": forward_logits, "step_logits": step_logits}
        if fp_forward_logits is not None:
            ref["fp_forward_logits"] = fp_forward_logits
        # Per-layer hidden states: a deliberate debugging investment. A whole-model logit
        # mismatch is nearly impossible to localize by hand across 24 layers; "first divergence
        # at layer 7" is the difference between an afternoon and a week. ~200 KB at toy scale.
        for i, h in enumerate(ref_model.hidden_states(tokens)):
            ref[f"hidden.{i}"] = _np_f32(h)
        # Per-layer head-averaged mixing matrices (#264), a second interpretability oracle
        # alongside hidden.*: the #100 mixing-match accessor, gated at the same strict fp32
        # tolerance. Only for non-quantized fixtures — quantization is a lossy path and must
        # not drag the loose #168/#266 tolerance hook into this strict gate (same carve-out
        # `packed.safetensors` already uses).
        mixing_layers: list[int] = []
        if quant_bits is None:
            mixing = ref_model.mixing_matrices(tokens)          # [(layer_idx, (B,L,L))]
            for i, m in mixing:
                ref[f"mixing.{i}"] = _np_f32(m)
            mixing_layers = [i for i, _ in mixing]
            # An empty list would make the whole Swift mixing-matrix section a silent no-op
            # for any fixture whose config actually has a Mamba layer to mix.
            has_mamba_layer = any(isinstance(l, MambaBlock) for l in ref_model.layers)
            if has_mamba_layer and not mixing_layers:
                raise SystemExit(
                    f"refusing to write {out_dir}: mixing_matrices() returned no layers for a "
                    "config with at least one Mamba layer — the mixing-matrix oracle would be "
                    "a silent no-op")
        # `load.{i}` (#265), keyed by ABSOLUTE layer index like `mixing.{i}` — the per-expert
        # token counts from the pinned forward pass above. Present only when `count_loads`,
        # so the six non-participating fixtures (no MoE layers, or quantized) keep
        # byte-identical metadata.
        for i, counts in zip(moe_load_layers, moe_loads):
            ref[f"load.{i}"] = np.asarray(counts, dtype=np.float32)
        _save_finite(ref, str(out / "reference.safetensors"))
        if dequant_ref:
            _save_finite(dequant_ref, str(out / "dequant_ref.safetensors"))

        # --- prefill.safetensors: the state-handoff oracle (#169) -----------------------------
        # One entry per STATEFUL leaf, keyed by layer index and slot, derived from the layer's
        # CLASS (not tensor shapes) — so Swift, which iterates its OWN layers and demands
        # exactly the keys its own LayerState case implies, fails loudly on a missing key if the
        # two sides ever disagree about layer structure, instead of silently skipping a check.
        pre_logits, pre_state = ref_model.prefill(tokens)
        pre_last, _ = ref_model.prefill(tokens, last_only=True)
        prefill_out = {
            "prefill_logits": _np_f32(pre_logits),
            "prefill_last_logits": _np_f32(pre_last),
        }
        for i, (layer, st) in enumerate(zip(ref_model.layers, pre_state)):
            if isinstance(layer, MambaBlock):
                conv, ssm = st
                prefill_out[f"state.{i}.conv"] = _np_f32(conv)
                prefill_out[f"state.{i}.ssm"] = _np_f32(ssm)
            elif isinstance(layer, AttentionBlock):
                k, v = st
                prefill_out[f"state.{i}.k"] = _np_f32(k)
                prefill_out[f"state.{i}.v"] = _np_f32(v)
            elif isinstance(layer, MoEBlock):
                pass   # stateless — emitting a zero-sized placeholder would only add a
                       # round-tripping risk for no coverage; Swift's .moe case carries nothing.
            else:
                raise SystemExit(
                    f"refusing to write {out_dir}: layer {i} has unrecognized type "
                    f"{type(layer).__name__} — teach the exporter its state slot names")
        _save_finite(prefill_out, str(out / "prefill.safetensors"))

        # --- packed.safetensors: the packing-aware seg_ids oracle (#68/#263) -------------------
        # Only for non-quantized fixtures (`quant_bits is None`): quantized packing is a
        # separate concern and would drag the loose #168/#266 tolerance hook into a gate this
        # issue keeps STRICT (fp32 rtol=1e-4/atol=1e-5, same as the other fixture arrays).
        # Uses the SAME packing rule `check_doc_boundary_parity` uses — that function is the
        # contract, not reinvented here — so this artifact is a Swift-consumable snapshot of
        # exactly what the Python conformance gate already checks.
        packed_meta_doc_lengths = None
        packed_meta_seq_len = None
        if packed_doc_lengths is not None and quant_bits is None:
            Q = cfg.chunk_size or 64
            doc_lens = _resolve_packed_doc_lengths(packed_doc_lengths, Q)
            rng = np.random.default_rng(seed)
            docs = [rng.integers(0, cfg.vocab_size, size=n).tolist() for n in doc_lens]

            packed: list[int] = []
            seg: list[int] = []
            for d, doc in enumerate(docs):
                n = len(doc)
                plen = ((n + Q - 1) // Q) * Q
                packed.extend(doc + [0] * (plen - n))   # pad_id=0, following the real tokens
                seg.extend([d] * plen)

            packed_arr = np.asarray(packed, dtype=np.int32)[None]      # (1, Lp)
            seg_arr = np.asarray(seg, dtype=np.int32)[None]            # (1, Lp)

            # A single-document (or otherwise degenerate) fixture would make the whole Swift
            # P6 gate vacuous — assert this in the export path, not only by eye (the plan's
            # "Verification" step 5 rule).
            if len(set(seg)) < 2:
                raise SystemExit(
                    f"refusing to write {out_dir}/packed.safetensors: packed_doc_lengths="
                    f"{doc_lens!r} produced only {len(set(seg))} distinct document id(s) — "
                    "the anti-no-op gate would be vacuous")
            expected_len = sum(((n + Q - 1) // Q) * Q for n in doc_lens)
            assert seg_arr.shape[1] == expected_len, (
                f"packed sequence length {seg_arr.shape[1]} != expected {expected_len} for "
                f"doc_lengths={doc_lens!r}, chunk_size={Q}")

            packed_logits = np.asarray(
                ref_model.forward(packed_arr, seg_arr), dtype=np.float32)   # (1, Lp, V)

            _save_finite({
                "packed_tokens": packed_arr,
                "packed_seg_ids": seg_arr,
                "doc_lengths": np.asarray(doc_lens, dtype=np.int32),
                "packed_logits": packed_logits,
            }, str(out / "packed.safetensors"))
            packed_meta_doc_lengths = doc_lens
            packed_meta_seq_len = int(seg_arr.shape[1])

        # --- generation.safetensors: the greedy-id parity oracle (#167 AC1) --------------------
        # `prompt_ids` is the first `min(8, L)` ids of the token batch's first row; `greedy_ids`
        # are `gen_steps` ids produced by driving `model.step` token-by-token (the same shape
        # `monica-parity` reproduces in Swift) with temperature=0 (argmax, first-max-on-tie).
        # #266: derived from the precision band (2x it — the `greedy_margin_floor`
        # doubling in src/conformance/tolerances.py), not hardcoded fp32 constants — a
        # fp16/bf16 fixture needs a wider margin for the SAME reason a quantized one does
        # below (a looser kernel/dtype tolerance can flip a near-tie argmax).
        rtol_gen, atol_gen = 2 * rtol_fp, 2 * atol_fp
        if quant_bits is not None:
            # Quantization shrinks logit margins, and Swift will decode this fixture through
            # the TRUE quantized kernel (not the dequantized fake-quant reference this
            # margin check runs against) — accumulation-order differences can plausibly flip
            # a near-tie argmax even where fp32-vs-fp32 would not. Require a much wider
            # margin here than the base fp32 parity band demands.
            atol_gen = max(atol_gen, 0.25)
        prompt_len = min(8, seq)
        prompt_ids = [int(t) for t in tokens[0, :prompt_len].tolist()]

        state = ref_model.init_state(1)
        logits = None
        for t in prompt_ids:
            logits, state = ref_model.step(np.array([t], dtype=np.int64), state)

        greedy_ids: list[int] = []
        margins: list[float] = []
        top1s: list[float] = []
        for _ in range(gen_steps):
            row = _np_f32(logits).reshape(-1)
            nxt = int(np.argmax(row))
            top1 = float(row[nxt])
            row2 = row.copy()
            row2[nxt] = -np.inf
            top2 = float(row2.max())
            margins.append(top1 - top2)
            top1s.append(top1)
            greedy_ids.append(nxt)
            logits, state = ref_model.step(np.array([nxt], dtype=np.int64), state)

        # A near-tie greedy argmax makes cross-implementation id equality ill-posed: two fp32
        # implementations that agree to 1e-4 relative can still cross a near-zero margin and pick
        # different tokens. Refusing to write a flaky fixture is the point — see the module
        # docstring's rationale and swift/engine/Fixtures/README.md.
        bad_steps = [i for i, (m, t1) in enumerate(zip(margins, top1s))
                    if m < atol_gen + rtol_gen * abs(t1)]
        if bad_steps:
            raise SystemExit(
                f"refusing to write {out_dir}: greedy margin at step(s) {bad_steps} is inside the "
                f"parity band (atol={atol_gen} + rtol={rtol_gen}*|top1|) — a near-tie argmax makes "
                "cross-implementation id equality flaky. Bump --seed or --gen-steps and retry "
                f"(margins={margins})")

        # --- prefill parity: assert the SessionStore/generate.py path agrees, id-for-id ---------
        # Ties the fixture to "the same token ids as scripts/generate.py greedy", not to an
        # exporter-private step loop.
        store = SessionStore(ref_model, max_concurrent=1)
        store.create("fixture")
        greedy_sampler = partial(sampling.sample, temperature=0.0)
        via_generate = py_generate(store, "fixture", prompt_ids, sampler=greedy_sampler,
                                   to_numpy=np.array, max_new_tokens=gen_steps)
        store.remove("fixture")
        if via_generate != greedy_ids:
            raise SystemExit(
                f"refusing to write {out_dir}: step-driven greedy_ids {greedy_ids} disagree with "
                f"src.serve.generate's prefill-based path {via_generate} — the two Python code "
                "paths (step-by-step vs one-shot prefill) must produce identical greedy ids "
                "(src/conformance/prefill_decode_parity.py's contract)")

        _save_finite({
            "prompt_ids": np.array(prompt_ids, dtype=np.int32),
            "greedy_ids": np.array(greedy_ids, dtype=np.int32),
            "margins": np.array(margins, dtype=np.float32),
        }, str(out / "generation.safetensors"))

        meta = {
            "config": os.path.basename(config_path),
            "batch": batch,
            "seq": seq,
            "seed": seed,
            "precision": cfg.precision,
            "vocab_size": cfg.vocab_size,
            "gen_steps": gen_steps,
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "moe_bias": moe_bias,
            "forward_step_max_abs_diff": verdict["max_abs_diff"],
            "prefill_decode_max_abs_diff": prefill_verdict["max_abs_diff"],
            "prefill_decode_max_abs_state_diff": prefill_verdict["max_abs_state_diff"],
            "greedy_margin_min": min(margins) if margins else None,
        }
        if quant_bits is None:
            meta["mixing_layers"] = mixing_layers
        if count_loads and moe_load_layers:
            meta["moe_load_layers"] = moe_load_layers
            meta["moe_route_margin_min"] = moe_route_margin_min
        if moe_load_omitted_reason is not None:
            meta["moe_load_omitted_reason"] = moe_load_omitted_reason
        if lowp_self_mean_kl is not None:
            # A MEASUREMENT, never a threshold — Swift reads thresholds only from its own
            # precisionBands/lowpKLMax table, never from a fixture-carried value (that
            # would reopen the "measure, then widen until green" loop #266 must not create).
            meta["lowp_self_mean_kl"] = lowp_self_mean_kl
        if packed_meta_doc_lengths is not None:
            # Human-readable only (#68/#263) — monica-parity's P6 section reads the shapes it
            # needs (doc_lengths, chunk_size) straight out of packed.safetensors/the loaded
            # model config, not from here.
            meta["packed_doc_lengths"] = packed_meta_doc_lengths
            meta["packed_seq_len"] = packed_meta_seq_len
        if quant_bits is not None:
            q = check_quant_parity(fp_forward_logits, forward_logits, bits=quant_bits)
            # A quantized fixture carries its OWN rtol/atol (#168's minimal #266 hook) —
            # deliberately looser than the fp32 gate's 1e-4/1e-5, and never applied to a
            # fixture that omits them (monica-parity defaults absent rtol/atol to the fp32
            # constants unchanged).
            meta["rtol"] = 2e-2
            meta["atol"] = 2e-2
            meta["quant_bits"] = quant_bits
            meta["quant_group_size"] = quant_group_size
            meta["quant_targets"] = sorted(targets)
            meta["quant_top1_agreement"] = q["top1_agreement"]
            meta["quant_mean_kl"] = q["mean_kl"]
            meta["quant_max_abs_drift"] = q["max_abs_drift"]
            meta["quant_ok_vs_thresholds"] = q["ok"]
            if not q["ok"]:
                raise SystemExit(
                    f"refusing to write {out_dir}: quantized-vs-fp quality gate failed "
                    f"(top1={q['top1_agreement']:.4f} kl={q['mean_kl']:.4f}) — try "
                    "--quant-head-bits 8, a larger --quant-group-size, or a different seed")
        if train_steps > 0:
            if quant_bits is not None:
                raise SystemExit(
                    "--train-steps and --quant-bits are mutually exclusive: training a "
                    "quantized checkpoint is out of scope (packed uint32 codes have no "
                    "meaningful weight gradient) — see the plan's fixture-selection table")
            # #195: the K-step training trajectory oracle. Built from a PRISTINE RELOAD of
            # the weights just written above — critical ordering, so the inference oracles
            # above are computed on untrained weights (the exporter must never train the
            # model the logit fixtures were exported from).
            meta.update(_export_train_oracle(cfg, weights_path, tokens, out=out,
                                             train_steps=train_steps, seed=seed))
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        return meta
    finally:
        mx.set_cache_limit(_prev_cache_limit)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a config/*.yaml")
    p.add_argument("--out", required=True, help="fixture directory to write")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=40)
    p.add_argument("--precision", default=None, choices=["fp32", "fp16", "bf16"],
                   help="override the config's precision (the harness gates fp32)")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="override the config's vocab_size (#167's toy-gen: a "
                        "monica-tokenize-trained tokenizer's vocab needs a wider model than "
                        "toy.yaml's default 256)")
    p.add_argument("--gen-steps", type=int, default=16,
                   help="greedy-decode steps written to generation.safetensors (#167 AC1)")
    p.add_argument("--moe-bias", action="store_true",
                   help="activate a per-expert route bias (#213) so the fixture carries "
                        "moe_route_bias.* keys and exercises the biased-ranking branch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quant-bits", type=int, default=None, choices=(2, 4, 8),
                   help="#168: write a QUANTIZED fixture — weights.safetensors holds the "
                        "packed checkpoint + quant sidecar, and the logit/hidden/"
                        "generation references come from the fake-quant (dequantized) "
                        "model rather than the fp one")
    p.add_argument("--quant-group-size", type=int, default=64)
    p.add_argument("--quant-head-bits", type=int, default=None,
                   help="override bits for the tied embedding/head; defaults to 8 "
                        "automatically when --quant-bits 4 (see quant_targets)")
    p.add_argument("--train-steps", type=int, default=0,
                   help="#195: write train.safetensors, a K-step Swift/MLX train_step "
                        "trajectory oracle (loss/grad_norm/grad-tree/post-step-weights). "
                        "0 (default) writes no training oracle, unchanged from before #195")
    p.add_argument("--packed-doc-lengths", default=None,
                   help="#68/#263: write packed.safetensors (the Swift monica-parity P6 "
                        "packing-aware seg_ids oracle) with these comma-separated, "
                        "chunk-length-relative doc lengths, e.g. 'Q,2*Q,5' — 'Q' resolves "
                        "to the config's chunk_size (default 64). Ignored for quantized "
                        "fixtures (--quant-bits set)")
    p.add_argument("--allow-moe-load-omit", action="store_true",
                   help="#266 fallback ladder: if the MoE route-margin refusal would fire, "
                        "omit load.{i}/moe_load_layers instead of raising, and record why "
                        "in meta.json's moe_load_omitted_reason — only use after a --seed "
                        "search and --moe-bias have both failed to clear the margin guard")
    args = p.parse_args()

    quant_head_bits = args.quant_head_bits
    if quant_head_bits is None and args.quant_bits == 4:
        quant_head_bits = 8

    meta = build_fixture(args.config, args.out, batch=args.batch, seq=args.seq,
                         precision=args.precision, moe_bias=args.moe_bias, seed=args.seed,
                         vocab_size=args.vocab_size, gen_steps=args.gen_steps,
                         quant_bits=args.quant_bits, quant_group_size=args.quant_group_size,
                         quant_head_bits=quant_head_bits, train_steps=args.train_steps,
                         packed_doc_lengths=args.packed_doc_lengths,
                         allow_moe_load_omit=args.allow_moe_load_omit)
    print(f"wrote {args.out}: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
