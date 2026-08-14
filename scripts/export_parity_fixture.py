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
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

from src.conformance.forward_step_parity import check_forward_step_parity
from src.conformance.prefill_decode_parity import check_prefill_decode_parity
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


def build_fixture(config_path: str, out_dir: str, *, batch: int, seq: int,
                  precision: str | None = None, moe_bias: bool = False,
                  seed: int = 0, vocab_size: int | None = None,
                  gen_steps: int = 16, quant_bits: int | None = None,
                  quant_group_size: int = 64, quant_head_bits: int | None = None,
                  train_steps: int = 0) -> dict:
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
    from safetensors.numpy import save_file

    from src.eval.quantize import (
        dequantize_portable_state_dict, quant_targets, quantize_portable_state_dict)
    from src.model.mlx_backend import AttentionBlock, MambaBlock, MLXMambaModel, MoEBlock
    from src.serve import sampling
    from src.serve.generate import generate as py_generate
    from src.serve.sessions import SessionStore
    from src.train.checkpoint import save_weights

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
        fp_forward_logits = np.array(model.forward(tokens), dtype=np.float32)
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

    # A fixture that fails PYTHON's own forward/step gate must never reach disk — it would
    # make the Swift gate compare against a reference that is itself internally inconsistent.
    verdict = check_forward_step_parity(ref_model, tokens, to_numpy=np.array)
    if not verdict["ok"]:
        raise SystemExit(
            f"refusing to write {out_dir}: the Python model fails its own forward/step "
            f"parity check (max_abs_diff={verdict['max_abs_diff']:.3e})")

    # Same rule for #169's state handoff: a fixture whose own Python prefill state
    # disagrees with its stepped state (or whose prefill logits disagree with forward, or
    # whose prefill-then-decode disagrees with pure step-by-step) can never reach disk —
    # it would make the Swift gate compare against an internally-inconsistent oracle.
    prefill_verdict = check_prefill_decode_parity(
        ref_model, tokens, to_numpy=np.array, n_decode=min(4, seq - 1))
    if not prefill_verdict["ok"]:
        raise SystemExit(
            f"refusing to write {out_dir}: the Python model fails its own prefill/decode "
            f"parity check ({prefill_verdict})")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    weights_path = str(out / "weights.safetensors")
    if weights_sd is None:
        ref_model.save(weights_path)        # weights.safetensors + .config.json sidecar
    else:
        save_weights(weights_sd, weights_path, config=cfg, quant=quant_block)

    save_file({"tokens": tokens}, str(out / "inputs.safetensors"))

    forward_logits = np.array(ref_model.forward(tokens), dtype=np.float32)
    state = ref_model.init_state(batch)
    steps = []
    for t in range(seq):
        logits_t, state = ref_model.step(tokens[:, t], state)
        steps.append(np.array(logits_t, dtype=np.float32))
    step_logits = np.stack(steps, axis=1)

    ref = {"forward_logits": forward_logits, "step_logits": step_logits}
    if fp_forward_logits is not None:
        ref["fp_forward_logits"] = fp_forward_logits
    # Per-layer hidden states: a deliberate debugging investment. A whole-model logit
    # mismatch is nearly impossible to localize by hand across 24 layers; "first divergence
    # at layer 7" is the difference between an afternoon and a week. ~200 KB at toy scale.
    for i, h in enumerate(ref_model.hidden_states(tokens)):
        ref[f"hidden.{i}"] = np.array(h, dtype=np.float32)
    save_file(ref, str(out / "reference.safetensors"))
    if dequant_ref:
        save_file(dequant_ref, str(out / "dequant_ref.safetensors"))

    # --- prefill.safetensors: the state-handoff oracle (#169) -----------------------------
    # One entry per STATEFUL leaf, keyed by layer index and slot, derived from the layer's
    # CLASS (not tensor shapes) — so Swift, which iterates its OWN layers and demands
    # exactly the keys its own LayerState case implies, fails loudly on a missing key if the
    # two sides ever disagree about layer structure, instead of silently skipping a check.
    pre_logits, pre_state = ref_model.prefill(tokens)
    pre_last, _ = ref_model.prefill(tokens, last_only=True)
    prefill_out = {
        "prefill_logits": np.array(pre_logits, dtype=np.float32),
        "prefill_last_logits": np.array(pre_last, dtype=np.float32),
    }
    for i, (layer, st) in enumerate(zip(ref_model.layers, pre_state)):
        if isinstance(layer, MambaBlock):
            conv, ssm = st
            prefill_out[f"state.{i}.conv"] = np.array(conv, dtype=np.float32)
            prefill_out[f"state.{i}.ssm"] = np.array(ssm, dtype=np.float32)
        elif isinstance(layer, AttentionBlock):
            k, v = st
            prefill_out[f"state.{i}.k"] = np.array(k, dtype=np.float32)
            prefill_out[f"state.{i}.v"] = np.array(v, dtype=np.float32)
        elif isinstance(layer, MoEBlock):
            pass   # stateless — emitting a zero-sized placeholder would only add a
                   # round-tripping risk for no coverage; Swift's .moe case carries nothing.
        else:
            raise SystemExit(
                f"refusing to write {out_dir}: layer {i} has unrecognized type "
                f"{type(layer).__name__} — teach the exporter its state slot names")
    save_file(prefill_out, str(out / "prefill.safetensors"))

    # --- generation.safetensors: the greedy-id parity oracle (#167 AC1) --------------------
    # `prompt_ids` is the first `min(8, L)` ids of the token batch's first row; `greedy_ids`
    # are `gen_steps` ids produced by driving `model.step` token-by-token (the same shape
    # `monica-parity` reproduces in Swift) with temperature=0 (argmax, first-max-on-tie).
    rtol_gen, atol_gen = 1e-4, 1e-5
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
        row = np.array(logits, dtype=np.float32).reshape(-1)
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

    save_file({
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
    if quant_bits is not None:
        from src.conformance.quant_parity import check_quant_parity
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
        # #195: the K-step training trajectory oracle. Built from a PRISTINE RELOAD of the
        # weights just written above — critical ordering, so the inference oracles above
        # are computed on untrained weights (the exporter must never train the model the
        # logit fixtures were exported from).
        meta.update(_export_train_oracle(cfg, weights_path, tokens, out=out,
                                         train_steps=train_steps, seed=seed))
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


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
    args = p.parse_args()

    quant_head_bits = args.quant_head_bits
    if quant_head_bits is None and args.quant_bits == 4:
        quant_head_bits = 8

    meta = build_fixture(args.config, args.out, batch=args.batch, seq=args.seq,
                         precision=args.precision, moe_bias=args.moe_bias, seed=args.seed,
                         vocab_size=args.vocab_size, gen_steps=args.gen_steps,
                         quant_bits=args.quant_bits, quant_group_size=args.quant_group_size,
                         quant_head_bits=quant_head_bits, train_steps=args.train_steps)
    print(f"wrote {args.out}: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
