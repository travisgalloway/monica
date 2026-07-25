# Smoke gate & eval

[← Index](README.md)

## The gate: resume must be exact

[`scripts/smoke_test.py`](../../scripts/smoke_test.py) is the milestone-4 gate — the
project's most important test. From its docstring:

> The single most important test in the project. Most projects silently break at
> checkpoint resume and dataloading, not in the model. Do NOT proceed past this gate
> until resume is verifiably exact and eval runs.

The insight: a model can be correct and a project still fail, because the failure is
usually in the *plumbing* — checkpoint resume and the data loader — not the math. So
the gate targets exactly that plumbing.

### Procedure

From the docstring (toy model, tiny data, fixed seed, fp32 ⇒ effectively exact):

> 1. Reference run: train N steps uninterrupted; record the loss trajectory.
> 2. Interrupted run (same seed, same fixed batch stream, same LR schedule): train
>    N/2 steps, SAVE portable weights + a within-backend resume bundle (optimizer
>    state + step), tear the model/optimizer down, REBUILD, LOAD, and train the rest.
> 3. Assert the post-resume trajectory matches the reference within tolerance.
> 4. Run a held-out val-perplexity eval end to end.

Determinism is engineered, not assumed:

> we drive training over a PRE-MATERIALIZED fixed batch list so the batch at global
> step s is identical in both runs (independent of where the "kill" falls) — the
> resume exactness check would otherwise be confounded by data ordering.

If the post-resume max loss diff exceeds `atol` (default 1e-4), the script exits
non-zero with `SMOKE TEST FAILED`. fp32 + fixed seed makes a correct resume
effectively bit-exact, so this is a real pass/fail, not a fuzzy threshold.

### Verified result

Run on Apple Silicon (macOS arm64, Python 3.14.3, mlx 0.31.2):

```
[reference] step0 loss=8.06889  step49 loss=1.53557
[resumed]   resumed at step=25  step49 loss=1.53557
[match] post-resume max|loss diff| over steps 25..49 = 1.192e-07
[eval] val_loss=1.5416  val_perplexity=4.6721
SMOKE TEST PASSED ✅  resume is exact and eval runs.
```

Resume matches to ~1e-7 (far under the 1e-4 gate), loss drops 8.07 → 1.54 over 50
steps, and held-out eval runs end to end. This single run exercises the whole M1–M4
stack: MLX model `forward`+`step`, the injected `train_step` + loop, the
two-concern [checkpoint](05-training.md), and [eval](#eval-the-success-metric).

## Eval: the success metric

The POC has no external benchmark requirement. From
[`src/eval/val_loss.py`](../../src/eval/val_loss.py):

> Tier-1 evaluation: held-out validation loss / perplexity.
>
> This is the primary pipeline-health signal for the POC: a smoothly decreasing val
> perplexity IS the success criterion (no external harness needed). The numeric core
> (`cross_entropy`, `perplexity`) is pure numpy and testable anywhere; `evaluate`
> orchestrates it over a loader using only `ModelInterface.forward`.

`evaluate` weights each batch's mean cross-entropy by its token count, so a smaller
final batch (`drop_last=False`) doesn't bias the result. The numeric core is
backend-free numpy; `to_numpy` converts backend logits at the boundary.

## OLMES / lm-eval is implemented, but stays Tier-2

**Status: built.** This section previously described
[`src/eval/olmes_adapter.py`](../../src/eval/olmes_adapter.py) as a deferred stub. It is not —
the adapter implements the harness's model class over `ModelInterface.forward`, and
`scripts/eval_olmes.py` runs loglikelihood and generative tasks end to end. It keeps the same
split as `val_loss`: a pure-numpy scoring core (`score_continuation`,
`disjoint_rolling_windows`) that is testable anywhere, and a thin lm-eval shell built by
`make_lm_eval_adapter`. lm-eval is a heavy optional dependency (some versions pull in torch),
so it is imported **only inside the factory** — which is how the module stays above the seam.

The documented indexing trap is real and is what the numpy core exists to pin down: `forward`
logits at position `i` predict the token at `i+1`, so the model input is `(ctx + cont)[:-1]`
and the continuation is scored by the **last `len(cont)` logit rows**.

What keeps OLMES at **Tier-2** is no longer implementation effort but the finding: at POC scale
scores land near chance (confirmed on the completed ~205M `poc-qwen` run). So **Tier-1 (val
perplexity / BPB) defines POC success**, and the harness's job is to *run*, not to place on a
leaderboard. For the M12 code model, **BPB** is the primary small-model metric
([13-code-model-moe.md](13-code-model-moe.md)) and the code-specific suite is #221.

## Related

- [Training](05-training.md) — the checkpoint machinery the gate stresses.
- [Data pipeline](04-data-pipeline.md) — the disjoint val shard eval reads.
- [Conformance](03-conformance.md) — the other correctness guard.
