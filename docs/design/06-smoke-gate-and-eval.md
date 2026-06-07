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

## OLMES / lm-eval is deferred (Tier-2)

Standardized benchmarks are explicitly out of scope for the POC. From
[`src/eval/olmes_adapter.py`](../../src/eval/olmes_adapter.py):

> Tier-2 evaluation: OLMES / lm-evaluation-harness adapter — STUB (deferred).
>
> Evaluating a custom MLX Mamba requires implementing the harness's model class (the
> loglikelihood-style methods) ... This is its own milestone-sized task, NOT wiring.
>
> Known trap: off-by-one errors in loglikelihood token indexing. ... For a 100M
> model, absolute scores will be poor — judge the harness by whether it runs end to
> end, not by leaderboard position.
>
> Deferred for the POC (success = Tier-1 val perplexity).

So: Tier-1 (val perplexity) defines POC success; Tier-2 (OLMES) is a later,
milestone-sized effort with a documented indexing pitfall, and a reminder that a 100M
model's job is to *run* the harness, not top a leaderboard.

## Related

- [Training](05-training.md) — the checkpoint machinery the gate stresses.
- [Data pipeline](04-data-pipeline.md) — the disjoint val shard eval reads.
- [Conformance](03-conformance.md) — the other correctness guard.
