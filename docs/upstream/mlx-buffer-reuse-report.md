# Upstream report — MLX silently corrupts a later, unrelated computation after a lazy array is forked to two consumers

**Prepared 2026-08-22 for `ml-explore/mlx`. NOT FILED.**

This document is a maintainer-actionable bug report, written and checked in so that filing it is a
one-step decision for a human. **It has not been posted to `ml-explore/mlx`, or to any other
third-party project, and must not be posted autonomously.** Opening an issue on someone else's
repository is a communication act with the project's maintainers, not a code change; that call
belongs to this repository's owner. If you are that person: the sections below are written in
issue-report form and can be pasted as-is.

Tracking issue in this repository: [#298](https://github.com/travisgalloway/monica/issues/298).
Full measurement record: [`../design/14-inference-engine.md`](../design/14-inference-engine.md),
section *"MLX 0.32.0 buffer reuse (#298)"* §§D1–D6.

---

## Summary

On Apple Silicon, calling an accessor that forks a **lazy** array into two independent consumers
without materialising it first causes a **later, unrelated** computation on the same model to
return wrong values. There is no exception, no log line, and no shape change — only the numbers are
wrong, and they are wrong catastrophically rather than marginally (fp32 logit `max|d|` of **6–9**;
argmax token ids off by 222 out of a 256-vocab).

The failure is **deterministic within a process and varies between processes**, so a rerun
"fixes" it. In CI it is indistinguishable from flake.

It reproduces on **0.31.2, 0.32.0 and 0.32.1** at the same order of magnitude, so it is not a
recent regression and there is no released version to upgrade or downgrade to.

## Severity, and why this is worth an upstream fix

Silent wrong results are the worst failure mode a numerical library has: every downstream check
agrees with the wrong number. In this project the concrete consequence is that a *checked-in
oracle* — the frozen reference a Swift port is gated against — can be generated from a corrupted
computation and then permanently enshrine the corrupted values. We guard against that with a
two-fresh-process byte-for-byte export check, but that is a workaround for consumers, not a fix.

## Environment

| | |
|---|---|
| Chip | Apple M1 Pro |
| OS | macOS 26.5.2 (arm64) |
| Python | 3.14.6 |
| MLX | 0.32.0 (installed); also reproduced on 0.31.2 and 0.32.1 |
| MLX releases checked | 0.31.2, 0.32.0, 0.32.1. **0.32.1 (2026-08-18) is the newest release on both GitHub and PyPI as of 2026-08-22; there is no newer version, tested or untested.** |
| Repo commit these numbers come from | `c57284ec05a31876189f1c054e862972e3f26b70` (branch `feat/298-mlx-buffer-reuse-corruption`) |
| Second host | A hosted GitHub Actions `macos-latest` runner — see §"Second host" below |

## What the code does

The model is a Mamba-2/SSD hybrid built on `mlx.nn`. The triggering accessor walks the layer stack
and, for each Mamba layer, feeds the *same lazy* hidden state `h` to **two** consumers before `h`
has been evaluated:

```python
h = cast(model.embedding(mx.array(tokens)), dtype)
out = []
for i, (layer, layer_fn) in enumerate(zip(model.layers, model._layer_fns)):
    if is_mamba_layer(i):
        out.append((i, layer.mixing_matrix(h).mean(axis=1)))   # consumer A (lazy)
    h = layer_fn(h)                                            # consumer B (lazy)
```

Then, **on the same model and with no further accessor calls**, an ordinary prefill runs:

```python
logits, _ = model.prefill(mx.array(tokens), last_only=True)
mx.eval(logits)
```

`logits` is wrong. The values read out of `mixing_matrix` are *not* the ones that come back wrong —
the corruption lands on the later, unrelated call.

Inserting a single `mx.eval(h)` before the fork reduces the rate substantially but **does not
eliminate it** (see §Remedy). What eliminates it on this host is disabling MLX's buffer cache.

## Reproduction

The reproducer is currently expressed against this project's model rather than as a standalone
script. **Reducing it to an MLX-only snippet was deliberately left undone** — the search for a
minimal graph is unbounded research, and this report would rather be accurate about what was
actually measured than approximate about a smaller case. What a minimisation attempt needs to
preserve, based on what we did measure:

- Repeated construction of **freshly initialised models** in one process (a single model reused
  does not reproduce; the allocator has to be recycling buffers).
- A **lazy** array forked to two consumers, evaluated only afterwards.
- A **shape-dependent** trigger. On this host `toy-moe`/`toy-hybrid` configs at `seq=129` reproduce
  and `toy` at `seq=512` does not, so a minimisation that lands on the wrong shape will read as
  "cannot reproduce" when the defect is still present. Any minimisation attempt must carry a
  positive control for exactly this reason.
- MLX's **buffer cache enabled** (the default). `mx.set_cache_limit(0)` suppresses it entirely.

The measuring instrument is `scripts/probe_mlx_buffer_reuse.py` in this repository. It compares
every trial against an **accessor-free reference** (not against trial 0 — with the accessor in the
loop roughly half the trials *including trial 0* are corrupt, so a trial-0 reference undercounts by
~2×), re-computes the reference at the end of the run, and reports `BLIND` rather than "clean" if
the reference moved or the positive control failed to fire.

```bash
# rates per MLX version (fresh venv per version)
python -m venv .venv-mlx031 && .venv-mlx031/bin/pip install "mlx==0.31.2" numpy safetensors pyyaml
PYTHONPATH=. .venv-mlx031/bin/python scripts/probe_mlx_buffer_reuse.py \
    --pattern mixing --trials 40 --seq 129 --config config/toy.yaml

# the shape sweep on the installed version
PYTHONPATH=. .venv/bin/python scripts/probe_mlx_buffer_reuse.py \
    --pattern mixing --trials 40 --seq 129 --config config/toy-moe.yaml

# with the mx.eval(h) barrier in place — still corrupts
PYTHONPATH=. .venv/bin/python scripts/probe_mlx_buffer_reuse.py \
    --pattern mixing --trials 40 --seq 129 --config config/toy.yaml --barrier

# with the buffer cache disabled — clean
PYTHONPATH=. .venv/bin/python scripts/probe_mlx_buffer_reuse.py \
    --pattern mixing --trials 40 --seq 129 --config config/toy.yaml \
    --barrier --cache-limit zero
```

## Measured rates

Corrupt trials out of 40, barrier removed, buffer cache on (M1 Pro, 2026-08-19):

| mlx | `toy` | `toy-hybrid` | `toy-moe` |
|-----|-------|--------------|-----------|
| 0.31.2 (2026-04-22) | 35/40 | 16/40 | 14/40 |
| 0.32.0 (2026-07-07) | 30/40 | 18/40 | 11/40 |
| 0.32.1 (2026-08-18) | 27/40 | 24/40 | 16/40 |

Same order across all three released versions.

## Remedy and its cost (what we do downstream)

| mitigation | corrupt trials / 40 |
|---|---|
| none | 27–35 (`toy`, version-dependent) |
| `mx.eval(h)` before the fork | 18/40 (`toy`), 20/40 (`toy-moe`), 0/40 (`toy-hybrid`) |
| `mx.eval(h)` **+** `mx.set_cache_limit(0)` | **0/40** on all three configs, on 0.32.0 and 0.31.2 |

So the barrier alone is **not** sufficient; disabling the buffer cache is what actually holds.

Cost of `set_cache_limit(0)`, measured as forward + one AdamW training step, 30 iterations:

| config | cache on | limit 0 | slowdown |
|---|---|---|---|
| `toy` | 0.317 s | 0.358 s | **+13.0%** |
| `toy-moe` | 0.477 s | 0.623 s | **+30.5%** |

Toy scale overstates the penalty for a real run (allocator cost dominates where allocation dominates
compute), but a double-digit throughput cost is why we confine the mitigation to oracle-writing
paths rather than applying it globally.

## Second host

The same probe runs on a hosted GitHub Actions `macos-latest` runner via the
`mlx-buffer-reuse-probe` job in `.github/workflows/scheduled-parity.yml` (`--report-only`, so a
runner that does not reproduce is recorded as **BLIND**, never as clean). Its numbers, with runner
image, MLX version and run id, are in
[`../design/14-inference-engine.md`](../design/14-inference-engine.md) §D6.

## Upstream search — what was searched, and the result

Re-run **2026-08-22** (a first pass was run 2026-08-19 and is recorded in §D1). Queries via
`gh api search/issues -f q="repo:ml-explore/mlx <query>"`, covering issues **and** pull requests,
open and closed:

| query | total | nearest hits |
|---|---|---|
| `buffer cache corruption` | 8 | #3856 (quantized-MoE silent corruption, open), #3186 (kernel panic), #3866 (int32 corruption under `async_eval` + in-place slice updates, closed) |
| `set_cache_limit` | 6 | #3350 (cache pool retains unusable buffers — a *memory-growth* bug, closed), #3566, #3554, #3849 |
| `allocator reuse wrong results` | 4 | #3912 (fp quantized matmul corruption, open) |
| `silent wrong results` | 42 | #4370 (`mx.dequantize` wrong on a non-contiguous slice, regression on 0.32.1, open), #4261 (`gather_mm`, closed), #3979 (conv2d all-zero output, open) |
| `hazard tracking` | 5 | #3461 (buffer destroyed mid-flight under *untracked hazard mode with custom kernels*, closed), #1496, #1509 |
| `untracked hazard` | 2 | #3462, #3461 (both closed) |
| `lazy graph fork wrong results` | 0 | — |
| `nondeterministic results same input` | 0 | — |

**Result: no upstream issue matches this defect.** Every near neighbour is a different bug — all of
them are either quantization-specific (#3856, #3912, #4370), custom-kernel/`async_eval`-specific
(#3461/#3462/#3866), operator-specific (#4261, #3979), or about memory growth rather than
correctness (#3350). This repro is fp32, unquantized, no custom kernels, no `async_eval`.

Two hits are **new since the 2026-08-19 pass** and were re-checked against this defect: #4370
(`mx.dequantize`, 0.32.1 regression) and #3856 (quantized MoE, sequence length % 32). Neither
applies — no quantization is involved here, and the failure predates 0.32.1.

Releases checked for a fix in the notes: **v0.32.0** and **v0.32.1**. No newer release exists.

## Status of filing

**Not filed.** No issue, discussion, or pull request has been opened on `ml-explore/mlx` or any
other third-party repository from this project. Posting this report is a maintainer decision; when
it is made, this section should be replaced with the upstream issue link and #298 updated to track
it.
