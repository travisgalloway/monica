# Model: the Mamba block + selective SSM

[← Index](README.md)

The model is the standard Mamba block (Gu & Dao): a diagonal selective state-space
model with input-dependent B, C, and delta. Config lives in
[`src/model/blocks.py`](../../src/model/blocks.py); the MLX implementation in
[`src/model/mlx_backend.py`](../../src/model/mlx_backend.py).

## Block dataflow

From the `src/model/blocks.py` module docstring:

```
input projection
  -> split into `main` and `gate`
  -> short causal depthwise conv on `main` (width `d_conv`)
  -> SiLU
  -> selective SSM (diagonal A; input-dependent B, C, delta; parallel scan)
  -> multiply by SiLU(gate)
  -> output projection
```

Each block is wrapped **pre-norm (RMSNorm) with a residual**. The full model is:
token embedding → N residual blocks → final RMSNorm → tied LM head.

## RMSNorm

A standard RMS normalization (`x * rsqrt(mean(x²) + eps) * weight`), chosen over
LayerNorm to match the Mamba reference and for cheaper compute (no mean-subtraction).

## Two compute paths, one result

The SSM is implemented twice and the two must agree (see
[conformance](03-conformance.md)). From the `mlx_backend.py` module docstring:

> * `parallel(x)`: a chunked closed-form selective scan over the full sequence
>   (training path). Chunking keeps the per-chunk cumulative decay bounded so `exp`
>   does not overflow — a global single-pass cumsum overflows fp32 even at modest
>   seq_len, so we always chunk (default chunk 32).
> * `recurrence(x, h)`: one-step state update (inference path).

### Why always chunk

The closed-form scan exponentiates a cumulative log-decay (`A_cum`). Over a long
sequence that cumulative sum grows without bound and `exp(A_cum)` overflows fp32 —
even at modest sequence lengths. Chunking bounds the working set so the decay inside
each chunk stays finite. The default chunk size is **32**; the per-chunk recurrence
carries state across chunk boundaries.

The closed-form per-chunk update (from `parallel`):

```
A_cum = cumsum(a_c)                              # inclusive log-decay
# h_j = exp(A_cum_j) * (h_carry + sum_{i<=j} exp(-A_cum_i) * bu_i)
inner = cumsum(exp(-A_cum) * bu_c)
h     = exp(A_cum) * (h_carry + inner)
```

`chunk_size` is also a `MambaConfig` field. From `blocks.py`:

> Chunked scan working-set bound. None => single-pass parallel scan, which is fine
> for seq_len up to ~2k. Set an int for long-context (prevents exp overflow).

(The `null` default in the configs still uses the internal default chunk of 32; an
explicit int is only needed for long-context inference.)

## Diagonal-A initialization

`A = -exp(A_log)`, with `A_log` initialized so `A = -(1..d_state)` broadcast across
channels — the standard **S4D-real** init. From `mlx_backend.py`:

> A = -exp(A_log). Init A = -(1..d_state) broadcast across channels (the standard
> "S4D-real" init).

## The load-bearing dt-bias

The single most important initialization in the model. From
`SelectiveSSM._init_dt_bias`:

> LOAD-BEARING dt-projection bias init (inverse-softplus into a small positive
> range). Without this the model fails to learn recall.
>
> ```
> dt   = uniform(log(dt_min), log(dt_max)).exp().clamp(min=dt_init_floor)
> bias = dt + log(-expm1(-dt))          # inverse softplus
> ```

The `dt` projection passes through softplus at runtime; initializing its bias to the
inverse-softplus of a log-uniform sample in `[dt_min, dt_max]` puts the initial
timescales in a usable range. The parameters (`dt_min=1e-3`, `dt_max=1e-1`,
`dt_init_floor=1e-4`) live in `MambaConfig` and are "carried into every backend" —
this is a model decision, not an MLX detail. This was verified empirically (issue
#5): on the toy model, loss decreases and long-range memory works.

## Causal depthwise convolution

`main` passes through a depthwise `Conv1d` (groups = channels) of width `d_conv`
before the SSM. It is made causal by padding `d_conv-1` on both sides and trimming to
the first `L` outputs:

> Causal depthwise conv: pad both sides (d_conv-1), keep the first L outputs.

In the recurrence path the conv is reconstructed from a small rolling window kept in
the state, so single-token inference matches the full-sequence conv exactly.

## State layout

`init_state` returns a per-layer list of `(conv_state, ssm_state)` tuples:

- `conv_state`: `(B, d_conv-1, d_inner)` — the rolling conv window
- `ssm_state`: `(B, d_inner, d_state)` — the SSM hidden state

This is the opaque blob the [seam](01-architecture-seam.md) snapshots and restores.

## Tied LM head

The output projection reuses the embedding matrix rather than learning a separate
head:

```python
def _head(self, h):
    if self._tie_embeddings:
        return h @ self.embedding.weight.T
    return self.lm_head(h)
```

At POC scale the embedding is ~38M of ~100M parameters, so tying is **mandatory**,
not optional — see [configs & decisions](07-configs-and-decisions.md). It also means
the portable state dict has no separate head param to reconcile when saving.

## Related

- [Architecture: the hardware seam](01-architecture-seam.md)
- [Conformance](03-conformance.md) — how `parallel` and `recurrence` are kept in agreement.
- [Configs & locked decisions](07-configs-and-decisions.md)
