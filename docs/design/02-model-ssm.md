# Model: the Mamba block + selective SSM

[← Index](README.md)

The model is a **Mamba-2 / SSD** block (Dao & Gu, *State Space Duality*): a
**scalar-A** selective state-space model with input-dependent B, C, and delta,
multi-head with one shared B/C group. Scalar A (one decay per head, not a per-state
diagonal) is the restriction that turns the scan into matmuls — see [the migration
note](#why-scalar-a-mamba-2). Config lives in
[`src/model/blocks.py`](../../src/model/blocks.py); the MLX implementation in
[`src/model/mlx_backend.py`](../../src/model/mlx_backend.py).

## Block dataflow

From the `src/model/blocks.py` module docstring:

```
input projection
  -> split into `main` and `gate`
  -> short causal depthwise conv on `main` (width `d_conv`)
  -> SiLU
  -> selective SSM (Mamba-2 / SSD: scalar A per head; input-dependent B, C, delta;
     chunked-matmul scan)
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

> * `parallel(x)`: the SSD chunked-matmul scan over the full sequence (training
>   path). Intra-chunk via matmul, a short recurrence across chunk-states. All decays
>   are exp of non-positive sums, so it is overflow-safe; chunk length Q comes from
>   `chunk_size`.
> * `recurrence(x, h)`: one-step state update (inference path).

### The SSD chunked-matmul scan

`d_inner` is split into `n_heads = d_inner // head_dim` heads of width `P = head_dim`.
Each head has a **scalar** decay `A = -exp(A_log)` (shape `(n_heads,)`); `B` and `C`
are a single group of width `N = d_state`, shared across heads. The per-head log-decay
is `g = delta * A` (`<= 0`). The scan ([Dao & Gu SSD, part 3](https://tridao.me/blog/2024/mamba2-part3-algorithm/))
splits the sequence into chunks of length `Q` and runs four steps — three of them
matmuls (tensor-core/Metal-friendly), one short recurrence:

```
1. intra-chunk (diagonal): Lmask = exp(segsum(g));  Y_diag = (Lmask ∘ CBᵀ) · Xin
2. chunk-final states:      states = Σ decay·Xin·B          (each chunk's end state)
3. inter-chunk recurrence:  carry states across chunks (the only scan, length nc)
4. off-diagonal:            Y_off = C · S_enter · exp(cumsum g)
Y = Y_diag + Y_off
```

`segsum` builds the lower-triangular log-decay mask `seg[i,j] = Σ_{j<k≤i} g_k`; its
`exp` is the within-chunk 1-semiseparable decay matrix (upper triangle `-inf → 0`,
enforcing causality). **Overflow-safety is structural**: every decay is `exp` of a sum
of non-positive `g`, so it lies in `[0, 1]` — no `exp(-A_cum)` term that can blow up
(the failure mode of the old diagonal-A cumsum scan). The sequence is padded up to a
multiple of `Q` (padded steps carry zero input, trimmed from the output). `chunk_size`
(`MambaConfig`) sets `Q`; `null` → the backend default of **64**.

### Why scalar A (Mamba-2)

The original POC used **Mamba-1** diagonal A (`A` shape `(d_inner, d_state)`). Its
training backward retained the full `(B, L, d_inner, d_state)` scan intermediates for
every layer at once — at poc scale (24 layers, seq 1024) ~76 GB, which swapped on a
32 GB M4 and made a step take ~180 s. **SSD's matmul form requires scalar A** (`A =
aI` per head) — that restriction is what collapses the per-`(channel,state)` decay
into a shared matmul. Migrating to scalar-A Mamba-2 (plus [gradient
checkpointing](#memory-gradient-checkpointing)) is what makes the scale run feasible.

## Scalar-A initialization

`A = -exp(A_log)`, with `A_log = log(1..n_heads)` — one decay per head, the **S4D-real**
init carried to the Mamba-2 head layout. From `mlx_backend.py`:

> Scalar decay A per head, stored as log: A = -exp(A_log). S4D-real init.

## Memory: gradient checkpointing

The training `forward` optionally wraps each layer in `mlx.nn.utils.checkpoint`
(`grad_checkpoint` config), recomputing the layer in the backward pass instead of
retaining its activations. Combined with the SSD scan this keeps the 24-layer poc
backward within unified memory (without it, it swaps). Inference (`step`) is
unaffected.

## The load-bearing dt-bias

The single most important initialization in the model. From
`SelectiveSSM._init_dt_bias`:

> LOAD-BEARING dt-projection bias init (inverse-softplus into a small positive
> range). Without this the model fails to learn recall. Now PER-HEAD (shape n_heads).
>
> ```
> dt   = uniform(log(dt_min), log(dt_max)).exp().clamp(min=dt_init_floor)
> bias = dt + log(-expm1(-dt))          # inverse softplus
> ```

The `dt` projection passes through softplus at runtime; initializing its bias (one
value **per head** in Mamba-2) to the inverse-softplus of a log-uniform sample in
`[dt_min, dt_max]` puts the initial timescales in a usable range. The parameters (`dt_min=1e-3`, `dt_max=1e-1`,
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
- `ssm_state`: `(B, n_heads, head_dim, d_state)` — the per-head SSM hidden state

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
