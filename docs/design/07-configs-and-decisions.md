# Configs & locked decisions

[← Index](README.md)

Two configs are the single source of truth for model dimensions and run parameters,
loaded into [`MambaConfig`](../../src/model/blocks.py): `toy.yaml` (correctness /
smoke) and `poc.yaml` (the ~100M scale run). The comments in these files *are* the
decision record — reproduced verbatim below.

## `config/toy.yaml`

```yaml
# Toy config for the milestone-1..4 smoke test.
# Tiny + fp32 so fixed-seed resume is exactly reproducible.
d_model: 64
n_layers: 2
d_state: 16            # SSM state width N (per head)
expand: 2              # d_inner = 128
d_conv: 4
head_dim: 16           # Mamba-2/SSD: 128/16 = 8 heads (scalar A per head)
dt_rank: auto

vocab_size: 256        # byte fallback tokenizer for offline smoke testing
seq_len: 128
tie_embeddings: true

precision: fp32        # correctness first; trivial exact resume
chunk_size: null       # SSD chunk length Q (null -> backend default 64)
grad_checkpoint: false # tiny model -> not needed; keep smoke exact-resume cheap

# dt-projection bias init (load-bearing)
dt_min: 0.001
dt_max: 0.1
dt_init_floor: 0.0001
```

The toy config exists to make the [smoke gate](06-smoke-gate-and-eval.md) fast and
**bit-exact**: tiny dims, `fp32` (so fixed-seed resume is reproducible), and
`vocab_size: 256` to run on the byte-fallback tokenizer with no network. `head_dim 16`
gives 8 heads — enough timescale spread for the dt-init recall test.

> Note on `chunk_size: null`: it means *the backend's default chunk length* (the MLX
> SSD scan uses **64**), not an unchunked pass. The SSD scan is overflow-safe by
> construction — see [the SSD scan](02-model-ssm.md#the-ssd-chunked-matmul-scan).

## `config/poc.yaml`

```yaml
# ~100M POC config. Target ~3B tokens (~Chinchilla for 100M), seq length 1024.
# The tied embedding (vocab x d_model ~= 50280*768 ~= 38M) is a large share of the
# budget -> tie_embeddings MUST stay true. d_model 768 x 24 layers lands near 100M.
d_model: 768           # d_inner = expand*d_model = 1536
n_layers: 24
d_state: 16            # SSM state width N (per head, shared B/C group)
expand: 2
d_conv: 4
head_dim: 64           # Mamba-2/SSD: 1536/64 = 24 heads (scalar A per head)
dt_rank: auto

vocab_size: 50280      # CONFIRMED: allenai/OLMo-7B-hf, vocab 50280 < 65536 (uint16)
seq_len: 1024
tie_embeddings: true

# CONFIRMED ON MLX (M1 micro-benchmark): fp16 ~3.96 TFLOP/s vs bf16 ~3.36 and
# fp32 ~3.40 on this Metal GPU -> fp16 is ~18% faster; bf16 gives no speedup.
# Use fp16 + loss scaling for the scale run (toy/smoke stay fp32 for exact resume).
precision: fp16
chunk_size: null       # SSD chunk length Q (null -> backend default 64)
grad_checkpoint: true  # REQUIRED at depth: recompute layers in backward so the
                       # 24-layer fp16 backward fits in unified memory (else it swaps)

# dt-projection bias init (load-bearing)
dt_min: 0.001
dt_max: 0.1
dt_init_floor: 0.0001
```

> Note on `chunk_size: null`: it means the backend's default SSD chunk length (**64**),
> not an unchunked pass. The migration to **Mamba-2 / SSD** (scalar A) plus
> `grad_checkpoint` is what makes the poc step fit in memory and run fast — see
> [the SSD scan](02-model-ssm.md#the-ssd-chunked-matmul-scan) and
> [why scalar A](02-model-ssm.md#why-scalar-a-mamba-2).

## The decisions, distilled

### Sizing: ~100M params, ~3B tokens

`d_model 768 × 24 layers` lands near 100M parameters; the target ~3B tokens is
roughly Chinchilla-optimal for that size. `seq_len 1024` runs the [SSD
scan](02-model-ssm.md#the-ssd-chunked-matmul-scan) with the default chunk length
`Q = 64`; an explicit `chunk_size` is only needed to tune that. `head_dim 64` splits
`d_inner = 1536` into 24 scalar-A heads.

### Tied embedding is mandatory at scale

The embedding matrix is `vocab × d_model ≈ 50280 × 768 ≈ 38M` — roughly a third of
the ~100M budget. Tying the input and output embeddings (rather than learning a
separate LM head) is therefore not a tuning knob but a requirement; see
[model](02-model-ssm.md) for the tied-head implementation.

### Precision: fp16 for poc, fp32 for toy/smoke

The fp16-vs-bf16 question was settled empirically on MLX in M1 (issue #3), not
assumed. The benchmark — reproducible in-repo via `scripts/bench_precision.py`, which
times the poc forward GEMM workload (per-layer in/x/out projections + the tied head)
in each dtype — on this Metal GPU: **fp16 ~4.37 TFLOP/s vs bf16 ~3.78 and fp32 ~3.33**.
fp16 is **~16% faster than bf16** (and ~31% over fp32); bf16 buys only a small edge on
fp32. So the scale run uses **fp16 + loss scaling** (the loss-scaling machinery lives
in [training](05-training.md); the precision→scaler wiring is `scaler_for_precision`
in `src/train/loss_scale.py`), while toy/smoke stay **fp32** for exact resume. Note
this contradicts the common assumption that bf16 is the safe default — on Metal it
isn't. Re-run `python scripts/bench_precision.py` on new hardware to re-confirm.

### Vocab is locked to uint16

`vocab_size: 50280` (OLMo-7B-hf) is confirmed `< 65536`, the bound required by the
[uint16 packing](04-data-pipeline.md) and enforced by `MambaConfig.validate()`.

### dt-bias parameters are shared

`dt_min` / `dt_max` / `dt_init_floor` appear in **both** configs identically — the
[load-bearing dt-bias init](02-model-ssm.md) is a model-wide decision carried into
every backend, not a per-run tuning choice.

## Related

- [Model: the Mamba block + selective SSM](02-model-ssm.md) — what these dims build.
- [Training](05-training.md) — how fp16 loss scaling is applied.
- [Data pipeline](04-data-pipeline.md) — the uint16 / vocab link.
- [GitHub issue #2](https://github.com/travisgalloway/monica/issues/2) — the milestone tracker.
