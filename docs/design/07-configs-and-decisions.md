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
d_state: 16
expand: 2
d_conv: 4
dt_rank: auto

vocab_size: 256        # byte fallback tokenizer for offline smoke testing
seq_len: 128
tie_embeddings: true

precision: fp32        # correctness first; trivial exact resume
chunk_size: null       # seq_len well under ~2k -> single-pass scan

# dt-projection bias init (load-bearing)
dt_min: 0.001
dt_max: 0.1
dt_init_floor: 0.0001
```

The toy config exists to make the [smoke gate](06-smoke-gate-and-eval.md) fast and
**bit-exact**: tiny dims, `fp32` (so fixed-seed resume is reproducible), and
`vocab_size: 256` to run on the byte-fallback tokenizer with no network.

> Note on `chunk_size: null`: the inline comments here predate the backend default
> and read as "single-pass". In practice `null` means *the backend's default chunk
> size* (the MLX backend uses 32) — the scan is always chunked. See
> [why always chunk](02-model-ssm.md#why-always-chunk).

## `config/poc.yaml`

```yaml
# ~100M POC config. Target ~3B tokens (~Chinchilla for 100M), seq length 1024.
# The tied embedding (vocab x d_model ~= 50280*768 ~= 38M) is a large share of the
# budget -> tie_embeddings MUST stay true. d_model 768 x 24 layers lands near 100M.
d_model: 768           # d_inner = expand*d_model = 1536
n_layers: 24
d_state: 16
expand: 2
d_conv: 4
dt_rank: auto

vocab_size: 50280      # CONFIRMED: allenai/OLMo-7B-hf, vocab 50280 < 65536 (uint16)
seq_len: 1024          # <= ~2k -> chunking NOT required for the training run
tie_embeddings: true

# CONFIRMED ON MLX (M1 micro-benchmark): fp16 ~3.96 TFLOP/s vs bf16 ~3.36 and
# fp32 ~3.40 on this Metal GPU -> fp16 is ~18% faster; bf16 gives no speedup.
# Use fp16 + loss scaling for the scale run (toy/smoke stay fp32 for exact resume).
precision: fp16
chunk_size: null       # set an int only for long-context inference

# dt-projection bias init (load-bearing)
dt_min: 0.001
dt_max: 0.1
dt_init_floor: 0.0001
```

## The decisions, distilled

### Sizing: ~100M params, ~3B tokens

`d_model 768 × 24 layers` lands near 100M parameters; the target ~3B tokens is
roughly Chinchilla-optimal for that size. `seq_len 1024` is comfortably under the ~2k
limit, so `chunk_size` can stay `null` — meaning the backend's default chunk (32),
not an unchunked pass; the scan is [always chunked](02-model-ssm.md#why-always-chunk).
An explicit `chunk_size` is only needed to tune long-context behavior.

### Tied embedding is mandatory at scale

The embedding matrix is `vocab × d_model ≈ 50280 × 768 ≈ 38M` — roughly a third of
the ~100M budget. Tying the input and output embeddings (rather than learning a
separate LM head) is therefore not a tuning knob but a requirement; see
[model](02-model-ssm.md) for the tied-head implementation.

### Precision: fp16 for poc, fp32 for toy/smoke

The fp16-vs-bf16 question was settled empirically on MLX in M1 (issue #3), not
assumed. The micro-benchmark on this Metal GPU: **fp16 ~3.96 TFLOP/s vs bf16 ~3.36
and fp32 ~3.40** — fp16 is ~18% faster and bf16 gives no speedup. So the scale run
uses **fp16 + loss scaling** (the loss-scaling machinery lives in
[training](05-training.md)), while toy/smoke stay **fp32** for exact resume. Note this
contradicts the common assumption that bf16 is the safe default — on Metal it isn't.

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
- [`../MAC_RUNBOOK.md`](../MAC_RUNBOOK.md) — the milestone sequencing that produced these.
