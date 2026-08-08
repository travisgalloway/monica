# Configs & locked decisions

[← Index](README.md)

`config/*.yaml` is the single source of truth for model dimensions and run parameters,
loaded into [`MambaConfig`](../../src/model/blocks.py). **The comments in these files
*are* the decision record.** Two of them carry the load-bearing decisions and are
reproduced verbatim below — `toy.yaml` (correctness / smoke) and `poc.yaml` (the ~100M
scale run) — but they are no longer the whole surface.

### The full config surface

| Config | Role | Status |
|---|---|---|
| `toy.yaml` | milestone-1..4 smoke test; tiny + fp32 so fixed-seed resume is exactly reproducible | **live** (the gate) |
| `toy-hybrid.yaml` | tiny Mamba-2 with config-gated attention layers (#67) — parity + sizing tests exercise both block types | live |
| `toy-moe.yaml` | tiny Mamba-2 with config-gated sparse-MoE FFN layers (#53) — parity + sizing tests exercise the MoE block | live |
| `toy-moe-dense.yaml` | degenerate `n_experts: 1, top_k: 1` MoE block (IS a plain dense SwiGLU FFN) — the sparse-upcycle SOURCE fixture (#214) | live |
| `toy-moe-fine.yaml` | 64 x top-6, `moe_d_ff = d_inner/8`, balancing ON — the sparse-upcycle TARGET fixture, pairs with `toy-moe-dense.yaml` (#214) | live |
| `toy-moe-muon.yaml` | `toy-moe.yaml` + `optimizer: muon` — covers expert gate/up/down (Muon) and the router (AdamW) in one real model (#214) | live |
| `toy-moe-8bit.yaml` | `toy-moe.yaml` + `optimizer_8bit: true` — config-surface fixture ONLY, not runnable without `bitsandbytes`/a CUDA host (#214) | live (surface-only) |
| `toy-muon.yaml` | toy.yaml's shape with `optimizer: muon` (#237) — exercises the hybrid optimizer | live |
| `small.yaml` | ~2.6M params, byte vocab — fast local iteration | live |
| `poc.yaml` | ~127M OLMo-vocab from-scratch scale run | reserve (validated foundation, #75) |
| `poc-small.yaml` | ~97M, real-but-slow local POC (the "≤100M trained locally" target) | reserve |
| `poc-qwen.yaml` | poc.yaml retargeted to Qwen2.5 — the **completed** ~205M run (val-ppl 75.7) | reserve (done) |
| `1b.yaml` | ~1B from-scratch, OLMo vocab | reserve (#75) |
| `student-1b.yaml` | ~1B hybrid distillation-student sweep seed | **history** (M10 dropped 2026-07-19) but still **live as a fixture**: `tests/test_packing_dtype.py` uses it as the uint32 case, and [09-hybrid-architectures.md](09-hybrid-architectures.md) cites it as the hybrid sizing example |

> The `student-1b-attn-lo.yaml` / `-attn-hi.yaml` sweep siblings were **deleted** (2026-07-25).
> They documented themselves as derived from `config/manifests/*.yaml` via `manifest_to_config`,
> and both the manifest directory and that converter were removed with #189/#248 — so they
> described tooling that no longer exists and nothing in the tree referenced them. The
> attention-fraction reasoning they encoded survives as prose in
> [09-hybrid-architectures.md](09-hybrid-architectures.md); the files themselves are in git history.

The M12 code model's own configs (small ~120M-active/700M-total, "Large A"
~700M-active/3.5B-total) do **not** exist yet — they arrive with #200/#219. See
[13-code-model-moe.md](13-code-model-moe.md).

**Note (#200) — the sparse-upcycle source is a specific shape, not just a matching width.**
Three things must hold, and only the first is about `d_model`:

- **`d_model` must match the target exactly.** `src/train/upcycle.py` hard-raises
  `UpcycleError` on a mismatch (expert replication copies a tensor, it cannot widen one).
  #200's currently-assumed `d_model 768` does **not** match Large A's current default of
  `d_model 1536` — see [13-code-model-moe.md](13-code-model-moe.md)'s "Open (#223)" note and
  **#272** before dimensioning #200.
- **…and so must 14 other fields.** `_MUST_MATCH` (`src/train/upcycle.py:48-52`) is the
  authority: `moe_every`, `moe_d_ff_resolved`, `attn_every`, `n_attn_heads_resolved`,
  `vocab_size`, `tie_embeddings` and the rest of the backbone. Matching only the non-expert
  dims is necessary but not sufficient.
- **#200 is trained as a degenerate `n_experts: 1, top_k: 1` MoE with `moe_d_ff ≈ d_inner/8`**,
  not as a plain dense config — a plain dense model has no `experts.0.*` keys to replicate, and
  the narrow `moe_d_ff` is what makes 64 fine-grained experts fit Large A's budget. The
  `toy-moe-dense.yaml` / `toy-moe-fine.yaml` pair above is the worked example;
  `scripts/upcycle.py --dry-run` checks a real pair in seconds.

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

### Vocab determines the packing dtype

`vocab_size: 50280` (OLMo-7B-hf) is confirmed `< 65536`, so `poc.yaml` packs as
**uint16** — half the shard bytes of the alternative. This is a *dtype selection*, not a
hard bound: since #90, `MambaConfig.packing_dtype` returns `'uint32'` at or above 65536,
and `validate()` only rejects above uint32 capacity (`2**32`). See
[dtype-aware packing](04-data-pipeline.md). The reserve `poc-qwen.yaml`/`student-1b.yaml`
configs are uint32; the M12 code BPE is uint16.

### dt-bias parameters are shared

`dt_min` / `dt_max` / `dt_init_floor` appear in **both** configs identically — the
[load-bearing dt-bias init](02-model-ssm.md) is a model-wide decision carried into
every backend, not a per-run tuning choice.

## Related

- [Model: the Mamba block + selective SSM](02-model-ssm.md) — what these dims build.
- [Training](05-training.md) — how fp16 loss scaling is applied.
- [Data pipeline](04-data-pipeline.md) — the uint16 / vocab link.
- [GitHub issue #2](https://github.com/travisgalloway/monica/issues/2) — the milestone tracker.
