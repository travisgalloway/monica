# Swift/MLX logit-parity fixtures (#166)

Each directory here is one **frozen oracle** for `monica-parity`: the Python MLX backend's
answer for a fixed config, a fixed token batch, and a fixed seed. The Swift port passes iff
it reproduces those numbers in **fp32** at **`rtol = 1e-4`, `atol = 1e-5`** — the same
contract `src/conformance/forward_step_parity.py` uses, with the Python array as the
reference operand (numpy's `allclose` formula is asymmetric).

Both code paths are gated, because they are two separate implementations of the same
function and a mismatch between them is exactly the bug this harness exists to catch:

* `forward_logits` — the SSD chunked-matmul scan
* `step_logits` — the same tokens fed one at a time through the one-step recurrence

## Files in a fixture

| File | What |
| --- | --- |
| `weights.safetensors` | Portable weights, written by `MLXMambaModel.save` |
| `weights.safetensors.config.json` | The `MambaConfig` sidecar — self-describing, so Swift needs no config override |
| `inputs.safetensors` | `tokens` `int32 (B, L)` |
| `reference.safetensors` | `forward_logits`, `step_logits` `fp32 (B, L, V)`, plus `hidden.{i}` for `i in 0..n_layers` |
| `meta.json` | config name, B/L, seed, precision, mlx version, and Python's own forward-vs-step max-abs-diff |

`hidden.{i}` is the per-layer output (`hidden.0` is the embedding, `hidden.{i+1}` is layer
`i`'s output — the HF convention). `monica-parity` only reads it on failure, to report
**which layer diverged first**; localizing a whole-model logit mismatch by hand across 24
layers is otherwise a week's work.

## The fixtures

| Fixture | Config | B × L | Why it exists |
| --- | --- | --- | --- |
| `toy/` | `config/toy.yaml` | 2 × 129 | Pure Mamba, **3 chunks** (`2*64 + 1`) — exercises the inter-chunk state handoff AND a ragged final chunk. At a single chunk the scan degenerates and the likeliest train/infer divergence is never hit. |
| `toy-hybrid/` | `config/toy-hybrid.yaml` | 2 × 40 | RoPE + KV-cache growth (attention at layers 1 and 3) |
| `toy-moe/` | `config/toy-moe.yaml` | 2 × 40 | Router top-2-of-4, bias inactive — the pre-#213 ranking path |
| `toy-moe-biased/` | `config/toy-moe.yaml` | 2 × 40 | Loss-Free-Balancing biased ranking (#213) + the `moe_route_bias.*` load path |

## Regenerating

```bash
.venv/bin/python scripts/export_parity_fixture.py --config config/toy.yaml \
    --out swift/engine/Fixtures/toy --batch 2 --seq 129
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-hybrid.yaml \
    --out swift/engine/Fixtures/toy-hybrid --batch 2 --seq 40
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-moe.yaml \
    --out swift/engine/Fixtures/toy-moe --batch 2 --seq 40
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-moe.yaml \
    --out swift/engine/Fixtures/toy-moe-biased --batch 2 --seq 40 --moe-bias
```

The exporter refuses to write a fixture whose Python model fails its own forward/step
check, so a reference that is internally inconsistent can never reach disk.

`tests/test_parity_fixture_export.py` re-exports `toy` into a tmpdir and compares it to the
checked-in reference. That is what stops a future change to `mlx_backend.py`'s math from
silently leaving the Swift gate testing a stale oracle.

## `poc` is deliberately NOT checked in

At fp32 the `config/poc.yaml` weights alone are ~508 MB. The exporter supports it and
`monica-parity --fixtures <dir>` takes any fixture directory, so the poc run is a **manual,
local acceptance step**, not a CI gate:

```bash
.venv/bin/python scripts/export_parity_fixture.py \
    --config config/poc.yaml --precision fp32 --batch 1 --seq 128 \
    --out /tmp/monica-poc-fixture
cd swift/engine && swift run monica-parity --fixtures /tmp/monica-poc-fixture
```

poc shares 100% of its code paths with `toy` (pure Mamba, no attention, no MoE) at larger
dimensions, so gating it in CI would need Git LFS or a ~10-minute in-runner generation step
for no new coverage. The known upgrade path, if it is ever wanted: have the existing
`full-macos` job (which already installs MLX) generate the fixture and hand it to a
dependent job as an artifact.
