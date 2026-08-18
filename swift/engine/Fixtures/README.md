# Swift/MLX logit-parity fixtures (#166 / #167 / #169)

Each directory here is one **frozen oracle** for `monica-parity`: the Python MLX backend's
answer for a fixed config, a fixed token batch, and a fixed seed. Every fixture declares
its `precision` in `meta.json`; `monica-parity` looks up that precision's band from
`src/conformance/tolerances.py` (hand-mirrored in `main.swift`'s `precisionBands`) and
compares against it, with the Python array as the reference operand (numpy's `allclose`
formula is asymmetric). fp32 fixtures — everything below `toy-fp16`/`toy-bf16`/
`toy-moe-fp16`/`toy-hybrid-fp16` in the table — get **`rtol = 1e-4`, `atol = 1e-5`**, the
same contract `src/conformance/forward_step_parity.py` uses; the four low-precision
fixtures get a DIFFERENT, looser, dtype-derived contract — see "Three tolerance regimes"
below and `docs/design/03-conformance.md`'s "The low-precision contract (#266)".

Five code paths are gated, because each is a separate implementation of the same
function and a mismatch between them is exactly the bug this harness exists to catch:

* `forward_logits` — the SSD chunked-matmul scan
* `step_logits` — the same tokens fed one at a time through the one-step recurrence
* `greedy_ids` (#167 AC1) — greedy (`temperature=0`) decode ids from `monica-generate`'s
  `Sampler`/`Generator`, compared **exactly** (not at a tolerance) against Python's
  step-driven argmax
* `prefill_logits`/`prefill_last_logits` + per-layer `state.{i}.{slot}` (#169) — one
  parallel scan over the whole prompt, and the exact recurrent state that scan must hand
  off to `step` for generation to continue correctly. State is the load-bearing part: a
  wrong carry-out is silent (the prompt's own logits still look right) and only corrupts
  tokens generated afterwards, so it is checked element-wise, not just via logits.
* `packed_logits` (#68/#263, `monica-parity`'s P6) — multiple documents packed into one
  sequence with `seg_ids`, exercising the packing-aware forward path (chunk-boundary SSM
  mask, boundary-aware conv, block-diagonal attention). See "The packed (`seg_ids`) oracle"
  below.
* `hidden.{i}` (#264) — per-layer hidden states, checked UNCONDITIONALLY (not only on a
  forward/step failure, as before #264) and `mixing.{i}` (#264/#100) — each Mamba layer's
  head-averaged mixing matrix. See "Interpretability accessors (#264)" below.
* `load.{i}` (#265) — each MoE layer's per-expert token counts from `set_moe_load_counting`,
  compared **exactly** (never a tolerance), plus a route-bias WRITE-BACK check that has no
  checked-in tensor of its own. See "MoE load counting + route-bias write-back (#265)" below.

## The checkpoint round trip (#196) adds NO fixture

`monica-parity`'s round-trip section (`Checkpoint.save`/`portableStateDict`, #196) proves the
Swift **writer** produces a checkpoint Swift can read back byte-for-byte, and
`scripts/check_swift_checkpoint.py` proves Python can too — but neither check needed, or got,
a new checked-in tensor. Every comparison is computed **in-process** from the fixture already
loaded above: save the already-loaded model, reload it, and diff against the SAME model
object's own tensors/config/logits, all in the same run. This is deliberate, not an oversight:
#195 (the Swift train step) shipped a checked-in **gradient** oracle that turned out to
diverge across macOS machines (`grad.0.layers.0.conv.weight` by `max|d|=1.226e-02` on the
hosted CI runner vs. locally) — forward/step *logit* and *weight* fixtures have never shown
that failure mode, but the safest fix is to not add another checked-in tensor of any kind. The
round trip's two EXACT (no-tolerance) comparisons — save→load tensor bit-identity, and the
mutation round trip's exact-preservation check — are therefore same-process, same-machine
comparisons by construction, which makes them immune to the cross-machine drift that broke
#195's fixture regardless of what hardware CI runs on next.

## Files in a fixture

| File | What |
| --- | --- |
| `weights.safetensors` | Portable weights, written by `MLXMambaModel.save` |
| `weights.safetensors.config.json` | The `MambaConfig` sidecar — self-describing, so Swift needs no config override |
| `inputs.safetensors` | `tokens` `int32 (B, L)` |
| `reference.safetensors` | `forward_logits`, `step_logits` `fp32 (B, L, V)`, plus `hidden.{i}` for `i in 0..n_layers`, (non-quantized fixtures only) `mixing.{i}` for each Mamba layer's absolute index (#264), and (when `moe_load_layers` is present) `load.{i}` `fp32 (n_experts,)` for each MoE layer's absolute index (#265) — exact integer counts stored as fp32 |
| `generation.safetensors` | `prompt_ids` `int32 (P,)`, `greedy_ids` `int32 (S,)`, `margins` `fp32 (S,)` — #167's greedy-decode oracle |
| `prefill.safetensors` | `prefill_logits` `fp32 (B, L, V)`, `prefill_last_logits` `fp32 (B, V)`, plus `state.{i}.conv`/`state.{i}.ssm` (Mamba layers) or `state.{i}.k`/`state.{i}.v` (attention layers) — #169's state-handoff oracle. MoE layers are stateless and emit no keys. Also #264's `verifyBlock` oracle — see below; it adds no new tensor. |
| `packed.safetensors` (OPTIONAL) | `packed_tokens`/`packed_seg_ids` `int32 (1, Lp)`, `doc_lengths` `int32 (D,)` (real, pre-pad token count per doc), `packed_logits` `fp32 (1, Lp, V)` — #68/#263's packing-aware `seg_ids` oracle. Present only for `toy`/`toy-hybrid`/`toy-moe`; see "The packed (`seg_ids`) oracle" below. |
| `train.safetensors` (#195, `toy`/`toy-hybrid`/`toy-moe` only) | `mb.{j}.inputs`/`mb.{j}.targets` int32 micro-batches, and per step `k` in `0..<train_steps`: `loss.{k}`, `grad_norm.{k}` (scalars), `grad.{k}.<param>` (the full gradient tree, pre-clip), `weights_after.{k}.<param>` (the full parameter tree, post-optimizer-update) — the K-step training-trajectory oracle `monica-train` gates against |
| `meta.json` | config name, B/L, seed, precision (#266: REQUIRED — `monica-parity`'s band lookup key, missing/unrecognised is a hard failure), vocab_size, gen_steps, mlx version, Python's own forward-vs-step max-abs-diff, the prefill/decode max-abs-diff and max-abs-state-diff, the minimum greedy margin, (when `packed.safetensors` was written) `packed_doc_lengths`/`packed_seq_len`, (non-quantized fixtures only) `mixing_layers` — the ABSOLUTE layer indices `mixing.{i}` was exported for (#264), (when the fixture has unquantized MoE layers with `top_k < n_experts`) `moe_load_layers` — the ABSOLUTE layer indices `load.{i}` was exported for — and `moe_route_margin_min` — the exactness-hazard guard for `load.{i}`'s exact comparison (#265), (low-precision fixtures only) `lowp_self_mean_kl` — a MEASUREMENT, never a threshold (`monica-parity` reads thresholds only from its own table), (only if the #266 `--allow-moe-load-omit` fallback fired) `moe_load_omitted_reason` — a DECLARED skip of the load-count oracle, never a silent one, and (when `train.safetensors` was written) `train_steps`/`train_grad_accum`/`train_grad_clip`/`train_lrs`/`train_loss`/`train_grad_norm`/`train_clipped_steps`/`train_rtol`/`train_atol` — see "A third tolerance regime: training (#195)" below |

`hidden.{i}` is the per-layer output (`hidden.0` is the embedding, `hidden.{i+1}` is layer
`i`'s output — the HF convention). As of #264 `monica-parity` checks it UNCONDITIONALLY on
every fixture (previously only on a forward/step failure, to localize it) — a whole-model
logit mismatch is still localized to "first divergence at layer N" the same way, but an
accessor bug that leaves `forward`'s own separate loop unaffected (e.g. appending a hidden
state before applying the layer, an off-by-one that shifts every later index) would
previously never have fired this check at all.

## Interpretability accessors (#264)

Two more accessors, both `MonicaModel.mixingMatrices`/`SelectiveSSM.mixingMatrix`/
`MonicaModel.verifyBlock` ports of the `#100`/`#52` Python interpretability/speculative-
decoding auxiliaries (`mlx_backend.py:280-298`/`:426-435`/`:938-950`/`:897-920`), gated at
the SAME strict fp32 contract as everything else here — interpretability accessors are not
a lossy path, so neither uses the #168/#266 per-fixture `rtol`/`atol` hook.

* **`mixing.{i}`** — each Mamba layer's head-averaged `(B, L, L)` mixing matrix, keyed by
  its ABSOLUTE layer index (attention AND MoE layers are skipped — a MoE block is pointwise,
  not a mixer). `meta.json`'s `mixing_layers` records which indices to expect; a fixture
  whose config has at least one Mamba layer but an EMPTY `mixing_layers` would make the
  whole section a silent no-op, so the exporter refuses to write one. `monica-parity`
  additionally re-derives the FULL (un-averaged) per-head matrix of the first Mamba layer
  in-process and checks two properties a numeric compare against the head-averaged oracle
  alone cannot: strict causality (`max|upper triangle| == 0` exactly — `segsum`'s `-inf`
  above the diagonal `exp`s to exact zero) and the defining identity
  `einsum("bhij,bjhp->bihp", M, X) == ssm.parallel(x)`. Both catch a transposed or
  wrongly-oriented matrix that a plain numeric diff against a head-averaged reference could
  miss (averaging over heads can wash out an antisymmetric structural defect that survives
  in the full tensor). Present only for non-quantized fixtures — quantization is a lossy
  path and must not drag the loose #168/#266 tolerance hook into this strict gate (same
  carve-out `packed.safetensors` already uses); `monica-parity` prints an explicit
  `SKIP (no mixing oracle: quantized fixture)` line for `toy-moe-int8`/`toy-moe-int4`
  rather than silently doing nothing.
* **`verifyBlock`** — the #172 speculative-decoding prerequisite: consumes a batch of
  tokens through `step` from a given state in ONE `MLX.eval`, returning per-token logits
  and per-token states. Adds **no new checked-in tensor** (same rationale as the checkpoint
  round trip above, and the reason #195's checked-in gradient oracle is not reproduced
  here): `monica-parity` gates row 0's per-token logits against the already-checked-in
  `step_logits`, the FINAL returned state against `prefill.safetensors`'s `state.{i}.{slot}`
  sliced to row 0 (prefill's state after `L` tokens is, by contract, the stepped state
  after `L` tokens — already gated Python-side by `check_prefill_decode_parity`), and an
  in-process rollback identity — `states[k]` must equal a sequential `step` walk over
  `tokens[0...k]` at `k=0` and mid-sequence — proving the per-token state list is not
  simply the final state repeated. `verify_block` is MLX-only per
  `docs/design/14-inference-engine.md`: not on `ModelInterface`, no CUDA port.

## MoE load counting + route-bias write-back (#265)

Two training-side surfaces #166 deliberately left out and #195 deferred again: per-expert
load counting (`set_moe_load_counting`/`pop_moe_load`) and the `set_moe_biases` WRITE path
(the bias READ path — `moe_route_bias.*` in `weights.safetensors` — was already ported for
`toy-moe-biased`, #196). Load counting does **not** depend on the balancer: every config in
the tree has `moe_balance_rate: null` (#217), but `set_moe_load_counting` is an independent
switch (`mlx_backend.py:992-996`), so `toy-moe`/`toy-moe-biased` exercise it directly rather
than waiting on a train step or a config change.

* **Why counts are compared exactly.** `load.{i}` is per-token routing bookkeeping — an
  integer count stored as fp32 — so it is compared with `==`, never `rtol`/`atol` (the
  #168/#266 tolerance hook does not apply here, and never will: a "close" count is not a
  meaningful concept). That makes it exactness-hazard-prone the same way `greedy_ids` is:
  a near-tie in the router's ranking could make two correct fp32 implementations pick a
  different top-`k` set and produce different (but both "valid") counts. `meta.json`'s
  `moe_route_margin_min` is the `greedy_margin_min` analogue — the smallest gap, over every
  token and every MoE layer, between the k-th and (k+1)-th largest SELECTION score
  (`logits + route_bias` when the bias is active, else `logits`) — and the exporter refuses
  to write a fixture below `1e-5`, printing a warning below `1e-3`. `toy-moe`'s margin
  (`~9.2e-5`) intentionally trips that warning; it is ~65x clear of the fixture's own
  `forward_step_max_abs_diff` (`~1.4e-6`), so it is not a live flake risk today, but the
  warning text is what a future one gets diagnosed from — do not raise the refusal
  threshold to silence it, and do not lower `1e-5` to make a future fixture fit; regenerate
  with a different `--seed` instead.
* **Why the sum-identity check exists.** An unexercised load counter that always returns
  zero would pass every comparison against a fixture that ALSO happened to be all-zero —
  the exporter refuses to write a `load.{i}` oracle unless `moe_load_layers` is non-empty,
  not every count is zero, and each layer's counts sum to exactly `tokens.size * top_k` (the
  cheapest possible proof the counter observed the real routing, not a no-op).
* **`monica-parity`'s gate**, run on `toy-moe`/`toy-moe-biased` after the mixing-matrix
  section: (1) exact per-layer equality against `load.{i}`; (2) the same sum-identity check,
  independent of (1); (3) counting OFF → forward → pop returns all-zero (catches a counter
  that ignores its own gate — a leak that would otherwise grow the lazy MLX graph for a
  whole run); (4) a route-bias WRITE-BACK check with **no checked-in tensor of its own**
  (same "adds no fixture" pattern as the #196 checkpoint round trip and #264's `verifyBlock`
  above): a saturating bias (`[1e4, 0, 0, ...]`) pushed via `setMoeBiases` on a freshly
  reloaded model must drive expert 0's count to exactly `B*L`, `moeBiases()` must read back
  what was written, an out-of-range layer/vector count must throw, and restoring the
  original bias must reproduce `forward_logits` again (proving the write path is reversible
  and corrupts nothing). A fixture with no MoE layers, or a quantized one (`toy-moe-int8`/
  `toy-moe-int4` — a lossy path must not feed this strict exact gate, same carve-out
  `packed.safetensors`/`mixing.{i}` already take), gets an explicit `SKIP` line rather than
  silently doing nothing; a fixture WITH unquantized MoE layers but no `moe_load_layers` in
  `meta.json` is a stale-fixture FAILURE, not a skip.
* **The parameter-tree canary.** `monica-parity`'s existing checkpoint key-set equality
  check fires on every fixture if the Swift load-count accumulator ever leaks into
  `parameters()` — no new assertion needed; the failure would be immediate and total.

Python-side, `tests/test_parity_fixture_export.py` carries the same two rules the `toy`
staleness test already established for logits: `test_checked_in_toy_moe_fixture_load_counts_match_todays_backend`
re-exports `toy-moe` and asserts `load.{i}` matches exactly (plus the reference key set and
`moe_route_margin_min`'s presence/threshold), and
`test_route_bias_write_lands_in_the_logits_and_the_counts` pushes `toy-moe-biased`'s own
checked-in bias into a freshly-loaded `toy-moe` model and asserts both the logits AND the
counts reproduce `toy-moe-biased`'s checked-in oracle — the strongest available proof the
write path lands, runnable locally (mlx-swift cannot execute on this host).

Carried over from Python unchanged, as comments rather than behaviour to reproduce (neither
can fire in this inference-only engine): `grad_checkpoint: true` doubles every count
uniformly (both consumers are scale-invariant), and counts survive an fp16
overflow-skipped training step (harmless — the routing really happened). #217's routing-
entropy diagnostic (`pop_routing_stats`/`_entropy_sum`), gated by the same Python flag, is
explicitly NOT ported — out of scope for #265.

`hiddenStates`/`mixingMatrices` under `seg_ids` are out of scope for #264: the packed
(`seg_ids`) oracle above only carries `packed_logits`, and per-layer hidden states or
mixing matrices under packing would need a new exporter key this issue does not add.

`monica-generate --dump-activations <file.safetensors>` (#264) is the CLI surface these
accessors get: given `--weights` + `--prompt-ids` (the tokenizer-free debug path, so no
tokenizer is required), it writes `hidden.{i}`/`mixing.{i}` for the prompt batch using the
SAME key convention as `reference.safetensors` above, so a user can diff a Swift dump
against a Python one directly.

`generation.safetensors`'s `prompt_ids` are the first `min(8, L)` ids of the token batch's
row 0; `greedy_ids` are `gen_steps` (default 16) ids produced by driving `model.step`
token-by-token with `temperature=0` (argmax, first-max-on-tie) — the same shape
`monica-parity` reproduces in Swift. `margins[i]` is `top1_logit - top2_logit` at step `i`;
the exporter **refuses to write a fixture** whose margin at any step falls inside the parity
band (`atol + rtol*|top1|`), because a near-tie greedy argmax makes cross-implementation id
equality flaky rather than well-posed — random-init toy weights produce near-uniform logits,
so this check is load-bearing, not theoretical. The exporter also asserts that
`src/serve/generate.py`'s prefill-based path (`SessionStore` + `generate()`) reproduces the
same `greedy_ids`, so the fixture is anchored to the real CLI's behavior, not to an
exporter-private step loop.

## The fixtures

| Fixture | Config | B × L | Why it exists |
| --- | --- | --- | --- |
| `toy/` | `config/toy.yaml` | 2 × 129 | Pure Mamba, **3 chunks** (`2*64 + 1`) — exercises the inter-chunk state handoff AND a ragged final chunk. At a single chunk the scan degenerates and the likeliest train/infer divergence is never hit. Also `monica-parity`'s primary #169 P3 fixture, for the same reason. |
| `toy-short/` | `config/toy.yaml` | 1 × 2 | #169: `L=2 < d_conv-1=3` — the ONLY fixture that exercises `convWindow`'s LEFT-PAD branch (`toy` at L=129 and `toy-gen` at L=8 never reach it). Mirrors `tests/test_mlx_parity.py::test_prefill_short_sequence_conv_window`. |
| `toy-hybrid/` | `config/toy-hybrid.yaml` | 2 × 40 | RoPE + KV-cache growth (attention at layers 1 and 3) |
| `toy-moe/` | `config/toy-moe.yaml` | 2 × 40 | Router top-2-of-4, bias inactive — the pre-#213 ranking path |
| `toy-moe-biased/` | `config/toy-moe.yaml` | 2 × 40 | Loss-Free-Balancing biased ranking (#213) + the `moe_route_bias.*` load path |
| `toy-gen/` | `config/toy.yaml`, `--vocab-size 512` | 1 × 8 | #167: exercises `monica-generate`'s real `--prompt` path (tokenizer → ids → model) end to end in CI. A `monica-tokenize`-trained tokenizer's vocab (256 base bytes + specials) cannot fit `toy.yaml`'s default 256-wide model, so this fixture widens the model instead of shrinking the tokenizer. Not in `monica-parity`'s default fixture list — it is consumed directly by `monica-generate` in the `swift-engine` CI job. |
| `toy-moe-int8/` | `config/toy-moe.yaml`, `--quant-bits 8` | 2 × 40 | #168: int8 weight-only quantization (group 64, affine, `mx.quantize`-compatible). Exercises the MoE experts, Mamba projections, AND the tied embedding/head in one artifact. |
| `toy-moe-int4/` | `config/toy-moe.yaml`, `--quant-bits 4 --quant-head-bits 8` | 2 × 40 | #168: int4, with the tied head kept at int8 — plain int4-on-embedding pushed this particular random-init toy model's KL past the int4 threshold (an accuracy risk the design doc flags), so `--quant-head-bits 8` is the recorded default for int4 fixtures. |
| `toy-fp16/` | `config/toy.yaml`, `--precision fp16` | 2 × 40 | #266: the fp16 elementwise-band + KL-tier contract, plain Mamba. fp16 is poc's native compute precision and #167's fast-decode target. |
| `toy-bf16/` | `config/toy.yaml`, `--precision bf16` | 2 × 40 | #266: the bf16 contract. One dense fixture only — no config in the tree trains at bf16 today, and `toy.yaml` has no MoE layers, sidestepping the wider bf16 route-margin requirement (`128 * u_bf16 = 0.5`) a MoE bf16 fixture would impose. |
| `toy-moe-fp16/` | `config/toy-moe.yaml`, `--precision fp16 --seed 15 --moe-bias` | 2 × 40 | #266: fp16 + MoE routing. `--seed 15 --moe-bias` — no seed in `0..59` at the DEFAULT (unbiased) ranking cleared the fp16 route-margin guard (`128 * u_fp16 = 6.25e-2`; the toy model's near-uniform random-init router logits produce margins an order of magnitude too small), so this fixture uses the same `--moe-bias` fallback `toy-moe-biased` already established, then a short seed sweep among biased seeds (margin 0.371 at `--seed 15`, ~6x clear of the guard). |
| `toy-hybrid-fp16/` | `config/toy-hybrid.yaml`, `--precision fp16` | 2 × 40 | #266: fp16 + RoPE/attention — a distinct numeric path from plain Mamba (attention softmax and the KV cache round differently at low precision) worth its own fixture. |

### Which fixtures get a training oracle (#195)

Three of the seven above additionally carry `train.safetensors` (via `--train-steps 3`):
**`toy`** (pure Mamba/SSD backward), **`toy-hybrid`** (attention + RoPE backward), and
**`toy-moe`** (router argsort-mask + expert backward) — one representative fixture per
distinct backward-pass shape in the tree. The other four are excluded, each for a stated
reason:

* `toy-short` (L=2) — the conv left-pad edge case is inference/prefill-specific; there is
  no training-side analogue.
* `toy-moe-biased` — the route bias is an inference-side ranking input; the Loss-Free-
  Balancing *write* path is DEFERRED (see `.claude/plans/issue-195.md`), so this fixture
  would add no training coverage over `toy-moe`.
* `toy-moe-int8`/`toy-moe-int4` — training a quantized checkpoint is out of scope
  entirely: `QuantizedLinear`'s packed uint32 codes have no meaningful weight gradient.
  The exporter refuses `--train-steps` combined with `--quant-bits`.

`monica-train` carries its OWN default fixture list (`toy`, `toy-hybrid`, `toy-moe`,
narrower than `monica-parity`'s seven) and treats a missing `train.safetensors` in a
fixture it was asked to check as a **FAILURE, not a skip** — the same standing rule every
other oracle file in this directory follows.

## The packed (`seg_ids`) oracle (#68/#263)

`monica-parity`'s P6 section gates the three packing-aware forward-path arms (`chunkSegMask`
in `SelectiveSSM.swift`, `convSeqSeg` in `MambaBlock.swift`, the block-diagonal mask in
`AttentionBlock.swift`) against `packed.safetensors`, present on `toy`/`toy-hybrid`/`toy-moe`
only. Regenerated with `--packed-doc-lengths` (a comma-separated, chunk-length-relative doc
length spec, e.g. `"Q,2*Q,5"` — `Q` resolves to the config's `chunk_size`, default 64),
which builds the SAME chunk-aligned multi-document packing
`src/conformance/doc_boundary_parity.py::check_doc_boundary_parity` uses — that function is
the contract, not reinvented in the exporter.

P6 runs three assertions per fixture, deliberately in this order so a failure is easy to
place:

1. **Cross-language.** `model.forward(packedTokens, segIds: packedSegIds)` vs
   `packed_logits`, at the file-level strict fp32 gate (`rtol=1e-4/atol=1e-5` — never a
   per-fixture override, unlike the #168 quantized fixtures).
2. **Self-consistency**, the real correctness claim: each document's logit slice from the
   packed-aware forward vs that SAME document's standalone Swift forward, computed
   in-process. This needs no checked-in numbers, so it is immune to the cross-machine
   drift that blocked #195/PR #293's checked-in gradient oracle (`grad.0.layers.0.
   conv.weight` diverging `max|d|=1.226e-02` between the hosted CI runner and a local Mac).
3. **Anti-no-op.** The packing-BLIND forward (`model.forward(packedTokens)`, no `segIds`)
   vs each document's standalone forward (doc 0 excluded — it never leaks a PRIOR
   document's state, so it proves nothing about packing-awareness). This must exceed
   `1e-2` for every later document, or the fixture cannot distinguish a real port from one
   that silently discards `seg_ids` — see "Why the threshold is checked, not assumed"
   below.

A missing `packed.safetensors` SKIPS P6 for that fixture (packing is not exercised by
every fixture); a present-but-unreadable one is a FAILURE, the same rule this file already
applies to `generation.safetensors`/`prefill.safetensors`.

### Why the threshold is checked, not assumed

If assertion 3's gap were ever comfortably inside `1e-2` on a real fixture, the FIXTURE
GEOMETRY is wrong (a boundary too near the end, or a document too short for state to
accumulate) — never the threshold. The exporter itself refuses to write a `packed.
safetensors` whose packed sequence carries fewer than 2 distinct document ids (a
single-document fixture is vacuous by construction), and `monica-parity` re-asserts
`spans.count >= 2` on the Swift side for the same reason. On the three checked-in
fixtures, computing the blind-vs-standalone gap directly (`.venv/bin/python` against the
Python oracle, the exact numbers a silently no-op Swift port would reproduce) gives, per
non-first document: `toy` 9.1e-2/8.3e-2, `toy-hybrid` 3.36/4.74, `toy-moe` 1.1e-1/1.0e-1 —
all far clear of `1e-2`, and doc 0 in every fixture sits at ~1e-6 (floating-point noise),
confirming the "doc 0 proves nothing" exclusion is itself correct.

## Three tolerance regimes (#168 / #266)

Every fp32 fixture (`toy` through `toy-moe-int4` in the table above, minus the two
quantized ones' own override) is gated at the SAME fp32 contract (`rtol=1e-4`/`atol=1e-5`)
described above. Three regimes share the same lookup-by-`precision` machinery:

1. **fp32, strict.** `meta.precision == "fp32"`, no override — the historical contract,
   unchanged.
2. **fp32 + quantized (#168).** The two quantized fixtures carry their OWN `rtol`/`atol`
   in `meta.json` (currently `2e-2`/`2e-2`, deliberately a *sanity* bound rather than a
   precision claim) — accepted ONLY because they also declare `quant_bits`, and only ever
   LOOSENING the fp32 band (`resolved = max(band, override)`), never tightening it. This
   is the minimal hook #266's general precision→tolerance contract builds on top of; #266
   does not build a second one.
3. **Low precision (#266).** `meta.precision == "fp16"`/`"bf16"`, no override — a
   DIFFERENT contract entirely (elementwise band + a load-bearing KL tier), derived in
   `docs/design/03-conformance.md`'s "The low-precision contract (#266)" and
   `src/conformance/tolerances.py`. Never combined with the quantized override: a
   low-precision fixture carrying `rtol`/`atol` without `quant_bits` is a hard FAILURE in
   `monica-parity`, not a silently-honoured loosening.

A quantized fixture is checked THREE separate ways, because "Swift's true quantized kernel
vs a Python reference" bundles two very different questions that must not be conflated:

1. **Format correctness (exact, 1e-6).** `dequant_ref.safetensors` holds the fp32 weight
   Python's `mlx_affine_dequantize` reconstructs from the packed checkpoint. Swift
   dequantizes its OWN loaded `.weight`/`.scales`/`.biases` (via `MLX.dequantized`) and
   compares against that reference at `1e-6` — this never touches the quantized GEMM
   kernel, so a mismatch here means the packing/scale/bias CONTRACT is wrong, not that the
   kernel is imprecise.
2. **Kernel/logit agreement (loose, the fixture's own `rtol`/`atol`).** `forward_logits`/
   `step_logits`/`hidden.*` in `reference.safetensors` come from the FAKE-QUANT Python
   reference (dequantized weights loaded into the unmodified `MLXMambaModel` — no change to
   `mlx_backend.py`). The weights are numerically IDENTICAL to what Swift's real quantized
   kernel operates on; the only source of divergence is accumulation order/precision inside
   `quantizedMM`, which is why this comparison gets its own, looser tolerance instead of the
   fp32 one.
3. **Quality vs the fp model (statistical, never `allclose`).** `fp_forward_logits` in
   `reference.safetensors` is the ORIGINAL fp model's logits. `monica-parity` computes
   top-1 agreement + mean KL of Swift's quantized logits against it and gates on
   bits-dependent thresholds (`src/conformance/quant_parity.QUANT_THRESHOLDS`, mirrored in
   `monica-parity/main.swift`'s `quantThresholds`): int8 top1≥0.99/KL≤1e-3, int4
   top1≥0.95/KL≤5e-2. This is the statistical acceptance criterion #168 asks for.

`generation.safetensors`'s greedy-margin guard (see above) uses a WIDER minimum margin for
quantized fixtures (`atol` floored at `0.25` in the exporter) — quantization shrinks logit
margins, and Swift decodes through the true quantized kernel, so a margin comfortably clear
of the loose kernel tolerance is needed for cross-implementation greedy-id equality to stay
well-posed rather than flaky.

## A third tolerance regime: training (#195)

`train.safetensors`'s four surfaces (`loss`, `grad_norm`, the full `grad.{k}.<param>`
tree, and the full `weights_after.{k}.<param>` tree) are compared at their OWN
`train_rtol`/`train_atol` in `meta.json` — new keys, deliberately not `rtol`/`atol` (those
are the quantized fixtures' logit-gate keys, `2e-2`/`2e-2`, which would be far too loose
here and would gate nothing). A training step is not a forward pass: gradients accumulate
error through the whole backward graph, and AdamW moments compound it across steps.
`loss` is a pure forward quantity and stays at the same tight regime the fp32 logit gate
uses; `grad_norm`/`grad`/`weights_after` sit in a looser `2e-4`/`1e-6` band — tighter atol
than the fp32 gate's `1e-5` (which would be *looser than the grad values themselves* on
small toy leaves and gate nothing there), traded against a looser rtol to absorb the fp32
summation-order difference between Python's `tree_flatten`-order reduction and Swift's
sorted-key-order one. See `docs/design/14-inference-engine.md`'s #195 entry for the full
rationale, including why `MLXOptimizers.clipGradNorm` is NOT used (R2: its strict `<`
clip boundary differs from Python's unconditional `min(1.0, clip/(norm+eps))`).

## Regenerating

```bash
.venv/bin/python scripts/export_parity_fixture.py --config config/toy.yaml \
    --out swift/engine/Fixtures/toy --batch 2 --seq 129 --train-steps 3 \
    --packed-doc-lengths "Q,2*Q,5"
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-hybrid.yaml \
    --out swift/engine/Fixtures/toy-hybrid --batch 2 --seq 40 --train-steps 3 \
    --packed-doc-lengths "Q,7,Q+3"
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-moe.yaml \
    --out swift/engine/Fixtures/toy-moe --batch 2 --seq 40 --train-steps 3 \
    --packed-doc-lengths "Q,7,Q+3"
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-moe.yaml \
    --out swift/engine/Fixtures/toy-moe-biased --batch 2 --seq 40 --moe-bias
.venv/bin/python scripts/export_parity_fixture.py --config config/toy.yaml \
    --out swift/engine/Fixtures/toy-gen --batch 1 --seq 8 --vocab-size 512
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-moe.yaml \
    --out swift/engine/Fixtures/toy-moe-int8 --batch 2 --seq 40 \
    --quant-bits 8 --quant-group-size 64 --seed 1
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-moe.yaml \
    --out swift/engine/Fixtures/toy-moe-int4 --batch 2 --seq 40 \
    --quant-bits 4 --quant-group-size 64 --quant-head-bits 8 --seed 1
.venv/bin/python scripts/export_parity_fixture.py --config config/toy.yaml \
    --out swift/engine/Fixtures/toy-short --batch 1 --seq 2 --gen-steps 8
.venv/bin/python scripts/export_parity_fixture.py --config config/toy.yaml \
    --out swift/engine/Fixtures/toy-fp16 --batch 2 --seq 40 --precision fp16
.venv/bin/python scripts/export_parity_fixture.py --config config/toy.yaml \
    --out swift/engine/Fixtures/toy-bf16 --batch 2 --seq 40 --precision bf16
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-moe.yaml \
    --out swift/engine/Fixtures/toy-moe-fp16 --batch 2 --seq 40 \
    --precision fp16 --seed 15 --moe-bias
.venv/bin/python scripts/export_parity_fixture.py --config config/toy-hybrid.yaml \
    --out swift/engine/Fixtures/toy-hybrid-fp16 --batch 2 --seq 40 --precision fp16
```

**`toy`/`toy-hybrid`/`toy-moe`'s `train.safetensors` + its `meta.json` `train_*` keys are
CI-generated, not locally-generated (#195/PR #293).** Running the commands above will
regenerate the base 6-7 files fine, but the `--train-steps 3` addition will NOT reproduce
the checked-in `train.safetensors` on a local Mac.

Measured evidence (the #195/PR #293 investigation; full write-up in
`docs/design/14-inference-engine.md`'s #195 entry): `tests/
test_train_parity_fixture_export.py` passed on every local Mac but failed reproducibly on
the hosted `full-macos` CI runner. A dedicated CI diagnostic
(`.claude/plans/issue-195-unblock.md` Phase 0) established the split precisely:

- **Same MLX version, both sides** — `meta.json`'s `mlx_version` read `0.32.0` on both the
  locally-generated checked-in oracle and the CI runner. Not a stale-version artifact.
- **CI is internally stable.** Two independent generations, in two fresh processes on two
  separate `macos-latest` runners, agreed to **~1.5e-08** (fp32 noise floor) on every key.
  Not within-host nondeterminism.
- **A real, deterministic, host-family difference.** Both CI runners disagreed with the
  local (physical M1 Pro) checked-in oracle by the SAME amounts — up to
  `max|d| = 1.83e-02` on `grad.*.conv.weight`, EXCEEDING that tensor's own `absmax`
  (`1.20e-02`) — not accumulated rounding. Confined to gradients that reduce over a
  shifted/padded input window (`conv.weight` dominant, `{in,out}_proj.weight` a smaller
  echo); `conv.bias` (a plain sum) and every non-gradient key passed at the strict band.
  MLX's Metal conv1d weight-gradient reduction order is host-family-dependent — CI's
  "Apple M1 (Virtual)" runner vs a physical M1 Pro.

The canonical regeneration path is the CI job pair `train-fixture-oracle` (generate) +
`train-fixture-oracle-verify` (a numeric-tolerance cross-check, at the SAME
`train_rtol=2e-4`/`train_atol=1e-6` the production staleness guard uses — see the
comment on the GATE step in `.github/workflows/ci.yml` for why this is tolerance-based
rather than poc's byte-exact check) before the output is trusted:
`gh workflow run ci.yml --ref <branch>`, then download the `train-fixture-oracle`
artifact and copy `train.safetensors`/`meta.json` for the three fixtures into their
directories here. The other 6-7 files per fixture (forward-only — weights/logits/state,
never a gradient) are unaffected by this and can still be regenerated locally with the
plain commands above.

**#266's low-precision fixtures carry no `--packed-doc-lengths`** — P6's packing gate
reuses the file-level DTYPE band already, so a low-precision packed fixture would work,
but none of the four adds one (the fp32 packed fixtures already cover the packing code
paths; adding packing here would only grow CI runtime for no new coverage).

**`toy-moe-fp16`'s `--seed 15 --moe-bias`** is the #266 fallback ladder in action: the
fp16 route-margin guard (`128 * u_fp16 = 6.25e-2`) is far stricter than fp32's historical
`1e-5`, and `toy-moe.yaml`'s near-uniform random-init router logits could not clear it at
the DEFAULT (unbiased) ranking for any seed in `0..59`. Falling back to `--moe-bias` (the
same asymmetric bias `toy-moe-biased` already uses) widened the achievable margins
considerably; `--seed 15` was the first of a short sweep (`0..19`, biased) to clear the
guard, at margin `0.371` — about 6x clear. If a future regeneration needs a new seed, the
documented ladder is unchanged: seed search at the default ranking, then `--moe-bias` +
seed search, then (only if both fail) `--allow-moe-load-omit`, which records
`moe_load_omitted_reason` in `meta.json` instead of raising — `monica-parity` accepts a
DECLARED omission as a legitimate skip, never a silent one.

**A local MLX (0.32.0) buffer-protocol gap surfaced while adding `toy-bf16`:** converting
a `bfloat16` MLX array straight to numpy via the buffer protocol fails
(`RuntimeError: Item size 2 ... does not match ... item size 1`) for INTERMEDIATE arrays
(hidden states, mixing matrices, per-layer state) — final logits were never affected
(the head computes in fp32 already). The exporter's `_np_f32` helper casts to fp32 inside
MLX before the numpy conversion, sidestepping it. This is separate from #298 (the
buffer-CACHE-reuse corruption bug); `mx.clear_cache()`/`mx.set_cache_limit(0)` stays
exactly as documented below and did not need to change for this.

`--seed 1` is pinned for the quantized fixtures because the toy model's RANDOM-INIT
weights produce near-uniform logits — small quantization perturbations move a
disproportionate amount of KL relative to a trained model, and the exporter refuses to
write a fixture whose measured top-1/KL misses `QUANT_THRESHOLDS` (see
`src/conformance/quant_parity.py`). `--seed 0` (the other fixtures' default) happens to
fail that check for this config at int8; `--seed 1` passes comfortably at both bit-widths.
This is a toy-scale artifact of near-uniform random logits, not evidence the thresholds are
wrong for a trained model.

`toy-short` uses `--gen-steps 8` (down from the default 16) — at `L=2` on random-init toy
weights, the greedy-margin guard (below) has less room before a longer decode run drifts
into a near-tie; 8 steps clears it comfortably at `--seed 0`. If a future regeneration
trips the margin guard here, the documented fallback ladder is: bump `--seed`, then reduce
`--gen-steps` further; if neither works, drop the fixture and cover the conv-window
left-pad branch with a synthetic shape/zero-row assertion in `monica-parity` instead.

The exporter refuses to write a fixture whose Python model fails its own forward/step
check (or its own prefill/decode-state check — `src/conformance/prefill_decode_parity.py`,
#169), or whose greedy margins are inside the parity band, or whose prefill-based
`generate()` path disagrees with the step-driven `greedy_ids` — see above — so a reference
that is internally inconsistent can never reach disk.

`tests/test_parity_fixture_export.py` re-exports `toy` into a tmpdir and compares it to the
checked-in reference (logits, `greedy_ids`, `prefill.safetensors`'s logits/state, AND
`packed.safetensors`'s tokens/seg_ids/doc_lengths (exact) and `packed_logits` (tolerance)).
That is what stops a future change to `mlx_backend.py`'s math from silently leaving the
Swift gate testing a stale oracle. `tests/test_train_parity_fixture_export.py` does the same
for `train.safetensors` (#195), re-exporting `toy` with `--train-steps 3` and comparing every
key at the training tolerance above — the guard against `mlx_train_step.py`'s math silently
drifting from `monica-train`'s oracle. **This guard's `grad.*.conv.weight` comparison has a
known cross-machine-drift caveat** — see the checkpoint round-trip note above and
`docs/design/14-inference-engine.md`'s #195 entry for the measured numbers and the fix
applied.

## `poc` is deliberately NOT checked in

At fp32 the `config/poc.yaml` fixture is **571 MB** total (`weights.safetensors` 507 MB,
`reference.safetensors` 62.9 MB, `prefill.safetensors` 28.8 MB, the remaining 4 files under
2 KB combined) — too large to check into git or Git LFS for a fixture that shares 100% of its
code paths with `toy` (pure Mamba, no attention, no MoE) at larger dimensions.

It IS still gated in CI (#267), just not on every PR: jobs `poc-fixture-oracle` +
`poc-parity` in `.github/workflows/ci.yml`, triggered only on `workflow_dispatch` or a
weekly `schedule` (Monday 09:17 UTC) — **never on `pull_request` or `push`**. Generation is
cheap (measured on an M1 Pro): **5.2-5.8 s wall**, ~2 GB peak RSS — not the ~10 minutes an
earlier draft of this note assumed — so the fixture is never checked in, and the **571 MB
fixture output itself never crosses the network**: `poc-fixture-oracle` generates it, hashes
it, and uploads only a **~2 KB sha256 manifest + `meta.json`** (those two small files *do*
cross the network as a CI artifact); `poc-parity` regenerates its own independent copy on a
second runner and `cmp`s the two manifests before trusting either as an oracle (the
cross-process #298 guard below) — measured **bit-identical, 7/7 files**, during planning.

Run it locally the same way CI does:

```bash
.venv/bin/python scripts/export_parity_fixture.py \
    --config config/poc.yaml --precision fp32 --batch 1 --seq 128 \
    --out /tmp/monica-poc-fixture
cd swift/engine && swift run monica-parity --fixtures /tmp/monica-poc-fixture
```

**Why generate-on-runner isn't vacuous.** The oracle (Python/MLX,
`scripts/export_parity_fixture.py`) and the consumer (`swift/engine`'s hand-written
mlx-swift port) share nothing but the `.safetensors` bytes — running them in the same CI
run changes nothing about that independence; it only avoids paying to move 571 MB over the
network. What this gate buys over `toy` is **scale coverage of the tolerance contract**,
not new code-path coverage: at poc, `forward_step_max_abs_diff = 3.62e-05` (25x toy's
1.43e-06) and `greedy_margin_min = 23.94`, so `src/conformance/tolerances.py`'s fp32 band
(`rtol=1e-4/atol=1e-5`, derived on toy configs and assuming an absolute-dominated band at
`|logit| ~ 8`) becomes **relative-dominated** at poc's larger logits. Headroom is still
~66x, so the contract holds — but nothing kept that true before this gate existed.

**The #298 cross-process guard.** #298 is silent, deterministic-per-process numerical
corruption during MLX 0.32.0 export. Two independently-generated fixtures, on two separate
runners, in two fresh processes, would have to corrupt identically to pass the manifest
`diff` — so a `poc-fixture-oracle`/`poc-parity` manifest mismatch is treated as a #298
sighting, not a flake to rerun past.

**Why dispatch/schedule only, not per-PR.** `poc-parity` is a second macOS runner carrying
an mlx-swift xcodebuild (warm-cache: minutes; cold: up to 40) plus a `pip install mlx` —
against zero new code-path coverage over `toy`, that is not a per-PR trade worth making. See
`docs/design/14-inference-engine.md` §Staged roadmap for the full record.
