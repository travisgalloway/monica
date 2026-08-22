# The M13 native inference + training engine (Swift + MLX)

[← Index](README.md)

The design record for **M13** ([issue #163](https://github.com/travisgalloway/monica/issues/163)):
a performant, native, **no-Python-runtime** engine for the Mamba-2 hybrid — *"like llama.cpp,
tailored for our model"* — targeting **Apple Silicon first**, written in **Swift on MLX**. This
document is #164; the issues it records are #165–#172 and #195–#197.

> **Read this as the design record; much of it has since shipped.** As of 2026-08-19 the tree
> holds both the **tokenizer** — `swift/Sources/MonicaTokenizer` + the `monica-tokenize` /
> `monica-selfcheck` executables (#191/#245, `swift/Package.swift`) — and the **Swift engine**:
> `swift/engine/Sources/MonicaEngine/` (`MonicaModel`, `MambaBlock`, `AttentionBlock`,
> `SelectiveSSM`, `MoEBlock`, `Generator`, `Sampler`, `Checkpoint`, `Quantization`, `Bench`,
> and `TrainStep`/`LossScaler`) plus the `monica-parity` / `monica-generate` / `monica-bench` /
> `monica-train` runners, landed across #166/#169/#170/#195/#196/#197 and the #265–#267
> MoE/parity work. The Swift **train step** shipped with #195 (PR #293) — see its detailed
> record at the end of this document. Remaining unbuilt: the fused Metal kernel (#171) and
> Swift speculative decoding (#172). On the Python side, `prefill` **landed via #165**
> (on the seam, in both backends, gated by `src/conformance/prefill_decode_parity.py`);
> `step_batch` still does not exist on `ModelInterface`, and `verify_block` exists on the MLX
> backend **only**
> (`src/model/mlx_backend.py:897`) — not on the seam, and with no CUDA equivalent.

The scope was **widened from inference-only to inference *and* training** when M12 landed: the
Swift backend must train the M12 small rung natively (#195 train step, #196 checkpoint I/O) and
give the LSP-in-the-loop fast decode loop a native harness (#197). #164's original body predates
that widening, and was filed against `docs/design/12-inference-engine.md` before slots 12 and 13
were taken — hence the 12 → 14 renumber recorded on the issue.

## Goals & non-goals

**Goals.**

- **Cut single-stream latency on local hardware.** Two costs dominate today. *Prefill* is
  `O(prompt_len)` sequential recurrence: `src/serve/generate.py:52-53` walks the prompt one token
  at a time through `store.step(...)`, even though the training path already consumes a whole
  sequence in one chunked-matmul scan. *Decode* is a batch-1 `step` loop bounded by per-token
  kernel launch and host sync — the **94.7 tok/s** M7 baseline (batch 1, 256 tokens, 32 warmup,
  `scripts/bench_train_step.py --mode decode`), posted on #30 and cited in
  `scripts/spec_decode.py:8`. That number is the record to beat.
- **Ship a native macOS artifact** — a CLI (and optionally an app) that loads the existing
  safetensors checkpoints, tokenizes with `swift/Sources/MonicaTokenizer`, generates, and quantizes, with
  no Python runtime installed. This is the POC's stated success metric doing its job: **the win is
  the local-hardware win — context length + tok/s** (CLAUDE.md; [13-code-model-moe.md](13-code-model-moe.md)),
  not benchmark scores.
- **Train the M12 small rung natively** (#195/#196) and **give the LSP fast loop a native
  harness** (#197).
- **Exploit the architecture's serving superpower: constant-size state.** A Mamba layer's state is
  a conv window `(d_conv-1, d_inner)` plus an SSM state `(n_heads, head_dim, d_state)` — fixed,
  independent of how many tokens have been consumed. `src/serve/sessions.py:40`
  (`per_session_state_floats`) computes it from **config alone**, and
  `src/serve/sessions.py:52` (`per_session_state_bytes`) turns it into a conservative admission
  budget. Only the ~12.5% attention layers carry a KV cache that grows with context. So the
  engine needs **no paged attention and no radix cache** — the machinery vLLM-class servers
  exist to provide is structurally unnecessary here.

**Non-goals** (from #163, and they are load-bearing — each one is a scope cut, not an oversight):

- Not a from-scratch C++/vLLM/TensorRT rewrite; the heavy lifting stays in MLX.
- No paged attention / radix cache (above).
- **Greedy-only** speculative decoding in v1 — no temperature>0 rejection sampling.
- **Weight-only** post-training quantization; no W8A8 activation quant.
- No Swift DPO/GRPO step factories in v1. **Python stays authoritative for post-training.**

And the standing constraint: **the parity gates are preserved, not replaced.** Everything the
Swift engine claims is checked against the Python seam in **fp32 at ~1e-4 relative**, the same
contract `src/conformance/` already enforces for forward-vs-step and backend-vs-backend — see
[03-conformance.md](03-conformance.md) for why bf16's epsilon is too coarse to gate on.

## Architecture — what "the engine" is

"The engine" is **prefill + decode + sampling** composed over the seam's primitives
(`src/model/interface.py`: `forward`, `step`, `init_state`, `save`/`load`). There are deliberately
**two implementations**:

- **The Python engine on the seam is the parity oracle.** `src/model/mlx_backend.py`,
  `src/model/mlx_train_step.py`, and the portable `src/serve/` layer define what is correct. They
  do not go away.
- **The Swift/MLX engine is the shippable artifact.** It is what a user runs.

Every Swift claim is checked against Python at fp32 ~1e-4: logits (#166), greedy token ids (#167),
prefill state (#169), loss + grad-norm (#195), checkpoint round-trip (#196).

The Swift engine sits **outside** the backend registry. `src/model/backend.py:45`'s `get_backend`
returns a `Backend` dataclass of **Python callables** — a native artifact is not a third branch
of it, and `tests/test_import_guard.py` does not cover Swift at all. This is the
**same third category the tokenizer already occupies** (CLAUDE.md's seam section): neither
above-seam portable Python nor a hardware backend behind `ModelInterface`.

## The four perf levers, and the seam methods they need

Each lever ends in a conformance gate. Being precise about **what already exists** is the point of
this section — three of the four are further along than they look.

| Lever | Seam method | Status today | Gate |
|---|---|---|---|
| **Prefill via parallel scan** (#165 → Swift #169) | `prefill(token_batch, seg_ids=None, *, last_only=False) -> (logits, State)` — **genuinely new** to `ModelInterface` | The scan **already computes the carry-out state and throws it away**: `new_states` is `(B, nc+1, H, P, N)` at `src/model/mlx_backend.py:252`, and `S_enter = new_states[:, :-1]` at `:253` drops the last entry. CUDA does exactly the same. **Shipped in #165**: `parallel(..., return_state=True)` surfaces the carry-out (the CUDA fused path passes `return_final_states=True`), `forward_prefill` on each block class pairs it with the conv window and the attention KV, and `prefill` is on the seam in both backends. `seg_ids` and continuing a non-fresh session are explicit non-goals in v1 (the packing-aware scan masks the carry-out row to zeros; RoPE positions are seeded from 0). `src/serve/generate.py` now prefills the prompt in one call | `src/conformance/prefill_decode_parity.py` (shipped): prefill-then-decode logits == pure step-by-step decode, **and** the extracted `State` == the step-produced state, element-wise, fp32 ~1e-4 |
| **Quantization** (#168 — **done**, see below) | quantized `load` — `Checkpoint.load` decodes an optional `quant` block from the `.config.json` sidecar (`mode`, `group_size`, `targets: {module_path: bits}`) and applies `MLXNN.quantize` before loading. Absence of the block = an fp checkpoint, byte-identical to before | `src/eval/quantize.py`'s `mlx_affine_quantize`/`mlx_affine_dequantize` (the ACTUAL `mx.quantize`-compatible format — see below; `quantize_dequantize` is a DIFFERENT, older variant, #51's measurement contract only) + `scripts/quantize_checkpoint.py` driver, weight-only W8/W4/W2 | `src/conformance/quant_parity.py`: top-1 agreement + KL vs the fp model, bits-dependent thresholds. The fp32 gates stay on the fp model, unchanged |
| **Speculative decoding** (#172) | `verify_block(tokens, state) -> (logits_list, state_list)` — implemented on **MLX only, both languages now** (`src/model/mlx_backend.py:897` / `swift/engine/Sources/MonicaEngine/MonicaModel.swift`'s `verifyBlock`, #264). It is **not** on `ModelInterface` and CUDA has no equivalent; promoting it to the seam is still the open work — #264 was a Swift-only port of the existing MLX-only shape, not a seam change | Drafter + accept rule are already portable: `src/serve/spec_decode.py` (`propose:27`, a prompt-lookup drafter needing no second model; `first_mismatch:53`). Driver: `scripts/spec_decode.py`, which calls `verify_block` at `:127`. **Greedy-only by construction** — `first_mismatch` compares against the verifier's *argmax*, stated in the module docstring | output **byte-identical to greedy decode** — what `tests/test_spec_decode.py:85` already asserts on MLX — plus accept-rate and speedup reported by the bench (#170). The Swift port is gated by `monica-parity` (row-0 logits vs `step_logits`, final state vs `prefill.safetensors`, an in-process rollback identity) — no new checked-in fixture |
| **Continuous batching** (**deferred**, per #163) | `step_batch` | `step` already takes a `(B,)` token array against a `(B, …)` state (`init_state(batch_size)`, `src/model/mlx_backend.py:681`), so **uniform** batching works today. What is missing is per-sequence admission / eviction / ragged lengths layered on `SessionStore`'s LRU (`src/serve/sessions.py:65`) | batched decode == per-sequence decode for the same inputs, fp32 ~1e-4. Specify the gate now; the lever stays **deferred** until the single-stream Mac path works |

**Prefill, concretely.** The carry-out is `new_states[:, -1]` — the entry `S_enter` discards —
read at layer `L-1`; tail chunks are zero-padded with `delta = 0`, so the padded positions
contribute identity decay and the carry-out is the true end-of-sequence state. The conv half of
the state is the last `d_conv-1` positions of the post-`in_proj` stream, matching what
`MambaBlock.step` (`src/model/mlx_backend.py:378`) expects to be handed. Attention layers contribute the
RoPE'd K/V produced by the sequence forward; MoE layers contribute the stateless placeholder pair
(`src/model/mlx_backend.py:692-694`).

> **The constant-size state is what makes all four levers cheap.** No paged attention, no radix
> cache, and turn-boundary snapshot/undo rides the same opaque blob — `RewindTree`
> (`src/serve/rewind.py:42`) snapshots a **fixed-size** `State`, not a cache that grows with the
> conversation. **Composition point and entry point (#305):** `SessionHistory`
> (`src/serve/sessions.py`) joins `SessionStore` to `RewindTree` — `commit_turn` is
> `tree.commit(store.get_state(...))`, `rewind_to` is `store.set_state(..., tree.rewind(...))` —
> and `scripts/generate.py --interactive` is the stateful continuation REPL a person types
> (`/rewind`, `/tree`, `/help`, sized by `--rewind-depth`). Turns after the first, and every turn
> after a rewind, call `serve.generate.generate(..., prefill=False)`, because `SessionStore.prefill`
> is fresh-session-only and a restored snapshot's position is unknowable above the seam.

## Quantization (#168)

**The premise the issue stated turned out to be wrong, and the fix was found empirically.**
The issue described `src/eval/quantize.py`'s existing `quantize_dequantize` as "group-wise
affine, the same scheme as `mx.quantize`". It is not: `quantize_dequantize` is classic
min/max affine (`scale = (hi-lo)/(2**bits-1) >= 0`, `bias = lo`) — #51's fake-quant
MEASUREMENT contract, and it stays exactly as it is. `mx.quantize`'s affine mode is a
different, zero-preserving, edge-exact variant, verified against `mx.quantize` 0.31.2 for
bits ∈ {2,4,8} × group_size ∈ {32,64,128}:

```
n     = 2**bits - 1
mn,mx = group min, group max
mask  = |mn| > |mx|
s0    = max(mx - mn, 1e-7) / n ;  s0 = mask ? s0 : -s0
edge  = mask ? mn : mx
q0    = round(edge / s0)
scale = (q0 == 0) ? s0   : edge / q0      # scale may be NEGATIVE
bias  = (q0 == 0) ? 0.0  : edge
codes = clip(round((w - bias) / scale), 0, n)
```

With this, numpy reproduces `mx.dequantize(mx.quantize(w))` to <1e-5 and `scales`/`biases` to
`rtol=1e-5`, and the packed `uint32` layout (little-endian within each word, `32/bits` codes
per word, groups along the last axis) is byte-identical to `mx.quantize`'s own output. This
is `src/eval/quantize.py`'s `mlx_affine_quantize`/`mlx_affine_dequantize`/
`pack_uint32_codes`, gated by `tests/test_quantize_mlx_format.py` — the load-bearing test
the rest of #168 trusts.

**Format.** `quantize_portable_state_dict` replaces each targeted module's `{path}.weight`
with the packed `uint32` codes and adds `{path}.scales`/`{path}.biases` (both fp32 — what
`mx.quantize` returns, and what mlx-swift's `update(parameters:verify: .all)` expects; MLX
has no symmetric mode, so the on-disk format always carries both). `quant_targets` narrows
`is_quantizable`'s 2-D-float floor by excluding `dt_proj`/`router` by name (load-bearing dt
path; tiny argmax router) and any tensor whose last axis doesn't evenly divide `group_size`
(MLX has no ragged-group fallback, unlike `_effective_group_size`'s whole-row fallback for
the OTHER quant scheme). `save_weights(..., quant=quant_block)` merges `{"mode": "affine",
"group_size": ..., "targets": {module_path: bits}}` into the `.config.json` sidecar; absence
leaves the sidecar byte-identical to before, and `load_config_sidecar` drops the `quant` key
silently (no "unknown field" note) rather than raising.

**Validation is three DIFFERENT comparisons, deliberately not conflated.** The fp32 gate
(`rtol=1e-4/atol=1e-5`) stays exactly as it is and protects the unquantized path only; a
quantized path cannot meet it by definition.

1. **Format correctness (exact, 1e-6).** The weights Swift dequantizes from the packed
   checkpoint (`MLX.dequantized` on its loaded `.weight`/`.scales`/`.biases`) must equal what
   Python packed (`dequant_ref.safetensors`) — a tight gate on the packing/scale/bias
   contract, not involving the quantized kernel at all.
2. **Kernel/logit agreement (loose, fixture-carried tolerance).** The Python reference for
   `forward_logits`/`step_logits`/`hidden.*` is a **fake-quant** model (dequantized weights
   loaded into the unmodified `MLXMambaModel` — no change to `mlx_backend.py`), while Swift
   runs the TRUE quantized GEMM (`quantizedMM`). The weights are numerically identical; the
   only divergence is accumulation order/precision. This gets its own `rtol`/`atol`, carried
   per-fixture in `meta.json` (`2e-2`/`2e-2` for the checked-in fixtures — a deliberate
   *sanity* bound, not a precision claim).
3. **Quality vs the fp model (statistical, never `allclose`).** `src/conformance/
   quant_parity.py`'s `check_quant_parity` — top-1 agreement + mean KL vs the ORIGINAL fp
   model's logits, bits-dependent thresholds (`QUANT_THRESHOLDS`: int8 top1≥0.99/KL≤1e-3,
   int4 top1≥0.95/KL≤5e-2). This is the statistical acceptance criterion the issue asks for.

`monica-parity` reads a fixture's optional `rtol`/`atol` from `meta.json`, defaulting to the
fp32 constants when absent (**this is the ONLY piece of #266's general precision→tolerance
contract #168 needed** — a minimal per-fixture hook, not a general table or an fp16/bf16
fixture family; #266 builds the general contract on top of this hook rather than duplicating
it), and runs all three checks above for any fixture whose `meta.json` declares
`quant_bits`.

**The tied head.** `MonicaModel.head` was changed to call `embedding.asLinear(h)` instead of
`matmul(h, embedding.weight.transposed(1, 0))` — semantically identical for the fp `Embedding`
base class (a no-op for the existing fp32 gate, landed and verified as its own commit), but
load-bearing under quantization: `QuantizedEmbedding.weight` is the packed codes, so the old
matmul would silently compute nonsense on it, while `asLinear` dispatches to
`QuantizedEmbedding`'s `quantizedMM` override. Since the tied head shares the embedding's
weight, quantizing the embedding quantizes the head too — the accuracy risk the issue flags.
`quant_targets(..., head_bits=...)` is the lever: a `--quant-head-bits` override (default 8
whenever `--bits 4`) is a data change, not a code branch. The checked-in `toy-moe-int4`
fixture needed it — plain int4-on-embedding pushed this random-init toy model's KL past the
int4 threshold; `toy-moe-int8`/`toy-moe-int4` both needed `--seed 1` for the same underlying
reason (a toy model's near-uniform random-init logits amplify KL disproportionately vs a
trained model — see `swift/engine/Fixtures/README.md`).

**Not measured by #168.** Decode-speed and real memory-footprint wins need a real trained
checkpoint, not a toy fixture — `scripts/quantize_checkpoint.py` reports the packed/original
byte ratio as the footprint evidence, and throughput is left to a future bench issue.

## Training on the Swift engine (#195/#196/#197)

Half of M13's scope, and the half with the most unknowns.

**The train step (#195).** Forward + backward + optimizer in Swift via mlx-swift autodiff,
mirroring `make_train_step` (`src/model/mlx_train_step.py:74`): gradient accumulation over
micro-batches, global-norm gradient clipping, and the contract `src/train/loop.py` injects —
`TrainStepFn = Callable[[ModelInterface, list, float], dict]` (`src/train/loop.py:47`), i.e.
`(model, micro_batches, lr) -> {loss, grad_norm, …}` where `micro_batches` is a list of
`(inputs, targets)` of length `grad_accum`. Only the base LM step is in scope; the SFT factory
(`src/model/mlx_train_step.py:107`) is a natural follow-on, and the DPO/GRPO factories
(`src/model/mlx_train_step.py:163` and `:197`) are explicit **non-goals** for v1.

**Loss scaling.** Mirror the split, not just the numbers: `src/train/loss_scale.py:20`'s
`DynamicLossScaler` is a **portable policy** that decides the scale and reacts to
overflow, while the *backend* performs the inf/nan check and skips the offending step. A Swift
implementation reproduces the policy; it does not re-invent it. See
[05-training.md](05-training.md).

**The gate.** Loss and grad-norm versus the Python MLX train step on a fixed-seed toy batch,
in the spirit of `src/conformance/backend_parity.py` — fp32, ~1e-4.

**Checkpoint I/O (#196) — shipped.** Both directions against `src/train/checkpoint.py:75`
(`save_weights`): safetensors plus the `<path>.config.json` sidecar. Python-written checkpoints
load in Swift and Swift-written checkpoints load in Python, bit-for-bit on the tensors.

*The reader half already existed.* `Checkpoint.load`/`loadInto`
(`swift/engine/Sources/MonicaEngine/Checkpoint.swift`) landed with #166/#168 — decode the
sidecar, apply an optional `quant` block, pop the non-parameter `moe_route_bias.{i}` keys, and
`update(parameters:verify: .all)`. #196's net-new work was the **writer**
(`Checkpoint.save`/`portableStateDict`) and the Swift→Python direction of the round trip.

*The writer* mirrors `_portable_state_dict` exactly: `model.parameters().flattened()` plus,
for each `MoEBlock` whose route bias is ACTIVE, `moe_route_bias.{i}` — emitted conditionally,
because an unconditional key would break Python's `num_parameters()` accounting
(`tests/test_moe.py`, `tests/test_sizing_mlx.py`; the bias is training-side routing state, not
a parameter, on either side of the seam). It rejects `bfloat16` at save time (Python's
`safetensors.numpy` reader has no bf16 numpy dtype — better to fail at the write, not three
files later on the read), and writes atomically (`.safetensors`-suffixed temp path,
`replaceItemAt`/`moveItem`), mirroring `checkpoint.py`'s `_atomic_write_bytes` discipline.

*The sidecar-fidelity decision.* `MambaConfig.to_dict()` on the Python side is
`dataclasses.asdict(self)` — **every** field (Muon / `torch_compile` / `fp8_experts` /
data-side knobs / `quant`). Swift's `MambaConfig` only *decodes* the ~22 fields the inference
path reads. Re-encoding just that subset on save would silently reset every other field to a
Python dataclass default on the next Python read — a quiet corruption of the cross-backend
bridge. Instead, `MambaConfig.load(sidecar:)` decodes the file **twice**: once into
`MambaConfig` (typed, known fields), once into a raw `[String: JSONValue]`
(`swift/engine/Sources/MonicaEngine/Config.swift`'s minimal `Any`-Codable enum, order-sensitive
— `Int` is tried before `Double` so `"d_model": 64` doesn't become `64.0`). `save(sidecar:)`
writes the raw dict back, **overlaid** with the known fields' current values, so an unmodeled
field (or `quant`) survives verbatim while a config actually mutated in Swift is still
reflected. This is deliberately **not** byte-identical to Python's `json.dumps(cfg, indent=2)`
(dataclass field order can't be reproduced without hardcoding it); the gate is *semantic* dict
equality (`scripts/check_swift_checkpoint.py`), not `cmp`. A config built directly in Swift
(no `rawSidecar`) writes only the known fields — lossy by construction, documented at the call
site, and not a path any checkpoint destined for Python should go through.

*The round-trip gate* has two halves, because a Swift binary alone cannot prove the direction
that matters to Python. `monica-parity --roundtrip-out <dir>` (Swift side, no new checked-in
fixture — see `swift/engine/Fixtures/README.md`) exercises, per fixture: (a) an identity round
trip — save, reload, assert every parameter tensor is **bit-identical** (save/load is lossless
by construction, so `rtol`/`atol` would be the wrong tool here), the sidecar round-trips to an
equal `MambaConfig`, and forward logits still match the Python oracle; (b) a mutation round
trip — perturb one parameter, save, reload, confirm the perturbation survived exactly, which is
what catches a writer that degenerates into a file copy; (c) route-bias presence/absence
matching `_portable_state_dict`'s conditional; (d) for the quantized fixtures, that the `quant`
block re-decodes to the same `mode`/`group_size`/`targets` (the packed tensors themselves are
already covered by (a)'s exact comparison). `scripts/check_swift_checkpoint.py` (Python side)
then loads what that step wrote: sidecar dict equality, quant-block equality, key-set equality,
and forward logits vs each fixture's `reference.safetensors`. It explicitly **SKIPs** — never
silently passes — the two quantized fixtures' model-construction/forward checks, because
`src/eval/quantize.py` is a fake-quant *measurement* spike (see "Quantization" above): the
Python MLX backend never builds an `nn.QuantizedLinear`/`nn.QuantizedEmbedding`, so there is no
loader reachable from a plain `MLXMambaModel` that can consume packed
`.weight`/`.scales`/`.biases`. Sidecar/quant-block equality still gates those two fixtures; the
packed-tensor and quant-block round trip is gated Swift-side by (a)/(d) above. An empty
`--roundtrip-dir` is itself a FAIL in the Python script, not a pass — a checker that can't see
its target must never read green.

*Explicitly out of scope, and why.* The **slot-a/slot-b resume bundle**
(`CheckpointStore`, `src/train/checkpoint.py:121` — step / loss-scale / RNG / optimizer state /
dataloader position) is **not** implemented here. That's the within-backend concern the
two-concern split exists to keep separate from portable weights (optimizer state does not need
to port across backends), and #196 delivers exactly concern (1) of that split. Concretely: at
the time #196 landed, #195 (the Swift train step) had not merged — `main` had no Swift
optimizer type, no `monica-train`, and no `train.safetensors` fixture — so there was nothing
real to serialize yet, and inventing an optimizer-state format against nonexistent types would
have collided with #195 when it eventually lands. `Checkpoint.save` takes only a model and a
destination URL, so a future optimizer serializer is an orthogonal file written into the same
directory, not a refactor of this one. Swift resume (optimizer state + step) should be filed as
its own issue once #195 lands, referencing this note.

**The LSP fast loop (#197) — SHIPPED.** A native persistent `typescript-language-server` client
plus a per-step completion-list logit-mask hook, ported from `src/lsp/ts_service.py` /
`src/lsp/jsonrpc.py` / `src/serve/constrained.py` / `src/lsp/completion_mask.py`. The
constrained-decode hook itself (`Generator.allowedIdsFor`) already existed from #167/#169 — #197
built the native **producer** and wired it through `monica-generate`.

*Package placement.* `MonicaLSP` is a new **target** in `swift/` (the zero-dependency
`MonicaTokenizer` package), not a new package and not a new dependency edge —
`swift/Package.swift` stays `dependencies: []`. It needs no MLX at all (`Process` + pipes + JSON
+ string scanning), so the framing/demux/trie/scanner majority of the code — the parts with real
correctness risk — is built and self-checked (`swift run monica-lsp --self-test`, binary-free, no
subprocess) on **both** `swift-macos` and `swift-linux`, with no mlx-swift build in the loop. Only
the wiring lives in `swift/engine/`: `monica-generate` gains `--lsp-mask`/`--lsp-project`/
`--lsp-file` flags that construct a `TsLspClient` + `CompletionMasker` and hand
`allowedIdsFor:` to `Generator.generate` (`.product(name: "MonicaLSP", package: "swift")` — same
path-dependency identity trap as the tokenizer edge, see #167's note two sections up). Absent the
flags, `allowedIdsFor` stays `nil`: an exact no-op, so every existing `monica-generate`
invocation is byte-identical in behavior to before this issue.

*Why this doesn't reduce `diagnostics` latency* — and does not claim to. #278 diagnosed ~350ms of
the measured `diagnostics` round trip as a **client-side debounce hardcoded inside
`typescript-language-server` 5.3.0** (`cli.mjs:17868`/`:20516`), not type-checking; that debounce
lives in the Node process on the far side of the pipe, so no Swift client can cut it. The fast
loop instead runs on `completions` (debounce-free, 2,500+ calls/s on the Python reference,
[12-lsp-in-the-loop.md](12-lsp-in-the-loop.md)'s #197 section has the Swift-vs-Python numbers),
and the masker issues exactly **one completion query per identifier span, not per token** — the
load-bearing amortization, reported as queries per 100 generated tokens in the bench output.

*Reaping.* `ProcessSupervisor` (`swift/Sources/MonicaLSP/ProcessSupervisor.swift`) is the only
type in the codebase allowed to own the child process: a scoped `withServer { }` entry point
reaps on return/throw via `defer`, `shutdown()` is idempotent and follows the same
terminate-then-bounded-wait-then-kill ladder as `src/lsp/ts_service.py:446`, and a process-wide
`sig_atomic_t` + `SIGINT`/`SIGTERM`/`SIGHUP`/`atexit` backstop kills the last-spawned child even
on a signal Swift's `defer` never runs for. `typescript-language-server` 5.3.0 also empirically
honors the LSP `processId` parent-death watch (verified via `monica-lsp --probe-reap`'s SIGKILL
scenario on this pin — the server exits on its own even when the parent is killed with no chance
to run a handler), so a `SIGKILL`ed `monica-lsp` still leaves no orphan in practice.

*Out of scope, filed as follow-ups rather than built here*: a Swift port of #279's raw-`tsserver`-
protocol (`semanticDiagnosticsSync`) transport, if #279 lands and the bypass proves worth a second
client; tree-sitter grammar masking for the fast loop (the Python extractor,
`src/lsp/ts_boundaries.py`, is an optional `[eval]` extra with a C dependency the zero-dependency
`swift/` package must not take).

## The native-engine investigation (B1–B4)

Four options were evaluated. The decision is **B1**.

**B1 — Swift + MLX (chosen, Mac-first).** Reuses our MLX numerics, weights, and quantization
wholesale. mlx-swift ops mirror the Python backend close to 1:1 (`einsum`, `cumsum`, `exp`,
`conv1d`, softmax), so porting `SelectiveSSM.parallel` (`src/model/mlx_backend.py:198`) and
`SelectiveSSM.recurrence` (`src/model/mlx_backend.py:282`) is mechanical and, more importantly, **directly
parity-checkable** against the oracle. The surrounding ecosystem is turnkey: `mlx-swift-lm` for
generation/sampling/KV/quantized loading, `mlx-swift-examples`' `llm-tool` as a CLI shell to fork
— and the **tokenizer is ours** (`swift/Sources/MonicaTokenizer`), already cross-platform, already
bit-identical, already CI-gated, so there is no swift-transformers dependency and no separate
Linux-tokenizer question. **Not Mac-locked:** MLX has a CUDA backend and mlx-swift builds with
CUDA on Linux (ml-explore/mlx-swift **#320**, *"new[CMake, CI]: add CUDA GPU backend to Linux CMake
Swift build option"*, merged **2025-12-18**, with a `cuda-12.9 / swift-6.2.3 / ubuntu-24.04 /
x86_64` CI job), so the same Swift codebase can extend to Linux/NVIDIA later. **Deferred** —
Apple Silicon first.

**B2 — C++ + MLX (`mlx-c` / the MLX C++ API).** Equal Apple-Silicon payoff, more boilerplate: an
own tokenizer binding and an own generation loop. Prefer only if an **embeddable C++ library with
no Swift toolchain** becomes a requirement.

**B3 — `mx.fast.metal_kernel`.** A fused chunked-SSD-scan + conv/recurrence kernel (#171). This is
a **profile-gated optimization, not a path**: sequenced *after* the port lands and only if a
profile shows the generic ops are the bottleneck. It is **Metal-only**, so it does not travel to
CUDA. #30's rejected-lever list already flagged a custom SSD kernel as high-risk to the
`parallel`/`step` parity invariant; that risk is unchanged, which is why the gate is the same
fp32 ~1e-4 comparison rather than a weaker one.

**B4 — llama.cpp / ggml.** The widest reach by far (CPU/Metal/CUDA/Windows/some ROCm) and ggml
already carries Mamba-2 + hybrid ops. But it needs a converter and a new architecture written in
C++, and **MoE-Mamba is not supported in ggml** — disqualifying for the M12 MoE spine (#198). It
is also **inference-only**: no training path, which is now half of M13's scope. Best **future**
production/server target, gated on architecture freeze + ggml MoE support.

> **Locked: B1 — Swift + MLX, Mac-first.** Linux/CUDA is reachable via mlx-swift#320 but
> **deferred**; ggml/llama.cpp is **deferred** behind architecture freeze + MoE-Mamba support.

**The prior contrary decision, stated honestly.** Issue #30 (closed) contains a **2026-06-09
assessment that rejected a Swift/mlx-swift rewrite**, on these grounds: mlx-swift binds the same
C++ mlx core via `mlx-c` (identical kernels, identical graph engine); Python host work is ~2% of
the poc training step; published Swift-vs-Python comparisons show parity at best; and mlx-swift
exposed **no gradient-checkpointing API** while `config/poc.yaml:22` requires
`grad_checkpoint: true`. That assessment was correct **for what it evaluated** — a
training-throughput rewrite of the Python MLX train step, where the host language was never the
bottleneck, so the rewrite bought nothing. M13 is a different proposition: a **native
no-Python-runtime artifact**. Its payoff is a shippable macOS CLI/app, single-stream *latency*
levers (prefill-via-scan, quantization, speculative decoding, optionally a fused kernel), and a
low-latency LSP fast loop — none of which are "make the same graph run faster in the same
process". The grad-checkpoint gap, however, **survives as a live risk** on the training half; see
Risks below.

## Library reference

| Library | What we'd use it for | macOS | Linux/CUDA | Notes |
|---|---|---|---|---|
| **`swift/Sources/MonicaTokenizer` (ours)** | tokenize / detokenize; the corpus packer | ✅ | ✅ | #191/#245. Own byte-level BPE, **no external deps** (`swift/Package.swift`), bit-identical across platforms, CI-gated by `swift-macos` / `swift-linux` / `swift-parity` (#246). **Displaces swift-transformers entirely** |
| `mlx-swift` | tensors, autodiff, the model port | ✅ | ✅ (build) | The core dependency. CUDA build via ml-explore/mlx-swift#320 (merged 2025-12-18) — available, deferred |
| `mlx-swift-lm` | generation loop, sampling, KV cache, quantized loading | ✅ | untested | **NOT used — a deliberate deviation, #167.** `mlx-swift-lm`'s `LanguageModel`/`KVCache` protocols assume a transformer whose state *is* a KV cache; `MonicaModel`'s state is a `[LayerState]` enum carrying Mamba conv windows + SSM state, so the adapter would be larger than the ~200-line native port it replaces (`Sampler.swift`, `Generator.swift`). It would also pull `swift-transformers` + Hub into the `swift-engine` CI job for a tokenizer we deliberately do not use, and its MLX-RNG samplers cannot reproduce `src/serve/sampling.py`'s numpy draws — porting the sampler directly is what makes "matches `src/serve/sampling.py`'s semantics" a checkable claim (AC1: greedy ids are bit-for-bit; sampled draws are semantically but not id-identical, since numpy's `Generator` stream cannot be reproduced in Swift) |
| `mlx-swift-examples` (`llm-tool`) | CLI shell to fork for #167 | ✅ | untested | **Not forked** — `monica-generate` is a new executable target in `swift/engine/`, written against `scripts/generate.py`'s flag surface (the design doc's own table already called this example code, not a dependency) |
| `swift-transformers` | — | — | — | **Not used.** Superseded by our own tokenizer (#163's tokenizer note, #167) |
| `mlx-c` | the B2 C++ path | ✅ | ✅ | Only if an embeddable C-ABI library becomes a requirement |
| `mx.fast.metal_kernel` | fused SSD-scan/conv kernel (#171) | ✅ | ❌ | **Metal-only.** A CUDA sibling kernel would be separate work (#171) |
| `metal-cpp` | hand-written Metal from C++ | ✅ | ❌ | Below B3; no current need |
| `ggml` / `llama.cpp` | the B4 port | ✅ | ✅ | Widest reach, but needs a converter + new arch, **no MoE-Mamba**, inference-only |

## Staged roadmap

The #163 dependency order:

1. **#164** (this document) and **#165** (`prefill` on the seam, both Python backends) — no
   dependencies, start immediately.
2. **#166** — port the model to mlx-swift + the logit-parity harness. Everything else waits on it.
3. **#167** (generation CLI — **done**), **#195** (train step + optimizer), **#196** (checkpoint
   I/O) — in parallel once #166 lands.
4. **#197** (LSP harness — **done**), **#168** (quantization — **done**), **#169** (Swift
   prefill — **done**), **#170** (Apple-Silicon benchmark harness — **done**).
5. Stretch: **#171** (fused Metal kernel), **#172** (speculative decoding).
6. **#267 — done: the poc-scale Swift parity gate, generate-on-runner (CI, dispatch/schedule
   only).** #166 gated `swift-engine` against four checked-in *toy*-scale fixtures and
   verified `config/poc.yaml` (d_model 768, 24 layers, vocab 50280) manually, once, locally.
   #267 turns that into a standing gate without checking the 571 MB poc fixture into git:
   two new jobs, `poc-fixture-oracle` (macOS, Python+MLX, no Swift toolchain — generates the
   fixture, hashes it, uploads a ~2 KB sha256 manifest + `meta.json`) and `poc-parity`
   (macOS, `needs: [poc-fixture-oracle]` — regenerates its own independent copy, `cmp`s the
   two manifests, THEN runs `monica-parity --fixtures` against the verified fixture; #267
   also gave it a `needs: swift-engine` edge purely to inherit that job's warm xcodebuild
   cache, which #302 dropped when the two jobs ended up in different workflow files —
   `needs:` cannot cross files, and the cache is still shared via `restore-keys`). Measured
   on an M1 Pro
   during planning: **5.2-5.8 s wall, ~2 GB peak RSS, 571 MB output**, and the two
   independent generations were **bit-identical, 7/7 files** — which is what makes the
   manifest `diff` a real guard against **#298** (MLX 0.32.0's deterministic-per-process
   buffer-reuse corruption) rather than a coin flip: two fresh processes on two runners would
   have to corrupt identically to pass it. Both jobs run on `workflow_dispatch` or a weekly
   Monday `schedule:` — **never `pull_request`/`push`** — because poc adds
   zero new *code-path* coverage over `toy` (pure Mamba, no attention/MoE/quant); what it adds
   is *scale* coverage of the tolerance contract (R2 below), which does not change PR to PR,
   so per-PR cost (a second macOS mlx-swift build) isn't worth paying. #267 expressed that as
   an `if: github.event_name == …` guard on each job inside `ci.yml`; **#302 (entry 7 below)
   replaced the guard with a separate workflow file that has no `pull_request`/`push` trigger
   at all** — the exclusion is now structural. `swift-engine` gained
   no `needs:` and no existing job's cache key/`if:` changed. No change to
   `src/conformance/tolerances.py`, `scripts/export_parity_fixture.py`, or any checked-in
   fixture. See `swift/engine/Fixtures/README.md` §poc for the operator-facing version and
   the local reproduction command.
7. **#302 — done: the weekly schedule now fires only the parity gate.** #267's `schedule:` was
   declared at *workflow* level in `ci.yml`. GitHub fires the whole workflow on that event and the
   four `if:` guards only filter the guarded jobs, so every Monday the other 8 jobs ran too —
   including three macOS runners, one of them `swift-engine`'s 60-minute budget. The comment above
   the trigger said the schedule existed for the poc gate; the behaviour disagreed. By the time
   #302 was executed it was **4 of 12** jobs guarded, not the 2 of 10 the issue reported: #195/#293
   had added the `train-fixture-oracle`/`-verify` pair after filing.

   *Three options were on the table.* **(1)** Invert the guards: add
   `if: github.event_name != 'schedule'` to the 8 unguarded jobs. **(2)** Move the 4 guarded jobs
   into their own workflow file with its own `schedule:`, and delete `ci.yml`'s. **(3)** Keep the
   behaviour and fix the misleading comment.

   *Option 2 was chosen*, on the constraint CLAUDE.md states as hard — *the poc jobs must NEVER run
   on `pull_request`/`push`*. Under option 1 that guarantee stays **expression-level**: it is an
   assertion eight `if:` expressions have to keep true, and each one is a chance to typo a PR gate
   into permanent silence (which is the failure mode the issue's own criterion 4 names). Under
   option 2 it becomes **structural** — `scheduled-parity.yml` has no `pull_request` key, so there
   is no event to be skipped under, and a PR produces no check entry for that workflow at all
   rather than a skipped one. That is the same move the repo already makes elsewhere: `swift/`'s
   zero-dependency property is preserved by *separating packages*, not by conditioning build edges.
   Option 3 was rejected as contradicting the issue outright. **All four** guarded jobs moved, not
   just #267's two: `ci.yml` lost its `schedule:`, so a `train-fixture-oracle` left behind would
   have had a dead `|| github.event_name == 'schedule'` clause and would have silently degraded to
   dispatch-only — verbatim the "gate stops running and nobody notices" failure.

   *One accepted deviation.* The issue's criterion 3 asked that `workflow_dispatch` still fire
   everything. It now takes **two** commands (`gh workflow run ci.yml`, `gh workflow run
   scheduled-parity.yml`). The capability — every job manually dispatchable, no inputs — is intact;
   the single command is not. That is inherent to option 2, and a `workflow_call` shim would
   reintroduce the coupling the split exists to remove.

   *What this deliberately gives up.* The weekly full run was not purely waste: it caught
   **environmental drift** that per-PR CI attributes to flake. Two instances in the preceding week
   — `swift selfcheck (macOS)`'s failure on PR #301 (the #279 tsserver/LSP debounce race) and
   `swift model parity (macOS, mlx-swift)` failing once and passing on retry on PR #310 — are
   exactly that class: read as noise against a changing diff, read as *the environment moved* when
   they happen on unchanged `main`. After #302 nothing ran those jobs on a cadence. That was an
   accepted, recorded cost, not an oversight, and it was **restored by #312** — see entry 8: once
   the files are split, a cron on `ci.yml` fires only `ci.yml`'s own 8 jobs and cannot reach a
   heavy one, so the coverage comes back without re-opening what #302 closed. Also dropped:
   `poc-parity`'s `needs: swift-engine` edge, which was cache-warming only
   (`swift-engine` produces no artifact it consumes) and cannot cross workflow files — the cache is
   still restored by the `swift-engine-${{ runner.os }}-` `restore-keys` prefix from the last push
   to `main`, with a cold 20-40 min vendored-C++ build as the accepted worst case inside the
   existing 90-minute timeout.

   *The gate.* `act` was rejected (needs Docker, cannot model macOS runners, and would be a manual
   step rather than a gate). Instead `tests/test_workflow_triggers.py` parses both workflow files
   and asserts the full trigger×job matrix — it runs in `ci.yml`'s `portable` job on every PR, so
   adding a `pull_request:` to `scheduled-parity.yml`, giving any `ci.yml` job an `if:`, retiming
   either cron (#312), or losing a job in a future move all fail a test. Two details in it
   are load-bearing: PyYAML's `safe_load` is YAML 1.1, where a bare `on:` key parses as the
   **boolean `True`** rather than the string `"on"` (the loader looks under both and asserts it
   found one, so a future PyYAML change fails loudly instead of reading as "no triggers, therefore
   nothing runs, therefore everything passes"); and the `if:`-expression evaluator **raises** on any
   form it does not recognise, because an unparsed guard is *unknown*, never *"the job runs"*.
8. **#312 — done: the drift cadence is back, on `ci.yml` only.** Entry 7 recorded the cost of the
   split as accepted; this reverses that half of it. `ci.yml` gained
   `schedule: - cron: "43 9 3 * *"` — 09:43 UTC on the 3rd of each month — which fires its own 8
   PR/push gates against **unchanged `main`**. That stable ref is the whole point: the failure
   class being detected (the hosted runner image moved — Xcode, Homebrew, npm, the mlx-swift
   toolchain) is attributed to *flake* when it lands on a changing diff and to *the environment
   moved* when it lands on a diff-free run. The two instances entry 7 names (PR #301, PR #310) are
   that class.

   *Three shapes were on the table*, from the issue. **(1)** A `schedule:` on `ci.yml`.
   **(2)** A third `drift.yml`. **(3)** Convert the two observed failures into deterministic tests
   and keep no cadence.

   *Shape 1 was chosen.* The key point, and the reason entry 7's parenthetical was **wrong as
   written** (it said a cron on `ci.yml` would re-open what #302 closed, and it is corrected in
   place above rather than annotated): what #302 removed was *coupling* — one workflow-level cron
   in a file holding both the cheap gates and the heavy poc gate, with the separation carried by
   four `if:` expressions. The file split removed that coupling **structurally**. A cron on
   `ci.yml` today fires exactly its 8 jobs and cannot reach a heavy job, because the heavy jobs are
   in a different file with its own trigger set. Shape 1 therefore restores only the half of the
   old weekly run that was worth keeping.

   *Shape 2 was rejected.* Duplicating the macOS job bodies into a second file creates a copy that
   silently drifts from the real ones — a drift detector running a stale copy of the job it is
   meant to watch is the BLIND-monitor failure, worse than no detector. The non-duplicating
   variant (a `workflow_call:` on `ci.yml` with `drift.yml` calling it) adds a trigger and a whole
   caller file to obtain what one `schedule:` key gives for free, and entry 7 already rejected a
   `workflow_call` shim as reintroducing the coupling the split exists to remove.

   *Shape 3 was rejected as a substitute*, and is partly already done: #279 is closed — the
   tsserver debounce race got a real fix — and closing it restored no cadence. The #310 mlx-swift
   retry-pass is not reducible to a deterministic test, because "the runner image moved" is not
   something an in-repo test can encode. Shape 3 also produces no stable-ref signal, which is the
   specific thing that was lost.

   *Monthly, not weekly.* Cost is not the deciding factor — `monica` is public, so GitHub-hosted
   macOS minutes are free. Signal-to-noise is: a red run on unchanged `main` has to be rare enough
   that someone reads it, and the drift sources move on a multi-week cadence. Going weekly later is
   a one-token edit to the cron plus the literal in the test.

   *The concurrency fix is part of the coverage, not a tidy-up.* `ci.yml`'s group was
   `${{ github.workflow }}-${{ github.ref }}`, shared by push and schedule runs on
   `refs/heads/main`. GitHub allows one in-progress plus one pending run per group and **cancels
   the previously pending run** when a newer one queues, so a monthly drift run queued behind an
   in-flight push run is cancellable by the next push — vanishing with no failure and no signal,
   which is exactly the BLIND failure this cadence exists to remove. The group now ends in
   `-${{ github.event_name }}`; `cancel-in-progress` still evaluates true only for `pull_request`,
   so PR-supersede behaviour is unchanged.

   *The gate.* `tests/test_workflow_triggers.py` grew two tests (the cron literal, the concurrency
   group) and DoD-1 was **rewritten, not weakened**: the deleted assertion was
   `"schedule" not in _triggers(ci.yml)`, and it is replaced by the property that assertion stood
   in for — a `schedule` event in `ci.yml` fires exactly `PR_PUSH_JOBS` and intersects `HEAVY_JOBS`
   in nothing. Trigger absence was the *mechanism* #302 happened to use; not-reaching-a-heavy-job
   is the *property*, and asserting it directly is strictly stronger. The two crons collide
   harmlessly — different files, different concurrency groups, different minutes (43 vs 17).
   Two limits are recorded in the `ci.yml` comment rather than left to be rediscovered: GitHub
   arms `schedule:` only from the default branch (so the first real run is post-merge, on the 3rd,
   checked with `gh run list --workflow=ci.yml --event=schedule`), and it disables scheduled
   workflows in public repos after 60 days of repository inactivity.

**Deferred set:** Linux/CUDA for the Swift engine, the ggml port, continuous batching, and Swift
DPO/GRPO step factories.

## Risks & open questions

- **Gradient checkpointing on mlx-swift.** #30 found no API for it; `config/poc.yaml:22` requires
  `grad_checkpoint: true` because the 24-layer backward otherwise exceeds 32 GB unified memory and
  swaps. #195 concedes this — checkpointing is an explicit non-goal of its v1. **Open:** does the
  M12 **small rung** (~120M active) backward fit *without* it on the target Mac? Note the Swift
  training target is that rung, **not** poc-24-layers-on-32 GB, so the answer may well be yes —
  but it is unmeasured.
- **MoE in Swift — RESOLVED (#166).** The concern was porting an *unfinished* router to a third
  implementation. #213 settled it (`15d699b`: the aux-loss-free load-balancing policy plus the MLX
  router), so #166 ported `MoEBlock` in full — the top-k double-argsort ranking, the biased-ranking
  branch, and the `moe_route_bias.*` load path — gated by the `toy-moe` and `toy-moe-biased`
  fixtures. Still Swift-side out of scope, as *training* surfaces with no effect on logits: load
  counting (`_count_loads`/`pop_load`) and the `set_route_bias` write path, both for #195.
  `src/model/cuda_backend.py`'s CUDA `MoEBlock`/`_Expert` landed with #214 (dropless
  grouped-gather routing, shared expert) — the `NotImplementedError` this line used to cite
  is gone. Swift porting the CUDA-specific gather dispatch remains out of scope for #166
  (Swift is inference-only and evaluates every expert the same way the MLX/dense reference
  does; gather is a training-throughput optimization with no logit effect to port).
- **`seg_ids` packing-aware forward path — RESOLVED (#263).** The three forward-path arms
  (`_chunk_seg_mask` -> `SelectiveSSM.chunkSegMask`, `_conv_seq_seg` -> `MambaBlock.
  convSeqSeg`, `AttentionBlock.forward_seq`'s block-diagonal mask) are ported and gated by
  `monica-parity`'s P6 section against a new checked-in `packed.safetensors` oracle
  (`toy`/`toy-hybrid`/`toy-moe`; see `swift/engine/Fixtures/README.md`). Deliberately
  scoped to inference: `forwardPrefill`/`prefill` still take no `segIds` — that is the
  same #165 exclusion as before (a packing-aware carry-out reads as zeros; see
  `SelectiveSSM.scan`'s `carryOutRequested` precondition), not a gap #263 left open. Also
  out of scope: plumbing `seg_ids` through a Swift training loop, data loader, or shard
  reader — none of that exists yet (#195/#271 territory), and this port is a
  prerequisite of #195 rather than a dependent of it.
- **`SessionStore`'s budget math ignores the attention KV cache.** `per_session_state_floats`
  (`src/serve/sessions.py:40`) charges every layer a conv window plus an SSM state and has **no
  attention term**. That is exact for a pure-Mamba config, but the M12 hybrid's ~12.5% attention
  layers carry a KV cache that grows with context, so the number under-counts for the model this
  engine will actually serve — the wrong direction for an admission gate. Open for the native
  engine's memory admission, and a likely Python-side follow-up.
- **Parity tolerance vs quantization.** fp32 ~1e-4 **cannot** gate a quantized path. The quant
  gate is a different contract — top-1 agreement + KL (#168) — and the two must not blur. The
  fp32 gates keep running against the fp model.
- **Weight-layout twist on load.** `src/model/cuda_backend.py:809` (`_portable_state_dict`) documents the
  one layout difference in the portable bridge: the depthwise conv weight is `(out, in/groups, k)`
  in torch but **`(out, k, in/groups)` in MLX**, and the portable format is the **MLX** layout —
  so torch transposes on the way out (`.transpose(1, 2)`) and back on the way in
  (`_load_portable`, `src/model/cuda_backend.py:820`). A third loader gets exactly this kind of
  detail wrong; #166 must mirror it.
- **Two engines is two maintenance surfaces.** Every seam change — #165's `prefill`, a promoted
  `verify_block` — now lands twice. The parity gates are the mitigation, and they are the reason
  the gates are non-negotiable rather than nice-to-have.
- **Where the Swift engine lives — RESOLVED (#166): a sibling package at `swift/engine/`.**
  `swift/Package.swift` is not touched. The rejected alternative was adding mlx-swift to it with
  `condition: .when(platforms: [.macOS])` on the target dependency — which does not work, because
  **platform conditions apply to build *edges*, not to *resolution***: SwiftPM would still clone
  mlx-swift (which vendors the whole `mlx` C++ tree) on Linux, putting a large network fetch and a
  new upstream-outage failure mode into `swift-linux`, a job that today needs zero network for
  dependencies. It would also drag `swift/Package.swift` from `swift-tools-version: 5.9` /
  `.macOS(.v13)` up to `6.0` / `.macOS("14.0")` **for the tokenizer too**. A sibling package keeps
  `swift/`'s zero-dependency, cross-platform property intact — the property that makes the #246
  bit-identity gate cheap and credible. `swift/engine/` is not inside any target path declared by
  `swift/Package.swift`, so `cd swift && swift build` ignores it exactly the way it already ignores
  `Fixtures/`. CI gains one job, `swift-engine` (macOS only); jobs 4/5/6 are unchanged.
- **Cross-package dependency direction — RESOLVED (#167): engine -> tokenizer, never the
  reverse.** `monica-generate` needs `swift/Sources/MonicaTokenizer` for encode/decode, so
  `swift/engine/Package.swift` gained a **path** dependency, `.package(path: "..")`, and the
  `monica-generate` executable target references
  `.product(name: "MonicaTokenizer", package: "swift")` — the product's `package:` argument is
  the path dependency's **directory basename** (`"swift"`), *not* its `Package(name:)`
  (`"MonicaTokenizer"`); the wrong form fails with `unknown package … valid packages are:
  'swift'`, verified empirically on Swift 6.3.3. `swift/Package.swift` is untouched — it still
  declares `dependencies: []`, so `cd swift && swift build` on Linux still resolves nothing,
  and `swift package show-dependencies --format flatlist` from `swift/` still prints nothing
  (CI-guarded, `swift-engine` job). Path dependencies are not recorded in `Package.resolved`,
  so this does not perturb the `swift-engine` cache key. Only the `monica-generate`
  **executable** gained the tokenizer edge — the `MonicaEngine` library and `monica-parity`
  stay exactly as coupled as before (`Sampler.swift`'s RNG is a deliberate ~10-line duplicate
  of `MonicaTokenizer.SplitMix64` rather than a shared type, precisely to keep the library's
  dependency shape unchanged).
- **#169 shipped the parallel-scan prefill.** `MonicaModel.prefill(tokens, lastOnly:)` runs
  one SSD chunked-matmul scan over the whole prompt (`SelectiveSSM.parallelWithState`,
  `MambaBlock`/`AttentionBlock`/`MoEBlock.forwardPrefill`) and hands `Generator.generate`
  (default `usePrefill: true`) the exact `[LayerState]` an `L`-step walk of `model.step`
  would have left. The old per-token sequential prefill stays reachable as
  `usePrefill: false` (`monica-generate --no-prefill`) — the AC3 baseline and the AC1 A/B
  reference, not dead code. `monica-parity`'s P1-P5 checks (fp32 `rtol=1e-4`/`atol=1e-5`,
  unchanged) gate: prefill logits vs the Python oracle and vs `forward`; `lastOnly`
  honesty; **state handoff, element-wise, both cross-language (vs a new
  `prefill.safetensors` oracle) and intra-Swift (vs `step`'s own recurrence)** — the
  load-bearing check, since a wrong carry-out is silent in the prompt's own logits and only
  corrupts tokens generated afterward; prefill-then-decode vs pure step-by-step; and exact
  greedy-id equality through the prefill path. AC3 (`monica-generate --bench-prefill`,
  wired into CI's `swift-engine` job as an informational, non-gating step) measures
  sequential-vs-parallel prefill latency. First green-CI numbers, from run
  **31777284815 on `main`** (`Fixtures/toy`, prompt_len=128, iterations=5), on the
  **hosted macOS CI runner** — NOT a local-hardware measurement, see #170 below:

  ```
  bench-prefill: prompt_len=128 iterations=5  sequential=332.62ms  parallel-scan=6.01ms  speedup=55.30x
  bench-prefill: sequential argmax=186  parallel-scan argmax=186
  ```

- **#170 — the benchmark harness.** `swift/engine/Sources/MonicaEngine/Bench.swift` +
  the `monica-bench` executable (`Package.swift`; deps `MonicaEngine` + `MLX` only, no
  tokenizer edge — `swift/Package.swift` is untouched) generalize the AC3 measurement
  into a reusable harness: `--mode prefill/decode/memory/all`, `--weights` (a real
  checkpoint, quantized automatically via its `quant` sidecar) or `--config` (a
  shapes-only random-init sidecar for poc-scale runs, `swift/engine/Benchmarks/
  configs/*.json`, exported by `scripts/export_bench_config.py` and kept honest by
  `tests/test_bench_config_export.py`), `--quantize 8|4` for an in-engine
  throughput/footprint approximation of `quant_targets` on a `--config` model, a
  machine-identified `--json` record, and `--baseline` regression flagging that
  **refuses to compare across machine ids** (`SKIPPED`, never a false-green `OK`).
  `monica-generate --bench-prefill` now delegates to the same `Bench.prefill`, so the
  55.30x figure above stays the comparable historical baseline rather than being
  orphaned. `monica-bench --self-test` (deterministic: analytic byte arithmetic, the
  quant filter, baseline-comparison logic, arg validation — no weights, no timing) is a
  CI **gate**; the timing modes are informational, per the same noisy-hosted-runner
  reasoning as AC3 above. Python side: `scripts/bench_context.py` gained
  `--prefill-mode {sequential,parallel,both}` (the `parallel` arm calls `model.prefill`,
  matching the Swift default) and `--json` emitting the same `source`-tagged record
  shape as `monica-bench --json`, so the two harnesses' output is diffable directly.
  Results ledger, machine-keyed tables, and the CI-vs-developer-machine provenance
  split: `docs/benchmarks.md`. **No acceptance criterion in #170 depends on a number
  this repo cannot produce on the box that authored it** — mlx-swift cannot execute
  without Xcode's Metal toolchain (`swift build` compiles; running needs
  `default.metallib`), so every genuinely local-hardware Swift-engine number in
  `docs/benchmarks.md` is recorded as "not yet measured" with the exact command to fill
  it in. Python-MLX numbers (no such constraint) are measured locally where noted.
- **R2 (#267) — the fp32 parity band is relative-dominated at poc scale, not
  absolute-dominated like it was derived to be.** `src/conformance/tolerances.py`'s Step 2
  designs the fp32 band (`rtol=1e-4/atol=1e-5`) to be absolute-dominated for `|logit| ~ 8`
  ("`rtol = atol / 8` so the relative term contributes at most one `atol` at the largest
  logit"), calibrated on the `toy*.yaml` configs only. At `config/poc.yaml` scale
  (d_model 768, 24 layers, vocab 50280), measured `forward_step_max_abs_diff = 3.62e-05` is
  already 3.6x `atol` on its own, and `greedy_margin_min = 23.94` — logits ~3x larger than
  toy's — mean the check only passes because the relative term (`rtol * |logit| ~= 2.4e-3`)
  carries it: the band has flipped to relative-dominated. Net headroom is still ~66x (vs
  toy's ~70x), so the contract holds *today*, and #267's `poc-parity` CI job (dispatch/
  schedule-only, in `.github/workflows/scheduled-parity.yml` since #302 — both above) is what
  keeps that a monitored fact instead of a one-off measurement
  that silently goes stale. **If a future mlx-swift version introduces a benign op-order
  difference, this gate could trip at poc scale while every toy fixture stays green — that
  is a finding to triage (is the drift real, or just the relative term catching up), not a
  number to widen.** The only sanctioned escape hatch, if one is ever needed, is a *measured*
  poc-scale row added to `src/conformance/tolerances.py` with its derivation recorded the
  same way the existing toy-derived bands are — never an ad-hoc `rtol`/`atol` literal in the
  CI workflow YAML.

- **#195 — the Swift MLX training step + optimizer.** `swift/engine/Sources/MonicaEngine/
  TrainStep.swift` + `LossScaler.swift` mirror `src/model/mlx_train_step.py`'s pretraining
  `make_train_step`/`_accumulate_and_step` via mlx-swift's own autodiff
  (`MLXNN.valueAndGrad(model:_:)` differentiating `trainableParameters()`, exactly
  matching Python's `nn.value_and_grad`) — NOT a hand-rolled backward pass. `TrainStep.
  accumulateAndStep` is factored as the SAME shared accumulate -> (unscale) -> clip ->
  optimizer-step tail Python's `_accumulate_and_step` is, specifically so a future SFT
  masked-CE step is a small delta (a different `lossAndGrad`, reusing this function
  unchanged) rather than a rewrite. `makePretrainLossAndGrad` is exposed publicly (not
  inlined into `makeTrainStep`) so `monica-train`'s fixture gate can drive the exact
  production gradient-producing closure through a `captureGrads` hook, rather than
  duplicating it a second time.

  **Scope — REPRODUCE vs DEFER** (full table in `.claude/plans/issue-195.md`): grad
  accumulation/averaging, hand-rolled grad clipping, the dynamic fp16 loss-scale policy +
  skip control flow (numerics-independent; the general fp16/bf16 *numeric* parity band is
  #266's, not this issue's), and `AdamW` (mlx-swift's defaults — betas (0.9, 0.999), eps
  1e-8, weightDecay 0.01, biasCorrection false — verified identical to Python MLX's) are
  all REPRODUCE. `grad_checkpoint`, SFT masked-CE, DPO/GRPO, MoE Loss-Free-Balancing (load
  counting + the `setRouteBias` write path — `MoEBlock.swift` already records these as not
  ported in #166), and optimizer-state save/load (owned by #196, which only ever *reads*
  here) are all DEFERRED — the first three to a proposed follow-up issue, the last to #196.

  **Two risks worth recording explicitly** (both in the plan as R2/R3): (1)
  `MLXOptimizers.clipGradNorm` is NOT Python's clip — it uses a strict `totalNorm .<
  maxNorm` branch (`Optimizers.swift:895-905`), whereas Python applies
  `min(1.0, grad_clip/(norm+1e-6))` unconditionally; `accumulateAndStep` hand-rolls the
  Python form rather than calling the library helper. (2) `globalGradNorm` sums grad
  leaves in SORTED-KEY order for run-to-run determinism, but this does not reproduce
  Python's `tree_flatten`-order summation exactly — fp32 non-associativity puts a floor of
  ~1e-7 relative on `grad_norm`, which is why it (and the full gradient tree, and
  post-step weights) sit in a looser **`2e-4`/`1e-6`** band, not the fp32 forward/step
  gate's `1e-4`/`1e-5`. `loss` (a pure forward quantity) stays at the tight band. These
  live as NEW `meta.json` keys — `train_rtol`/`train_atol` — deliberately not `rtol`/
  `atol`, so they can never collide with the two quantized fixtures' own looser logit
  band.

  **The oracle.** `scripts/export_parity_fixture.py --train-steps K` (K=3) adds
  `train.safetensors` to `toy`/`toy-hybrid`/`toy-moe` only (of the seven checked-in
  fixtures) — see `swift/engine/Fixtures/README.md` for why the other four carry no
  training coverage. Built from a **pristine reload** of the just-written weights, so the
  inference oracles stay computed on untrained weights; per-step LRs are non-constant
  (`[1e-3, 5e-4, 2e-4]`) so a Swift port that set `learningRate` once at construction
  would fail the gate; `grad_clip` is chosen as the median of a clip-disabled dry run's
  per-step norms, and the exporter refuses to write unless at least one of the K steps
  clips and at least one does not. The per-parameter gradient tree is captured by
  duplicating only the "objective-specific piece" (`loss_fn` + `nn.value_and_grad`) on the
  Python side and recomputing it on the same pristine model state immediately before each
  *production* `train_step` call — MLX being deterministic given identical inputs and no
  randomness in this graph, this reproduces the pre-clip gradients `_accumulate_and_step`
  computes internally but never returns, without touching `mlx_train_step.py` itself.

  **The gate.** `monica-train` (same dependency-free-runner style as `monica-parity`/
  `monica-bench`: hand-rolled args, a `failures` array, `exit(1)` on any failure, achieved
  numbers always printed) has four modes: `--self-test` (the `DynamicLossScaler` policy —
  pure Swift, no MLX import, so it runs even where mlx-swift cannot execute on this
  Command-Line-Tools-only development host); the fixture gate (its OWN default list —
  `toy`/`toy-hybrid`/`toy-moe`, narrower than `monica-parity`'s seven — treating a missing
  `train.safetensors` in a checked fixture as a FAILURE, not a skip); `--overflow-check`
  (`initScale: 1e40` forces `loss*scale -> inf` in fp32, asserting the step is skipped, the
  scale halves, and every weight is bit-identical to before — the skip branch gated end to
  end with no fp16 fixture and no new tolerance); and `--train <fixture> --steps N`
  (free-running, the issue's literal decreasing-loss acceptance criterion). CI
  (`swift-engine`) runs all four after the existing `monica-parity` step. Tolerance
  calibration is CI-only: mlx-swift cannot execute on this development host at all — not
  just its Metal GPU path (`MONICA_ENGINE_CPU=1` still fails, "Failed to load the default
  metallib", because the checkout's Command Line Tools have no `metal` compiler) — so the
  first CI run is where the achieved max|d| numbers this design doc's tolerance table
  assumes get calibrated.

  **The checked-in gradient oracle export had a column-major layout bug — RESOLVED
  (#195/PR #293).** `tests/test_train_parity_fixture_export.py` (a Python-only staleness
  guard: re-export `toy`, diff against the checked-in `train.safetensors` at
  `train_rtol=2e-4`/`train_atol=1e-6`) passed on every local Mac but failed reproducibly on
  the hosted `full-macos` CI runner, printing
  `grad.0.layers.0.conv.weight   max|d| = 1.226e-02`. This was investigated through THREE
  wrong hypotheses before the real one, each ruled out by direct measurement:

  1. **Not a corrupted export.** Regenerating locally, with and without #264's
     `mx.set_cache_limit(0)` mitigation, reproduced the checked-in oracle to `9.313e-10`
     both ways.
  2. **Not an MLX-version mismatch.** Both the checked-in oracle and CI's hosted runner
     reported `mlx_version: 0.32.0` in `meta.json` — same version, still ~1.2–1.8e-2 apart.
  3. **Not within-host nondeterminism, and (WRONGLY, at first) concluded to be a real
     cross-host numeric difference.** A dedicated CI diagnostic
     (`.claude/plans/issue-195-unblock.md`'s Phase 0) ran the exporter twice, in two fresh
     processes on two independent `macos-latest` runners: the two agreed to **~1.5e-8**
     (fp32 noise floor) on every key, but both disagreed with a locally-generated (M1 Pro)
     oracle by the SAME ~1.2–1.8e-2 amounts. This looked exactly like a deterministic,
     host-family-dependent difference in MLX's Metal conv1d weight-gradient reduction —
     and was reported and documented as such — but it was a misdiagnosis: the CI diagnostic
     only ever compared CI generations against EACH OTHER and against the (also
     column-major-affected, see below) local oracle; it never tested whether the
     *underlying values*, not just the exported bytes, actually agreed.

  4. **The actual root cause: a column-major export bug, not a numeric difference.**
     Reinterpreting each of the 18 mismatched `grad.*` keys' raw buffer as column-major
     instead of row-major — `old.flatten(order="F")` vs `new.flatten(order="C")` — collapsed
     EVERY one down to `~1e-9` (fp32 noise). Every `conv.weight`/`in_proj.weight`/
     `out_proj.weight` across all 3 training steps, no exceptions. `weights_after.*` (real
     `model.parameters()` arrays, never a transpose-producing autodiff view) needed no
     reinterpretation and matched directly in plain row-major order — the clean control
     group proving the underlying training math was never in question. `mx.grad`/
     `value_and_grad` can hand back a column-major (transposed-view) array rather than the
     row-major layout `model.parameters()` always returns, and the exporter's plain
     `np.array(val, dtype=np.float32)` did not canonicalize this — apparently handled
     differently by MLX across host builds (a genuine MLX/export-boundary defect, not a
     computation difference: `swift model parity`'s `monica-train` step, running mlx-swift's
     own autodiff on the SAME CI runner family, matched the OLD (buggy-on-CI,
     coincidentally-correct-when-read-as-column-major-on-a-local-Mac) oracle almost exactly
     — `grad max|d|=8.941e-08` — which is what first exposed the bug: swapping to a
     freshly CI-generated oracle broke `swift model parity` at the SAME `1.226e-02`
     magnitude, the wrong direction for a "CI is canonical" fix to move.

  **The fix.** `scripts/export_parity_fixture.py`'s `_export_train_oracle` now calls
  `mx.contiguous(val)` (MLX's own API — "Force an array to be row contiguous. Copy if
  necessary", default `allow_col_major=False`) before converting every `grad.*`/
  `weights_after.*` leaf to numpy. Verified a no-op wherever the array was already
  row-major (0.0 diff, measured on a machine that never showed the bug) and load-bearing
  wherever it wasn't. **`train.safetensors` is host-portable again** — regenerating
  locally works exactly as it did for every other fixture; there was never a genuine
  cross-machine numeric difference to work around. Both `train_rtol`/`train_atol`
  (`2e-4`/`1e-6`) and the fp32 logit gate (`PARITY_TOLERANCES["fp32"] == (1e-4, 1e-5)`) are
  UNCHANGED — the guard was correctly detecting a real defect throughout; it just wasn't
  the defect either the first CI diagnostic or its own follow-up initially concluded. The
  `train-fixture-oracle`/`train-fixture-oracle-verify` CI job pair (dispatch/schedule-only,
  mirrors `poc-fixture-oracle`/`poc-parity`, #267; all four live in
  `.github/workflows/scheduled-parity.yml` since #302) still exists but is **not load-bearing
  for correctness** — it is a regression check that verifies this exporter's output on the
  specific host family where the column-major bug manifested, for this change and any
  future one; it would NOT have caught the original bug on its own, since both independent
  CI-runner generations were equally column-major and agreed with each other. This is a
  **different phenomenon from #298** (deterministic-per-process export corruption): #298's
  worst observed drift (`mixing.N`) was 2.7x its own tolerance; this one exceeded the
  tensor's own absmax entirely, and reinterpreting axis order (not tolerance) resolved it.
  Not folded into #298, and #298 is not folded into this.

## MLX 0.32.0 buffer reuse (#298)

**Status: OPEN upstream, GUARDED here.** This section is the written half of #298. Nothing below
claims the defect is fixed — it is an MLX bug this repository cannot fix. What #298 shipped is
(a) a guard that makes a corrupted oracle impossible to check in unnoticed and (b) these measured
answers. #298 stays open as an upstream-tracking item.

**The defect.** Calling `MLXMambaModel.mixing_matrices` (or any accessor that forks a lazy `h`
into two independent consumers) on a freshly-constructed model silently corrupts a *later,
unrelated* computation on that same model — `prefill(..., last_only=True)` in the observed case.
Correct shapes, no exception, no log, and the corruption is catastrophic rather than marginal:
`max|d|` on fp32 logits is **6-9**, not `1e-5`. It is deterministic within a process but varies
between processes, so a rerun "fixes" it and it reads as a flake.

All numbers below are from `scripts/probe_mlx_buffer_reuse.py` on this host (M1 Pro, macOS,
Python 3.14.6), 2026-08-19. The probe compares every trial to an **accessor-free reference**
(`prefill` on a fresh model with no interpretability call), *not* to trial 0 — with the accessor in
the loop roughly half of the trials including trial 0 are corrupt, so a trial-0 reference reports
the clean trials as the failures and undercounts by ~2x. The reference is recomputed at the end of
every run and a run whose reference moved is reported `BLIND`, never clean.

### D1 — Is it version-specific? Is there an upstream fix?

**No, and none found.** `pyproject.toml` pins only `mlx>=0.18`; three released versions were built
into throwaway venvs and run through the identical probe (`--pattern mixing --trials 40 --seq 129`,
barrier removed, buffer cache on):

| mlx | toy.yaml | toy-hybrid.yaml | toy-moe.yaml |
|-----|----------|-----------------|--------------|
| 0.31.2 (2026-04-22) | 35/40 | 16/40 | 14/40 |
| 0.32.0 (2026-07-07, the installed build) | 30/40 | 18/40 | 11/40 |
| 0.32.1 (2026-08-18, latest) | 27/40 | 24/40 | 16/40 |

The rates are the same order across all three, so **downgrading to 0.31.2 is not a remedy and
upgrading to 0.32.1 is not either.** Reproduce with:

```bash
python -m venv .venv-mlx031 && .venv-mlx031/bin/pip install "mlx==0.31.2" numpy safetensors pyyaml
PYTHONPATH=. .venv-mlx031/bin/python scripts/probe_mlx_buffer_reuse.py \
    --pattern mixing --trials 40 --seq 129 --config config/toy.yaml
```

**Upstream search (2026-08-19), what was searched and what was found.** `ml-explore/mlx` issues
and PRs via `gh api search/issues` for *buffer cache corruption*, *set_cache_limit*, *allocator
reuse wrong results*, *silent wrong results*, *hazard tracking*, *untracked hazard*, *lazy graph
fork wrong results*, *nondeterministic results same input*; plus the release notes for v0.32.0 and
v0.32.1. **Result: no upstream issue matching this defect.** The near neighbours are all different
bugs — #3866 (int32 corruption under `async_eval` + in-place slice updates, closed), #3461/#3462
(buffer destroyed mid-flight under *untracked hazard mode with custom kernels*, closed), #3912 /
#4253 / #4261 (quantized-matmul and `gather_mm` wrong results), #3350 (the cache pool retaining
unusable buffers — a memory-growth bug, not a correctness one). This is a **documented negative**:
no upstream fix exists to wait for, and no upstream issue has been filed by this project either.

**Re-run 2026-08-22 (round 2).** The same query set, plus a check for any MLX release later than
0.32.1: **there is none** — 0.32.1 (2026-08-18) is still the newest on both the GitHub releases
list and PyPI. The negative stands. Two *new* near neighbours appeared since 2026-08-19 and were
checked against this defect: #4370 (`mx.dequantize` wrong results on a non-contiguous slice, a
0.32.1 regression, open) and #3856 (silent numerical corruption on a quantized MoE when
`seq %% 32 != 0`, open). Neither applies — this repro is fp32, unquantized, no custom kernels, no
`async_eval`. The full query/hit table and the reproduce instructions now live in a
maintainer-actionable report at [`../upstream/mlx-buffer-reuse-report.md`](../upstream/mlx-buffer-reuse-report.md),
written so that filing it upstream is a one-step human decision. **It has not been filed** — see
§D6.

### D2 — Does the barrier belong on the training/inference path? **No.**

**Decision: the `mx.eval(h)` barrier and `mx.set_cache_limit(0)` both stay OFF the
training/inference path.** Two independent reasons, one structural and one measured.

*Structural.* `mixing_matrices` is the only place in the backend that forks a **lazy** `h` into two
independent consumers — `layer.mixing_matrix(h)` and `layer_fn(h)` — without materialising it
first. No training or inference path does that: `forward`/`forward_seq`, `prefill` and `step` each
consume `h` linearly, one consumer per value. The interpretability accessors (`mixing_matrices`,
`hidden_states`) are the exception, and they are not on any hot path.

*Measured.* `--pattern hotpath --trials 30` — `forward`, `prefill`, three stacked `step`s, and a
real `make_train_step` over 30 freshly-constructed models, buffer cache **on**:

| config | failures |
|--------|----------|
| toy.yaml | 0/30 |
| toy-hybrid.yaml | 0/30 |
| toy-moe.yaml | 0/30 |

That clean result is evidence **because the same instrument, on the same host, in the same
session, does see the defect** at 11-30/40 in the `mixing` pattern. A clean run from an instrument
that had never demonstrated detection would be BLIND, and the probe exits non-zero rather than
report it.

*Cost, so the decision is grounded rather than asserted.* `--pattern throughput --trials 30`
(forward + `make_train_step`, timed with the cache on and then at limit 0, toy scale):

| config | cache on | limit 0 | slowdown |
|--------|----------|---------|----------|
| toy.yaml | 0.317 s | 0.358 s | **+13.0%** |
| toy-moe.yaml | 0.477 s | 0.623 s | **+30.5%** |

Toy scale overstates the penalty for a real run (allocator cost is largest where allocation
dominates compute) but the sign and order are clear, and at ~99 s/step for the poc protocol a
double-digit percentage is not a cost to take on speculatively. If a future reader wants to widen
the scope, the cost is on record either way.

### D3 — The guard, and what it does and does not claim

`scripts/export_parity_fixture.py` now exports **twice, in two fresh processes**, and compares the
trees byte-for-byte (`src/conformance/fixture_digest.py`, portable) before leaving anything at
`--out`. On agreement `--out` is kept; on disagreement **nothing** is left at `--out`, both trees
are preserved as `--out.mismatch-1`/`-2`, the per-file verdicts print, and the process exits 2.
`--no-double-export` opts out (and is set automatically on the child, bounding the recursion).

The defect is deterministic *per process*, so two independent interpreters would have to corrupt
identically to agree — that is the whole content of the guard, and it is why the second export
lives in `main()` rather than inside `build_fixture` (an in-process repeat would prove nothing).

**Scope: intra-machine only.** Two processes on ONE host. Cross-machine byte identity is *not*
claimed and is known to be false — CI's `train-fixture-oracle`/`-verify` pair measures ~1e-8 drift
in `train.safetensors` between runner instances.

**Confinement, now checkable in one command.** One helper, `disable_buffer_cache()` /
`disable_buffer_cache_for_process()`, in `scripts/export_parity_fixture.py`, with **five** call
sites: `build_fixture`, `tests/test_mlx_mixing_matrix.py`, `tests/test_parity_fixture_export.py`,
`tests/test_train_parity_fixture_export.py`, `tests/test_lowp_parity_band.py`. The issue text says
two and the plan said three; both undercounted — #264's mitigation had already been copy-pasted
into two more test modules. `git grep -n set_cache_limit -- src scripts tests` now returns **no
call site** outside that one file: the remaining hits are prose comments plus the monkeypatch
stubs in `test_mlx_mixing_matrix.py` that falsify the fail-loud check. That is what makes the
confinement an observation rather than an assumption — and in particular there is no call on the
training/inference path. The helper **raises** (naming `mx.__version__`) if the limit does not
take effect, so an MLX API change fails loudly instead of silently no-op'ing; MLX 0.32.0 exposes no
`get_cache_limit`, so it confirms via a second `set_cache_limit(0)` plus `get_cache_memory()`.

**Determinism of the export graph, i.e. why byte-for-byte is the right predicate.** The export has
no dropout and no sampling: `generation.safetensors` is a **greedy** decode, `mx.random.seed(seed)`
is re-applied at the top of each `run()`, and `_export_train_oracle` already re-runs its trajectory
and refuses a non-reproducing one. Measured: 20/20 in-process `build_fixture("config/toy.yaml")`
runs reproduce the checked-in oracle exactly (D4), and the double-export agrees on all 8 files.

### D4 — The `mixing.N` near-zero flake (the escalation comment)

CI run 31806256071 failed `test_checked_in_toy_fixture_matches_todays_backend` once with
`mixing.1 max|d| = 2.694e-05` against `rtol=1e-4/atol=1e-5`, then passed on rerun. `--pattern
export --trials 20` regenerates `toy` twenty times and diffs every `mixing.{i}` key plus
`forward_logits` as a control, against the checked-in oracle:

| key | max\|d\| over 20 | nonzero trials |
|-----|------------------|----------------|
| `mixing.0` | 0.0 | 0/20 |
| `mixing.1` | 0.0 | 0/20 |
| `forward_logits` | 0.0 | 0/20 |

**Branch fired for the isolated exporter: no drift reproduces there, so nothing is loosened.** No tolerance changed — not for
`mixing.*`, and certainly not for `forward_logits`/`step_logits`/`greedy_ids`.
`src/conformance/tolerances.py` is untouched by #298. The near-zero concern is real in principle
(the causal mixing matrix's strict upper triangle is *exactly* zero, so at `|b| ~ 0` the
`np.allclose` band collapses to `atol=1e-5` with no `rtol` headroom, and the observed 2.694e-05 is
~2.7x over) — but widening a band to absorb an unreproduced failure would be picking a number to
make a run green, which is the one thing the tolerance contract forbids.

**But the CI flake IS reproducible — in a populated pytest process, and it is worse than the CI
report suggested.** `build_fixture` in isolation is clean; run it after a module that exercises the
interpretability accessors and it corrupts. Measured on `main` (i.e. this **pre-dates** #298's
guard and is not caused by it), 12 runs each:

| pytest invocation | failures |
|---|---|
| `pytest tests/test_parity_fixture_export.py` | **0/12** |
| `pytest tests/test_mlx_mixing_matrix.py tests/test_parity_fixture_export.py` | **3/12** |

```bash
# reproduce (~3 s per run; expect roughly 1 in 4 to fail)
.venv/bin/python -m pytest tests/test_mlx_mixing_matrix.py \
    tests/test_parity_fixture_export.py -q -p no:cacheprovider
```

The failing assertion is **not** `mixing.N` and **not** a tolerance at all — it is `greedy_ids`,
compared as exact integers:

```
greedy_ids drifted from the checked-in Swift-parity oracle
Mismatched elements: 16 / 16 (100%)
 [0]: 222 (ACTUAL), 4 (DESIRED)
```

16 of 16 ids wrong, by 222 rather than by one — the same catastrophic signature as the `mixing`
probe's 6-9 logit drift, not the 2.694e-05 the CI comment reported. So **D4's third branch is the
one that fires**: this is a live recurrence of #298, and the correct response is to record it, not
to loosen anything. No band moved.

Two consequences worth stating explicitly, because both are easy to assume away:

1. **`set_cache_limit(0)` does not compose across pytest modules.** Both modules in the failing
   pair apply the process-wide disable at import, and the corruption still occurs. The 0/40 clean
   results in D1-D3 were all measured in *dedicated* processes; a long-lived pytest process that
   has already run ~19 other MLX modules is a different regime, and the mitigation is not known to
   hold there.
2. **The CI comment's `mixing.1 max|d| = 2.694e-05` is therefore probably a *different* event**
   from the one reproduced here — a near-miss at the atol floor rather than this wholesale
   corruption. That one is still unreproduced (0/20 fresh exports), and widening a band to absorb
   an unreproduced near-miss would be picking a number to make a run green. Left alone.

### D5 — What stays open

- The upstream MLX defect itself. Unfixed in 0.31.2, 0.32.0 and 0.32.1; no upstream issue found;
  none filed from here. The report is now *written* —
  [`../upstream/mlx-buffer-reuse-report.md`](../upstream/mlx-buffer-reuse-report.md) — and
  deliberately **not posted**; see §D6. **#298 remains open as an upstream-tracking item.**
- **#264's `mx.eval(h)` barrier is not sufficient on its own.** With the barrier in place and the
  buffer cache on, the probe still measures 18/40 (toy.yaml) and 20/40 (toy-moe.yaml) corrupt
  trials; only 0/40 for toy-hybrid.yaml. The mitigation that actually holds on this host is
  `set_cache_limit(0)`: barrier **plus** limit 0 gives **0/40 on all three configs**, on 0.32.0 and
  on 0.31.2. Every path that writes a checked-in oracle already runs under that combination, so the
  fixtures are covered — but a caller invoking `mixing_matrices` outside the exporter and the four
  test modules is **not** protected by the barrier alone. Parked in `docs/parked-findings.md`;
  changing it is a behaviour decision beyond #298's contract.
- **The `full-macos` suite is flaky at ~1-in-4 for the module pair above**, pre-dating this work.
  #298's guard protects what gets *committed* as an oracle; it does nothing for a test process that
  corrupts mid-run. Making the suite robust (process isolation for the MLX fixture modules, e.g.
  `pytest-forked`/`-p xdist --forked`, or dropping the accessor tests into their own job) is a
  test-infrastructure decision beyond this issue's contract. Parked.
- Pinning `mlx==` in `pyproject.toml`. Out of scope for #298 (a repo-wide dependency decision) and
  pointless as a remedy anyway, since all three tested versions are affected.
- A PR-time CI job for the double-export. The guard is in the CLI the regeneration command already
  invokes, and the dispatch-only `poc-fixture-oracle`/`poc-parity` and `train-fixture-oracle`/
  `-verify` pairs already do the cross-runner form; an always-on macOS job is a runtime-cost
  decision for the user.

### D6 — Round 2: scope, second host, the #303/#315 question, and the rerun policy

Round 1 (#311) answered D1–D5. Round 2 answers the four escalation items added to #298 on
2026-08-20, and states plainly what this repository can and cannot do about the defect.

#### What is ours and what is upstream

**This repository cannot fix MLX's buffer-cache allocator, and this round attempts no fix.** The defect lives in MLX's Metal allocator/graph machinery; we have no patch to apply, no
released version to move to (D1: 0.31.2, 0.32.0 and 0.32.1 all reproduce at the same order), and
minimising it to a standalone upstream repro is unbounded research that was deliberately not
started. What *is* ours, and what round 2 delivers, is four things: **measurement** (the probe, now
on a second host), **guarding** (the two-fresh-process export check from round 1), **truthful
status** (the feature-matrix and test-plan rows below), and a **prepared upstream report**. No
tolerance moved, no fixture was regenerated, and no `set_cache_limit` call site moved.

#### Is the current mitigation sufficient? Stated, not implied

**No — not universally, and the gap is named.** The mitigation that actually holds on this host is
the `mx.eval(h)` barrier **plus** `mx.set_cache_limit(0)`. The barrier alone still leaves **18/40**
(`toy.yaml`) and **20/40** (`toy-moe.yaml`) trials corrupt (D5); only the combination reaches 0/40
on all three configs.

`git grep -n set_cache_limit -- src scripts tests` (2026-08-22, unchanged from round 1) returns the
helper's own definition in `scripts/export_parity_fixture.py` plus prose and the monkeypatch stubs
in `tests/test_mlx_mixing_matrix.py` that falsify the fail-loud check. The **five call sites** are
`build_fixture` and four test modules (`test_mlx_mixing_matrix.py`, `test_parity_fixture_export.py`,
`test_train_parity_fixture_export.py`, `test_lowp_parity_band.py`) — every one of them an
oracle-writing or oracle-checking path. **No training or inference path carries it**, by the §D2
decision, and that decision stands.

The consequence, stated so it is not assumed away: **a caller that invokes `mixing_matrices` or
`hidden_states` outside the exporter and those four test modules is protected by the barrier alone,
which is not enough.** `hidden_states` in particular has never been measured — it forks a lazy `h`
the same way `mixing_matrices` does, and its rate is **unquantified**, not zero. Both remain parked
in `docs/parked-findings.md`; widening the mitigation is a behaviour-and-throughput decision (D2's
+13.0%/+30.5%) beyond #298's contract.

#### The probe on a second host (CI macOS runner)

Every number in D1–D5 came from one M1 Pro dev Mac. `mlx-buffer-reuse-probe` in
`.github/workflows/scheduled-parity.yml` now runs the same instrument on a hosted `macos-latest`
runner — a different chip generation, image and wheel, and the family that actually runs
`full-macos`/`parity-macos`. It runs `--pattern mixing` (the positive control) and
`--pattern hotpath` across all three toy configs at `--trials 20 --seq 129`, uploads the JSON plus
host provenance, and emits a `::warning::` if **no** control config reproduced. It is
dispatch/schedule-only (it lives in the file with no `pull_request`/`push` trigger) and carries no
`needs:`, so it cannot delay `poc-parity`.

`--report-only` is what makes this honest rather than green-by-default: reproduction is
shape-dependent (D1), so a hosted runner may legitimately fail to reproduce, and that is a
*measurement*, not a build failure. The flag records `"blind": true` with its reason in the JSON and
exits 0 for **that one case only**. A probe that cannot run at all — no MLX wheel, unloadable
config, import error — still exits non-zero and reddens the job. There is no `|| true` and no
`continue-on-error:` in the job; an exit status is never discarded.

**First real run: 2026-08-22, run [32584500428](https://github.com/travisgalloway/monica/actions/runs/32584500428), commit `1b353122`.**
Runner provenance, from the job's own `host.txt` artifact: image `macos26 20260728.0273.1`,
**Apple M1 (Virtual)**, macOS 26.5.2 (build 25F84), arm64, **mlx 0.32.1** — note the runner
installs the newest wheel, so this is a *different MLX build* from the dev host's 0.32.0 as well as
a different machine.

`--pattern mixing` (positive control, barrier removed, buffer cache on), 20 trials each:

| config | corrupt trials | `max\|d\|` | verdict |
|---|---|---|---|
| `toy.yaml` | 0/20 | 0.0 | **BLIND** — control did not fire |
| `toy-hybrid.yaml` | **3/20** | 1.679 | reproduced |
| `toy-moe.yaml` | 0/20 | 0.0 | **BLIND** — control did not fire |

`--pattern hotpath` (forward / prefill / stacked `step` / real `make_train_step`), 20 trials each:

| config | corrupt trials |
|---|---|
| `toy.yaml` | 0/20 |
| `toy-hybrid.yaml` | 0/20 |
| `toy-moe.yaml` | 0/20 |

**What this second host establishes.**

1. **The defect is not an artifact of one machine.** It reproduces on a virtualised M1 GitHub
   runner, on mlx 0.32.1, at 3/20 — the same catastrophic magnitude (`max|d|` 1.679 on fp32 logits,
   not `1e-5`), not a marginal drift.
2. **§D2's hot-path conclusion holds on a second host, and holds *as evidence*.** 0/20 on all three
   configs, from an instrument that demonstrably saw the defect on this same runner in this same
   run. That is the whole reason the control runs alongside the hot path; a clean hot path from a
   blind instrument would have been worth nothing.
3. **Shape-dependence is real and the ranking is host-dependent — which is why one config would
   have been useless.** On the M1 Pro dev host `toy.yaml` is the *strongest* reproducer (27–35/40)
   and `toy-hybrid` among the weakest; on this runner it inverts exactly — `toy.yaml` and
   `toy-moe` are both BLIND and only `toy-hybrid` fires. A probe that ran the control on a single
   config would have reported BLIND here and, worse, would have reported "clean hot path" from a
   blind instrument. Any future minimisation attempt has to carry its own positive control for the
   same reason.
4. **The rate is lower here (3/20 vs 11–35/40), and that is not reassurance.** Two of three
   configs could not see the defect at all on this runner, so 3/20 is a floor on what this host
   does, not a measurement of how often it bites in a real long-lived pytest process.

**An incidental finding worth recording, because it shaped the job's configuration.** The identical
control step took **5 seconds** on run 32583285323's runner and **over 20 minutes** — a timeout
kill — on run 32583406103's, twelve minutes apart from the same commit, against 0.26 s per config
locally. The hosted `macos-latest` family therefore has multi-hundred-fold wall-clock variance on
this workload. `timeout-minutes` is 45 rather than 20 for that reason, recorded in the job's own
comment: a timeout kill on a measurement job would read as a probe regression, which is CONF-2's
failure shape (#315) all over again.


#### Did #303 raise the trigger rate? (with the sample size)

**Verdict: #298 fired once in the 8 post-#303 `ci.yml` runs on `main`, at the near-`atol`
magnitude, and has not fired in the 3 runs since #315/#324 — a sample far too small to claim the
rate moved either way. The mechanism is unchanged.**

Post-merge `ci.yml` runs on `main` from #303's fix (PR #314, `0cd860b`, 2026-08-20) to 2026-08-22:

| run id | sha | conclusion | failing job | #298? |
|---|---|---|---|---|
| 32339694911 | `0cd860b5` | failure | `swift selfcheck (macOS)` | no |
| 32353544909 | `93c187c8` | failure | `full suite + smoke gate + cross-backend parity (macOS)` | **YES** — `test_parity_fixture_export.py:77`, `reference.safetensors['mixing.1']` `max\|d\| = 1.561e-05` |
| 32365170104 | `88a7b89d` | success | — | no |
| 32371858248 | `c79ca58d` | failure | `swift selfcheck (macOS)` | no |
| 32536027371 | `cc373669` | success | — | no |
| 32540044632 | `9e1666fc` | success | — | no |
| 32542924915 | `6ceb8c80` | failure | `swift selfcheck (macOS)` | no |
| 32582697121 | `6a260c54` | success | — | no |

What that supports, and what it does not:

- **1 of 8** post-#303 runs failed on #298, and it is the *same* signature as the pre-#303 event
  D4 records (run 31806256071, `mixing.1 max|d| = 2.694e-05`): a near-`atol` `mixing.*` drift, not
  the catastrophic `greedy_ids` corruption D4 reproduced locally. So D4's "still unreproduced by a
  fresh export" note now has a second CI instance, and the local reproduction still gives no
  fresh-export drift. Both remain untouched — no band moved.
- **3 runs** have completed since #315 split the both-backend install off the suite job (`9e1666f`)
  and #324 trimmed the suite further, with 0 events. Three runs is not evidence of improvement.
  "Not reproduced in 3 runs" is the honest statement; "no longer an issue" is not.
- **The mechanism is unchanged.** `full-macos`'s `--ignore` set is `test_backend_parity.py` and
  `test_cuda_distributed.py` only, so `tests/test_mlx_mixing_matrix.py` and
  `tests/test_parity_fixture_export.py` **still share one process** — exactly the pair D4 reproduces
  at 3/12. Nothing about #303, #315 or #324 addressed that, and none of them tried to.
- Note the plan for this round attributed the `--ignore` trim to #322; #322's commits are not on
  `main`. The trim that is live landed as **#324** (`6a260c5`). Recorded here rather than left as a
  discrepancy for the next reader.

Bottom line: **#303's both-backend install plausibly changed the process's allocation profile, but
the run inventory cannot resolve a rate change at this sample size, and the module pair that
carries the risk was never separated.** The escalation item is answered; the underlying flake is
still parked.

#### The rerun / regeneration policy

**A red oracle gate is never answered by regenerating the oracle.** #298 makes a red
staleness gate ambiguous — an intended backend change and a per-process corruption produce the same
red — and regenerating until green is how a corrupted value becomes the contract every downstream
Swift gate is measured against.

The policy is `REGEN_ADVICE` in the portable `src/conformance/fixture_digest.py`, appended to every
staleness-failure message in `tests/test_parity_fixture_export.py` and
`tests/test_train_parity_fixture_export.py`, so a person hits it exactly where the decision is made:

1. **Re-run the failing test in a fresh process, on its own.** #298 is deterministic per process and
   varies between them; a failure that does not survive process isolation is #298, not drift, and
   nothing should be regenerated.
2. **Only if it reproduces in isolation, and only if the change was intended**, regenerate through
   `scripts/export_parity_fixture.py` with its default `--double-export` (two fresh processes,
   byte-for-byte; it refuses to write `--out` on disagreement). Never pass `--no-double-export` to
   get a fixture written.
3. **Never widen a tolerance in `src/conformance/tolerances.py` to absorb it.**

The content is pinned by `tests/test_fixture_digest.py`, which runs on the Linux `portable` job —
deliberately, since the two modules that *use* the constant are MLX-gated and skip there.

#### What round 2 did not do

Filing on `ml-explore/mlx` was **not** done. The report is written and checked in at
[`../upstream/mlx-buffer-reuse-report.md`](../upstream/mlx-buffer-reuse-report.md); opening an issue
on a third-party repository is a maintainer decision, not a coding-session action. Suite process
isolation (`pytest-forked`, or a dedicated job for the MLX fixture modules) stays parked, as does
the unmeasured `hidden_states` case. **#298 stays open.**

## See also

- [01-architecture-seam.md](01-architecture-seam.md) — the seam whose Python implementation is the
  parity oracle for everything here.
- [02-model-ssm.md](02-model-ssm.md) — the SSD chunked scan and the matching recurrence that get
  ported to Swift.
- [03-conformance.md](03-conformance.md) — the fp32 ~1e-4 contract every Swift gate mirrors, and
  why bf16 can't be the comparison dtype.
- [05-training.md](05-training.md) — the backend-free loop, the `TrainStepFn` contract, and the
  loss-scaling policy the Swift train step mirrors.
- [12-lsp-in-the-loop.md](12-lsp-in-the-loop.md) — why the LSP fast loop needs low latency (#197).
- [13-code-model-moe.md](13-code-model-moe.md) — the M12 model this engine serves and trains, and
  the source of the MoE sequencing constraint.
- [`../infrastructure.md`](../infrastructure.md) — the R2 + RunPod runbook, if the deferred
  Linux/CUDA Swift path is ever taken up.
