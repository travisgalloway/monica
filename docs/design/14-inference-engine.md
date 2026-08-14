# The M13 native inference + training engine (Swift + MLX)

[← Index](README.md)

The design record for **M13** ([issue #163](https://github.com/travisgalloway/monica/issues/163)):
a performant, native, **no-Python-runtime** engine for the Mamba-2 hybrid — *"like llama.cpp,
tailored for our model"* — targeting **Apple Silicon first**, written in **Swift on MLX**. This
document is #164; the issues it records are #165–#172 and #195–#197.

> **Read this as the target design, not as shipped state.** As of 2026-08-03 the only Swift in
> the tree is the **tokenizer** — `swift/MonicaTokenizer` + the `monica-tokenize` /
> `monica-selfcheck` executables (#191/#245, `swift/Package.swift`). There is **no Swift model,
> no Swift engine, and no Swift train step**. On the Python side, `prefill` **landed via #165**
> (on the seam, in both backends, gated by `src/conformance/prefill_decode_parity.py`);
> `step_batch` still does not exist on `ModelInterface`, and `verify_block` exists on the MLX
> backend **only**
> (`src/model/mlx_backend.py:626`) — not on the seam, and with no CUDA equivalent. Everything
> below is a plan the child issues implement.

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
  safetensors checkpoints, tokenizes with `swift/MonicaTokenizer`, generates, and quantizes, with
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
> conversation.

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
— and the **tokenizer is ours** (`swift/MonicaTokenizer`), already cross-platform, already
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
| **`swift/MonicaTokenizer` (ours)** | tokenize / detokenize; the corpus packer | ✅ | ✅ | #191/#245. Own byte-level BPE, **no external deps** (`swift/Package.swift`), bit-identical across platforms, CI-gated by `swift-macos` / `swift-linux` / `swift-parity` (#246). **Displaces swift-transformers entirely** |
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
  reverse.** `monica-generate` needs `swift/MonicaTokenizer` for encode/decode, so
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
