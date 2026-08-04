# The M13 native inference + training engine (Swift + MLX)

[← Index](README.md)

The design record for **M13** ([issue #163](https://github.com/travisgalloway/monica/issues/163)):
a performant, native, **no-Python-runtime** engine for the Mamba-2 hybrid — *"like llama.cpp,
tailored for our model"* — targeting **Apple Silicon first**, written in **Swift on MLX**. This
document is #164; the issues it records are #165–#172 and #195–#197.

> **Read this as the target design, not as shipped state.** As of 2026-08-03 the only Swift in
> the tree is the **tokenizer** — `swift/MonicaTokenizer` + the `monica-tokenize` /
> `monica-selfcheck` executables (#191/#245, `swift/Package.swift`). There is **no Swift model,
> no Swift engine, and no Swift train step**. On the Python side, `prefill` and `step_batch` do
> not exist on `ModelInterface`, and `verify_block` exists on the MLX backend **only**
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
| **Prefill via parallel scan** (#165 → Swift #169) | `prefill(token_batch, seg_ids=None, *, last_only=False) -> (logits, State)` — **genuinely new** to `ModelInterface` | The scan **already computes the carry-out state and throws it away**: `new_states` is `(B, nc+1, H, P, N)` at `src/model/mlx_backend.py:252`, and `S_enter = new_states[:, :-1]` at `:253` drops the last entry. CUDA does exactly the same (`src/model/cuda_backend.py:398-399`; `parallel` at `:329`). **Nothing in either backend surfaces it.** | new `src/conformance/prefill_decode_parity.py`: prefill-then-decode logits == pure step-by-step decode, **and** the extracted `State` == the step-produced state, fp32 ~1e-4 |
| **Quantization** (#168) | quantized `load` — `load` itself exists (`src/model/interface.py:87`); what is new is a **`quant` block in the `.config.json` sidecar** (`bits`, `group_size`, `symmetric`, `targets`). Absence of the block = an fp checkpoint, byte-identical to today | Portable numeric core already shipped: `src/eval/quantize.py` (group-wise affine, `mx.quantize`-compatible — `quantize_dequantize:41`, `is_quantizable:119`, `quantize_state_dict:131`) plus the `scripts/quantize.py` driver (#51), weight-only W8/W4 | new `src/conformance/quant_parity.py`: top-1 agreement + KL vs the fp model on a fixed prompt, with bits-dependent thresholds. The fp32 gates stay on the fp model |
| **Speculative decoding** (#172) | `verify_block(tokens, state) -> (logits_list, state_list)` — **already implemented, on MLX only** (`src/model/mlx_backend.py:626`). It is **not** on `ModelInterface` and CUDA has no equivalent; promoting it to the seam is the open work | Drafter + accept rule are already portable: `src/serve/spec_decode.py` (`propose:27`, a prompt-lookup drafter needing no second model; `first_mismatch:53`). Driver: `scripts/spec_decode.py`, which calls `verify_block` at `:127`. **Greedy-only by construction** — `first_mismatch` compares against the verifier's *argmax*, stated in the module docstring | output **byte-identical to greedy decode** — what `tests/test_spec_decode.py:85` already asserts on MLX — plus accept-rate and speedup reported by the bench (#170) |
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

**Checkpoint I/O (#196).** Both directions against `src/train/checkpoint.py:75` (`save_weights`):
safetensors plus the `<path>.config.json` sidecar. Python-written checkpoints must load in Swift
and Swift-written checkpoints must load in Python, bit-for-bit on the tensors. The **slot-a/slot-b
resume bundle** (`CheckpointStore`, `src/train/checkpoint.py:121`) is explicitly **out of
scope** — that is the within-backend concern the two-concern split exists to keep separate
(optimizer state does not need to port).

**The LSP fast loop (#197).** A native persistent `tsserver` client — the Python reference is
`src/lsp/ts_lsp.py` plus `src/lsp/harness.py`, driven through the `LMAdapter` seam
(`src/lsp/lm.py:29`), whose only implementation is `src/model/mlx_lm_adapter.py:73` — together
with a per-step logit-mask hook analogous to the Python `sampler` wrapper (`src/serve/sampling.py:26`).
That hook is what SSI's constrained decode (#226) needs. The **latency motive** comes straight
from [12-lsp-in-the-loop.md](12-lsp-in-the-loop.md): the harness's cost is dominated by round-trips
between the model and the language server, so a native fast loop is the lever that makes the
experiment affordable.

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
| `mlx-swift-lm` | generation loop, sampling, KV cache, quantized loading | ✅ | untested | Saves writing the decode scaffolding by hand |
| `mlx-swift-examples` (`llm-tool`) | CLI shell to fork for #167 | ✅ | untested | Example code, not a runtime dependency |
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
3. **#167** (generation CLI), **#195** (train step + optimizer), **#196** (checkpoint I/O) — in
   parallel once #166 lands.
4. **#197** (LSP harness), **#168** (quantization), **#169** (Swift prefill — needs #165),
   **#170** (Apple-Silicon benchmark harness).
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
- **MoE in Swift.** MoE is MLX-Python-only today: `src/model/cuda_backend.py:665` raises
  `NotImplementedError`, and the MLX `MoEBlock` (`src/model/mlx_backend.py:512`) is a toy softmax top-k
  router with no shared expert and no load balancing. Porting an unfinished router to a third
  implementation is premature — **sequence Swift MoE behind #213/#214**.
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
- **Where the Swift engine lives — open.** Extend the existing `swift/` package with new targets
  alongside `MonicaTokenizer`, or start a sibling package? #166 leaves it as "e.g. `swift/` or a
  sibling repo". The pull toward reusing `swift/` is real — the tokenizer is already there and the
  engine needs it. The cost is equally real: `swift/Package.swift` today has **zero** external
  dependencies, which is precisely what makes its Linux CI cheap and its bit-exactness credible
  (#246); adding mlx-swift changes that for the tokenizer too.

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
