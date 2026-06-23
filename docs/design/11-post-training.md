# Post-training (instruct → thinking → tool-use, + GRPO)

[← Index](README.md)

Once a capable hybrid base exists (via [distillation](10-distillation.md)), post-training builds
**three capability layers in order**. Instruct is the substrate that makes the model usable at
all; thinking is the headline skill of this POC; tool use is the extensibility layer enabled when
the product needs it. All three are taught by **SFT** on data from the shared post-training track,
with **GRPO** as a final reinforcement pass on the thinking layer. The tracker is
[issue #65](https://github.com/travisgalloway/monica/issues/65); the data lives under the
`shared/` prefix ([corpus pipeline](08-corpus-pipeline.md)) and reuses the M9 SFT/DPO machinery
(`make_sft_train_step`, `make_dpo_train_step`, the response-masked loaders).

The order matters: instruct is assumed by everything later, thinking is the point, tool use is
optional. GRPO is **polish**, not the source of reasoning — at ~1B the emergent self-reflection
that makes pure RL shine in large models does not reliably appear, so reasoning comes from trace
SFT and GRPO refines it.

## Instruct (#101, corpus #95)

**What.** Turn a text-continuer into a model that follows instructions, adopts a chat template,
and respects a system prompt — the foundation every later layer assumes.

**Why a stage at all.** The conversion teacher (`Qwen/Qwen3-4B-Thinking-2507`) is already
instruction-tuned (and reasoning/thinking-tuned), so the hybrid inherits much of this through the
matching — but the architecture conversion can blur instruction-following, so an explicit instruct
SFT stage re-establishes it.

**Method.** SFT on general instruction–response pairs under the Qwen chat template (a set such as
UltraChat plus the reasoning traces, which also carry instruction-following), optionally DPO (#77)
for format/preference polish, via TRL `SFTTrainer` / `DPOTrainer` shapes. References: InstructGPT,
FLAN, DPO.

## Thinking (#96 SFT, #103 GRPO)

**What.** Reason through a long trace before committing to an answer — the headline skill.

**Method.** SFT on reasoning traces formatted `<think> ... </think>` then `<answer> ... </answer>`
(#96), then GRPO with verifiable rewards as polish (#103). The primary trace corpus is open-r1
**Mixture-of-Thoughts** (~350k verified math/code/science traces distilled from R1, already on the
Qwen tokenizer), topped up from a larger R1 distill (14B/32B) only where coverage is thin. **Trace
SFT is the main event.**

`src/data/reasoning_sft.py` builds the corpus under `shared/sft/cleaned/reasoning-traces/` +
`shared/sft/tokenized/qwen3-8k/` (`reasoning_traces.py` does the `<think>/<answer>` formatting and
the Mixture-of-Thoughts / `load_topup` sources). It writes **two** forms: `reasoning.jsonl`
(response-masked records for `SFTLoader`) and `reasoning-packed/` — the long 8K packing where each
trace is one chunk-aligned document, so **no trace spans a sequence boundary** and `.bounds` marks
each start for the SSM reset (#68); over-length traces are dropped, never split. GSM8K + MATH and code-with-executable-tests supply the GRPO rewards and
evaluation; the [open-r1](https://github.com/huggingface/open-r1) harness provides GRPO with
code-execution rewards and 1.5B configs that adapt directly. References: Chain-of-Thought,
DeepSeek-R1.

**Two GRPO rules:** start with **math before code** (exact-match, no sandbox — the cheapest clean
reward loop) and require **≥5 tests per coding problem** (thin suites get gamed). For RLVR you need
only **problems + verifiers**, not reference solutions — the model generates, the verifier judges —
so problems whose reference solutions came from a restricted model are still usable.

## Tool use (#102, optional)

**What.** Emit a structured function call the runtime executes and feeds back, optionally in a
[ReAct](https://arxiv.org/abs/2210.03629) loop. Optional for the POC (headline skills are
reasoning/math/code); the path to enable if the product calls tools.

**Why it leans on attention.** At ~1B the achievable target is reliable function calling over a
**fixed tool set** — exact tool selection and argument fidelity are the associative-recall job pure
SSM layers do poorly, the same reason the hybrid keeps attention. TinyAgent (1.1B, runs locally on
Apple Silicon) is the direct precedent and maps onto the MLX serving target.

**Method.** SFT on function-calling data with **distractor tools as negatives** and **abstention
examples** (learn when *not* to call), validating every call against the schema in the runtime.
Sources: Glaive function-calling-v2, xLAM, ToolACE, plus When2Call for restraint; BFCL for eval.
References: ReAct, Toolformer, TinyAgent.

## The detail that bites: chat-template consistency

The Qwen base defines **`<|im_end|>`** as the chat EOS. Keep it **identical across SFT, RL, and
serving** — a mismatch degrades the model at serving time. This is a cross-cutting invariant for
all three layers and the GRPO pass. `src/data/chat_template.py` is the single source of truth for
the ChatML render + assistant-span masking (the assistant turn is trained up to and including its
trailing `<|im_end|>`, so the model learns to stop on it); the shared instruct corpus under
`shared/sft/` is produced by `src/data/instruct_sft.py` (#95).

## Shared with production

These corpora and RL sets are **class-shared**: both the POC student and the eventual production
model post-train the same way, so they are curated once (much of it teacher inference) and reused
unchanged on the production-reserve run (#75). Curating them is expensive — that is why they are
precomputed once and reused everywhere.

## Related

- [Corpus pipeline](08-corpus-pipeline.md) — the `shared/` SFT corpora and verifiable RL sets.
- [Distillation](10-distillation.md) — how the base these layers post-train is built.
- [Training](05-training.md) — the SFT/DPO machinery (M9) these layers reuse.
