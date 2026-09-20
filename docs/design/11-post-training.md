# Post-training (instruct → thinking → tool-use, + GRPO)

[← Index](README.md)

> **Status note (2026-07-19).** This is a design record; the base model it post-trains does not
> exist yet under either plan. It was written against the M10 distillation program (issue #65,
> **dropped 2026-07-19** — reserve under
> [`../reserve/10-distillation.md`](../reserve/10-distillation.md)). The live program is **M12**
> ([issue #198](https://github.com/travisgalloway/monica/issues/198)), whose post-training arms
> are **#101 (SFT)** and **#103 (RLVR)**, currently **parked** behind the MHM spine
> (corpus/tokenizer/MoE backend/ablation sweep/full run — see
> [`13-code-model-moe.md`](13-code-model-moe.md)). The instruct → thinking → GRPO shape and the
> chat-template invariant below carry over regardless of which base model they land on.

Once a capable hybrid base exists (via the **M12 MoE base run**, or as originally envisioned via
[distillation, reserve](../reserve/10-distillation.md)), post-training builds **three capability
layers in order**. Instruct is the substrate that makes the model usable at all; thinking is the
headline skill of this POC; tool use is the extensibility layer enabled when the product needs it.
All three are taught by **SFT** on data from the shared post-training track, with **GRPO** as a
final reinforcement pass on the thinking layer. This design record was tracked under
[issue #65](https://github.com/travisgalloway/monica/issues/65) (reserve); live tracking is under
issue #198's #101/#103. The data lives under the `shared/` prefix
([corpus pipeline](08-corpus-pipeline.md)) and reuses the M9 SFT/DPO machinery
(`make_sft_train_step`, `make_dpo_train_step`, the response-masked loaders).

The order matters: instruct is assumed by everything later, thinking is the point, tool use is
optional. GRPO is **polish**, not the source of reasoning — at ~1B the emergent self-reflection
that makes pure RL shine in large models does not reliably appear, so reasoning comes from trace
SFT and GRPO refines it.

## Instruct (#101, corpus #95)

**What.** Turn the from-scratch hybrid MoE code model into an instruction-following assistant that
strictly adopts the ChatML template, follows code refactoring directives, and adheres to system
prompts (`src/data/chat_template.py`).

**M12 Reframing.** Re-framed from the discontinued M10 distillation program to align with the M12
from-scratch hybrid MoE architecture. Without a teacher model, instruction-following and chat
discipline are established directly through the repository's SFT machinery (`scripts/sft.py`,
`src/data/sft_corpus.py`, and `src/data/instruct_sft.py`).

**Method & Machinery.** SFT on clean-license code instruction and multi-turn refactoring corpora
(`src/data/sft_sources.py`: `code`, `handauthored`, `architecture`, `oasst1`) rendered under the
ChatML template with response masking:
- `loss_mask` is strictly zero on system prompts, user turns, and turn headers (`<|im_start|>assistant\n`).
- `loss_mask` is active solely on assistant responses and the terminal `<|im_end|>`.
- Masked validation loss and masked val-perplexity are tracked and logged (`metrics.jsonl`).

```bash
python -m src.data.instruct_sft --sources handauthored code architecture --tokenizer qwen25 --out-root data/shared
.venv/bin/python scripts/sft.py --config config/poc.yaml \
    --data data/shared/sft/tokenized/qwen25-8k --corpus-form instruct \
    --init runs/poc/weights.safetensors --out runs/sft-instruct
```

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
each start for the SSM reset (#68); over-length traces are dropped, never split.

**The two forms take two different drivers (#306), and that is not an accident.**

```bash
# build once
python -m src.data.reasoning_sft --sources mot --tokenizer qwen3 --out-root data/shared

# (a) reasoning.jsonl -> the SFT driver (masked CE on the assistant span)
.venv/bin/python scripts/sft.py --config config/poc.yaml \
    --data data/shared/sft/tokenized/qwen3-8k --corpus-form reasoning \
    --init runs/poc/weights.safetensors --out runs/sft-reasoning

# (b) reasoning-packed/ -> the PRETRAINING driver (unmasked next-token CE)
python -m src.data.split --shards data/shared/sft/tokenized/qwen3-8k/reasoning-packed \
    --out data/reasoning-split --val-tokens 100000
.venv/bin/python scripts/train.py --config config/poc.yaml --data data/reasoning-split
```

`reasoning-packed/` is `shard.pack_atomic` output: a flat token stream whose `.bounds` marks
document *starts* only, with every document padded to `chunk_align` and every sequence tail padded
with `pad_id`. Nothing in it separates a document's real last token from its alignment padding, so
**a response mask cannot be reconstructed from it** — an "SFT over packed" mode would have to
invent one, and an invented mask is a silently wrong training signal. `scripts/sft.py
--corpus-form reasoning-packed` therefore fails by name and prints recipe (b) rather than
guessing. GSM8K + MATH and code-with-executable-tests supply the GRPO rewards and
evaluation; the [open-r1](https://github.com/huggingface/open-r1) harness provides GRPO with
code-execution rewards and 1.5B configs that adapt directly. References: Chain-of-Thought,
DeepSeek-R1.

**Two GRPO rules:** start with **math before code** (exact-match, no sandbox — the cheapest clean
reward loop) and require **≥5 tests per coding problem** (thin suites get gamed). For RLVR you need
only **problems + verifiers**, not reference solutions — the model generates, the verifier judges —
so problems whose reference solutions came from a restricted model are still usable.

### Data engineering, contracts, and query verifiers (#342)

Verifiable reward functions extend beyond unit tests and math solvers into structural domain contracts:
- **Relational SQL**: Evaluates query syntax via AST analysis and executes deterministic DML transactions in SQLite in-memory databases. Penalizes cartesian joins without join predicates and unconstrained wildcard queries (`SELECT *`).
- **Data and ML pipelines**: Inspects dataframe and tensor pipelines across Pandas, Polars, and PyTorch. Validates schema column selections and tensor dimensional transformations while penalizing row iterations (`iterrows`) and untyped object columns.
- **API contracts**: Enforces OpenAPI v3.1 and JSON Schema specifications. Verifies path parameter bindings and 2xx responses while rejecting untyped payloads and unconstrained string schemas.
- **GraphQL**: Validates Schema Definition Language types and fields, rejecting unbounded query depths and untyped scalars (`scalar Any`).
- **Protobuf and gRPC**: Ensures field tag uniqueness, valid tag ranges (1 to 536,870,911, excluding reserved range 19000 to 19999), and backward compatibility across message revisions.
- **Clean architecture boundaries**: Analyzes domain-driven architecture, enforcing layered flow from presentation down to domain entities while forbidding infrastructure or web framework imports inside core business logic.

### Web and backend application stack verifiers (#343)

Deterministic static analysis verifiers grade completions across web and enterprise backend application stacks without executing untrusted binaries:
- **TypeScript, React, Next.js, and NestJS**: Evaluates JSX and TypeScript structures. Verifies hook dependency rules, directive boundaries (`'use client'` and `'use server'`), NestJS dependency injection integrity, and route parameter typing. Penalizes suppression hacks (`as any`, `@ts-ignore`, `@ts-expect-error`, empty JSX fragments, and empty event handlers).
- **Python Web (FastAPI, Pydantic, Django ORM)**: Uses AST inspection to enforce Pydantic field schemas, validates FastAPI route path parameter bindings against function signatures, checks Django model relationships (`on_delete` specifications and `related_name`), and ensures migration determinism. Penalizes `# type: ignore`, bare `dict` returns, and empty handler bodies.
- **Go Web and Microservices**: Enforces struct tag syntax, unhandled error return detection, and goroutine leak patterns (unbounded loops without context cancellation guards). Penalizes `_ = err`, empty `select{}`, and panic-only stubs.
- **Java and Spring Boot 3**: Inspects Spring Bean injection, favoring constructor injection over field injection. Validates Jakarta Persistence entity mappings (`@Entity` primary key `@Id` and relationship mappings) and nullability annotations. Penalizes empty catch blocks and raw types.
- **C# and ASP.NET Core**: Validates Minimal API route parameter bindings, Entity Framework model configurations, and nullable reference annotations. Penalizes `#pragma warning disable`, `dynamic` escapes, and null-forgiving bypasses.
- **PHP and Laravel**: Enforces strict typing declarations (`declare(strict_types=1)`), Eloquent relationship return types (`: HasMany`, `: BelongsTo`), and Service Provider contracts. Penalizes `@phpstan-ignore` and untyped parameters.
- **Ruby and Rails**: Validates block balance, Rails model validation declarations, and Sorbet typed signatures (`sig { ... }`). Penalizes `# rubocop:disable` and `T.untyped`.
- **Driver and Unified Evaluator**: Provides `WebBackendVerifier` with stack auto-detection, exposes reward flags in `scripts/rlvr.py`, and records benchmark evaluation through `evaluate_web_backend` in `src/eval/code_suite.py`.

### Cloud, infra, containers, and automation verifiers (#344)

Deterministic static analysis and AST verifiers grade infrastructure as code, container definitions, CI/CD automation scripts, and frontend markup without deploying infrastructure or executing untrusted binaries:
- **Cloud IaC (Terraform / OpenTofu & HCL)**: Inspects HCL block structures, provider schemas, and variable typing. Resolves DAG symbol references across resources, variables, and local definitions, detecting undeclared dependencies and circular reference cycles. Anti-Goodhart guards reject hardcoded plaintext secrets (`password`, `aws_secret_key`, tokens) and unconstrained lifecycle drift escapes (`ignore_changes = all`).
- **Containers & Packaging (Docker & Containerfiles)**: Validates multi-stage build stage declarations and cross-stage copy references (`COPY --from=stage`). Enforces unprivileged execution (`USER <non-root>`), pinned base image tags, and layer hygiene. Anti-Goodhart rules penalize `:latest` tags, running as root, and missing or disabled healthchecks (`HEALTHCHECK NONE`).
- **Orchestration (Kubernetes YAML & Helm)**: Analyzes Kubernetes resource schemas across Pods, Deployments, Services, and StatefulSets. Checks resource specification completeness (`resources.requests` and `resources.limits`). Anti-Goodhart rules reject unbounded memory limits and missing `livenessProbe` / `readinessProbe` definitions on long-running workloads.
- **Shell Scripting & POSIX Automation (Bash / POSIX sh)**: Verifies script hygiene, block balancing, and strict execution modes (`set -euo pipefail`). Detects dangerous unquoted variable expansions in critical commands (`rm`, `cp`, `mv`) and flags legacy backtick subshells. Anti-Goodhart rules penalize `eval` injection patterns and `# shellcheck disable` suppression bypasses.
- **Web Core & UI Styling (HTML5, CSS3, Tailwind CSS)**: Inspects HTML document structure, requiring semantic elements (`<main>`, `<header>`, `<nav>`, `<article>`, `<section>`, `<footer>`) over `<div>` spam. Enforces ARIA accessibility contracts (valid roles, required `alt` on `<img>`, accessible inputs) and verifies Tailwind CSS utility tokens against standard utility specs.
- **Telemetry and Unified Verifier**: Provides `CloudInfraVerifier` with automatic domain inference, reports segregated telemetry metrics (tracking syntax vs. linter vs. escape-hatch penalties), wires CLI rewards into `scripts/rlvr.py`, and records benchmark evaluation via `evaluate_cloud_infra` in `src/eval/code_suite.py`.

### Repository context, symbol grounding, and compiler blast radius verifiers (#345)

Deterministic static analysis verifiers and probes evaluate repository-scale cross-file comprehension, symbol resolution, and compiler blast radius prediction without code execution:
- **Cross-File Symbol Grounding & RULER-over-Code**: Measures teacher-forced Top-1 Recall and Mean Reciprocal Rank (MRR) across scaled 8k to 32k token contexts in `src/eval/code_suite.py`. Probes span cross-file type annotations (`: UserProfile`), imported symbols in import clauses, and function invocations. Evaluates candidate rank by negative log-probability against near-miss exports.
- **Information-Gain Probe**: Measures cross-entropy delta (Delta CE = CE_without - CE_with) on implementation tokens with and without repository interface declarations. Validates that preceding interface context provides quantifiable information gain during code generation.
- **Compiler-Grounded Blast Radius Verifier**: Evaluates model predictions of impacted files and call-sites when function signatures or interface declarations mutate. Computes ground truth via headless compiler diagnostics (`tsc --noEmit` or `pyright`) and deterministic in-process AST static analysis. Rewards precision, recall, and F1 of predicted call-sites against exact compiler error locations (`TS2554: Expected N arguments`, file, line) in under 150ms.
- **Anti-Goodhart & Degeneracy Guards**: `BlastRadiusVerifier` in `src/train/verifiers/repository_context.py` rejects empty, whitespace, and comment-only completions. Anti-Goodhart filters penalize escape hatches (`@ts-ignore`, `eval`) with a reward of -1.0. Benchmark evaluation is recorded through `evaluate_blast_radius` in `src/eval/code_suite.py`.

### Database migration replay and API breaking-change verifiers (#348)

Deterministic static analysis and in-memory relational replay verifiers grade database schema evolutions and API backward compatibility without external databases or container infrastructure:
- **In-Memory Database Migration Replay**: Evaluates forward and rollback DDL migration scripts against pre-seeded records in ephemeral in-memory SQLite relational engines in under 50ms. Asserts 100% data integrity without silent data loss across table splits, column migrations, and index definitions.
- **Rollback Schema Parity Gate**: Executes rollback (`down`) migrations and compares post-rollback schema snapshots against pre-migration DDL, asserting exact parity across tables, columns, constraints, and indices.
- **Zero-Downtime Schema Safety Rules**: Rejects destructive operations violating Expand/Contract patterns (unmitigated `DROP TABLE`, `DROP COLUMN`, direct `RENAME COLUMN`, or `ADD COLUMN ... NOT NULL` without `DEFAULT` values).
- **API Contract Breaking-Change Detection**: Detects breaking changes across OpenAPI specs (v3.0/v3.1) and Protobuf schemas (`proto2`/`proto3`). Forbids endpoint, method, and field removals, tag and type mutations, and tightening parameter requirements or nullability, while scoring non-breaking additive extensions.
- **Telemetry and Unified Evaluators**: Provides `MigrationReplayVerifier` and `ContractDiffVerifier` in `src/train/verifiers/schema_evolution.py`, wires `--reward migration` and `--reward contract-diff` into `scripts/rlvr.py`, and records benchmark evaluations via `evaluate_schema_evolution` in `src/eval/code_suite.py`.


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

`src/data/tool_sft.py` builds the corpus (`tool.jsonl` + `tool-manifest.json` under
`shared/sft/tokenized/<tok>-<k>/`); the call is ordinary assistant content, so it trains through
the same SFT driver as the other masked forms (#306):

```bash
python -m src.data.tool_sft --sources xlam toolace when2call --tokenizer qwen3 --out-root data/shared
.venv/bin/python scripts/sft.py --config config/poc.yaml \
    --data data/shared/sft/tokenized/qwen3-8k --corpus-form tool \
    --init runs/poc/weights.safetensors --out runs/sft-tool
```

## Running an SFT pass over the shared corpora (#306)

`scripts/sft.py` is the **single** SFT driver for every masked form — instruct (#95), reasoning
(#96) and tool (#102). There is no per-form trainer, and adding one would be a regression against
the injected-`train_step` design (`docs/design/01-architecture-seam.md`).

`--data` accepts either layout:

| `--data` | `--corpus-form` | What happens |
|---|---|---|
| a dir with `train.jsonl` + `val.jsonl` (`src.data.sft_data`) | `auto` / `generic` | passed straight through, unchanged |
| a `shared/sft/tokenized/<tok>-<k>/` dir | `auto` | every masked form present is mixed |
| the same | `instruct` / `reasoning` / `tool` (one or several) | those forms, mixed |
| the same | `reasoning-packed` | **rejected by name**, with recipe (b) above |

The builders write one file per form and no held-out set, so the driver makes the split itself:
one seeded permutation over the concatenated records, `--val-frac` (default 0.05, min one record)
to the tail, and `train=/val=` printed so a run is reproducible from its log. Records longer than
`--max-len` (default: the config's `seq_len`) are **dropped and counted, never truncated** —
truncating clips the answer span off the end of the mask, which teaches the model to stop
mid-answer.

Mixing several forms in one run is supported **only** when their manifests agree on `chat_eos`,
`template`, `tokenizer`, `model_id` and `seq_len`; a disagreement is refused by name, because
concatenating two corpora built on different tokenizers trains one of them on the other's ids.

## The detail that bites: chat-template consistency

The Qwen base defines **`<|im_end|>`** as the chat EOS. Keep it **identical across SFT, RL, and
serving** — a mismatch degrades the model at serving time. This is a cross-cutting invariant for
all three layers and the GRPO pass. `src/data/chat_template.py` is the single source of truth for
the ChatML render + assistant-span masking (the assistant turn is trained up to and including its
trailing `<|im_end|>`, so the model learns to stop on it); the shared instruct corpus under
`shared/sft/` is produced by `src/data/instruct_sft.py` (#95).

Since #306 and #101 this is **enforced end-to-end across SFT, RLVR, and serving**:
1. **SFT (`src/data/sft_corpus.py`)**: verified at load. Every builder writes `chat_eos` and
   `template` into its manifest, and `resolve_sft_corpus` refuses a corpus whose `chat_eos` differs
   from `chat_template.CHAT_EOS` or whose `template` is not `qwen-chatml`. Response masking covers
   the assistant completion up to and including `<|im_end|>`.
2. **RLVR (`scripts/rlvr.py`)**: `enforce_chat_eos_consistency()` and `resolve_eos_ids()` ensure
   rollouts terminate on `<|im_end|>` rather than over-generating into invalid turns. Mismatched
   `--chat-eos` configurations fail fast with `ValueError`.
3. **Serving (`src/serve/generate.py` & `scripts/generate.py`)**: `generate()` supports composite
   EOS sets (`eos_id: int | Sequence[int] | set[int]`), and `scripts/generate.py --chat` defaults to
   ChatML template with `<|im_end|>` stop token and prompt isolation.

## Shared with production

These corpora and RL sets are **class-shared**: both the POC student and the eventual production
model post-train the same way, so they are curated once (much of it teacher inference) and reused
unchanged on the production-reserve run (#75). Curating them is expensive — that is why they are
precomputed once and reused everywhere.

## Related

- [Corpus pipeline](08-corpus-pipeline.md) — the `shared/` SFT corpora and verifiable RL sets.
- [Distillation, reserve](../reserve/10-distillation.md) — how the base these layers post-train
  was envisioned to be built under the (dropped) M10 program.
- [`13-code-model-moe.md`](13-code-model-moe.md) — the live M12 program; #101/#103 are these
  layers' current (parked) tracking.
- [Training](05-training.md) — the SFT/DPO machinery (M9) these layers reuse.
