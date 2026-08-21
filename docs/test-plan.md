# Test plan

Mirrors `docs/feature-matrix.md` row for row, keyed by the same IDs, tracking coverage by test
type so gaps are **visible rather than assumed**.

Columns match what this repository actually runs:

- **Unit** — a `tests/test_*.py`, or a case in the Swift `monica-selfcheck` / `monica-lsp
  --self-test` runners (`swift/` deliberately has no `.testTarget`; `swift test` is a no-op).
- **Integration** — a test wiring two or more subsystems, or a `src/conformance/` check.
- **E2E** — `scripts/smoke_test.py`, a job in `.github/workflows/ci.yml` or
  `.github/workflows/scheduled-parity.yml`, or `monica-parity` / `monica-bench`.

`n/a` means the tier cannot exist for that capability; `none` means it could and does not. The
two are never interchangeable. An empty Gaps cell is a claim, so it is only written when true.

Last audited 2026-08-18 (`/closure-audit`, whole repo: 81 capabilities × 131 test surfaces —
128 `tests/test_*.py` + `conftest.py` + 2 Swift native runners).

## Data pipeline

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| DATA-1 | `test_download_sources.py`, `test_filters.py`, `test_dedup.py`, `test_cleaned_jsonl.py` | `test_data_pipeline.py` | CI `smoke-linux`/`parity-macos` `--dummy` step | offline byte-fallback | live HF download path has no CI coverage (by design, `--dummy` only) |
| DATA-2 | none | `test_data_pipeline.py` | CI `--byte-fallback` tokenize step | byte fallback, dtype selection | no unit test isolating `src/data/tokenize.py` from the pipeline |
| DATA-3 | `test_shard.py`, `test_packing_dtype.py` | `test_data_pipeline.py` | CI pack step | uint16/uint32 boundary at vocab 65536 | none |
| DATA-4 | `test_split_shards.py` | `test_data_pipeline.py` | CI split step feeding `smoke_test.py` | shard-boundary split, val-token count | none |
| DATA-5 | `test_r2_sync.py`, `test_storage.py` | none | none | mocked/hermetic paths only | real-network R2 round-trip never exercised in CI (credentials-gated) |
| DATA-6 | `test_datatrove_pipeline.py` | none | none | offline-asset skip | scale pipeline runs in `.venv-dt`, outside CI entirely |
| DATA-7 | `test_stack_v2.py`, `test_ts_clean.py` | `test_build_ts_clean_corpus.py` | none | `resolve_tsc()`-gated real-tsc path | full 2–3B corpus build never run (#252) |
| DATA-8 | `test_sft_data.py`, `test_sft_loader.py`, `test_sft_sources.py`, `test_instruct_sft.py`, `test_reasoning_sft.py`, `test_reasoning_traces.py`, `test_tool_sft.py`, `test_chat_template.py`, `test_qwen_chat_template.py` | `test_sft_loader.py`, `test_sft_corpus.py` | `test_sft_driver_e2e.py` (builders -> `scripts/sft.py` -> real steps + resume; packed -> `split --shards` -> `PackedLoader`) | response masking through the driver, manifest EOS/template mismatch + missing key, form mixing (agree/disagree), over-length dropped not truncated, malformed record line, length-mismatched record, empty corpus, tool abstention / multi-call / schema-invalid payload | e2e driver test is MLX-only (no CUDA-backend equivalent); no real-tokenizer corpus is exercised in CI (byte fallback only) |
| DATA-9 | `test_dpo_data.py`, `test_dpo_loader.py`, `test_dpo_sources.py` | `test_dpo_loader.py` | none | licence/source tagging | `scripts/dpo.py` untested end to end |
| DATA-10 | `test_build_decontam_blocklist.py` | none | none | n-gram hashing | no integration test wiring the blocklist into a corpus build |
| DATA-11 | `test_vocab_sample.py` | none | none | fetch/filter/tagging, hermetic | `scripts/vocab_sweep.py` (the driver) has zero test references |
| DATA-12 | `test_curriculum.py`, `test_swift_fim.py` | CI FIM-pack steps (`swift-macos`/`swift-linux`) | CI `swift-parity` FIM-shard byte-diff | length-curriculum stage boundaries | none |

## Native tokenizer

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| TOK-1 | `monica-selfcheck` | n/a (zero-dependency package by design) | CI `swift-macos`/`swift-linux` train-the-parity-corpus step | merge-order determinism | none |
| TOK-2 | `monica-selfcheck` | `test_swift_fim.py` (build-gated) | CI FIM-pack steps | shard layout vs `src/data/shard.py` | none |
| TOK-3 | n/a (cross-platform property) | n/a | CI `swift-parity` (`cmp` of `tokenizer.json` + FIM shards) | byte-identity Mac vs Linux | none — the strongest-gated capability in the repo |
| TOK-4 | `test_swift_parquet.py` | n/a | CI `full-macos` (the suite job — still the only one with both pyarrow and a Swift toolchain; #315's split moved the parity/smoke gates off it, not this) | skips when `monica-tokenize` unbuilt | Linux CI never builds Swift, so Parquet parity is macOS-only |
| TOK-5 | `test_swift_fim.py` | CI FIM-pack | CI `swift-parity` diff | FIM span selection | none |

## Model and the seam

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| MODEL-1 | none dedicated | `load_config()` in `test_mlx_parity.py`, `test_cuda_parity.py`, `test_bench_config_export.py` | `scripts/smoke_test.py` | valid configs only | no test asserts `MambaConfig.validate()` **raises** on an invalid config (e.g. `head_dim ∤ d_inner`) |
| MODEL-2 | `test_mlx_parity.py` | `test_mlx_parity.py` | `smoke_test.py --backend mlx` (CI `parity-macos`) | chunk-boundary, seg_ids packing | none |
| MODEL-3 | `test_mlx_parity.py` (`forward_step_parity`) | same | same smoke gate | stacked-step vs forward | none |
| MODEL-4 | `test_cuda_parity.py`, `test_cuda_train_step.py`, `test_cuda_compile.py` | `test_backend_parity.py` (torch self-check half only) | `smoke_test.py --backend cuda` (CI `smoke-linux`, CPU torch) | CPU-torch path | real-GPU path (fused mamba-ssm kernel, MPS) skip-gated in CI — no GPU on hosted runners |
| MODEL-5 | `test_moe.py`, `test_moe_routing.py`, `test_moe_balance.py`, `test_moe_balance_mlx.py`, `test_cuda_moe.py`, `test_cuda_moe_balance.py`, `test_cuda_moe_gather.py` | `test_cuda_moe_fixture_parity.py`, `test_backend_parity.py` | CI `smoke-linux` MoE-config smoke step | dropless grouped-gather, shared expert, route-bias | none |
| MODEL-6 | `test_upcycle.py` (mlx-gated) | none | none | `_MUST_MATCH` 15-field guard | `scripts/upcycle.py` untested end to end |
| MODEL-7 | `test_sizing.py`, `test_sizing_mlx.py`, `test_train_time.py` | `test_bench_config_export.py`, `test_bench_context.py` | none (informational) | tied-embedding accounting | none |
| MODEL-8 | `test_quantize.py`, `test_quantize_mlx_format.py` | `test_quant_checkpoint.py`, `test_quant_parity.py` | CI `swift-engine` int8 `monica-bench` step | int8 group size, scale/zero-point | none |

## Training

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| TRAIN-1 | `test_train_loop.py` | `test_train_loop.py` | `scripts/smoke_test.py` (CI `smoke-linux`/`parity-macos`) | grad accum, grad checkpointing | none |
| TRAIN-2 | `test_train_loop.py` resume tests | `test_checkpoint.py` | `scripts/smoke_test.py` (resume exactness is its headline check) | bit-exact fp32 resume at toy scale | `scripts/train.py`'s **stream-resume** path (data position rebuilt from seed+step) is a separate code path the smoke gate does not cover |
| TRAIN-3 | `test_loss_scale.py`, `test_lowp_parity_band.py` | `test_mlx_train_step.py`, `test_cuda_train_step.py` | `scripts/smoke_test.py` | overflow skip, scale halving | smoke gate runs `toy.yaml` (fp32), so the overflow/skip path never executes end to end |
| TRAIN-4 | `test_stream.py` | none | none | shard rotation | real R2 streaming untested in CI (credentials-gated) |
| TRAIN-5 | `test_schedule.py` | `test_train_loop.py` | `scripts/smoke_test.py` | warmup/stable/decay boundaries | none |
| TRAIN-6 | `test_cuda_muon.py`, `test_muon_taxonomy.py` | `test_cuda_train_step.py` | none | param taxonomy (Muon vs AdamW split) | none beyond CPU-torch gating |
| TRAIN-7 | `test_parallel.py` (portable sizing/partition math) | `test_cuda_distributed.py` | none | single-process mocks, expert-partition policy | real multi-GPU/multi-node FSDP2 has no CI path — structurally unverifiable on a single hosted runner |
| TRAIN-8 | `test_cuda_8bit_optimizer.py`, `test_cuda_fp8.py` | none | none | dependency-absent path | hardware-unverified: no Hopper+ GPU, no `transformer_engine`/`bitsandbytes` in CI |
| TRAIN-9 | none | `test_train_loop.py` (`log_every` wiring only) | `scripts/smoke_test.py` (runs, output not asserted) | none | `src/train/logging.py` has no direct unit test; its output format is never asserted on |

## Post-training

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| POST-1 | `test_sft_train_step.py` | `test_sft_train_step.py` | none | response masking in the loss | `scripts/sft.py` untested end to end |
| POST-2 | `test_dpo_math.py` | `test_dpo_train_step.py` | none | reference-model logratio, beta | `scripts/dpo.py` untested end to end |
| POST-3 | `test_grpo.py` | `test_grpo_train_step.py`, `test_verifiers.py` | none | group advantage, KL penalty | `test_verifiers.py:39` gates real code execution behind `RUN_CODE_VERIFIER`, which **no CI job sets** — the execution path never runs in the standard matrix |
| POST-4 | none | `test_dpo_sources.py` (source label only, never invokes generation) | none | none | `scripts/gen_onpolicy_prefs.py` has zero test references anywhere |

## Serving

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| SERVE-1 | `test_generate.py` | `test_generate.py` | `scripts/smoke_test.py`, CI `swift-engine` `monica-generate` steps | EOS handling, max tokens | none |
| SERVE-2 | `test_repetition_penalty.py`, `test_generate.py`, `test_constrained_sampling.py` | same | `scripts/smoke_test.py` | temp 0, top-p renormalization, all-non-finite fallback | none |
| SERVE-3 | `test_serve.py` (`RewindTree` + `SessionHistory`) | `test_rewind_entry_point.py` (whole REPL, portable), `test_serve.py` `test_mlx_rewind_restores_the_opaque_state_and_the_continuation` (mlx-gated, toy scale) | none | over-deep rewind, uncommitted session, root rewind, `/rewind 0/-1/abc/#unknown`, unknown command, double branch + rewind across the branch point, LRU eviction, `--rewind-depth 0`, negative `--rewind-depth` | no CI job drives the CLI REPL itself — the mlx-gated integration test is the closest surface; CUDA and Swift are uncovered (the path is backend-agnostic but exercised on neither) |
| SERVE-4 | `test_constrained_sampling.py`, `test_masked_decode.py`, `test_completion_mask.py` | CI `swift-macos` mask-set parity (Python vs Swift `VocabTrie`) | CI `swift-engine` `monica-generate --lsp-mask` | empty/all-out-of-range/duplicate-bearing `allowed_ids` | none |
| SERVE-5 | `test_spec_decode.py` (mlx-gated) | none | none | greedy-equivalence assertion | Swift side has only the `verifyBlock` prerequisite; full spec decode not built (#172) |

## Evaluation

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| EVAL-1 | `test_val_loss.py`, `test_domain_bpb.py` | `test_train_loop.py` | `scripts/smoke_test.py` | per-domain split, BPB vs ppl | none |
| EVAL-2 | `test_olmes_adapter.py`, `test_olmes_generate.py` | none | none | `lm_eval`-gated skip | full benchmark run has no CI coverage |
| EVAL-3 | `test_long_context.py` (mlx-gated) | none | none | extension beyond trained length | never runs in CI |
| EVAL-4 | `test_code_suite.py`, `test_external_sets.py`, `test_build_humaneval_ts_set.py` | `test_eval_code_suite.py`, `test_build_decontam_blocklist.py` | `test_external_sets.py::test_live_pull_returns_documented_shape` (opt-in) | real upstream schemas per adapter; struct-valued prompt rejected; SAFIM missing `{{completion}}`; McEval `task_id`-derived language; empty-prefix infill; unpinned entry refused; stale pin fails by name | **CI covers the offline fixture path only.** All 7 pins were pulled live and normalized end-to-end locally on 2026-08-20 (counts in `eval_sets/external/README.md`; none is schema-verified-only), but that is an **opt-in** run — `MONICA_EXTERNAL_LIVE=1`, deliberately not wired into any job (#315 macOS headroom + contamination). CI therefore cannot detect a pin that goes stale upstream |
| EVAL-5 | `test_fim_eval.py` | none | none | prefix/suffix/middle split | no CI job runs FIM eval |
| EVAL-6 | `test_code_recall.py`, `test_code_needle.py` | none | none | distractor sampling | no CI job runs these |
| EVAL-7 | `test_retrieval_probe.py` (mlx-gated), `test_probes.py` | none | none | key/value distinctness | no CI job runs these |
| EVAL-8 | `test_moe_routing.py` | `test_backend_parity.py` routing-stats section | Swift `monica-parity` `mixing_matrix` | routing entropy | none |
| EVAL-9 | `test_bfcl_adapter.py` | none | none | call-shape normalization | `scripts/eval_bfcl.py` untested end to end |
| EVAL-10 | `test_quant_parity.py` | `test_quant_parity.py` | CI `swift-engine` quantized bench (informational) | int8 tolerance band | none |
| EVAL-11 | `test_ssi_contract.py` | `test_ssi_contract.py` | none | escape-hatch gate | no CI job runs the SSI contract end to end |
| EVAL-12 | `test_ts_error_eval.py` (node/tsc-gated) | none | none | 96-item injection set | `scripts/validate_ts_error_set.py` untested; needs `npm install` no Python CI job performs |

## LSP / structural-signal integration

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| LSP-1 | `test_opengrep_oracle.py`, `test_eval_code_suite.py` | `test_ts_lsp.py`, `test_lsp_harness.py` | CI `swift-macos` live-tsserver steps | multi-push diagnostics ordering (#211) | none |
| LSP-2 | `test_lsp_diagnostics.py` | `test_lsp_harness.py` | none | debounce, reap | none |
| LSP-3 | see SERVE-4 | see SERVE-4 | see SERVE-4 | trie prefix masking | none |
| LSP-4 | `test_opengrep.py` (binary-gated), `test_opengrep_oracle.py` | none | none | rule-match scoring | `scripts/opengrep_soak.py` untested end to end |
| LSP-5 | `test_lsp_tsc.py` (tsc-gated) | none | none | pinned tsconfig | none beyond toolchain gating |
| LSP-6 | `test_prettier.py` (PATH-gated) | none | none | idempotent format | none |
| LSP-7 | `test_lsp_chat.py`, `test_lsp_lm.py`, `test_mlx_lm_adapter.py` | none | none | model-availability gating | none |
| LSP-8 | `test_lsp_execute.py` (tsc-gated) | none | none | timeout, non-zero exit | none beyond toolchain gating |
| LSP-9 | `monica-lsp --self-test` | `monica-lsp --probe-reap` (CI `swift-macos`) | `monica-lsp --bench` vs `scripts/bench_ts_lsp.py` | framing/demux/trie/scanner | `swift-linux` runs only `--self-test` (no node toolchain) — live-server path is macOS-only, by design |
| LSP-10 | `test_ts_server_direct_mechanism.py` (binary-free, the CI gate) | `test_ts_server_direct.py` (toolchain-gated): 96-record × 2-direction parity vs `TsLspService`, plus the anti-vacuity `expected_diagnostic` half | `scripts/bench_ts_lsp.py` `diagnostics_direct` column + `verdict_direct`, BLIND-guarded | all 15 enumerated: write framing (newline, never `Content-Length`), read framing (reused `jsonrpc.read_message`), events never resolve waiters, out-of-order `request_seq`, `success: false`, missing `body`, EOF/dead-child restart, cold load excluded from op stats, syntactic+semantic merge, `category` filter, `TS`-prefixed code, 1-based coordinates, whole-file `updateOpen`, no-toolchain skip, clean→clean unambiguous | No Linux CI coverage of the live half (the `portable` job has no node); the parity gate costs ~74 s, ~70 s of it the LSP client's own debounce. `TsServerDirect` has no consumer yet — adoption by `CompositeOracle`/the harness is a separate decision, so nothing tests it *in* a loop |

## Swift engine

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| ENGINE-1 | n/a | n/a | CI `swift-engine` `monica-parity` forward + stacked-step | fp32 `rtol=1e-4/atol=1e-5` | none |
| ENGINE-2 | `monica-bench --self-test` | n/a | CI `swift-engine` `--mode all` (informational) | argmax agreement gated | timing regressions are informational only, never threshold-gated |
| ENGINE-3 | n/a | n/a | `monica-parity` MoE sections | load counting, route-bias write-back | none |
| ENGINE-4 | n/a | n/a | `monica-parity` | fp16/bf16 tolerance band | none |
| ENGINE-5 | n/a | n/a | `monica-parity` extras | `mixing_matrix`, `hidden_states`, `verify_block` | none |
| ENGINE-6 | n/a | `test_workflow_triggers.py` (the trigger matrix) | `scheduled-parity.yml`'s `poc-fixture-oracle` + `poc-parity` | 571 MB poc-scale fixture, cross-process corruption guard (#298) | **dispatch/schedule-only** — since #302 these live in a workflow with no `pull_request`/`push` trigger at all, so a green PR never implies poc-scale parity |
| ENGINE-7 | n/a | `scripts/check_swift_checkpoint.py` | CI `swift-engine` Swift→Python round-trip (GATE) | the direction `monica-parity` alone cannot prove | none |
| ENGINE-8 | n/a | n/a | n/a | n/a | n/a — not built (#171) |
| ENGINE-9 | n/a | n/a | n/a | n/a | n/a — only the `verifyBlock` prerequisite exists (#172) |
| ENGINE-10 | `test_fixture_digest.py` (portable), `test_mlx_mixing_matrix.py`'s `disable_buffer_cache` cases | `test_parity_fixture_export.py`, `test_train_parity_fixture_export.py` | `export_parity_fixture.py --double-export` (default on); CI `poc-fixture-oracle` + `poc-parity`, `train-fixture-oracle` + `-verify` | flipped byte, dropped file, size mismatch, nested file, **empty tree is BLIND not clean**, MLX API change fails loudly, near-zero `mixing.*` entries | **intra-machine only** — two processes on ONE host; cross-machine determinism is not claimed and `train.safetensors` is known to drift ~1e-8 across runner instances. No PR-time CI job runs the double-export; the underlying MLX defect is unfixed upstream (design/14 §#298) |

## Conformance

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| CONF-1 | `test_mlx_parity.py`, `test_cuda_parity.py` | `src/conformance/forward_step_parity.py` | `scripts/smoke_test.py` | fp32 ~1e-4 rel | none |
| CONF-2 | `test_backend_parity.py`, `test_ci_backend_matrix.py` (portable YAML contract), `test_ci_macos_budget.py` (portable wall-clock contract, #315) | `test_backend_parity.py` | CI `parity-macos` (#315) — step *"Cross-backend parity"*, `MONICA_REQUIRE_BOTH_BACKENDS=1` over `.[dev,data,mlx,cuda]`, ~2 min against `timeout-minutes: 15` | the 5 real MLX↔torch comparisons (logits, hybrid/attention, `seg_ids` packing, portable-weights round-trip both directions, MoE routing-entropy) at fp32 `rtol=1e-4/atol=1e-5` on `config/toy.yaml`/`toy-hybrid`/`toy-moe`; plus the torch self-check on `cuda-cpu`; a missing backend on the designated job **errors** rather than skips, and re-adding a skip marker or dropping the flag from `ci.yml` is caught on the Linux `portable` job. #315 split this gate off `full-macos` so a 35-min suite can no longer kill it as an unexplained `ETIMEDOUT`; `test_ci_macos_budget.py` pins the budget literal, both `timeout-minutes`, and `budget + 300s <= timeout` so the legible guard fires first | Gated at **toy scale, fp32 only** (`B=2, L=24`) against **torch on CPU** — real-GPU CUDA kernels and poc-scale shapes are compared by **no** job. `cuda-cpu` (Linux) still skips 5 of 6: mlx has no Linux wheel, so it is not parity coverage. The budget contract checks the *shape* of the guard from `ci.yml`, never a runner's actual wall clock — only a real macOS run measures that |
| CONF-3 | `test_doc_boundary_parity.py`, `test_cuda_doc_boundary_parity.py` | `src/conformance/doc_boundary_parity.py` | none | seg_ids block-diagonal masking | none |
| CONF-4 | `test_mlx_parity.py`, `test_cuda_parity.py` | `src/conformance/prefill_decode_parity.py` | `monica-parity` | prefill vs stacked step | none |
| CONF-5 | `test_quant_parity.py` | `src/conformance/quant_parity.py` | CI `swift-engine` | int8 tolerance | none |

## Operations

| ID | Unit | Integration | E2E | Edge cases covered | Gaps |
|----|------|-------------|-----|--------------------|------|
| OPS-1 | n/a | n/a | `scripts/smoke_test.py`, CI `smoke-linux` + `parity-macos` | resume exactness + eval | does not cover `train.py`'s stream-resume (see TRAIN-2) |
| OPS-2 | `test_workflow_triggers.py` | n/a | `ci.yml` (9 jobs, PR/push + monthly schedule) + `scheduled-parity.yml` (4 jobs, dispatch/weekly schedule) | portable/seam guard, both backends, Swift parity; the trigger×job matrix itself; `ci.yml`'s cron literal `"43 9 3 * *"` and its event-scoped `concurrency.group` (#312) | 4 of 13 jobs are in a workflow with no PR/push trigger (see ENGINE-6); the monthly cadence bounds drift-detection latency to ~1 month, and nothing alerts on a red scheduled run beyond GitHub's default notification |
| OPS-3 | `test_bench_config_export.py`, `test_bench_context.py` | none | `monica-bench` CI steps (informational) | provenance tagging | nothing checks `docs/benchmarks.md` stays in sync with CI bench output |
