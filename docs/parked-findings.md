# Parked findings

Observations made while working that were **out of scope at the time**. This file is a dated log,
appended to and never rewritten — unlike `docs/feature-matrix.md` and `docs/test-plan.md`, which
assert current truth and are edited in place.

The list is deliberately **flat and unranked**. A ranked parking file is a second backlog wearing
a disguise. Nothing here is scheduled; if one of these is wanted, it becomes a new task with its
own done contract.

Format: date, `file:line`, what was seen, what task surfaced it, rough severity.

---

- [2026-08-18] `scripts/gen_onpolicy_prefs.py`, the on-policy preference generation driver has
  zero test references anywhere in `tests/`; `test_dpo_sources.py` only asserts a pre-existing
  `"onpolicy"` source label, never invokes the generator. Found during `/closure-audit` pass C.
  Severity: coverage, non-blocking.

- [2026-08-18] `scripts/vocab_sweep.py`, the vocab-sizing sweep driver has no test; only its
  underlying sampler `src/data/vocab_sample.py` is unit-tested. Found during `/closure-audit`
  pass C. Severity: coverage, non-blocking.

- [2026-08-18] `src/train/logging.py`, no dedicated unit test; exercised only incidentally via
  `test_train_loop.py`'s `log_every` wiring, so the module's actual output format is never
  asserted on. Found during `/closure-audit` pass C. Severity: coverage, low.

- [2026-08-18] `src/model/blocks.py`, `MambaConfig.validate()` has no negative-case test — nothing
  asserts it raises on an invalid config such as `head_dim` not dividing `d_inner`. Coverage is
  implicit via every other test's successful `load_config()` on already-valid YAML. Found during
  `/closure-audit` pass C. Severity: coverage, low.

- [2026-08-18] Driver-script tier generally: `scripts/{sft,dpo,upcycle,eval_code_suite,eval_bfcl,
  validate_ts_error_set,opengrep_soak}.py` each have solid unit coverage for their underlying
  `src/` logic, but no test exercises the script's own argument parsing and wiring, and none run
  in any CI job. Found during `/closure-audit` pass C. Severity: coverage, non-blocking.

- [2026-08-18] `tests/test_verifiers.py:39`, the RLVR code-verifier's real execution path is gated
  behind `RUN_CODE_VERIFIER`, which no CI job sets. Unlike the other 45 skips in the suite (all
  hardware or toolchain absence), this one is opt-in and therefore never runs in the standard
  matrix. Found during `/closure-audit` pass C. Severity: coverage, medium.

- [2026-08-18] `src/train/parallel.py` + `src/model/cuda_distributed.py`, real multi-GPU and
  multi-node FSDP2 correctness has no CI path — single hosted runners cannot form a real process
  group. Structurally unverifiable rather than a probe failure. Found during `/closure-audit`
  pass C. Severity: structural, non-blocking.

- [2026-08-18] `docs/benchmarks.md` is a hand-maintained ledger; nothing automatically checks it
  stays in sync with the `monica-bench` output CI produces. Found during `/closure-audit` pass C.
  Severity: process, low.

- [2026-08-18] `docs/reserve/{10-distillation,path-b-run}.md` and
  `docs/reserve/runbooks/{m10-phase-bprime-append,m10-pod-chain}.md` collectively cite ~40 paths
  removed by #189 (`scripts/distill.py`, `src/model/teacher.py`, `config/manifests/`, and others).
  Every one of these docs carries a "Reserve / historical, superseded 2026-07-19" banner, so the
  drift is self-disclosed at the point of citation. Found during `/closure-audit` pass D.
  Severity: doc hygiene, lowest.

- [2026-08-18] `src/lsp/jsonrpc.py:181,242`, `src/lsp/harness.py:300`, `src/lsp/oracle.py:90`,
  `src/lsp/ts_lsp.py:227`, `src/data/tool_sources.py:342`, `src/eval/code_suite.py:350` — bare
  `except: pass` swallow sites. Each looked deliberate in context, none were verified individually.
  Found during `/closure-audit` pass B. Severity: robustness, low.

- [2026-08-18] `tests/test_cuda_parity.py:250,252,281,283,334,339`, the real-GPU paths (CUDA/MPS
  device, fused mamba-ssm kernel) are skip-gated and therefore never exercised on hosted CI
  runners. Legitimate hardware gating, recorded so the coverage claim is not mistaken for
  automation. Found during `/closure-audit` pass C. Severity: structural, non-blocking.

- [2026-08-18] ~~`src/eval/external_sets.py`, only the MultiPL-E entries carry `repo_verified=True`;
  the SAFIM / CrossCodeEval / RepoBench / MCEval `hf_repo`/`config` identifiers are best-known
  guesses that must be confirmed at pin time. Distinct from the missing revision pins themselves,
  which are ticketed. Found during `/closure-audit` pass B. Severity: correctness risk, deferred.~~
  **RESOLVED by #304** (2026-08-20): the guesses were wrong — two `hf_repo` ids returned 401, one
  was an unloadable script dataset, and two `config=None` values were invalid. All seven
  identifiers are now confirmed against the live hub, `repo_verified=True` everywhere, with the
  method and date in each entry's `note`.

- [2026-08-18] `/closure-audit` pass A left two sub-checks **UNAUDITED** rather than reporting a
  false clean: per-field consumer verification of `MambaConfig`'s ~25 fields in
  `src/model/blocks.py` (skimmed, none looked orphaned, not mechanically verified), and the
  reverse env-var direction (a documented or CI-set variable that no code reads). Re-run with
  `area=src/model`. Severity: audit completeness, non-blocking.

- [2026-08-18] `CLAUDE.md`'s CI paragraph omits the `cuda-cpu` job ("CUDA test suite, Linux, CPU
  torch") while naming the other nine. An omission rather than a false claim, so it was corrected
  in the same pass; recorded here because the pattern — a doc listing jobs by hand — will drift
  again the next time a job is added. Severity: process, low.

- [2026-08-19] `src/lsp/ts_service.py:578` (`TsLspService.diagnostics`), the #278 residual
  in-flight notification race costs measurably more signal than previously quantified: on the
  5000-file bench project the LSP client observes only **64%** of introduced `TS2339`s while
  `TsServerDirect` observes **100%** on the identical edit cycle. Found while adding #279's
  `diagnostics_direct` bench column. Severity: correctness, non-blocking (the direct transport
  sidesteps it; closing it in the LSP client needs a per-edit request/response barrier).

- [2026-08-19] `src/lsp/oracle.py` / `src/lsp/harness.py`, nothing consumes
  `src/lsp/ts_server_direct.py` yet — `CompositeOracle`'s `"ts"` arm is still `TsLspOracle`.
  #279 deliberately scoped adoption out; whether the eval arm should switch transports (and what
  `is_incomplete`/frontier filtering looks like under whole-program rather than open-document
  semantics) is an unmade decision. Found while closing #279. Severity: scope, deferred.

- [2026-08-19] `tests/test_ts_server_direct.py`, the parity gate costs ~74 s of the macOS suite,
  ~70 s of which is the LSP client's own #278 debounce paid 192 times. There is no `slow`/`live`
  pytest marker in this repo to gate it behind. Found while adding the gate. Severity: test
  runtime, non-blocking.

- [2026-08-19] `swift/engine/Sources/MonicaEngine/Sampler.swift:146-148`, the Swift port's
  all-non-finite fallback carries the same duplicate bias #285 fixed on the Python side, and
  rebuilds the in-range `ids` array a second time rather than reusing the one built for the mask.
  Found while fixing #285. Severity: cosmetic, non-blocking (sampled draws are not a
  cross-language parity contract; only greedy is).

- [2026-08-19] `src/model/mlx_backend.py:948`, #264's `mx.eval(h)` barrier in
  `mixing_matrices` reduces but does not eliminate the #298 corruption: measured 18/40 (toy.yaml)
  and 20/40 (toy-moe.yaml) corrupt trials with the barrier in place and the MLX buffer cache on
  (`scripts/probe_mlx_buffer_reuse.py --pattern mixing --barrier --trials 40`). Only barrier +
  `set_cache_limit(0)` gives 0/40. Every path that writes a checked-in oracle already runs under
  both, so the fixtures are covered — an arbitrary caller of `mixing_matrices`/`hidden_states` is
  not. Found while closing #298. Severity: correctness, non-blocking (deciding what a bare caller
  should get is a behaviour change beyond #298's contract).

- [2026-08-19] `src/model/mlx_backend.py:938` `hidden_states` forks a lazy `h` the same way
  `mixing_matrices` does (`layer_fn(h)` advancing while the previous `h` is retained in `hs`) and
  carries no barrier at all. Unmeasured — the #298 probe only exercises `mixing_matrices`. Found
  while closing #298. Severity: correctness, unquantified.

- [2026-08-19] `pyproject.toml:30` pins only `mlx>=0.18` while the repo's numerical contracts
  (fp32 `1e-4/1e-5`, the #266 low-precision bands, every checked-in oracle) are all calibrated on
  one build. #298 measured three releases behaving differently under the same probe. Whether to
  pin a floor/ceiling is a repo-wide dependency decision. Found while closing #298. Severity:
  reproducibility, non-blocking.

- [2026-08-19] No PR-time CI job runs `export_parity_fixture.py --double-export`; the guard only
  fires when a human regenerates a fixture. The cross-runner equivalents
  (`poc-fixture-oracle`/`poc-parity`, `train-fixture-oracle`/`-verify`,
  `.github/workflows/ci.yml`) are dispatch/schedule-only. Adding an always-on macOS job is a
  runtime-cost decision. Found while closing #298. Severity: coverage, non-blocking.

- [2026-08-19] `CLAUDE.md` and `.claude/plans/issue-298.md`'s V8 both give the Swift gate as
  `cd swift/engine && swift run monica-parity`, which cannot work: mlx-swift's `default.metallib`
  is an Xcode-only build product, so `swift run` builds fine and then dies with "Failed to load
  the default metallib". `.github/workflows/ci.yml:446-487` documents this and uses `xcodebuild`
  + the product binary instead. On a host with only Command Line Tools installed the gate is not
  runnable at all. Found while closing #298. Severity: docs/tooling, non-blocking (CI runs the
  real form).

- [2026-08-19] `tests/test_parity_fixture_export.py:90`, the macOS suite is flaky at ~1-in-4 when
  `tests/test_mlx_mixing_matrix.py` runs before it in the same process (3/12 measured on `main`,
  0/12 for the export module alone). The failure is `greedy_ids` as EXACT ints — 16/16 wrong, by
  222 — i.e. #298 corruption, not a tolerance. Both modules already apply `set_cache_limit(0)`, so
  the mitigation does not compose across pytest modules. Fixing it means process isolation for the
  MLX fixture modules (pytest-forked, or a separate CI job), which is test-infrastructure work.
  Found while verifying #298. Severity: CI reliability, non-blocking but recurring.

- [2026-08-19] `.github/workflows/` (both files), `actionlint` is installed on the dev host and
  reports both workflows clean, but nothing in CI or in the pytest suite runs it — workflow
  syntax/expression errors are only caught by GitHub at dispatch time. Found while splitting
  `ci.yml` for #302 (a move of 383 lines of YAML with no local schema gate). Severity: CI
  tooling, non-blocking.

- [2026-08-20] ~~`src/eval/external_sets.py:238` (`normalize_repobench`), the row `id` is
  `f"{repo_name}::{file_path}"`, which is not unique: a live `cross_file_first` pull yields 8033
  rows carrying only 4588 distinct ids, because RepoBench v1.1 has several completion points per
  file. Anything that keys results by `id` would silently collapse them. Found while pinning the
  external suites for #304. Severity: correctness risk in downstream scoring, non-blocking here.~~
  **RESOLVED by #304** (2026-08-20): the id now folds in `token_num` and `gold_snippet_index` —
  both stable per-row disambiguators present on every upstream row — and both are required by
  `_require(...)` so schema drift on either field is caught rather than silently degrading back
  to a colliding id. Confirmed against a live pull: 8033 rows now carry 8020 distinct ids (was
  4588). The 13 residual collisions are duplicate rows *upstream* — byte-identical `prompt` and
  `answer` — so no distinct instance is collapsed; the id is a faithful instance key, not a
  literally-unique row key.

- [2026-08-20] `scripts/build_decontam_blocklist.py:138`, the `--min-words` filter (default 13)
  drops every text from `external:safim`, `external:repobench` and `external:real-fim-eval` in
  fixture mode — three of the seven external sources contribute 0 lines to the blocklist. Verified
  identical on `main`, so it predates #304, but it means those suites are not actually
  decontaminated against. Found while checking the #304 fixture rewrite lost no texts. Severity:
  contamination risk, non-blocking.

- [2026-08-20] `src/serve/rewind.py:49`, `RewindTree` exposes `__contains__`/`__len__` but no
  public way to enumerate its retained node ids — `_nodes` is the only source. `SessionHistory`
  (#305) therefore keeps its own commit-order list and filters it through `__contains__`, which
  works but duplicates bookkeeping the tree already has and would silently drift if anything ever
  committed to the tree without going through `SessionHistory`. Found while wiring the rewind
  entry point for #305. Severity: minor API gap, non-blocking.

- [2026-08-20] `src/data/tool_sft.py:96-115` (`_iter_calls`), a `<tool_call>` block whose payload is
  syntactically malformed JSON is silently *skipped* rather than counted: the row then declares zero
  calls, passes `_row_schema_valid` vacuously, and is written to `tool.jsonl` with the broken text
  trained verbatim. `n_schema_invalid` counts only well-formed-JSON calls that violate the declared
  schema. Pinned by `tests/test_sft_corpus.py::test_syntactically_malformed_payload_is_a_named_outcome`.
  Found while wiring the SFT training driver for #306 (changing builder output was out of scope).
  Severity: training-signal quality, non-blocking.

- [2026-08-20] `src/data/loader.py:36-38`, `PackedLoader`'s `seq_len + 1` stride re-straddles the
  atomic documents `shard.pack_atomic` was careful to chunk-align — a `reasoning-packed/` corpus
  packed so that no trace spans a sequence boundary loses that property as soon as the loader cuts
  it into `seq_len+1` windows, and `.bounds` is dropped by `split_shards` anyway. Found while
  proving the packed form trains via the pretraining driver for #306. Severity: correctness of the
  #68 boundary-reset claim on this path, non-blocking.

- [2026-08-20] `src/data/instruct_sft.py:157`, the instruct builder writes its manifest as
  `manifest.json` while `reasoning_sft.py` and `tool_sft.py` write `<kind>-manifest.json`. The
  resolver carries a per-form name table because of it (`src/data/sft_corpus.py:FORMS`), and the
  bare name is one `pack_atomic` output away from a collision if a future form ever writes shards
  into the same directory. Found while writing the #306 resolver. Severity: naming inconsistency,
  non-blocking.

- [2026-08-20] `scripts/dpo.py` / `scripts/rlvr.py` consume neither the `shared/sft/` layout nor
  `src/data/sft_corpus.py` — #306 gave the masked SFT forms a driver, but DPO and GRPO over the same
  corpora still have no path. Found while closing #306 (explicitly out of its scope). Severity:
  missing consumer, non-blocking.

- [2026-08-21] `.github/workflows/ci.yml`, `full-macos` — `pytest-xdist` (`-n 3`, the hosted macOS
  runner is 3-core) is the obvious next lever on the macOS suite's wall clock and was deliberately
  not taken in #315: it adds a dependency and test-scheduling nondeterminism to a repo whose
  central claim is bit-exact parity. Found while splitting the macOS job for #315. Severity: CI
  throughput, non-blocking.

- [2026-08-21] `docs/design/14-inference-engine.md:1056`, the recorded ~1-in-4 `full-macos` flake
  for the engine module pair pre-dates #303 and is untouched by #315's split — a flaky suite job
  now also consumes the budget guard's headroom on a rerun. Found while splitting the macOS job
  for #315. Severity: CI reliability, non-blocking.

- [2026-08-21] `docs/test-plan.md:100`, `MONICA_EXTERNAL_LIVE=1` is still wired into no CI job; the
  row defers this citing #315, which did not take it on. Found while auditing the test-plan rows
  that name CI jobs for #315. Severity: missing coverage, non-blocking.

- [2026-08-21] `.github/workflows/ci.yml`, `full-macos` — the hosted macOS runner is ~6× slower
  than an M1 Pro on this same suite (1577 passed / 90 skipped in 2165 s on CI vs 1759 passed /
  20 skipped in 361 s locally): *fewer* tests, six times the wall clock. Unexplained, and the
  single largest term in the macOS CI budget. Found while profiling for #315. Severity:
  CI throughput, non-blocking.

- [2026-08-22] RESOLVED by #322 — the 2026-08-21 "`full-macos`: the hosted macOS runner is ~6×
  slower than an M1 Pro" finding above. It is not an environment gap: `--durations=25` on run
  32542924915 shows `tests/test_cuda_distributed.py` alone is 1610.94s of a 2086.73s suite (19.06s
  locally, ~85×) because macOS has no `fork` start method for `torch.multiprocessing.spawn` and its
  ~26 gloo workers each re-import torch/mlx (~62s apiece). Everything else runs at ≈1.4× local. The
  file is `--ignore`d on `full-macos` (it is already gated on `cuda-cpu`). See
  `docs/design/03-conformance.md` §"Cutting the suite".

- [2026-08-22] RESOLVED by #322 — the 2026-08-21 "`pytest-xdist` (`-n 3`) is the obvious next
  lever" finding above. Rejected on evidence, not taste: the measured cost was nested-subprocess
  import time, not schedulable test time, and `-n 3` on a 3-core runner whose slowest tests each
  spawn two gloo workers would oversubscribe it. The decision is recorded in
  `docs/design/03-conformance.md`; this item is closed rather than re-parked.

- [2026-08-22] `.github/workflows/ci.yml`, `swift-macos` (`swift selfcheck (macOS)`) fails
  spuriously — run 32542924915 on unchanged `main` is red on that job with the rest of `main`
  green, and it also failed on PR #323. Found while cutting the macOS suite for #322 (explicitly
  out of its scope). Severity: CI reliability, non-blocking.
