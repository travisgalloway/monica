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

- [2026-08-18] `src/eval/external_sets.py`, only the MultiPL-E entries carry `repo_verified=True`;
  the SAFIM / CrossCodeEval / RepoBench / MCEval `hf_repo`/`config` identifiers are best-known
  guesses that must be confirmed at pin time. Distinct from the missing revision pins themselves,
  which are ticketed. Found during `/closure-audit` pass B. Severity: correctness risk, deferred.

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
