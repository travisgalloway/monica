# Requirements review, 2026-09-23

## 1. Scope

Date: 2026-09-23. Repository: travisgalloway/monica. Scope: whole repo, no tokens.

Pass R1 checked acceptance criteria against delivered evidence, extracting 340 criteria across 127
issues in scope. Twenty-one of those issues carried no checkable criteria. Twenty-three issues were
fully criterion-verified: #101, #103, #104, #171, #219, #220, #222, #223, #240, #252, #341, #342,
#343, #346, #348, #356, #357, #360, #370, #371, #386, #387, #388. The remaining 104 issues in
scope were not individually criterion-verified this pass; they are listed under Not audited,
below. The first verification attempt on this pass stalled with no output. The counts above are
from the successful rerun.

## 2. Findings

All 13 REQ findings from this run, ranked by severity, then by verdict, with CONFIRMED before
PLAUSIBLE.

### High

**REQ-3256f522** · `data/sample_slice` · issue #252 · rule `criterion-unmet` · severity high ·
verdict CONFIRMED unmet

Issue #252 asked for the #193 pipeline run at full scale. The request included shards landed on
R2 and a recorded token count. The repo holds only `data/sample_slice/`, 3.7 MB total. That is not
the 50 to 70 billion token corpus #222 needs, nor the 150 billion token corpus #223 needs. No R2
manifest recording a real token count exists locally.

Disposition: covered by #418, #419, #420.

**REQ-279eefd9** · `docs/benchmarks.md` · issue #223 · rule `criterion-unmet` · severity high ·
verdict CONFIRMED unmet

Issue #223's own Verification section names `runs/large-a/metrics.jsonl` as the evidence path for
a completed or deliberately stopped run. That path does not exist anywhere in the repo.
`tests/test_large_moe_run.py`'s own docstring states it verifies offline mock pipeline simulation,
not an executed run.

Disposition: covered by #430, #432.

**REQ-4f5abcac** · `src/model/metal_kernels.py` · issue #171 · rule `criterion-unmet` · severity
high · verdict CONFIRMED unmet

Issue #171's acceptance criterion 2 required a measured tok/s or prefill-latency improvement in
the bench. Otherwise, the kernel was to be dropped. No such figure appears in `docs/benchmarks.md`
or `results/`. `docs/feature-matrix.md`'s ENGINE-8 row had not been updated from Planned before
this run, despite the kernel shipping in PR #333.

Disposition: ticketed as #443.

**REQ-3c6f62d4** · `docs/benchmarks.md` · issue #222 · rule `criterion-unmet` · severity high ·
verdict PLAUSIBLE unmet

Issue #222's accept line requires passing recall evals and domain-separated routing histograms
from an actual run. `docs/feature-matrix.md:139` states the #222 throughput figures are analytical
projections calibrated against datacenter GPU specifications, not local measurements. No `runs/`
artifact for a code-small-moe run exists.

Disposition: covered by #421, #422.

**REQ-98111e6a** · `docs/benchmarks.md` · issue #223 · rule `criterion-unmet` · severity high ·
verdict PLAUSIBLE unmet

Issue #223's checklist requires val-perplexity decreasing smoothly, a stable grad norm, and BPB
reported as the primary metric. No `runs/large-a/` directory or metrics file exists to check this
against.

Disposition: parked. REQ-279eefd9, above, already covers issue #223's missing run artifact.

### Medium

**REQ-0fa5914d** · `tests/test_mtp.py` · issue #356 · rule `criterion-unmet` · severity medium ·
verdict CONFIRMED unmet

Issue #356 requires an acceptance rate above 60% on code-completion benchmarks for native MTP
proposals. The only test trains a toy model for 70 steps on a hand-built, six-token repeating
integer sequence. That sequence has no tokenizer and no source code. No `eval_sets/` run backs the
claimed figure, anywhere in `results/` or `docs/benchmarks.md`.

Disposition: ticketed as #446.

**REQ-efef865e** · `docs/benchmarks.md` · issue #223 · rule `criterion-unmet` · severity medium ·
verdict PLAUSIBLE unmet

Issue #223's own body states the checkpoint-resume kill-and-resume rehearsal "has not happened," as
of its 2026-08-08 update. No later comment or artifact records that it happened since.

Disposition: parked.

**REQ-36d2b0da** · `tests/test_serve_critic.py` · issue #388 · rule `criterion-untested` ·
severity medium · verdict PLAUSIBLE unmet

The 70%-pruning-rate test hand-supplies critic probability values, eight of ten chosen below the
threshold by construction. The test never runs the trained critic head on real generations,
demonstrating the filter's arithmetic, not the criterion.

Disposition: parked.

**REQ-51785ec1** · `tests/test_serve_critic.py` · issue #388 · rule `criterion-untested` ·
severity medium · verdict PLAUSIBLE unmet

The three-times latency-speedup test skips seven of eight mock LSP calls by construction, from a
hand-picked probability list. The measured speedup follows from the test's own inputs, regardless
of the critic's real discriminative power.

Disposition: parked.

**REQ-c798238d** · `scripts/cache_critic_features.py` · issue #387 · rule `criterion-untested` ·
severity medium · verdict PLAUSIBLE unmet

The recorded calibration run, in `results/critic_training_metrics.json`, reports a calibration
error of 0.0 and an AUC-ROC of 1.0. The test held only 22 samples. Every negative label is the same
hardcoded syntax-error string, contrasted against clean labels. That is a trivially separable
synthetic set, not realistic near-miss completions.

Disposition: parked.

**REQ-9424bdf5** · `tests/test_cuda_fp8.py` · issue #240 · rule `criterion-untested` · severity
medium · verdict PLAUSIBLE unmet

The bf16-versus-fp8 forward-equivalence acceptance test needs real Hopper-or-later hardware. It
skips in every CI job, including `cuda-cpu`. No `results/` or `runs/` artifact records the
comparison having been run.

Disposition: parked.

**REQ-7bd2e6ac** · `tests/test_grammar_decoding.py` · issue #360 · rule `criterion-untested` ·
severity medium · verdict PLAUSIBLE unmet

The 100-prompt zero-syntax-error test builds each completion with its own hand-written,
deterministic string-closer. That closer reads only the grammar engine's open-bracket state. The
test never calls the model's real `sample()`/`generate()` loop under grammar constraints.

Disposition: parked.

**REQ-f465e57b** · `swift/Sources/MonicaTokenizer/Pretokenizer.swift` · issue #357 · rule
`criterion-untested` · severity medium · verdict PLAUSIBLE unmet

The claimed 5% compression-ratio improvement, on indentation-heavy TypeScript files, has no
recorded measurement. Nothing appears in `swift/`, `docs/`, or `results/`, before or after.

Disposition: parked.

## 3. Decisions needed

None from this pass. Every finding above resolved to a ticket, a coverage note on an already
ticketed issue, or a park. No pattern here required a standalone human decision.

## 4. Not audited

104 of 127 in-scope issues were not individually criterion-verified this pass. Named-burst issues
received file-existence triage only, not full criterion verification. The full list:

#339, #340, #344, #345, #347, #349, #350, #351, #352, #353, #354, #355, #358, #359, #361, #362,
#363, #364, #365, #366, #367, #368, #16, #30, #42, #43, #44, #45, #46, #47, #48, #49, #50, #102,
#141, #163, #164, #165, #166, #167, #168, #169, #170, #172, #174, #176, #177, #188, #189, #191,
#192, #193, #194, #195, #196, #197, #198, #199, #200, #201, #204, #211, #213, #214, #215, #216,
#217, #218, #221, #224, #225, #226, #227, #230, #237, #238, #239, #246, #247, #249, #251, #263,
#264, #265, #266, #267, #271, #272, #278, #279, #281, #285, #288, #302, #303, #304, #305, #306,
#307, #312, #315, #318, #322, #328.
