# Mac Runbook — Mamba POC next steps

Ordered, local checklist for finishing the POC on Apple Silicon. Mirrors the
GitHub tracker (parent issue #2) and milestones M1–M4. Each step lists the files
to touch, the command to run, and the acceptance gate.

> Backend rule: only the backend modules — `src/model/mlx_backend.py` and
> `src/model/mlx_train_step.py` — may import `mlx`. Keep everything above the seam
> backend-free (`tests/test_import_guard.py` enforces this).

---

## 0. Environment setup
- [ ] Clone + check out the branch: `git switch claude/mamba-poc-plan-UPhGd`
- [ ] Create a venv: `python3 -m venv .venv && source .venv/bin/activate`
- [ ] Install portable + dev deps: `pip install -e ".[dev,data]"`
- [ ] Install MLX (Apple Silicon only): `pip install mlx`
- [ ] Confirm green baseline: `pytest` → expect 10 passed (data/schedule/val_loss/import-guard)

## 1. M1 — MLX model + parity  (issues #3, #5, #6, #7, #8, #9)
- [ ] **dt-bias init (#5):** implement `SelectiveSSM._init_dt_bias` in `src/model/mlx_backend.py`
      (log-uniform dt in `[dt_min,dt_max]`, clamp `dt_init_floor`, bias = inverse-softplus).
- [ ] **Selective scan (#7):** implement `SelectiveSSM.parallel` (cumsum closed form, no Python loop).
      Add a test: parallel vs sequential reference agree in **fp32**, ~1e-4 rel.
- [ ] **Block + model (#8):** implement `RMSNorm`, `MambaBlock.forward_seq`/`.step`,
      `MLXMambaModel.forward/step/init_state/get_state/set_state`, tied head,
      `_portable_state_dict`/`_load_portable`.
- [ ] **forward/step parity (#9):** run `src/conformance/forward_step_parity.py`
      on the toy model → must pass in fp32 (~1e-4 rel).
- [ ] **Precision decision (#3):** benchmark fp16+loss-scaling vs bf16 on MLX; set
      `precision` in `config/poc.yaml` with a one-line rationale. (toy stays fp32.)
- [ ] **Chunked scan (#6):** implement `chunk_size` path; optional now (off the smoke
      path since seq 1024 < ~2k), required before long-context. Verify it matches the
      single-pass scan in fp32.
- [ ] Gate: scan-vs-sequential test green **and** forward_step_parity green.

## 2. M2 — Data pipeline at scale  (issues #4, #10)
- [x] **Tokenizer check (#4):** CONFIRMED `allenai/OLMo-7B-hf` loads via HF, vocab 50280
      (eos 50279) < 65536; `vocab_size` set in `config/poc.yaml`. No compatibly-licensed
      pre-tokenized <65536-vocab subset exists on HF → tokenize raw text ourselves.
- [ ] **Download (#10):** implement `download_fineweb_edu_slice` in `src/data/download.py`
      (stream `HuggingFaceFW/fineweb-edu` `sample-10BT`, ODC-By).
- [ ] Run the real pipeline (~2–5B tokens; plan ~10–20GB raw, several GB packed):
      `download → tokenize → pack → split`. (`pack`/`split`/`loader` are done + tested.)
- [ ] Gate: loader yields contiguous batches; val shard disjoint from train.
- [ ] Optional sanity now (offline, no network):
      ```
      python -m src.data.download --dummy --out data/raw --max-docs 2000
      python -m src.data.tokenize --in data/raw/dummy.txt --out data/ids.npy --byte-fallback
      python -m src.data.pack  --in data/ids.npy --out data/packed.bin
      python -m src.data.split --packed data/packed.bin --out data/split --val-tokens 2000
      ```

## 3. M3 — Training loop  (issue #11)
- [ ] Implement an MLX `train_step(model, inputs, targets, lr) -> {loss, grad_norm}`
      (`nn.value_and_grad` + optimizer, grad accumulation, grad clipping, mixed precision /
      loss scaling per #3).
- [ ] Finish the `TODO[mac]` checkpoint + logging hooks in `src/train/loop.py`.
- [ ] Wire logging from step 1: loss, val loss/perplexity (`src/eval/val_loss.py`), LR,
      grad norm, tokens/sec (W&B or similar).
- [ ] Gate: toy ~50 steps — train loss **and** held-out val perplexity both improve.

## 4. M4 — SMOKE TEST gate  (issue #12) — do not pass until green
- [ ] Implement `scripts/smoke_test.py`: toy model, tiny data, ~50 steps, fixed seed.
- [ ] Reference (uninterrupted) run vs second run that **saves → kills → resumes → continues**.
- [ ] Assert post-resume trajectory matches reference within tolerance (fp32 toy ⇒ ~exact),
      using `save_weights`/`save_resume`/`load_resume` from `src/train/checkpoint.py`.
- [ ] Run a held-out val-perplexity eval end to end (`eval.val_loss.evaluate`).
- [ ] Run: `python scripts/smoke_test.py --data data/split`
- [ ] **Gate: resume is verifiably exact and eval runs. Stop here if not.**

## 5. M5 — Scale to ~100M  (issue #13)  *(the POC result)*
- [ ] Train `config/poc.yaml` (d_model 768 / 24 layers / seq 1024) on 2–5B tokens.
- [ ] Watch val perplexity + grad norm.
- [ ] Gate: a smoothly decreasing val-perplexity curve. (Benchmark scores NOT required.)

## Deferred (after the core POC) — only if wanted
- [ ] **#14 (M6):** OLMES / lm-eval adapter (`src/eval/olmes_adapter.py`) — mind loglikelihood off-by-one.
- [ ] **#15 (M7):** serving (`src/serve/sessions.py`) + rewind (`src/serve/rewind.py`).
- [ ] **#16 (M8):** CUDA backend (`src/model/cuda_backend.py`) + `backend_parity` (fp32) on a CUDA box.

---
After each milestone, tick its box in tracker issue #2 and push your work to
`claude/mamba-poc-plan-UPhGd` (or open a PR when ready).
