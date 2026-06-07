# Design docs

Why the Mamba POC is built the way it is. These files explain the **design
choices and their rationale** — the *what* and *why*. M1–M4 are implemented and
verified; the milestone tracking and remaining work (M5–M8) live in
[GitHub issue #2](https://github.com/travisgalloway/monica/issues/2). For the
project overview, see the root [`README.md`](../../README.md).

Every claim here is sourced from a docstring or config comment in the code, with a
`src/...` or `config/...` path so you can jump to the source of truth.

## Topics

1. [Architecture: the hardware seam](01-architecture-seam.md) — one abstraction
   (`ModelInterface`) isolates MLX/CUDA so everything above it stays portable.
2. [Model: the Mamba block + selective SSM](02-model-ssm.md) — block dataflow,
   diagonal-A init, load-bearing dt-bias, the chunked parallel scan, recurrence.
3. [Conformance: fp32 parity](03-conformance.md) — forward-vs-step and
   backend-vs-backend equivalence checks, and why they run in fp32.
4. [Data pipeline](04-data-pipeline.md) — uint16 packing, the OLMo tokenizer,
   disjoint val split, the mmap loader.
5. [Training](05-training.md) — backend-free loop, LR schedule, grad clipping /
   loss scaling, and the two-concern checkpoint split.
6. [Smoke gate & eval](06-smoke-gate-and-eval.md) — why resume-exactness is *the*
   gate, and val perplexity as the success metric.
7. [Configs & locked decisions](07-configs-and-decisions.md) — `toy.yaml` /
   `poc.yaml` in full, plus the precision benchmark and sizing math.

## Locked decisions at a glance

| Decision | Choice | Why | Source |
|---|---|---|---|
| Hardware isolation | one seam (`ModelInterface`) | clean MLX→CUDA migration | `src/model/interface.py` |
| Token storage | uint16 packing | compact; vocab must be < 65536 | `src/data/pack.py` |
| Tokenizer | `allenai/OLMo-7B-hf` (vocab 50280) | fits uint16; matches AI2 for comparison | `src/data/tokenize.py` |
| Embedding | tied (input = output) | ~38M of ~100M budget at POC scale | `config/poc.yaml` |
| dt-bias init | inverse-softplus, log-uniform | **load-bearing** — model can't learn recall without it | `src/model/mlx_backend.py` |
| Selective scan | chunked closed form (default 32) | the `exp(-A_cum)` term overflows fp32 unchunked | `src/model/mlx_backend.py` |
| Precision (poc) | fp16 + loss scaling | ~18% faster than bf16 on Metal (M1 benchmark) | `config/poc.yaml` |
| Precision (toy/smoke) | fp32 | exact, reproducible resume | `config/toy.yaml` |
| Conformance | compare in fp32, ~1e-4 rel | bf16 epsilon (~8e-3) too large to be meaningful | `src/conformance/` |
| Checkpoints | portable weights + separate resume bundle | weights port across backends; optimizer state doesn't need to | `src/train/checkpoint.py` |
| Success metric | held-out val perplexity (Tier-1) | a smoothly decreasing curve *is* the POC goal | `src/eval/val_loss.py` |
| OLMES / lm-eval | deferred (Tier-2) | its own milestone-sized task; not needed for the POC | `src/eval/olmes_adapter.py` |
