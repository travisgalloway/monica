PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
TSC_PATH ?= eval_sets/ts_error_injection/node_modules/.bin
CONFIG ?= config/toy.yaml
DATA ?= data/split
OUT ?= runs/demo_run
STEPS ?= 20
BATCH_SIZE ?= 2
MODEL ?= mlx-community/mamba2-130m
LIMIT ?= 5
PROMPT ?= function add(a: number, b: number): number {
MAX_TOKENS ?= 20

POC_CONFIG ?= config/code-small-dense.yaml
POC_OUT ?= runs/tier1-poc
POC_STEPS ?= 50
POC_BATCH_SIZE ?= 2
POC_BASE_LR ?= 3e-4

.PHONY: help smoke train train-poc quantize-poc generate generate-poc generate-q4 build-swift bench-lsp eval-ar eval-diagnostic eval-cross-path eval-code eval e2e e2e-poc test-ssi clean

help:
	@echo "Available commands:"
	@echo "  make smoke            Run resume-exactness gate & val-perplexity smoke test"
	@echo "  make train            Train model on MLX and save weights.safetensors"
	@echo "  make train-poc        Train Tier 1 Mac POC model (code-small-dense.yaml)"
	@echo "  make quantize-poc     Quantize Tier 1 POC weights to Mixed W4+Head8"
	@echo "  make generate         Generate text from trained demo weights"
	@echo "  make generate-poc     Generate text from Tier 1 POC native weights"
	@echo "  make generate-q4      Generate text from Tier 1 POC quantized weights"
	@echo "  make build-swift      Build native Swift tokenizer and engine tools"
	@echo "  make bench-lsp        Benchmark native Swift LSP service"
	@echo "  make eval-ar          Run AR harness ablation across 4 cells (Issue #201)"
	@echo "  make eval-diagnostic  Run diagnostic supervision evaluation (Issue #227)"
	@echo "  make eval-cross-path  Run cross-path efficiency comparison (Issue #204)"
	@echo "  make eval-code        Run code evaluation suite (recall, needle, FIM)"
	@echo "  make eval             Run all evaluation pipelines"
	@echo "  make e2e              Run demo flow: smoke -> train -> generate -> eval"
	@echo "  make e2e-poc          Run full Tier 1 Mac POC flow: smoke -> train -> quantize -> gen -> evals"
	@echo "  make test-ssi         Run SSI test suite"
	@echo "  make clean            Clean up output run directories"

smoke:
	$(PYTHON) scripts/smoke_test.py --config $(CONFIG) --data $(DATA) --steps 10 --batch-size 2 --out runs/smoke_test

train:
	$(PYTHON) scripts/train.py --config $(CONFIG) --data $(DATA) --out $(OUT) --total-steps $(STEPS) --batch-size $(BATCH_SIZE) --log-every 5 --eval-every 10 --ckpt-every 10

train-poc:
	$(PYTHON) scripts/train.py --config $(POC_CONFIG) --data $(DATA) --out $(POC_OUT) --total-steps $(POC_STEPS) --batch-size $(POC_BATCH_SIZE) --base-lr $(POC_BASE_LR) --log-every 5 --eval-every 10 --ckpt-every 10

quantize-poc:
	$(PYTHON) scripts/quantize_checkpoint.py --weights $(POC_OUT)/weights.safetensors --out $(POC_OUT)/weights.q4.safetensors --bits 4 --head-bits 8

generate:
	$(PYTHON) scripts/generate.py --config $(CONFIG) --weights $(OUT)/weights.safetensors --prompt "$(PROMPT)" --byte-fallback --max-new-tokens $(MAX_TOKENS)

generate-poc:
	$(PYTHON) scripts/generate.py --config $(POC_CONFIG) --weights $(POC_OUT)/weights.safetensors --prompt "$(PROMPT)" --byte-fallback --max-new-tokens $(MAX_TOKENS)

generate-q4:
	$(PYTHON) scripts/generate.py --config $(POC_CONFIG) --weights $(POC_OUT)/weights.q4.safetensors --prompt "$(PROMPT)" --byte-fallback --max-new-tokens $(MAX_TOKENS)

build-swift:
	cd swift && swift build -c release --build-system native

eval-ar:
	PATH="$(TSC_PATH):$$PATH" $(PYTHON) scripts/eval_ar_ablation.py --model $(MODEL) --limit $(LIMIT)

eval-diagnostic:
	PATH="$(TSC_PATH):$$PATH" $(PYTHON) scripts/eval_diagnostic_supervision.py --limit $(LIMIT)

eval-cross-path:
	$(PYTHON) scripts/eval_cross_path.py

eval-code:
	$(PYTHON) scripts/eval_code_suite.py --config $(POC_CONFIG) --checkpoint $(POC_OUT)/weights.safetensors --backend mlx --byte-tokenizer --suites recall,needle,fim,external --limit $(LIMIT) --output results/poc_code_suite.json --transcript results/poc_code_suite.jsonl

bench-lsp:
	swift/.build/release/monica-lsp --bench --eval-set-dir eval_sets/ts_error_injection

eval: eval-diagnostic eval-cross-path eval-ar eval-code

e2e: smoke train generate eval-diagnostic eval-cross-path

e2e-poc: smoke train-poc quantize-poc generate-poc generate-q4 eval-code eval-ar eval-cross-path

test-ssi:
	$(PYTEST) tests/test_ar_ablation.py tests/test_diagnostic_supervision.py tests/test_cross_path.py

clean:
	rm -rf runs/demo_run runs/smoke_test /tmp/smoke_out
