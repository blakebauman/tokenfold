# Model weights live on the external SSD, never the internal disk (it is nearly full).
# Override with `make baseline HF_HUB_CACHE=...`.
export HF_HUB_CACHE ?= /Volumes/External SSD 1TB/huggingface/hub

.PHONY: help hf-cache install install-llama lint fmt type test test-cov check dev dev-vllm dev-llama calibrate-mock build-eval baseline recalibrate

help:
	@echo "tokenfold dev targets:"
	@echo "  install          uv sync --dev"
	@echo "  install-llama    + llama-cpp-python built with Metal (local model backend)"
	@echo "  lint/fmt/type/test/test-cov/check"
	@echo "  dev              run the server on the mock backend (no GPU)"
	@echo "  dev-vllm         run against VLLM_BASE_URL with calibration.json"
	@echo "  dev-llama        run against a local GGUF model (TOKENFOLD_LLAMA_MODEL)"
	@echo "  calibrate-mock   generate a mock dataset and run the calibration harness end to end"
	@echo "  build-eval       sample the public task suite into data/eval/ (cached in data/raw/)"
	@echo "  baseline         score data/eval/all.jsonl with the llama backend -> data/reports/"
	@echo "  recalibrate      refit from the saved baseline logits, no model needed"
	@echo "  Single test: uv run pytest tests/unit/test_answers.py::test_noul_is_p_yes"

hf-cache:
	@test -d "$$(dirname "$(HF_HUB_CACHE)")" || { echo "HF_HUB_CACHE parent missing: $(HF_HUB_CACHE) (is the SSD mounted?)"; exit 1; }
	@mkdir -p "$(HF_HUB_CACHE)"

install:
	uv sync --dev

install-llama:
	CMAKE_ARGS="-DGGML_METAL=on" uv sync --dev --extra llama

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

type:
	uv run ty check src

test:
	uv run pytest -q

test-cov:
	uv run pytest -q --cov --cov-report=term:skip-covered

check: lint type test-cov
	uv run ruff format --check .

dev:
	uv run tokenfold-server --backend mock --port $${TOKENFOLD_PORT:-8080}

dev-vllm:
	uv run tokenfold-server --backend vllm --calibration calibration.json --port $${TOKENFOLD_PORT:-8080}

dev-llama: hf-cache
	uv run tokenfold-server --backend llama --calibration data/calibration.llama.json --port $${TOKENFOLD_PORT:-8080}

calibrate-mock:
	mkdir -p data
	uv run python examples/make_mock_dataset.py --out data/mock.jsonl
	uv run tokenfold-evaluate --data data/mock.jsonl --backend mock --out data/calibration.mock.json

build-eval:
	uv run tokenfold-build-eval --n $${N:-100}

baseline: hf-cache data/eval/all.jsonl
	uv run tokenfold-evaluate --data data/eval/all.jsonl --backend llama \
		--out data/calibration.llama.json --report data/reports/baseline-llama.json \
		--save-logits data/logits/baseline-llama.json

# refit + re-report from the saved baseline logits (seconds, no model)
recalibrate:
	uv run tokenfold-evaluate --data data/eval/all.jsonl --from-logits data/logits/baseline-llama.json \
		--out data/calibration.llama.json --report data/reports/baseline-llama.json

data/eval/all.jsonl:
	$(MAKE) build-eval
