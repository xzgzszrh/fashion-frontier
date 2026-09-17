.PHONY: help install test lint clean cpu-route teacher students export quantize \
        bench results docker web-data demo

PYTHON ?= python
ARTIFACTS := artifacts

help: ## Show this help
	@# ".*?" is a PCRE construct that BSD grep rejects outright, so ^...:.*## .*$ it is.
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Editable install (CPU-only route)
	$(PYTHON) -m pip install -e .

install-quant: ## Editable install including Brevitas/QONNX
	$(PYTHON) -m pip install -e ".[quant]"

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Ruff lint
	$(PYTHON) -m ruff check src tests

## ---------------- pipeline ----------------

teacher: ## Train baseline + EfficientNet-B0 teacher, export soft targets
	$(PYTHON) -m fashionfrontier train --config configs/teacher/01_baseline_cnn.yaml
	$(PYTHON) -m fashionfrontier train --config configs/teacher/02_efficientnet_b0_transfer.yaml
	$(PYTHON) -m fashionfrontier export-soft-targets \
		--config configs/teacher/04_export_soft_targets.yaml \
		--checkpoint outputs/efficientnet_b0_transfer_95/best_model.pt \
		--out $(ARTIFACTS)/teacher_targets_full_train.npz \
		--split full_train --temperature 1.0

students: ## Train the four CPU-only delivery tiers
	$(PYTHON) -m fashionfrontier train --config configs/cpu/00_tinyplus_base.yaml
	$(PYTHON) -m fashionfrontier train --config configs/cpu/01_tinyplus_kd.yaml
	$(PYTHON) -m fashionfrontier train --config configs/cpu/02_tinyplus_kd_fulltrain.yaml
	$(PYTHON) -m fashionfrontier train --config configs/cpu/03_tinyfast_xs.yaml
	$(PYTHON) -m fashionfrontier train --config configs/cpu/04_tinyfast_xxs.yaml
	$(PYTHON) -m fashionfrontier train --config configs/cpu/05_tinyfast_xxxs.yaml

export: ## Export the trained students to ONNX
	$(PYTHON) -m fashionfrontier export-onnx --model tiny_fashion_cnn --variant tinyfast_xs \
		--checkpoint outputs/tinyfast_xs/best_model.pt \
		--out $(ARTIFACTS)/tinyfast_xs_fp32.onnx
	$(PYTHON) -m fashionfrontier export-onnx --model tiny_fashion_cnn --variant tinyplus \
		--checkpoint outputs/tinyplus_kd_fulltrain/best_model.pt \
		--out $(ARTIFACTS)/tinyplus_kd_fulltrain_fp32.onnx
	$(PYTHON) -m fashionfrontier export-onnx --model tiny_fashion_cnn --variant tinyfast_xxs \
		--checkpoint outputs/tinyfast_xxs/best_model.pt \
		--out $(ARTIFACTS)/tinyfast_xxs_fp32.onnx
	$(PYTHON) -m fashionfrontier export-onnx --model tiny_fashion_cnn --variant tinyfast_xxxs \
		--checkpoint outputs/tinyfast_xxxs/best_model.pt \
		--out $(ARTIFACTS)/tinyfast_xxxs_fp32.onnx

quantize: ## Static INT8 quantisation of the exported models
	for m in tinyplus_kd_fulltrain tinyfast_xs tinyfast_xxs tinyfast_xxxs; do \
		$(PYTHON) -m fashionfrontier quantize --model $(ARTIFACTS)/$${m}_fp32.onnx \
			--out $(ARTIFACTS)/$${m}_int8.onnx --calib-size 256 || exit 1; \
	done

cpu-route: teacher students export quantize ## Full CPU-only delivery route

test-set: ## Dump the Fashion-MNIST test set as NPZ for the board
	$(PYTHON) -m fashionfrontier export-test-set --out $(ARTIFACTS)/fashion_mnist_test.npz

bench: ## Benchmark every ONNX model in artifacts/ on this machine
	$(PYTHON) -m fashionfrontier bench --model-dir $(ARTIFACTS) \
		--test-set $(ARTIFACTS)/fashion_mnist_test.npz

results: ## Print the authoritative results table
	$(PYTHON) -m fashionfrontier results --only-deployed

models: ## List TinyFashionCNN variants and parameter counts
	$(PYTHON) -m fashionfrontier models

## ---------------- web demo ----------------

web-data: ## Regenerate web/data/results.json from benchmarks/results.csv
	$(PYTHON) scripts/build_web_data.py

demo: ## Serve the repo root on :8000 for the in-browser demo
	@echo "open http://localhost:8000/web/"
	$(PYTHON) -m http.server 8000

docker: ## Build the container image
	docker build -t fashion-frontier -f docker/Dockerfile .

clean: ## Remove build artefacts (keeps configs, docs and results)
	rm -rf outputs $(ARTIFACTS) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
