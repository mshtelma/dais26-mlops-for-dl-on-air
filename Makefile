.PHONY: install lint test test-integration bundle-validate bundle-deploy-dev bundle-run-train bundle-run-embeddings bundle-run-drift warmup pin-cache build help

install:                           ## Install project in editable mode with dev deps
	uv pip install -e ".[dev]"

lint:                              ## Run ruff linter
	ruff check src/ tests/ notebooks/ scripts/

test:                              ## Run unit tests
	pytest tests/unit/ -q --tb=short

test-integration:                  ## Run integration tests (requires Databricks workspace)
	pytest tests/integration/ -q --tb=short -m integration

build:                             ## Build the wheel via uv
	uv build

bundle-validate:                   ## Validate DAB configuration
	databricks bundle validate -t dev

bundle-deploy-dev:                 ## Deploy UC + jobs (NOT endpoints)
	databricks bundle deploy -t dev

bundle-run-train:                  ## Run DAB quickstart: train + register + confirm @challenger
	databricks bundle run train_detector -t dev

bundle-run-embeddings:             ## Run embedding precompute (champion job task, prod)
	databricks bundle run deploy_champion_job -t prod --only precompute_embeddings

bundle-run-drift:                  ## Run drift monitor lane on prod
	databricks bundle run drift_monitor -t prod

warmup:                            ## Pre-warm serving endpoints
	python scripts/warmup_endpoints.py

pin-cache:                         ## Cache model weights (C-RADIOv4 + DINOv2 fallback) to UC Volume
	python scripts/pin_model_cache.py

discover:                          ## Run AIR runtime discovery (Day 1)
	python scripts/discover_air_runtime.py

help:                              ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'
