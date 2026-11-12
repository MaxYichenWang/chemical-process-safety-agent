.PHONY: help format format-all check-fmt check-lint lint test clean install check-data verify web

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

format: ## Format code with black and isort
	black src/ --line-length 88
	isort src/ --profile black --line-length 88

format-all: format ## Format and lint code (complete workflow)
	ruff check --fix src/

check-fmt: ## Check code formatting without modifying files
	black src/ --line-length 88 --check
	isort src/ --profile black --line-length 88 --check

check-lint: ## Check linting without modifying files
	ruff check src/

lint: check-lint ## Alias for check-lint

test: ## Run tests with pytest
	pytest

install: ## Install the package and development tools
	python -m pip install -e '.[dev]'

clean: ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

check-data: ## Check analytical data file availability
	python -m src.agentic_rag.analytical_tools --check

web: ## Run the web UI server (localhost by default; LAN access requires a token)
	python -m src.web

verify: ## Run the test suite and lint checks
	pytest tests/ -v
	ruff check src/
