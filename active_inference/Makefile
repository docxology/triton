# Active Inference Framework - uv Makefile
# Convenience targets for common development tasks

.PHONY: help install install-dev install-all test test-verbose test-coverage examples validate lint format type-check clean setup

help: ## Show this help message
	@echo "Active Inference Framework - uv Makefile"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install core dependencies
	uv sync

install-dev: ## Install with development dependencies
	uv sync --dev

install-all: ## Install everything including Triton
	uv sync --dev && uv add triton

test: ## Run tests
	uv run pytest tests/

test-verbose: ## Run tests with verbose output
	uv run pytest tests/ -v

test-coverage: ## Run tests with coverage
	uv run pytest --cov=src tests/

examples: ## Run all examples
	uv run python run_all_examples.py

validate: ## Run Triton validation
	uv run python validate_triton_usage.py

lint: ## Run all linters
	uv run black . && uv run isort . && uv run flake8 .

format: ## Format code
	uv run black . && uv run isort .

type-check: ## Run type checking
	uv run mypy src/

clean: ## Clean up generated files
	rm -rf .venv/
	rm -rf __pycache__/
	rm -rf *.pyc
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf outputs/
	rm -f uv.lock

setup: ## Run the uv setup script
	./setup_uv.sh

# Default target
.DEFAULT_GOAL := help
