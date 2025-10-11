# Makefile for AIUnitTest
# ======================

# Variables
PYTHON := python3
PIP := pip
VENV_BIN := env/bin
PYTEST := $(VENV_BIN)/pytest
BLACK := $(VENV_BIN)/black
ISORT := $(VENV_BIN)/isort
FLAKE8 := $(VENV_BIN)/flake8
MYPY := $(VENV_BIN)/mypy
COVERAGE := $(VENV_BIN)/coverage
PRE_COMMIT := $(VENV_BIN)/pre-commit

# Directories
SRC_DIR := src
TEST_DIR := tests
VENV_DIR := env

# Configurations
.PHONY: help install install-dev test test-unit test-integration lint format type-check \
        coverage clean build publish pre-commit setup-dev run-app activate-env \
        test-fast test-all check-all docs reset-fake-project

# Default target
.DEFAULT_GOAL := help

# Check if virtual environment exists
check-venv:
	@if [ ! -f $(VENV_BIN)/python ]; then \
		echo "❌ Virtual environment not found at $(VENV_DIR)"; \
		echo "Run 'make create-venv' to create or activate existing virtual environment"; \
		exit 1; \
	fi

help: ## Show this help message
	@echo "Available commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and setup
install: ## Install the project in development mode
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements.txt

install-prod: ## Install production dependencies only
	$(PIP) install -e .

setup-dev: install-dev ## Configure complete development environment
	@if [ -f $(VENV_BIN)/pre-commit ]; then \
		$(PRE_COMMIT) install; \
	else \
		echo "⚠️  pre-commit not found in virtual environment, trying global version"; \
		pre-commit install || echo "❌ Error configuring pre-commit"; \
	fi
	@echo "✅ Development environment configured!"

# Virtual environment
create-venv: ## Create a new virtual environment
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "✅ Virtual environment created at $(VENV_DIR)"
	@echo "To activate: source $(VENV_DIR)/bin/activate"

activate-env: ## Show how to activate virtual environment
	@echo "To activate the virtual environment, run:"
	@echo "source $(VENV_DIR)/bin/activate"

# Tests
test: check-venv ## Run unit tests with coverage (default)
	$(PYTEST) $(TEST_DIR)/unit -v

test-unit: check-venv ## Run unit tests only
	$(PYTEST) $(TEST_DIR)/unit -v

test-unit-cov: check-venv ## Run unit tests with coverage
	$(PYTEST) $(TEST_DIR)/unit --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-integration: check-venv ## Run integration tests
	$(PYTEST) $(TEST_DIR) -v -m "integration"

test-fast: check-venv ## Run fast tests (exclude slow tests)
	$(PYTEST) $(TEST_DIR) -v -m "not slow"

test-all: check-venv ## Run all tests including slow ones
	$(PYTEST) $(TEST_DIR) -v --tb=short

test-watch: check-venv ## Run tests in watch mode (re-run when files change)
	$(PYTEST) $(TEST_DIR) -f

test-simple: ## Run tests using global pytest (fallback)
	@if command -v pytest >/dev/null 2>&1; then \
		pytest $(TEST_DIR) -v; \
	else \
		echo "❌ pytest not found. Run 'make install-dev' first"; \
		exit 1; \
	fi

# Code coverage
coverage: check-venv ## Generate coverage report
	$(PYTEST) $(TEST_DIR)/unit --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "📊 HTML coverage report generated at htmlcov/index.html"

coverage-xml: check-venv ## Generate XML coverage report
	$(PYTEST) $(TEST_DIR)/unit --cov=$(SRC_DIR) --cov-report=xml

# Code quality
lint: check-venv ## Run linting with flake8
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)

format: check-venv ## Format code with black and isort
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(ISORT) $(SRC_DIR) $(TEST_DIR)

format-check: check-venv ## Check formatting without modifying files
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)

type-check: check-venv ## Run type checking with mypy
	$(MYPY) $(SRC_DIR)

# Combined checks
check-all: format-check lint type-check test ## Run all quality checks
	@echo "✅ All checks passed!"

pre-commit-run: ## Run pre-commit on all files
	$(PRE_COMMIT) run --all-files

pre-commit-install: ## Install pre-commit hooks
	$(PRE_COMMIT) install

# Build and publishing
build: clean ## Build the package
	$(PYTHON) -m build

clean: ## Remove temporary and build files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

publish-test: build ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	twine upload dist/*

# Application execution
run-app: check-venv ## Run the main application
	$(VENV_BIN)/ai-unit-test --help

run-cli: check-venv ## Run the CLI with example arguments
	$(VENV_BIN)/ai-unit-test --help

# Project utilities
reset-fake-project: ## Reset fake project for testing
	$(PYTHON) reset_fake_project.py

docs: ## Generate documentation (placeholder)
	@echo "📚 Documentation will be implemented soon"

# Quick development commands
dev-install: create-venv activate-env install-dev setup-dev ## Complete setup for new developers
	@echo "🚀 Development environment ready!"
	@echo "Don't forget to activate the virtual environment: source $(VENV_DIR)/bin/activate"

quick-test: format lint test-fast ## Quick check before commit
	@echo "✅ Quick check completed!"

full-check: format lint type-check coverage ## Complete verification
	@echo "✅ Full check completed!"

# Debug and logs
show-logs: ## Show recent logs
	@echo "📋 Available logs:"
	@ls -la logs/ 2>/dev/null || echo "No logs found"

tail-logs: ## Follow logs in real time
	tail -f logs/*.log 2>/dev/null || echo "No logs to follow"

# Project information
info: ## Show project information
	@echo "📦 AIUnitTest - Project Information"
	@echo "=================================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Current branch: $(shell git branch --show-current 2>/dev/null || echo 'N/A')"
	@echo "Last commit: $(shell git log -1 --format=%cd --date=short 2>/dev/null || echo 'N/A')"

# Special commands for CI/CD
ci-install: ## Installation for CI/CD
	$(PIP) install -e ".[dev]"

ci-test: ## Tests for CI/CD
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=xml --cov-report=term

ci-check: ci-install format-check lint type-check ci-test ## Complete CI/CD verification
	@echo "✅ CI/CD verification completed!"
