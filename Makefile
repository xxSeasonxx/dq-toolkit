.PHONY: install test lint format demo clean

install:  ## Editable install with dev dependencies
	pip install -e ".[dev]"

test:  ## Run the test suite with coverage
	pytest

lint:  ## Lint and check formatting
	ruff check src tests
	ruff format --check src tests

format:  ## Auto-format and apply safe fixes
	ruff format src tests
	ruff check --fix src tests

demo:  ## End-to-end: detect anomalies, resolve entities, print metrics
	python -m dqkit.demo

clean:  ## Remove caches and local Spark artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov spark-warehouse metastore_db derby.log
	find . -type d -name __pycache__ -exec rm -rf {} +
