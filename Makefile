.PHONY: help install kb train test lint format run docker deploy clean

PYTHON ?= python

help:
	@echo "Argus dev targets:"
	@echo "  install   — Install package + dev dependencies (editable)"
	@echo "  kb        — Build the reference knowledge base (SQLite)"
	@echo "  train     — Train the DDI severity model (synthetic data)"
	@echo "  test      — Run pytest"
	@echo "  lint      — Ruff check + mypy (best-effort)"
	@echo "  format    — Ruff format"
	@echo "  run       — Start the MCP server (dev mode)"
	@echo "  docker    — Build the Docker image"
	@echo "  deploy    — fly deploy (requires flyctl)"
	@echo "  clean     — Remove caches, build artifacts, SQLite KB"

install:
	$(PYTHON) -m pip install -e ".[dev]"

kb:
	$(PYTHON) -m argus.reference.build_kb

train:
	$(PYTHON) scripts/train_ddi_model.py --n-samples 20000

test:
	pytest -v

lint:
	ruff check argus tests scripts
	-mypy argus

format:
	ruff check --fix argus tests scripts
	ruff format argus tests scripts

run:
	$(PYTHON) -m argus.server

docker:
	docker build -t argus-mcp:latest .

deploy:
	fly deploy

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	rm -rf argus/reference/reference.sqlite argus/ml/artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
