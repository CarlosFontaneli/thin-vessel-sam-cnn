# Reproduction targets of the repository (see README.md and docs/reproduction.md).
PYTHON ?= python

.PHONY: install test lint clean

install:  ## install the package with the development tools
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## unit tests (CPU, about ten seconds)
	$(PYTHON) -m pytest

lint:  ## ruff lint and format check (the vendored third_party/ code is excluded)
	ruff check . && ruff format --check .

clean:  ## remove the tool caches
	rm -rf .pytest_cache .ruff_cache
