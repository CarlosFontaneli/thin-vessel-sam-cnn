# Reproduction targets of the repository (see README.md and docs/reproduction.md).
PYTHON ?= python

.PHONY: install test lint figures paper clean

install:  ## install the package with the development tools
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## unit tests (CPU, about ten seconds)
	$(PYTHON) -m pytest

lint:  ## ruff lint and format check (the vendored third_party/ code is excluded)
	ruff check . && ruff format --check .

figures:  ## statistics, every figure and table of the paper from results/, then the byte check
	PYTHON="$(PYTHON)" paper/figures_tables/run_all.sh

paper:  ## compile the paper to outputs/paper/main.pdf (needs TeX Live with latexmk)
	cd paper/manuscript && latexmk -pdf -interaction=nonstopmode -outdir=../../outputs/paper main.tex

clean:  ## remove the LaTeX build files and the generated previews (keeps results/ and paper/manuscript/figures/)
	rm -rf outputs/paper outputs/figures outputs/statistics
