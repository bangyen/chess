.PHONY: init fmt lint type test web all

init: ## install tooling
	python -m pip install -U pip
	pip install -e ".[dev]"
	pre-commit install

fmt:  ## format code
	black .

lint: ## lint code
	ruff check .

type: ## type-check
	mypy .

test: ## run tests
	python -m pytest

web:  ## run web dashboard
	python -m chess_ai.web.app

all: fmt lint type test
