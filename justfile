# Task runner for the chess AI project

# Auto-detect uv - falls back to plain python if not available
PYTHON := `command -v uv >/dev/null 2>&1 && echo "uv run python" || echo "python"`

# install tooling
init:
    #!/usr/bin/env bash
    if command -v uv >/dev/null 2>&1; then
        echo "Using uv..."
        uv sync --extra dev
        uv run pre-commit install
    else
        echo "Using pip..."
        python -m pip install -U pip
        pip install -e ".[dev]"
        pre-commit install
    fi

# format code
fmt:
    {{PYTHON}} -m black .

# lint code
lint:
    {{PYTHON}} -m ruff check .

# type-check
type:
    {{PYTHON}} -m mypy .

# run tests
test:
    {{PYTHON}} -m pytest

# run web dashboard
web:
    echo "Starting chess AI web dashboard..."
    {{PYTHON}} -m chess_ai.web.app

# build the Rust extension module via maturin (dev mode)
build:
    {{PYTHON}} -m maturin develop --release

# run all checks (fmt, lint, type, test)
all: fmt lint type test
    echo "All checks completed!"

