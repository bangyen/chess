# Task runner for the project

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
    echo "Installing JavaScript dependencies..."
    npm install

# format code
fmt:
    {{PYTHON}} -m black .

# lint code (Python)
lint_python:
    {{PYTHON}} -m ruff check .

# lint code (JavaScript)
lint_js:
    npm run lint:js

# lint styles (CSS)
lint_css:
    npm run lint:css

# lint all code
lint: lint_python lint_js lint_css
    @echo "All lint checks completed!"

# type-check
type:
    {{PYTHON}} -m mypy .

# run tests
test:
    {{PYTHON}} -m pytest

# run all checks (fmt, lint, type, test)
all: fmt lint type test
    @echo "All checks completed!"

# start Flask dashboard
dashboard:
    @echo "Starting dashboard..."
    @echo "Dashboard will be available at http://localhost:5050"
    {{PYTHON}} dashboard/main.py

