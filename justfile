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
test *ARGS:
    {{PYTHON}} -m pytest {{ARGS}}

# run web dashboard
web:
    echo "Starting chess AI web dashboard..."
    {{PYTHON}} -m chess_ai.web.app

# build the Rust extension module via maturin (dev mode)
build:
    {{PYTHON}} -m maturin develop --release

# run explainability audit against Stockfish (default: 400 positions, depth 16)
audit *ARGS:
    {{PYTHON}} -m chess_ai.cli.audit --engine "$(command -v stockfish)" --baseline_features {{ARGS}}

# compare audit metrics between two commits (pass --positions/--depth/--threads to override)
benchmark OLD NEW *ARGS:
    {{PYTHON}} scripts/benchmark_commits.py {{OLD}} {{NEW}} {{ARGS}}

# profile Rust and Python feature-extraction probes
profile:
    {{PYTHON}} scripts/profile_probes_rust.py
    {{PYTHON}} scripts/benchmark_python_probes.py

# download Syzygy 3/4/5-piece tablebase files
fetch-syzygy:
    {{PYTHON}} scripts/fetch_syzygy.py

# verify Syzygy tablebase integration on known endgames
verify-syzygy:
    {{PYTHON}} scripts/verify_syzygy_endgames.py

# run all checks (fmt, lint, type, test)
all: fmt lint type
    {{PYTHON}} -m pytest
    echo "All checks completed!"

