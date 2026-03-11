# Task runner for the chess AI project

CARGO := "cargo"
CARGO_OPTS := "--manifest-path rust/Cargo.toml"
CHESS_AI := "{{CARGO}} run {{CARGO_OPTS}} --bin chess-ai --"

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
    {{CARGO}} build {{CARGO_OPTS}}

# format code
fmt:
    {{CARGO}} fmt {{CARGO_OPTS}}
    {{PYTHON}} -m black .

# lint code
lint:
    {{CARGO}} clippy {{CARGO_OPTS}}
    {{PYTHON}} -m ruff check .

# type-check (Python scripts only)
type:
    {{PYTHON}} -m mypy .

# run tests
test *ARGS:
    {{CARGO}} test {{CARGO_OPTS}}
    {{PYTHON}} -m pytest {{ARGS}}

# run web dashboard (Rust-native)
web:
    echo "Starting chess AI web dashboard (Rust Axum)..."
    {{CHESS_AI}} server

# build the project
build:
    {{CARGO}} build {{CARGO_OPTS}} --release
    # Also build maturin for Python interop
    {{PYTHON}} -m maturin develop --release

# train the surrogate model (Rust-native)
train *ARGS:
    {{CHESS_AI}} train {{ARGS}}

# run explainability audit (Rust-native)
audit *ARGS:
    {{CHESS_AI}} audit {{ARGS}}

# play an interactive game (Rust-native)
play:
    {{CHESS_AI}} play

# compare audit metrics between two commits
benchmark OLD NEW *ARGS:
    {{PYTHON}} scripts/benchmark_commits.py {{OLD}} {{NEW}} {{ARGS}}

# profile feature-extraction probes
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
all: fmt lint type test
    echo "All checks completed!"
