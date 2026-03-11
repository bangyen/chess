# Task runner for the chess AI project (Rust Native)

CARGO := "cargo"
CARGO_OPTS := "--manifest-path rust/Cargo.toml"
CHESS_AI := "{{CARGO}} run {{CARGO_OPTS}} --bin chess-ai --"

# format code
fmt:
    {{CARGO}} fmt {{CARGO_OPTS}}

# lint code
lint:
    {{CARGO}} clippy {{CARGO_OPTS}}

# run tests
test *ARGS:
    {{CARGO}} test {{CARGO_OPTS}}

# run web dashboard (Rust-native)
web:
    echo "Starting chess AI web dashboard (Rust Axum)..."
    {{CHESS_AI}} server

# build the project
build:
    {{CARGO}} build {{CARGO_OPTS}} --release

# train the surrogate model (Rust-native)
train *ARGS:
    {{CHESS_AI}} train {{ARGS}}

# run explainability audit (Rust-native)
audit *ARGS:
    {{CHESS_AI}} audit {{ARGS}}

# play an interactive game (Rust-native)
play:
    {{CHESS_AI}} play

# run all checks (fmt, lint, test)
all: fmt lint test
    echo "All checks completed!"
