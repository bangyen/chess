OPTS := "--manifest-path rust/Cargo.toml"
RUN := "cargo run " + OPTS + " --bin chess-ai --"

# format code
fmt:
    cargo fmt {{OPTS}}

# lint code
lint:
    cargo clippy {{OPTS}}

# run tests
test *ARGS:
    cargo test {{OPTS}} {{ARGS}}

# run web dashboard (Rust-native)
web:
    @echo "Starting chess AI web dashboard (Rust Axum)..."
    {{RUN}} server

# build the project
build:
    cargo build {{OPTS}} --release

# train the surrogate model (Rust-native)
train *ARGS:
    {{RUN}} train {{ARGS}}

# run explainability audit (Rust-native)
audit *ARGS:
    {{RUN}} audit {{ARGS}}

# play an interactive game (Rust-native)
play:
    {{RUN}} play

# run all checks (fmt, lint, test)
all: fmt lint test
    echo "All checks completed!"
