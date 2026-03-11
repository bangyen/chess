RUN := "cargo run --bin chess-ai --"

# format code
fmt:
    cargo fmt

# lint code
lint:
    cargo clippy

# run tests
test *ARGS:
    cargo test {{ARGS}}

# run web dashboard (Rust-native)
web:
    @echo "Starting chess AI web dashboard (Rust Axum)..."
    {{RUN}} server

# build the project
build:
    cargo build --release

# train the surrogate model (Rust-native)
train *ARGS:
    {{RUN}} train {{ARGS}}

# run explainability audit (Rust-native)
audit *ARGS:
    {{RUN}} audit {{ARGS}}

# play an interactive game (Rust-native)
play:
    {{RUN}} play

# download syzygy tablebases (3-5 piece)
syzygy-download dest="~/syzygy":
    {{RUN}} syzygy download --dest {{dest}}

# verify syzygy tablebase integration
syzygy-verify path="~/syzygy":
    {{RUN}} syzygy verify --syzygy-path {{path}}

# run all checks (fmt, lint, test)
all: fmt lint test
    @echo "All checks completed!"
