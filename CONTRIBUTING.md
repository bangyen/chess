# Contributing to Chess AI

Welcome! Thank you for your interest in contributing to the Chess AI project. This project provides explainable chess analysis tools, now fully implemented in Rust.

## Developer Quickstart

1.  **Dependencies**: We use `rust` and `just`.
    *   Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
    *   Install `just`: `brew install just` (or see [justfile](https://github.com/casey/just))

2.  **Environment**: Build the project:
    ```bash
    just build
    ```

3.  **Stockfish**: Ensure Stockfish is installed and accessible.
    *   macOS: `brew install stockfish`
    *   Set `STOCKFISH_PATH` if it's not in your PATH.

4.  **Running Tests**:
    ```bash
    just test
    ```

## Development Workflow

-   **Formatting**: Use `just fmt`.
-   **Linting**: Use `just lint` (runs `cargo clippy`).
-   **Full Verification**: Run `just all`.

## Project Structure

-   `rust/`: Core engine, ML surrogate model, and web server.
    -   `src/engine/`: Stockfish interface and explainable engine logic.
    -   `src/features/`: Chess feature extraction for the surrogate model.
    -   `src/ml/`: Native Rust ML implementation using `linfa`.
    -   `src/web_server.rs`: Axum-based web dashboard.
-   `web/`: Front-end assets (templates and static files).

## Refactoring Guidelines

-   **Modularize**: Keep files small and focused. Avoid monolithic logic files.
-   **Pruning**: Remove unused code and over-abstractions.
-   **Tests**: Ensure tests are organized and follow the internal module structure.
