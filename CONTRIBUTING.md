# Contributing to Chess AI

Welcome! Thank you for your interest in contributing to the Chess AI project. This project aims to provide explainable chess analysis tools.

## Developer Quickstart

1.  **Dependencies**: We use `uv` and `just`.
    *   Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    *   Install `just`: `brew install just` (or see [justfile](https://github.com/casey/just))

2.  **Environment**: Initialize the environment:
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
-   **Linting**: Use `just lint`.
-   **Type Checking**: Use `just type`.
-   **Full Verification**: Run `just all`.

## Project Structure

-   `src/chess_ai/`: Core logic.
    -   `audit/`: Feature set audit and worker processes.
    -   `engine/`: Stockfish interface, Syzygy, and recommendations.
    -   `features/`: Chess feature extraction.
-   `tests/`: Unit and integration tests.
-   `rust/`: Performance-critical Rust extensions.

## Refactoring Guidelines

-   **Modularize**: Keep files small and focused. Avoid monolithic logic files.
-   **Pruning**: Remove unused code and over-abstractions.
-   **Tests**: Ensure tests are organized and follow the `src` structure.
