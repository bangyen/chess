"""Utilities for finding Stockfish on the system."""

import os
import shutil


def find_stockfish() -> str:
    """Find Stockfish executable on the system.

    Checks common paths and the system PATH.

    Returns:
        Path to Stockfish binary

    Raises:
        FileNotFoundError: If Stockfish cannot be found
    """
    # Common paths for Stockfish
    common_paths = [
        "/opt/homebrew/bin/stockfish",  # macOS Homebrew
        "/usr/local/bin/stockfish",  # macOS/Linux local install
        "/usr/bin/stockfish",  # Linux package manager
        "/usr/games/stockfish",  # Ubuntu/Debian
        "stockfish",  # In PATH
    ]

    # Check each path
    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    # Try to find in PATH
    stockfish_path = shutil.which("stockfish")
    if stockfish_path:
        return stockfish_path

    raise FileNotFoundError(
        "Stockfish not found! Please install Stockfish:\n"
        "  • Ubuntu/Debian: sudo apt install stockfish\n"
        "  • macOS: brew install stockfish\n"
        "  • Windows: Download from https://stockfishchess.org/\n"
        "  • Google Colab: !apt install stockfish\n"
        "  • Or add Stockfish to your PATH"
    )
