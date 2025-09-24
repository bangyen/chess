#!/usr/bin/env python3
"""
CLI for the Explainable Chess Engine

Usage: python -m src.chess_feature_audit.cli_explainable
"""

import argparse
import os
import shutil
import sys

from ..explainable_engine import ExplainableChessEngine


def find_stockfish() -> str:
    """Find Stockfish executable on the system."""
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Explainable Chess Engine - Learn chess with AI explanations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.chess_ai.cli.explainable
  python -m src.chess_ai.cli.explainable --engine /path/to/stockfish
  python -m src.chess_ai.cli.explainable --depth 20 --strength intermediate
        """,
    )

    parser.add_argument(
        "--engine",
        help="Path to Stockfish engine (auto-detected if not specified)",
    )

    parser.add_argument(
        "--depth", type=int, default=16, help="Search depth for analysis (default: 16)"
    )

    parser.add_argument(
        "--strength",
        choices=["beginner", "novice", "intermediate", "advanced", "expert"],
        default="beginner",
        help="Stockfish opponent strength (default: beginner)",
    )

    args = parser.parse_args()

    # Find Stockfish path
    if args.engine:
        stockfish_path = args.engine
    else:
        try:
            stockfish_path = find_stockfish()
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    try:
        with ExplainableChessEngine(
            stockfish_path, args.depth, args.strength
        ) as engine:
            engine.play_interactive_game()
    except KeyboardInterrupt:
        print("\n👋 Thanks for playing!")
    except FileNotFoundError:
        print(f"❌ Error: Could not find Stockfish at {stockfish_path}")
        print("Please install Stockfish or specify the correct path with --engine")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
