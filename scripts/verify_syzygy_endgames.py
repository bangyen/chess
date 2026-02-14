"""Verify Syzygy tablebase integration on known endgame positions.

Consolidates endgame checks (3-5 pieces) to confirm that the
ExplainableChessEngine properly queries Syzygy tablebases and
surfaces tablebase-backed reasons in move explanations.

Env vars:
    SYZYGY_PATH    - directory containing .rtbw/.rtbz files (default ~/syzygy)
    STOCKFISH_PATH - path to the Stockfish binary
"""

import os
import sys

import chess

from chess_ai.explainable_engine import ExplainableChessEngine

# Each tuple: (label, FEN, expected outcome hint)
ENDGAMES: list[tuple[str, str]] = [
    # 3-piece
    ("KQK", "4k3/8/8/8/8/8/8/4K1Q1 w - - 0 1"),
    ("KRK", "4k3/8/8/8/8/8/8/4K1R1 w - - 0 1"),
    # 4-piece
    ("KPK", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),
    ("KBNK", "8/8/8/8/8/8/2B1N3/4K2k w - - 0 1"),
    ("KBBK", "4k3/8/8/8/8/8/8/2K1BB2 w - - 0 1"),
    # 5-piece
    ("KBPK", "4k3/8/8/8/8/4P3/4K1B1/8 w - - 0 1"),
]


def verify_endgames() -> None:
    """Run Syzygy verification across all endgame positions.

    Prints per-position diagnostics so a developer can quickly
    confirm tablebase data is flowing into the explanation layer.
    """
    syzygy_path = os.environ.get("SYZYGY_PATH", "~/syzygy")
    stockfish_path = os.environ.get("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")

    print(f"Verifying Syzygy integration  (path={syzygy_path})\n")

    failures = 0

    try:
        with ExplainableChessEngine(
            stockfish_path, syzygy_path=syzygy_path, depth=12
        ) as engine:
            for name, fen in ENDGAMES:
                print(f"--- {name} ---")
                board = chess.Board(fen)
                engine.board = board

                analysis = engine.analyze_position()
                stockfish_score = analysis.get("stockfish_score", "N/A")
                syzygy_data = analysis.get("syzygy", {})

                print(f"  Stockfish score : {stockfish_score}")
                print(f"  Syzygy data     : {syzygy_data}")

                best_move = engine.get_best_move()
                if not best_move:
                    print("  FAIL - no best move found")
                    failures += 1
                    continue

                explanation = engine.explain_move(best_move)
                print(f"  Best move       : {board.san(best_move)}")
                print(f"  Explanation     : {explanation.overall_explanation}")

                found_syzygy_reason = False
                for reason_name, score, text in explanation.reasons:
                    if "syzygy" in reason_name:
                        print(f"  OK   Syzygy reason: {text} (score={score})")
                        found_syzygy_reason = True

                if not found_syzygy_reason:
                    print("  WARN - no Syzygy reason in explanation")
                    failures += 1

                print()
    except Exception as e:
        print(f"Error during verification: {e}")
        sys.exit(1)

    if failures:
        print(f"\n{failures} position(s) had issues - see output above.")
        sys.exit(1)

    print("All positions verified successfully.")


if __name__ == "__main__":
    verify_endgames()
