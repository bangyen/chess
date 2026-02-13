import os

import chess
import chess.engine

from chess_ai.explainable_engine import ExplainableChessEngine


def verify_endgames():
    syzygy_path = os.environ.get("SYZYGY_PATH", "~/syzygy")
    stockfish_path = os.environ.get("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")

    # Endgame positions (3-5 pieces) - Must be legal positions!
    endgames = [
        (
            "KpK",
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        ),  # Win/Draw depending on king position
        ("KQK", "4k3/8/8/8/8/8/8/4K1Q1 w - - 0 1"),  # Win
        ("KRK", "4k3/8/8/8/8/8/8/4K1R1 w - - 0 1"),  # Win
        ("KBBK", "4k3/8/8/8/8/8/8/2K1BB2 w - - 0 1"),  # Win
        ("KBPK", "4k3/8/8/8/8/4P3/4K1B1/8 w - - 0 1"),  # Win
    ]

    print(f"Verifying Syzygy impact with path: {syzygy_path}")

    try:
        with ExplainableChessEngine(
            stockfish_path, syzygy_path=syzygy_path, depth=12
        ) as engine:
            for name, fen in endgames:
                print(f"\n--- Testing {name} ---")
                board = chess.Board(fen)
                engine.board = board

                # Analyze
                analysis = engine.analyze_position()
                stockfish_score = analysis.get("stockfish_score", "N/A")
                syzygy_data = analysis.get("syzygy", {})

                print(f"Stockfish Score: {stockfish_score}")
                print(f"Syzygy Data: {syzygy_data}")

                # Get best move and explanation
                best_move = engine.get_best_move()
                if best_move:
                    explanation = engine.explain_move(best_move)
                    print(f"Best Move: {board.san(best_move)}")
                    print(f"Explanation: {explanation.overall_explanation}")

                    found_syzygy_reason = False
                    for reason_name, score, text in explanation.reasons:
                        if "syzygy" in reason_name:
                            print(f"✅ Found Syzygy reason: {text} (score={score})")
                            found_syzygy_reason = True

                    if not found_syzygy_reason:
                        print("❌ No Syzygy reason found in explanation.")
                else:
                    print("❌ No best move found.")
    except Exception as e:
        print(f"Error during verification: {e}")


if __name__ == "__main__":
    verify_endgames()
