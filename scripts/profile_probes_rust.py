import time

import chess
import chess.engine

from chess_ai.features.baseline import RUST_AVAILABLE
from chess_ai.rust_utils import find_best_reply


def profile_rust_vs_python():
    print(f"Rust Available: {RUST_AVAILABLE}")

    board = chess.Board()
    # A position with some complexity
    board.set_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 5")

    # 1. Profile find_best_reply (Rust)
    start = time.time()
    for _ in range(100):
        # Depth 2 is enough for simple tactical features
        find_best_reply(board.fen(), 2)
    rust_time = (time.time() - start) / 100
    print(f"Rust find_best_reply (depth 2): {rust_time*1000:.3f} ms")

    start = time.time()
    for _ in range(20):
        # Depth 4 with move ordering
        find_best_reply(board.fen(), 4)
    rust_time_4 = (time.time() - start) / 20
    print(f"Rust find_best_reply (depth 4): {rust_time_4*1000:.3f} ms")

    start = time.time()
    for _ in range(5):
        # Depth 6 with ID + move ordering
        find_best_reply(board.fen(), 6)
    rust_time_6 = (time.time() - start) / 5
    print(f"Rust find_best_reply (depth 6): {rust_time_6*1000:.3f} ms")

    # 2. Profile Stockfish Overhead (Simulated)
    # We can't easily profile internal baseline function without modifying it,
    # but we can profile a raw engine call.
    try:
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    except Exception:
        print("Stockfish not found, skipping comparison")
        return

    start = time.time()
    for _ in range(10):
        # Minimal depth to measure overhead
        engine.play(board, chess.engine.Limit(depth=1))
    sf_time = (time.time() - start) / 10
    print(f"Stockfish overhead (play depth 1): {sf_time*1000:.3f} ms")

    engine.quit()

    print(f"\nSpeedup: {sf_time / rust_time:.1f}x")


if __name__ == "__main__":
    profile_rust_vs_python()
