import time

import chess
import chess.engine

from chess_ai.features.baseline import baseline_extract_features


def profile_probes():
    board = chess.Board()

    try:
        engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    except FileNotFoundError:
        print("Stockfish not found")
        # For Rust-only test, we might get away with None if the wrapper handles it,
        # but the wrapper passes 'engine' to fallback.
        # let's try with None and see if it crashes (it won't if Rust works and we don't hit fallback)
        engine = None

    print("Extracting features (including probes)...")
    start = time.time()
    n = 20
    for _ in range(n):
        feats = baseline_extract_features(board)
        probes = feats.pop("_engine_probes", {})

        # Manually run the probes like audit.py
        if "hanging_after_reply" in probes:
            # depth=6
            probes["hanging_after_reply"](engine, board, depth=6)

        if "best_forcing_swing" in probes:
            # d_base=6
            probes["best_forcing_swing"](engine, board, d_base=6)

    total = time.time() - start
    print(f"{n}x full extraction: {total:.4f}s")
    print(f"Average per position: {total/n*1000:.2f}ms")

    if engine:
        engine.quit()


if __name__ == "__main__":
    profile_probes()
