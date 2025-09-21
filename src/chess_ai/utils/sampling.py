"""Position sampling utilities."""

import random
import sys
from typing import List

import chess
import chess.pgn

try:
    from tqdm import tqdm
except Exception:
    print(
        "tqdm is required for progress bars. Install with: pip install tqdm",
        file=sys.stderr,
    )
    raise


def sample_positions_from_pgn(
    path: str, max_positions: int, ply_skip: int = 8
) -> List["chess.Board"]:
    """Sample positions from a PGN file.

    Args:
        path: Path to the PGN file
        max_positions: Maximum number of positions to sample
        ply_skip: Skip every Nth ply when sampling

    Returns:
        List of chess board positions
    """
    boards = []
    print(f"Sampling positions from PGN file: {path}")
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            plies = 0
            for move in game.mainline_moves():
                board.push(move)
                plies += 1
                if plies % ply_skip == 0 and not board.is_game_over():
                    boards.append(board.copy(stack=False))
                    if len(boards) >= max_positions:
                        print(f"Sampled {len(boards)} positions from PGN")
                        return boards
    print(f"Sampled {len(boards)} positions from PGN")
    return boards[:max_positions]


def sample_random_positions(n: int, max_random_plies: int = 24) -> List["chess.Board"]:
    """Generate random chess positions.

    Args:
        n: Number of positions to generate
        max_random_plies: Maximum number of random plies to play

    Returns:
        List of chess board positions
    """
    boards = []
    print(f"Generating {n} random positions...")
    for _ in tqdm(range(n), desc="Generating positions"):
        b = chess.Board()
        # play random but legal moves to get middlegame-ish positions
        plies = random.randint(10, max_random_plies)
        for __ in range(plies):
            if b.is_game_over():
                break
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(random.choice(moves))
        if not b.is_game_over():
            boards.append(b.copy(stack=False))
    print(f"Generated {len(boards)} valid positions")
    return boards
