"""Piece-Square Tables for positional evaluation.

These tables encode standard positional heuristics (e.g. pawns should
advance, knights belong in the centre, kings should castle) as per-square
bonuses/penalties.  They are used by the Python fallback feature extractor
when the Rust extension is unavailable.
"""

import chess

# fmt: off

# Pawns (incentivize center control and advancement)
PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

# Knights (incentivize center)
PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

# Bishops (incentivize diagonal control key squares)
PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

# Rooks (incentivize 7th rank and center files slightly)
PST_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

# Queen (incentivize center but not too early)
PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

# King (Middlegame: safety)
PST_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

# King (Endgame: activity)
PST_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

# fmt: on


def pst_value(board: "chess.Board", side: bool, phase: float) -> float:
    """Phase-interpolated PST score.

    Uses a continuous phase factor (0.0 = endgame, 1.0 = opening)
    to smoothly blend middlegame and endgame king tables, giving
    more accurate positional scores in transitional positions.
    """
    score = 0.0
    # Continuous phase factor: 14 non-pawn/king pieces = 1.0 (opening).
    pf = min(phase / 14.0, 1.0)

    for pt, table in [
        (chess.PAWN, PST_PAWN),
        (chess.KNIGHT, PST_KNIGHT),
        (chess.BISHOP, PST_BISHOP),
        (chess.ROOK, PST_ROOK),
        (chess.QUEEN, PST_QUEEN),
    ]:
        if table is None:
            continue  # type: ignore[unreachable]
        for sq in board.pieces(pt, side):
            vis_r = (
                7 - chess.square_rank(sq)
                if side == chess.WHITE
                else chess.square_rank(sq)
            )
            vis_c = chess.square_file(sq)
            score += table[vis_r * 8 + vis_c]

    # King: smoothly interpolate between MG and EG tables.
    for sq in board.pieces(chess.KING, side):
        vis_r = (
            7 - chess.square_rank(sq) if side == chess.WHITE else chess.square_rank(sq)
        )
        vis_c = chess.square_file(sq)
        idx = vis_r * 8 + vis_c
        mg = PST_KING_MG[idx]
        eg = PST_KING_EG[idx]
        score += pf * mg + (1.0 - pf) * eg

    return float(score) / 100.0  # Standardize scale
