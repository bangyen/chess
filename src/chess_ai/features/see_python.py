"""Pure-Python Static Exchange Evaluation (SEE).

Provides a Python fallback for SEE when the Rust extension is
unavailable.  SEE simulates alternating least-valuable recaptures on a
square and returns the net material gain, giving far more accurate
tactical assessment than the binary "attacked and undefended" check.
"""

from typing import Optional, Tuple

import chess

_SEE_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Piece types ordered by ascending value for least-valuable-attacker.
_PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def _least_valuable_attacker(board: chess.Board, sq: int, side: bool) -> Optional[int]:
    """Return the square of the least-valuable attacker of *sq*
    belonging to *side*, or ``None`` if there is no attacker.

    Iterates piece types in ascending value order so the first
    hit is always the cheapest attacker -- the standard SEE
    convention.
    """
    for pt in _PIECE_ORDER:
        attackers = board.attackers(side, sq)
        for a_sq in attackers:
            if board.piece_type_at(a_sq) == pt:
                return a_sq
    return None


def _see(board: chess.Board, target: int, attacker_sq: int) -> int:
    """Static Exchange Evaluation for a capture on *target*
    initiated from *attacker_sq*.

    Simulates alternating least-valuable recaptures and returns
    the net material gain (positive = winning, negative = losing)
    from the initial attacker's perspective.

    WARNING: This function mutates *board* in place.  Always call
    via ``see_safe`` which preserves the board state.
    """
    attacker_pt = board.piece_type_at(attacker_sq)
    victim_pt = board.piece_type_at(target)
    if attacker_pt is None or victim_pt is None:
        return 0

    gain = [0] * 33
    depth = 0
    gain[0] = _SEE_PIECE_VALUES.get(victim_pt, 0)
    current_val = _SEE_PIECE_VALUES.get(attacker_pt, 0)
    side = board.color_at(attacker_sq)
    if side is None:
        return 0

    # Temporarily remove pieces to reveal x-ray attacks.
    board.remove_piece_at(attacker_sq)
    board.set_piece_at(target, chess.Piece(attacker_pt, side))

    while True:
        depth += 1
        side = not side
        gain[depth] = current_val - gain[depth - 1]

        if max(-gain[depth - 1], gain[depth]) < 0:
            break

        a_sq = _least_valuable_attacker(board, target, side)
        if a_sq is None:
            break

        pt = board.piece_type_at(a_sq)
        if pt is None:
            break
        current_val = _SEE_PIECE_VALUES.get(pt, 0)
        board.remove_piece_at(a_sq)
        board.set_piece_at(target, chess.Piece(pt, side))

    # Propagate backwards
    while depth > 1:
        depth -= 1
        gain[depth - 1] = -(max(-gain[depth - 1], gain[depth]))

    return gain[0]


def see_safe(board: chess.Board, target: int, attacker_sq: int) -> int:
    """Copy-safe wrapper around SEE that preserves board state.

    Makes a snapshot before the destructive ``_see`` call and
    restores the board afterwards.
    """
    # Use a scratch board to avoid modifying the original state at all
    scratch = board.copy()
    result = _see(scratch, target, attacker_sq)
    return result


def see_features(board: chess.Board, side: bool) -> Tuple[float, float]:
    """Compute SEE advantage and vulnerability for *side*.

    Returns ``(advantage, vulnerability)`` where advantage is the
    sum of positive SEE scores (in pawns) for captures our side
    can initiate, and vulnerability is the count of our pieces
    that the opponent can profitably capture.
    """
    them = not side
    advantage = 0.0
    vulnerability = 0.0

    # Advantage: positive-SEE captures we can make.
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == them and piece.piece_type != chess.KING:
            a_sq = _least_valuable_attacker(board, sq, side)
            if a_sq is not None:
                val = see_safe(board, sq, a_sq)
                if val > 0:
                    advantage += val / 100.0

    # Vulnerability: our pieces the opponent can profitably capture.
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == side and piece.piece_type != chess.KING:
            a_sq = _least_valuable_attacker(board, sq, them)
            if a_sq is not None:
                val = see_safe(board, sq, a_sq)
                if val > 0:
                    vulnerability += 1.0

    return advantage, vulnerability
