"""Heuristic-based move recommendations for chess AI."""

import chess


def get_heuristic_move(
    board: chess.Board, legal_moves: list[chess.Move]
) -> chess.Move | None:
    """Pick a reasonable move based on simple heuristics."""
    if not legal_moves:
        return None

    fullmove_number = board.fullmove_number

    # Opening: e4/d4
    if fullmove_number == 1:
        for move in legal_moves:
            if move.to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                return move

    # Development: Nf3/Nc3/Nf6/Nc6
    if fullmove_number <= 5:
        for move in legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                from_rank = chess.square_rank(move.from_square)
                if from_rank in [0, 7]:  # Starting ranks
                    return move

    # Center control priority
    for move in legal_moves:
        if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
            return move

    return legal_moves[0]
