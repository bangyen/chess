"""Tests for heuristic move recommendations."""

import chess

from chess_ai.engine.recommendations import get_heuristic_move


def test_get_heuristic_move_opening():
    """Test opening heuristic."""
    board = chess.Board()
    legal_moves = list(board.legal_moves)
    move = get_heuristic_move(board, legal_moves)
    # Should pick e4 or d4
    assert move.to_square in [chess.E4, chess.D4]


def test_get_heuristic_move_development():
    """Test development heuristic."""
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    legal_moves = list(board.legal_moves)
    # Should develop a knight or bishop or take center
    move = get_heuristic_move(board, legal_moves)
    assert move is not None


def test_get_heuristic_move_empty():
    """Test with no legal moves."""
    assert get_heuristic_move(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), []) is None
