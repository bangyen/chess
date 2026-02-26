"""Tests for explanation generation."""

from unittest.mock import Mock, patch

import chess

from chess_ai.explainable_engine import ExplainableChessEngine


def test_explain_move_basic():
    """Test basic move explanation."""
    engine = ExplainableChessEngine("/path/to/stockfish")
    engine.engine = Mock()
    move = chess.Move.from_uci("e2e4")

    with patch("chess_ai.explainable_engine.sf_eval", return_value=50.0):
        explanation = engine.explain_move(move)

    assert explanation.move == move
    assert isinstance(explanation.overall_explanation, str)


def test_generate_overall_explanation():
    """Test overall explanation formatting."""
    engine = ExplainableChessEngine("/path/to/stockfish")
    move = chess.Move.from_uci("e2e4")
    reasons = [("capture", 100.0, "Captures Piece")]

    explanation = engine._generate_overall_explanation(move, 60.0, reasons)
    assert "Excellent move!" in explanation
    assert "Captures Piece" in explanation


def test_show_best_move(capsys):
    """Test showing best move in console."""
    engine = ExplainableChessEngine("/path/to/stockfish")
    with patch.object(engine, "get_move_recommendation") as mock_rec:
        mock_rec.return_value = Mock(
            move=chess.Move.from_uci("e2e4"), overall_explanation="Good", reasons=[]
        )
        engine._show_best_move()

    captured = capsys.readouterr()
    assert "Best Move" in captured.out
