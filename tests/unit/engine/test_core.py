"""Core tests for explainable chess engine."""

from unittest.mock import Mock, patch

import chess

from chess_ai.explainable_engine import ExplainableChessEngine, MoveExplanation


class TestMoveExplanation:
    """Test MoveExplanation dataclass."""

    def test_move_explanation_creation(self):
        move = chess.Move.from_uci("e2e4")
        reasons = [("development", 2, "Develops pawn")]
        explanation = MoveExplanation(move, 50.0, reasons, "Move e4: Good")
        assert explanation.move == move
        assert explanation.score == 50.0
        assert explanation.reasons == reasons


class TestEngineCore:
    """Test core engine functionality."""

    def test_init(self):
        engine = ExplainableChessEngine("/path/to/stockfish")
        assert engine.stockfish_path == "/path/to/stockfish"
        assert engine.board == chess.Board()

    def test_reset_game(self):
        engine = ExplainableChessEngine("/path/to/stockfish")
        engine.board.push_san("e4")
        engine.reset_game()
        assert engine.board == chess.Board()

    def test_make_move_valid(self):
        engine = ExplainableChessEngine("/path/to/stockfish")
        assert engine.make_move("e4") is True
        assert engine.board.peek() == chess.Move.from_uci("e2e4")

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_context_manager(self, mock_popen):
        mock_engine = Mock()
        mock_popen.return_value = mock_engine
        with ExplainableChessEngine(
            "/path/to/stockfish", enable_model_explanations=False
        ) as e:
            assert e.engine == mock_engine
        mock_engine.quit.assert_called_once()
